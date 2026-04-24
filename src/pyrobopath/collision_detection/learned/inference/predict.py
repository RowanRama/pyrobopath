"""Pair prediction API for planner integration.

Provides:
- encode_task: Cache-aware Stage 1 encoding.
- predict_pair: Stage 2 clearance and collision prediction from embeddings.

The optional ``geometry_cache`` enables a three-layer geometric pre-filter
that gates pairs *before* Stage 1/2 are invoked:

  Layer 0 — MVEE algebraic separation  (O(1), closes-form, 2D)
  Layer 1 — Bhattacharyya coefficient  (O(1), Gaussian overlap)
  Layer 2 — Probabilistic chi² overlap (O(1), non-central χ²)

If any layer declares a pair safe, the neural network is never called.
When the heuristic filter is active, ``predict_pair`` also returns
``heuristic_pruned=True`` to allow downstream comparison of approaches.

Planner integration pattern:
  1. build_geometry_cache(tasks, ...)
  2. predictor = PairPredictor(..., geometry_cache=geo_cache, filter_mode='cascade')
  3. For each pair:
     c, p, pruned, how = predictor.predict_pair(...)
     if pruned: skip (heuristic says safe)
     elif c > tau and p < alpha: skip (neural net says safe)
     else: run exact checker
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from pyrobopath.collision_detection.learned.data.gcode_ingest import RawTask
from pyrobopath.collision_detection.learned.data.labeler import ArmParams
from pyrobopath.collision_detection.learned.data.resample import task_to_tensor
from pyrobopath.collision_detection.ellipsoid_filter import (
    EllipsoidRecord,
    GeometryCache,
    EllipsoidSeparationFilter,
    BhattacharyyaFilter,
    ProbabilisticOverlapFilter,
    CascadeFilter,
)
from pyrobopath.collision_detection.learned.inference.encode_cache import EmbeddingCache
from pyrobopath.collision_detection.learned.models.full_model import FullModel
from pyrobopath.collision_detection.learned.utils.config import Config

logger = logging.getLogger(__name__)

_FILTER_CLASSES = {
    "mvee_separation": EllipsoidSeparationFilter,
    "bhattacharyya":   BhattacharyyaFilter,
    "prob_overlap":    ProbabilisticOverlapFilter,
    "cascade":         CascadeFilter,
}


class PairPredictor:
    """High-level inference API for pairwise collision surrogate queries.

    Args:
        model: Trained :class:`FullModel`.
        cfg: Full configuration.
        device: Inference device.
        embedding_cache: Optional pre-computed embedding cache for fast lookup.
        tau: Clearance threshold for safe pre-filter decision (neural net layer).
        alpha: Probability threshold — prune when p_collision < alpha.
        geometry_cache: Optional :class:`GeometryCache` for geometric pre-filter.
        filter_mode: Which geometric filter to apply before Stage 2.
            One of: 'mvee_separation', 'bhattacharyya', 'prob_overlap', 'cascade'.
            Set to None to disable geometric pre-filter entirely.
        filter_kwargs: Extra keyword arguments forwarded to the chosen filter
            constructor (e.g. rho_threshold, p_threshold).
    """

    def __init__(
        self,
        model: FullModel,
        cfg: Config,
        device: torch.device,
        embedding_cache: Optional[EmbeddingCache] = None,
        tau: float = 0.03,
        alpha: float = 0.1,
        geometry_cache: Optional[GeometryCache] = None,
        filter_mode: Optional[str] = "cascade",
        filter_kwargs: Optional[dict] = None,
    ) -> None:
        self._model = model.eval().to(device)
        self._cfg = cfg
        self._device = device
        self._cache = embedding_cache
        self._tau = tau
        self._alpha = alpha
        self._L = cfg.model.encoder.seq_len

        # Geometric pre-filter setup
        self._geo_cache = geometry_cache
        self._geo_filter = None
        if geometry_cache is not None and filter_mode is not None:
            if filter_mode not in _FILTER_CLASSES:
                raise ValueError(
                    f"Unknown filter_mode '{filter_mode}'. "
                    f"Choose from {list(_FILTER_CLASSES.keys())}."
                )
            kwargs = filter_kwargs or {}
            self._geo_filter = _FILTER_CLASSES[filter_mode](geometry_cache, **kwargs)
            logger.info("Geometric pre-filter enabled: %s", filter_mode)

    def _get_geo_record(self, task: RawTask) -> Optional[EllipsoidRecord]:
        """Retrieve or compute EllipsoidRecord for a task."""
        if self._geo_cache is None:
            return None
        rec = self._geo_cache.get(task.gcode_id, task.task_id)
        if rec is None:
            # Compute on the fly and cache
            rec = EllipsoidRecord.from_polyline(task.gcode_id, task.task_id, task.polyline)
            self._geo_cache.put(rec)
        return rec

    @torch.no_grad()
    def encode_task(self, task: RawTask) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a single task to a sequence embedding (Stage 1).

        Uses the embedding cache when available. The returned embedding is
        robot-agnostic; reassigning arms does not invalidate it.

        Args:
            task: The task to encode.

        Returns:
            Tuple (embedding, feat_tensor):
            - embedding: shape (L', D).
            - feat_tensor: shape (L, F) for base-relative enrichment in Stage 2.
        """
        feat = task_to_tensor(task, self._L)  # (L, F)

        if self._cache is not None:
            cached = self._cache.get(task.gcode_id, task.task_id)
            if cached is not None:
                return cached.to(self._device), feat.to(self._device)

        feat_b = feat.unsqueeze(0).to(self._device)  # (1, L, F)
        use_amp = self._cfg.training.use_amp and torch.cuda.is_available()
        with torch.cuda.amp.autocast(enabled=use_amp):
            emb = self._model.encode_task(feat_b)[0]  # (L', D)

        if self._cache is not None:
            self._cache.put(task.gcode_id, task.task_id, emb.cpu())

        return emb, feat.to(self._device)

    @torch.no_grad()
    def predict_pair(
        self,
        task_a: RawTask,
        task_b: RawTask,
        base_a: tuple[float, float],
        base_b: tuple[float, float],
        dt: float,
        arm_params_a: ArmParams,
        arm_params_b: ArmParams,
    ) -> tuple[float, float, bool, str]:
        """Predict clearance and collision probability for a task pair.

        Runs geometric pre-filter first (if configured), then the neural net.

        Args:
            task_a: Task for arm A.
            task_b: Task for arm B.
            base_a: World-frame base of arm A.
            base_b: World-frame base of arm B.
            dt: Time offset of task B relative to A (seconds).
            arm_params_a: Arm A parameters.
            arm_params_b: Arm B parameters.

        Returns:
            Tuple (clearance_m, p_collision, is_safe_to_prune, pruned_by):
            - clearance_m: Predicted signed clearance (m).  NaN if heuristic-pruned.
            - p_collision: Predicted collision probability.  NaN if heuristic-pruned.
            - is_safe_to_prune: True if exact checker can be skipped.
            - pruned_by: One of 'mvee_separation', 'bhattacharyya', 'prob_overlap',
                         'neural_net', or 'none' (must run exact checker).
        """
        # ── Layer 0: geometric pre-filter ─────────────────────────────────
        if self._geo_filter is not None:
            rec_a = self._get_geo_record(task_a)
            rec_b = self._get_geo_record(task_b)
            if rec_a is not None and rec_b is not None:
                ba = np.array(base_a, dtype=np.float32)
                bb = np.array(base_b, dtype=np.float32)
                result = self._geo_filter.is_safe(rec_a, rec_b, base_a=ba, base_b=bb)
                # Cascade returns (safe, pruned_by, scores); single filters return (safe, scores_dict)
                if isinstance(result[0], bool) and len(result) == 3 and isinstance(result[2], dict):
                    geo_safe, geo_pruned_by, _ = result
                else:
                    geo_safe, _ = result
                    geo_pruned_by = self._geo_filter.name if geo_safe else "none"

                if geo_safe:
                    return float("nan"), float("nan"), True, geo_pruned_by

        # ── Layer 1: Stage 1 encode ────────────────────────────────────────
        emb_a, feat_a = self.encode_task(task_a)
        emb_b, feat_b = self.encode_task(task_b)

        # ── Layer 2: Stage 2 pairwise head ────────────────────────────────
        dev = self._device
        emb_a_b   = emb_a.unsqueeze(0)
        emb_b_b   = emb_b.unsqueeze(0)
        feat_a_b  = feat_a.unsqueeze(0)
        feat_b_b  = feat_b.unsqueeze(0)
        base_a_t  = torch.tensor(base_a, dtype=torch.float32, device=dev).unsqueeze(0)
        base_b_t  = torch.tensor(base_b, dtype=torch.float32, device=dev).unsqueeze(0)
        dt_t      = torch.tensor([dt], dtype=torch.float32, device=dev)
        arm_a_t   = torch.from_numpy(arm_params_a.to_array()).unsqueeze(0).to(dev)
        arm_b_t   = torch.from_numpy(arm_params_b.to_array()).unsqueeze(0).to(dev)

        use_amp = self._cfg.training.use_amp and torch.cuda.is_available()
        with torch.cuda.amp.autocast(enabled=use_amp):
            clearance, p_col = self._model.predict_from_embeddings(
                emb_a_b, emb_b_b, feat_a_b, feat_b_b,
                base_a_t, base_b_t, dt_t, arm_a_t, arm_b_t,
            )

        c = float(clearance[0])
        p = float(p_col[0])
        safe_to_prune = (c > self._tau) and (p < self._alpha)
        pruned_by = "neural_net" if safe_to_prune else "none"
        return c, p, safe_to_prune, pruned_by

    def predict_pair_batch(
        self,
        inputs: list[tuple],
    ) -> list[tuple[float, float, bool, str]]:
        """Batch predict for a list of pair inputs.

        Args:
            inputs: List of 7-tuples (task_a, task_b, base_a, base_b, dt, arm_a, arm_b).

        Returns:
            List of (clearance, p_collision, is_safe_to_prune, pruned_by) tuples.
        """
        return [self.predict_pair(*inp) for inp in inputs]
