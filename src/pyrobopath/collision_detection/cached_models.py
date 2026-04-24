from __future__ import annotations
from abc import abstractmethod
from typing import Any, Dict, Optional
from typing import ClassVar

import numpy as np

from .collision_model import CollisionModel
from .ellipsoid_filter import (
    EllipsoidRecord,
    EllipsoidSeparationFilter,
    BhattacharyyaFilter,
    CascadeFilter,
)


class CachedCollisionModel(CollisionModel):
    """Abstract collision model that caches per-contour geometry for a fast
    pairwise safety prefilter.

    Subclasses implement a specific filter (MVEE, Bhattacharyya, Cascade).
    The prefilter is conservative: it only reports a pair as safe when it is
    certain. If a contour id is not in the cache, the pair is reported as
    unsafe (falls back to the full FCL check).

    All instances of the same concrete subclass share a single class-level
    cache keyed by `type(self)`. This avoids rebuilding identical entries
    when multiple agents share the same filter type.
    """

    # Shared across ALL instances of the same concrete subclass
    _shared_caches: ClassVar[Dict[type, Dict[int, Any]]] = {}

    def __init__(self, safety_m: float = 0.0):
        super().__init__()
        self.safety_m = float(safety_m)
        cls = type(self)
        if cls not in CachedCollisionModel._shared_caches:
            CachedCollisionModel._shared_caches[cls] = {}

    @property
    def _cache(self) -> Dict[int, Any]:
        cls = type(self)
        if cls not in CachedCollisionModel._shared_caches:
            CachedCollisionModel._shared_caches[cls] = {}
        return CachedCollisionModel._shared_caches[cls]

    @classmethod
    def clear_cache(cls, target_cls=None) -> None:
        """Clear the shared cache for target_cls (or all classes if None).

        Call this between planning runs if contours change.
        """
        if target_cls is None:
            CachedCollisionModel._shared_caches.clear()
        else:
            CachedCollisionModel._shared_caches.pop(target_cls, None)

    def build_cache(self, contours) -> None:
        for contour in contours:
            if contour.id not in self._cache:
                self._cache[contour.id] = self._build_entry(contour)

    def is_safe_pair(self, id_a: int, id_b: int, base_a, base_b) -> bool:
        if id_a not in self._cache or id_b not in self._cache:
            return False
        return self._is_safe_cached(
            self._cache[id_a], self._cache[id_b], base_a, base_b
        )

    @abstractmethod
    def _build_entry(self, contour) -> Any:
        raise NotImplementedError

    @abstractmethod
    def _is_safe_cached(self, entry_a, entry_b, base_a, base_b) -> bool:
        raise NotImplementedError

    def in_collision(self, other: CollisionModel) -> bool:
        return False


def _contour_to_polyline(contour):
    return [(float(p[0]), float(p[1]), 0.0) for p in contour.path]


def _base_xy(base):
    b = np.asarray(base, dtype=np.float64).ravel()
    return b[:2]


class MVEECachedModel(CachedCollisionModel):
    """Cached collision prefilter using EllipsoidSeparationFilter (MVEE)."""

    def __init__(self, safety_m: float = 0.0):
        super().__init__(safety_m=safety_m)
        self._filter = EllipsoidSeparationFilter(None, safety_m=self.safety_m)

    def _build_entry(self, contour) -> EllipsoidRecord:
        return EllipsoidRecord.from_polyline(
            gcode_id="pyrobopath",
            task_id=str(contour.id),
            polyline=_contour_to_polyline(contour),
        )

    def _is_safe_cached(self, entry_a, entry_b, base_a, base_b) -> bool:
        safe, _ = self._filter.is_safe(
            entry_a, entry_b, _base_xy(base_a), _base_xy(base_b), self.safety_m
        )
        return bool(safe)


class BhattacharyyaCachedModel(CachedCollisionModel):
    """Cached collision prefilter using BhattacharyyaFilter."""

    def __init__(self, safety_m: float = 0.0, rho_threshold: float = 0.05):
        super().__init__(safety_m=safety_m)
        self._filter = BhattacharyyaFilter(
            None, rho_threshold=rho_threshold, safety_m=self.safety_m
        )

    def _build_entry(self, contour) -> EllipsoidRecord:
        return EllipsoidRecord.from_polyline(
            gcode_id="pyrobopath",
            task_id=str(contour.id),
            polyline=_contour_to_polyline(contour),
        )

    def _is_safe_cached(self, entry_a, entry_b, base_a, base_b) -> bool:
        safe, _ = self._filter.is_safe(
            entry_a, entry_b, _base_xy(base_a), _base_xy(base_b), self.safety_m
        )
        return bool(safe)


class CascadeCachedModel(CachedCollisionModel):
    """Cached collision prefilter using CascadeFilter."""

    def __init__(
        self,
        safety_m: float = 0.0,
        rho_threshold: float = 0.05,
        p_threshold: float = 0.02,
        n_sigma: float = 2.0,
    ):
        super().__init__(safety_m=safety_m)
        self._filter = CascadeFilter(
            None,
            rho_threshold=rho_threshold,
            p_threshold=p_threshold,
            n_sigma=n_sigma,
            safety_m=self.safety_m,
        )

    def _build_entry(self, contour) -> EllipsoidRecord:
        return EllipsoidRecord.from_polyline(
            gcode_id="pyrobopath",
            task_id=str(contour.id),
            polyline=_contour_to_polyline(contour),
        )

    def _is_safe_cached(self, entry_a, entry_b, base_a, base_b) -> bool:
        safe, _, _ = self._filter.is_safe(
            entry_a, entry_b, _base_xy(base_a), _base_xy(base_b), self.safety_m
        )
        return bool(safe)


def _contour_to_raw_task(contour):
    """Convert a pyrobopath Contour into the RawTask format used by the
    learned surrogate. Uses only (x, y) from 3D paths, with index as a
    synthetic timestamp (Contour.path has no timestamps)."""
    from pyrobopath.collision_detection.learned.data.gcode_ingest import RawTask

    polyline = [
        (float(p[0]), float(p[1]), float(i)) for i, p in enumerate(contour.path)
    ]
    duration = float(max(len(contour.path) - 1, 1))
    return RawTask(
        gcode_id="pyrobopath",
        task_id=str(contour.id),
        polyline=polyline,
        duration=duration,
    )


class LearnedCachedModel(CachedCollisionModel):
    """Cached collision prefilter using the two-stage neural network surrogate.

    Stage 1 (task encoder) embeddings are cached per contour — the same
    robot-agnostic caching strategy as used in PairPredictor.EmbeddingCache.
    Stage 2 (pairwise head) is called at query time with the two embeddings
    and the robot base positions.

    Args:
        model_path: Path to a saved FullModel checkpoint (.pt file).
        cfg: Full Config object (must match the checkpoint).
        device: Torch device to run inference on.
        tau: Clearance threshold — predict safe when clearance > tau.
        alpha: Probability threshold — predict safe when p_collision < alpha.
        safety_m: Additional geometric safety margin (kept for interface
                  compatibility; not used by the neural model directly).
        arm_params: ArmParams instance used for both agents as a placeholder
                    at prefilter level (no arm model in the pyrobopath side).
    """

    def __init__(
        self,
        model_path: str,
        cfg,
        device=None,
        tau: float = 0.03,
        alpha: float = 0.1,
        safety_m: float = 0.0,
        arm_params: Optional[Any] = None,
    ):
        super().__init__(safety_m=safety_m)
        import torch
        import tempfile
        from pyrobopath.collision_detection.learned.models.full_model import FullModel
        from pyrobopath.collision_detection.learned.inference.predict import PairPredictor
        from pyrobopath.collision_detection.learned.inference.encode_cache import (
            EmbeddingCache,
        )
        from pyrobopath.collision_detection.learned.data.labeler import ArmParams

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._cfg = cfg

        model = FullModel.from_config(cfg)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)

        tmp_cache_path = tempfile.mktemp(suffix=".pt")
        emb_cache = EmbeddingCache(tmp_cache_path)

        self._predictor = PairPredictor(
            model=model,
            cfg=cfg,
            device=device,
            embedding_cache=emb_cache,
            tau=tau,
            alpha=alpha,
            geometry_cache=None,
            filter_mode=None,
        )
        self._emb_cache = emb_cache
        self._tmp_cache_path = tmp_cache_path
        self._tau = tau
        self._alpha = alpha

        if arm_params is None:
            arm_params = ArmParams(
                width=0.1, safety_margin=0.02, min_length=0.2, max_length=0.8
            )
        self._arm_params = arm_params

    def _build_entry(self, contour):
        """Pre-compute Stage 1 embedding and store it in the predictor's
        EmbeddingCache. The shared class-level cache stores only the
        contour.id (an int) as a presence sentinel — the real per-task
        data lives entirely inside self._predictor's EmbeddingCache,
        keyed by (gcode_id="pyrobopath", task_id=str(contour.id)).
        The RawTask is reconstructed from that key at query time, so
        nothing heavyweight is stored here.
        """
        raw_task = _contour_to_raw_task(contour)
        self._predictor.encode_task(raw_task)  # populates _predictor._cache
        return contour.id  # sentinel: just the int id

    def _is_safe_cached(self, id_a: int, id_b: int, base_a, base_b) -> bool:
        """Reconstruct minimal RawTask keys and run Stage 2 via the predictor.

        The Stage 1 embeddings are already in self._predictor._cache from
        build_cache; predict_pair retrieves them by (gcode_id, task_id).
        """
        from pyrobopath.collision_detection.learned.data.gcode_ingest import RawTask
        # Reconstruct minimal RawTask with just enough info for the embedding
        # cache lookup — polyline and duration are not used by predict_pair
        # when the embedding is already cached.
        raw_a = RawTask(
            gcode_id="pyrobopath", task_id=str(id_a), polyline=[], duration=0.0
        )
        raw_b = RawTask(
            gcode_id="pyrobopath", task_id=str(id_b), polyline=[], duration=0.0
        )
        ba = tuple(float(x) for x in _base_xy(base_a).tolist())
        bb = tuple(float(x) for x in _base_xy(base_b).tolist())
        _c, _p, safe, _pruned_by = self._predictor.predict_pair(
            raw_a,
            raw_b,
            ba,
            bb,
            dt=0.0,
            arm_params_a=self._arm_params,
            arm_params_b=self._arm_params,
        )
        return bool(safe)

    def __del__(self):
        import os
        try:
            if hasattr(self, "_tmp_cache_path") and os.path.exists(
                self._tmp_cache_path
            ):
                os.remove(self._tmp_cache_path)
        except Exception:
            pass
