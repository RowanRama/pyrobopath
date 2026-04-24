"""Labeler: wraps user-provided collision checker and optional distance oracle.

The user must supply::

    check_collision(task_A, task_B, base_A, base_B, dt,
                    arm_params_A, arm_params_B) -> bool

Optionally::

    min_clearance(task_A, task_B, base_A, base_B, dt,
                  arm_params_A, arm_params_B) -> float  (signed metres)

If ``min_clearance`` is absent, the reference swept-box implementation below is
used as a placeholder. It is marked TODO so the user can replace it.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from pyrobopath.collision_detection.learned.data.gcode_ingest import RawTask

logger = logging.getLogger(__name__)


@dataclass
class ArmParams:
    """Physical parameters of one robot arm's swept box.

    Attributes:
        width: Box half-width (metres).
        safety_margin: Inflation margin for conservative checking (metres).
        min_length: Minimum arm reach (metres).
        max_length: Maximum arm reach (metres).
    """

    width: float
    safety_margin: float
    min_length: float
    max_length: float

    def to_array(self) -> np.ndarray:
        """Serialise to a 4-element numpy array.

        Returns:
            Float32 array [width, safety_margin, min_length, max_length].
        """
        return np.array(
            [self.width, self.safety_margin, self.min_length, self.max_length],
            dtype=np.float32,
        )

    def param_hash(self) -> str:
        """Return a short hash for cache-key construction.

        Returns:
            8-character hex string.
        """
        return hashlib.md5(json.dumps(self.to_array().tolist()).encode()).hexdigest()[:8]


@dataclass
class LabelRecord:
    """A labelled pair record for training.

    Attributes:
        task_a_id: Composite key (gcode_id, task_id) for task A.
        task_b_id: Composite key for task B.
        base_a: World-frame base position of arm A as (x, y) tuple.
        base_b: World-frame base position of arm B.
        dt: Time offset of task B relative to task A (seconds).
        arm_params_a: Physical parameters of arm A.
        arm_params_b: Physical parameters of arm B.
        collision: Binary collision label.
        clearance: Signed minimum clearance (metres); None if oracle unavailable.
    """

    task_a_id: tuple[str, str]
    task_b_id: tuple[str, str]
    base_a: tuple[float, float]
    base_b: tuple[float, float]
    dt: float
    arm_params_a: ArmParams
    arm_params_b: ArmParams
    collision: bool
    clearance: Optional[float] = None

    def cache_key(self) -> str:
        """Deterministic cache key for deduplication.

        Returns:
            MD5 hex string.
        """
        parts = [
            str(self.task_a_id),
            str(self.task_b_id),
            json.dumps(self.base_a),
            json.dumps(self.base_b),
            f"{self.dt:.6f}",
            self.arm_params_a.param_hash(),
            self.arm_params_b.param_hash(),
        ]
        return hashlib.md5("|".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Reference swept-box clearance (TODO: replace with user's distance oracle)
# ---------------------------------------------------------------------------

def _swept_box_clearance_reference(
    task_a: RawTask,
    task_b: RawTask,
    base_a: tuple[float, float],
    base_b: tuple[float, float],
    dt: float,
    arm_params_a: ArmParams,
    arm_params_b: ArmParams,
    n_time_samples: int = 100,
) -> float:
    """Reference implementation: minimum signed clearance via swept box geometry.

    TODO: Replace this with the user-supplied distance oracle for accurate
    clearance values.  This placeholder computes a conservative approximation
    using rectangular swept boxes at each sampled time step.

    Args:
        task_a: Task for arm A.
        task_b: Task for arm B (shifted by dt).
        base_a: World-frame base of arm A.
        base_b: World-frame base of arm B.
        dt: Time offset of task B start relative to task A start.
        arm_params_a: Arm A physical parameters.
        arm_params_b: Arm B physical parameters.
        n_time_samples: Number of time samples for approximation.

    Returns:
        Signed minimum clearance; negative means overlap.
    """
    poly_a = np.array(task_a.polyline, dtype=np.float64)  # (N, 3)
    poly_b = np.array(task_b.polyline, dtype=np.float64)  # (M, 3)

    dur_a = task_a.duration
    dur_b = task_b.duration

    def interp_pos(poly: np.ndarray, t_query: float) -> np.ndarray:
        t_arr = poly[:, 2]
        x_arr = poly[:, 0]
        y_arr = poly[:, 1]
        t_q = np.clip(t_query, t_arr[0], t_arr[-1])
        x = np.interp(t_q, t_arr, x_arr)
        y = np.interp(t_q, t_arr, y_arr)
        return np.array([x, y])

    def arm_half_width(params: ArmParams) -> float:
        return params.width / 2 + params.safety_margin

    t_start = 0.0
    t_end = max(dur_a, dur_b + dt)
    ts = np.linspace(t_start, t_end, n_time_samples)

    min_dist = float("inf")
    for t in ts:
        t_a = t
        t_b = t - dt

        in_a = 0.0 <= t_a <= dur_a
        in_b = 0.0 <= t_b <= dur_b

        if not (in_a and in_b):
            continue

        pos_a = interp_pos(poly_a, t_a)
        pos_b = interp_pos(poly_b, t_b)

        # Approximate clearance between centres minus sum of half-widths
        hw_a = arm_half_width(arm_params_a)
        hw_b = arm_half_width(arm_params_b)
        centre_dist = float(np.linalg.norm(pos_a - pos_b))
        clearance = centre_dist - hw_a - hw_b
        if clearance < min_dist:
            min_dist = clearance

    return min_dist if math.isfinite(min_dist) else 1.0


# ---------------------------------------------------------------------------
# Labeler class
# ---------------------------------------------------------------------------

class Labeler:
    """Wraps the user's collision checker and optional clearance oracle.

    Args:
        check_collision_fn: User-provided binary collision checker.
        min_clearance_fn: User-provided signed clearance oracle. If None,
            the reference swept-box implementation is used as a placeholder.
        cache_dir: Directory for persisting label cache.
        num_workers: Worker processes for parallel labeling.
    """

    def __init__(
        self,
        check_collision_fn: Callable[..., bool],
        min_clearance_fn: Optional[Callable[..., float]] = None,
        *,
        cache_dir: Optional[str | Path] = None,
        num_workers: int = 4,
    ) -> None:
        self._check_fn = check_collision_fn
        self._clearance_fn = min_clearance_fn or _swept_box_clearance_reference
        self._has_oracle = min_clearance_fn is not None
        self._num_workers = num_workers
        self._cache: dict[str, LabelRecord] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()

    # ------------------------------------------------------------------ cache
    def _cache_path(self) -> Path:
        assert self._cache_dir is not None
        return self._cache_dir / "label_cache.jsonl"

    def _load_cache(self) -> None:
        p = self._cache_path()
        if not p.exists():
            return
        count = 0
        with p.open() as f:
            for line in f:
                rec = json.loads(line.strip())
                key = rec.pop("_key")
                self._cache[key] = rec
                count += 1
        logger.info("Loaded %d cached labels from %s", count, p)

    def _append_cache(self, key: str, rec: dict) -> None:
        if not self._cache_dir:
            return
        with self._cache_path().open("a") as f:
            json.dump({"_key": key, **rec}, f)
            f.write("\n")

    # ------------------------------------------------------------------ label
    def label_pair(
        self,
        task_a: RawTask,
        task_b: RawTask,
        base_a: tuple[float, float],
        base_b: tuple[float, float],
        dt: float,
        arm_params_a: ArmParams,
        arm_params_b: ArmParams,
    ) -> LabelRecord:
        """Label a single pair.

        Args:
            task_a: Task for arm A.
            task_b: Task for arm B.
            base_a: World-frame base position of arm A.
            base_b: World-frame base position of arm B.
            dt: Time offset of task B relative to task A.
            arm_params_a: Physical params of arm A.
            arm_params_b: Physical params of arm B.

        Returns:
            :class:`LabelRecord` with binary and (if available) clearance labels.
        """
        rec = LabelRecord(
            task_a_id=(task_a.gcode_id, task_a.task_id),
            task_b_id=(task_b.gcode_id, task_b.task_id),
            base_a=base_a,
            base_b=base_b,
            dt=dt,
            arm_params_a=arm_params_a,
            arm_params_b=arm_params_b,
            collision=False,
        )
        key = rec.cache_key()
        if key in self._cache:
            cached = self._cache[key]
            return LabelRecord(
                task_a_id=(task_a.gcode_id, task_a.task_id),
                task_b_id=(task_b.gcode_id, task_b.task_id),
                base_a=base_a,
                base_b=base_b,
                dt=dt,
                arm_params_a=arm_params_a,
                arm_params_b=arm_params_b,
                collision=bool(cached["collision"]),
                clearance=cached.get("clearance"),
            )

        collision = bool(
            self._check_fn(task_a, task_b, base_a, base_b, dt, arm_params_a, arm_params_b)
        )
        clearance = float(
            self._clearance_fn(task_a, task_b, base_a, base_b, dt, arm_params_a, arm_params_b)
        )

        rec.collision = collision
        rec.clearance = clearance

        cache_val = {"collision": collision, "clearance": clearance}
        self._cache[key] = cache_val
        self._append_cache(key, cache_val)

        return rec

    def label_batch(
        self,
        pairs: list[tuple[RawTask, RawTask, tuple, tuple, float, ArmParams, ArmParams]],
    ) -> list[LabelRecord]:
        """Label a list of pairs (sequentially; parallel pool optional).

        Args:
            pairs: List of 7-tuples (task_a, task_b, base_a, base_b, dt, params_a, params_b).

        Returns:
            List of :class:`LabelRecord` objects in the same order.
        """
        return [self.label_pair(*p) for p in pairs]

    @property
    def has_clearance_oracle(self) -> bool:
        """True if a real distance oracle was supplied.

        Returns:
            Boolean.
        """
        return self._has_oracle
