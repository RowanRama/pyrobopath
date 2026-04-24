"""Fixed-length resampling and kinematic feature derivation for task trajectories.

This module is purely geometric/numeric; it has no knowledge of robot arms,
base positions, or time offsets. All coordinates remain in world frame.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from pyrobopath.collision_detection.learned.data.gcode_ingest import RawTask

# -----------------------------------------------------------------------
# Feature index map (must match DataConfig.num_features = 11)
# -----------------------------------------------------------------------
FEATURE_NAMES = [
    "x_world",          # 0  world-frame x position
    "y_world",          # 1  world-frame y position
    "t_abs",            # 2  absolute time from task start (seconds)
    "t_norm",           # 3  normalised time in [0, 1]
    "vx",               # 4  finite-diff x velocity (m/s)
    "vy",               # 5  finite-diff y velocity (m/s)
    "speed",            # 6  magnitude of velocity
    "heading",          # 7  atan2(vy, vx) in [-pi, pi]
    "curvature",        # 8  estimated signed curvature
    "dx",               # 9  x displacement from previous point
    "dy",               # 10 y displacement from previous point
]
NUM_FEATURES = len(FEATURE_NAMES)  # 11


def resample_polyline(
    polyline: list[tuple[float, float, float]],
    L: int = 128,
) -> np.ndarray:
    """Resample a (x, y, t) polyline to exactly L fixed-timestep points.

    Resampling is done in time: L equally-spaced time points are generated
    from t=0 to t=duration, and (x, y) are linearly interpolated.

    Args:
        polyline: List of (x, y, t) tuples, where t is seconds from task start.
            Must have at least 2 points.
        L: Target number of points.

    Returns:
        Array of shape (L, 3) with columns [x, y, t].
    """
    if len(polyline) < 2:
        # Degenerate: repeat the single point
        if len(polyline) == 0:
            return np.zeros((L, 3), dtype=np.float32)
        pt = polyline[0]
        arr = np.array([[pt[0], pt[1], pt[2]]] * L, dtype=np.float32)
        arr[:, 2] = np.linspace(0.0, max(pt[2], 1e-6), L)
        return arr

    xs = np.array([p[0] for p in polyline], dtype=np.float64)
    ys = np.array([p[1] for p in polyline], dtype=np.float64)
    ts = np.array([p[2] for p in polyline], dtype=np.float64)

    # Ensure monotone t
    for i in range(1, len(ts)):
        if ts[i] <= ts[i - 1]:
            ts[i] = ts[i - 1] + 1e-8

    t_new = np.linspace(ts[0], ts[-1], L)
    x_new = np.interp(t_new, ts, xs)
    y_new = np.interp(t_new, ts, ys)

    return np.stack([x_new, y_new, t_new], axis=-1).astype(np.float32)


def derive_features(xyt: np.ndarray) -> np.ndarray:
    """Compute kinematic derived features from a resampled (L, 3) array.

    Args:
        xyt: Array of shape (L, 3) with columns [x, y, t].

    Returns:
        Feature array of shape (L, NUM_FEATURES=11).
    """
    L = xyt.shape[0]
    x = xyt[:, 0]
    y = xyt[:, 1]
    t = xyt[:, 2]
    duration = t[-1] - t[0]

    # Normalised time
    t_norm = (t - t[0]) / (duration + 1e-8)

    # Finite differences
    dx = np.diff(x, prepend=x[0:1])
    dy = np.diff(y, prepend=y[0:1])
    dt = np.diff(t, prepend=t[0:1])
    dt = np.where(np.abs(dt) < 1e-8, 1e-8, dt)

    vx = dx / dt
    vy = dy / dt
    speed = np.sqrt(vx**2 + vy**2)
    heading = np.arctan2(vy, vx)

    # Curvature: d(heading)/ds ≈ d(heading)/dt / speed
    dheading = np.diff(heading, prepend=heading[0:1])
    # Unwrap phase jumps
    dheading = np.where(dheading > math.pi, dheading - 2 * math.pi, dheading)
    dheading = np.where(dheading < -math.pi, dheading + 2 * math.pi, dheading)
    curvature = np.where(speed > 1e-6, dheading / (dt * speed + 1e-10), 0.0)

    feats = np.stack(
        [x, y, t, t_norm, vx, vy, speed, heading, curvature, dx, dy], axis=-1
    )
    return feats.astype(np.float32)


def task_to_tensor(task: RawTask, L: int = 128) -> torch.Tensor:
    """Convert a :class:`RawTask` to a feature tensor of shape (L, NUM_FEATURES).

    All operations remain world-frame; no robot information is used here.

    Args:
        task: The raw task to encode.
        L: Target sequence length.

    Returns:
        Float32 tensor of shape (L, NUM_FEATURES).
    """
    xyt = resample_polyline(task.polyline, L)
    feats = derive_features(xyt)
    return torch.from_numpy(feats)


def compute_normalization_stats(
    tensors: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean and std from a list of feature tensors.

    Args:
        tensors: List of tensors, each shape (L, F).

    Returns:
        Tuple (mean, std), each shape (F,).
    """
    stacked = torch.stack(tensors, dim=0)  # (N, L, F)
    flat = stacked.reshape(-1, stacked.shape[-1])
    mean = flat.mean(dim=0)
    std = flat.std(dim=0).clamp(min=1e-6)
    return mean, std


class FeatureNormalizer:
    """Standardise feature tensors using pre-computed statistics.

    Args:
        mean: Per-feature mean, shape (F,).
        std: Per-feature std, shape (F,).
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.mean = mean
        self.std = std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Standardise a feature tensor.

        Args:
            x: Tensor of shape (..., F).

        Returns:
            Normalised tensor of same shape.
        """
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Reverse standardisation.

        Args:
            x: Normalised tensor of shape (..., F).

        Returns:
            Denormalised tensor.
        """
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def state_dict(self) -> dict:
        """Return serialisable state.

        Returns:
            Dict with 'mean' and 'std' tensors.
        """
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d: dict) -> "FeatureNormalizer":
        """Restore from a state dict.

        Args:
            d: Dict with 'mean' and 'std' keys.

        Returns:
            :class:`FeatureNormalizer` instance.
        """
        return cls(d["mean"], d["std"])
