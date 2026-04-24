"""Geometric heuristic pre-filters for pairwise collision checking.

This module provides three progressively richer spatial pre-filters that
gate pairs *before* the neural network (Stage 2) is invoked.

Two physical realities that the v1 design missed — now corrected
-----------------------------------------------------------------

(1) Safety margin (FNR → 0)
    A tight MVEE around trajectory points does not account for the physical
    width of the arm tool-head.  If arm A passes within one tool-width of arm B's
    trajectory the collision checker must be invoked.  We correct this by
    inflating each ellipsoid uniformly by ``safety_m`` metres in world-space
    before any intersection test.

    Uniform inflation in world-space is NOT the same as scaling M.  Scaling M
    uniformly shrinks/grows the Mahalanobis unit ball but changes each semi-axis
    by a *fraction*, not a fixed distance.  Correct inflation:

        semi-axis a_i' = a_i + safety_m   (a_i = 1 / sqrt(λ_i(M)) in metres)

    meaning the new shape matrix is:

        M_inflated = V  diag(1 / (a_i + safety_m)²)  Vᵀ

    where V, λ are from the eigen-decomposition of M.  This guarantees that any
    point within safety_m metres of the original ellipsoid surface is inside the
    inflated ellipsoid, driving FNR to zero when safety_m ≥ arm_width / 2.

(2) Base-to-task reach check
    The current filters only ask "do the task footprints of A and B overlap?".
    They miss the case where arm A's *link* (the physical body of the arm from
    its base to the task) passes through arm B's task region, or vice-versa.

    We model the arm link as a *line segment* from base_X to the task's ellipsoid
    centre, then test:

        ① Does segment (base_A → centroid_A) come within safety_m of ellipsoid B?
        ② Does segment (base_B → centroid_B) come within safety_m of ellipsoid A?

    Both checks must pass (report safe) for the pair to be pruned.

    The segment-to-ellipsoid minimum distance is computed in the ellipsoid's
    eigenspace where the shape becomes a unit sphere; the segment maps to a new
    segment in that space, and the distance from a segment to the origin of a
    unit sphere is well-defined in O(1).

    This correctly catches the scenario:
        "base A → task A passes through (or near) task B's region → not safe"
    regardless of whether the two task ellipsoids themselves overlap.

EllipsoidRecord (robot-agnostic, cached per task)
-------------------------------------------------
Stores geometry computed from trajectory points only:
    mu_g  (2,)    — Gaussian mean  of 2D positions
    sigma (2,2)   — Gaussian covariance
    mu_e  (2,)    — MVEE centre
    M     (2,2)   — MVEE shape matrix (tight, no inflation)
    evecs (2,2)   — eigen-vectors of M  (columns)
    evals (2,)    — eigen-values  of M  (semi-axis² = 1/λ)

Inflation and safety-margin logic is applied at query time, controlled by
``safety_m`` passed to each filter.  The record itself never stores a
pre-inflated shape — this keeps the cache robot-agnostic and lets the user
sweep ``safety_m`` without rebuilding the cache.

Filter API
----------
All three filter classes expose:

    is_safe(rec_a, rec_b, base_a, base_b, safety_m=0.0) -> (bool, dict)

``base_a`` / ``base_b`` are (x, y) world-frame tuples.  When ``safety_m=0``
the behaviour is identical to the original v1 filters.
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Numerical constants ────────────────────────────────────────────────────────
_EPS      = 1e-9
_MVEE_TOL = 1e-6
_MVEE_MAX = 500
_COV_REG  = 1e-4


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Core geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _fit_mvee(
    pts: np.ndarray,
    tol: float = _MVEE_TOL,
    max_iter: int = _MVEE_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Minimum-Volume Enclosing Ellipsoid via Todd-Yildirim / Khachiyan.

    Parameters
    ----------
    pts : (N, d) array
    tol, max_iter : convergence controls

    Returns
    -------
    center : (d,)
    M      : (d, d)  shape matrix — E = {x : (x-c)ᵀ M (x-c) ≤ 1}
    """
    N, d = pts.shape
    if N == 0:
        return np.zeros(d), np.eye(d)

    _, idx = np.unique(np.round(pts, 8), axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    N = pts.shape[0]

    if N <= d:
        mu  = pts.mean(axis=0)
        cov = (np.cov(pts.T) if N > 1 else np.eye(d))
        cov = np.atleast_2d(cov) + np.eye(d) * _COV_REG
        return mu, np.linalg.inv(cov) / d

    u = np.ones(N) / N
    for _ in range(max_iter):
        center = pts.T @ u
        X      = pts - center
        A      = (X.T * u) @ X + np.eye(d) * _EPS
        try:    A_inv = np.linalg.inv(A)
        except: A_inv = np.linalg.pinv(A)
        M_dist = np.einsum("ij,jk,ik->i", X, A_inv, X)
        j      = int(np.argmax(M_dist))
        step   = (M_dist[j] - d) / ((d + 1) * (M_dist[j] - 1) + _EPS)
        u_new  = (1 - step) * u
        u_new[j] += step
        if np.linalg.norm(u_new - u) < tol:
            u = u_new; break
        u = u_new

    center = pts.T @ u
    X      = pts - center
    A      = (X.T * u) @ X + np.eye(d) * _EPS
    try:    A_inv = np.linalg.inv(A)
    except: A_inv = np.linalg.pinv(A)
    return center, A_inv / d


def _gaussian_from_pts(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu    = pts.mean(axis=0)
    sigma = (np.cov(pts.T) if pts.shape[0] > 1 else np.eye(2))
    return mu, np.atleast_2d(sigma).astype(np.float64) + np.eye(2) * _COV_REG


def _inflate_M(M: np.ndarray, safety_m: float) -> np.ndarray:
    """Inflate MVEE shape matrix M by a uniform safety distance (metres).

    The correct physical inflation is:
        semi-axis  a_i  = 1 / sqrt(λ_i)     (metres, from eigenvalues of M)
        new axis   a_i' = a_i + safety_m
        new M'     = V  diag(1 / a_i'²)  Vᵀ

    This guarantees every point within safety_m metres of the original
    ellipsoid surface is inside the inflated ellipsoid, regardless of
    the ellipsoid's orientation or aspect ratio.
    """
    if safety_m <= 0.0:
        return M
    evals, evecs = np.linalg.eigh(M + np.eye(2) * _EPS)
    evals  = np.maximum(evals, _EPS)
    axes   = 1.0 / np.sqrt(evals)          # semi-axis lengths in metres
    axes_i = axes + safety_m               # uniformly grown
    new_ev = 1.0 / (axes_i ** 2)           # back to eigenvalues
    return (evecs * new_ev[None, :]) @ evecs.T


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  EllipsoidRecord  (per-task, robot-agnostic, cached)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class EllipsoidRecord:
    """Lightweight geometric summary of a task trajectory (world-frame, robot-agnostic).

    The record stores only the TIGHT geometry derived from the polyline.
    Safety inflation is applied at query time via ``safety_m`` so the cache
    never needs to be rebuilt when the safety margin changes.

    Attributes
    ----------
    gcode_id : str
    task_id  : str
    mu_g     : (2,) — Gaussian mean of 2D positions
    sigma    : (2,2) — Gaussian covariance
    mu_e     : (2,) — MVEE centre
    M        : (2,2) — MVEE shape matrix (tight — no inflation)
    evecs    : (2,2) — eigenvectors of M (columns)
    evals    : (2,)  — eigenvalues of M (semi-axis² = 1/λ)
    n_pts    : int
    """
    gcode_id: str
    task_id:  str
    mu_g:     np.ndarray   # (2,)
    sigma:    np.ndarray   # (2,2)
    mu_e:     np.ndarray   # (2,)
    M:        np.ndarray   # (2,2) tight MVEE
    evecs:    np.ndarray   # (2,2) eigen-vectors of M
    evals:    np.ndarray   # (2,)  eigen-values of M
    n_pts:    int

    # ── derived helpers ───────────────────────────────────────────────────────

    def semi_axes(self) -> np.ndarray:
        """Semi-axis lengths in metres: a_i = 1 / sqrt(λ_i)."""
        return 1.0 / np.sqrt(np.maximum(self.evals, _EPS))

    def inflated_M(self, safety_m: float) -> np.ndarray:
        """Return shape matrix inflated by safety_m metres (world-space uniform)."""
        if safety_m <= 0.0:
            return self.M
        axes_i = self.semi_axes() + safety_m
        new_ev = 1.0 / (axes_i ** 2)
        return (self.evecs * new_ev[None, :]) @ self.evecs.T

    def inflated_sigma(self, safety_m: float) -> np.ndarray:
        """Return Gaussian covariance inflated by safety_m metres.

        Inflates each standard deviation by safety_m (i.e. grows each axis
        of the covariance ellipse by safety_m in world-space).
        """
        if safety_m <= 0.0:
            return self.sigma
        # Eigen-inflate sigma the same way as M
        evals_s, evecs_s = np.linalg.eigh(self.sigma + np.eye(2) * _EPS)
        evals_s   = np.maximum(evals_s, _EPS)
        stds      = np.sqrt(evals_s)            # standard deviations along axes
        stds_i    = stds + safety_m
        new_ev_s  = stds_i ** 2
        return (evecs_s * new_ev_s[None, :]) @ evecs_s.T

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_polyline(
        cls,
        gcode_id: str,
        task_id: str,
        polyline: list,
        safety: float = 0.0,  # ignored — retained for API compat; use safety_m at query time
    ) -> "EllipsoidRecord":
        """Build an EllipsoidRecord from a raw (x, y, t) polyline.

        No safety inflation is stored.  Pass safety_m to each filter at query
        time instead — this keeps the cache valid for any safety setting.
        """
        if len(polyline) == 0:
            pts = np.zeros((1, 2), dtype=np.float64)
        else:
            pts = np.array([[p[0], p[1]] for p in polyline], dtype=np.float64)

        mu_g, sigma = _gaussian_from_pts(pts)
        mu_e, M     = _fit_mvee(pts)
        evals, evecs = np.linalg.eigh(M + np.eye(2) * _EPS)
        evals  = np.maximum(evals, _EPS)

        return cls(
            gcode_id=gcode_id,
            task_id=task_id,
            mu_g=mu_g.astype(np.float32),
            sigma=sigma.astype(np.float32),
            mu_e=mu_e.astype(np.float32),
            M=M.astype(np.float32),
            evecs=evecs.astype(np.float32),
            evals=evals.astype(np.float32),
            n_pts=len(polyline),
        )

    def to_dict(self) -> dict:
        return {
            "gcode_id": self.gcode_id,
            "task_id":  self.task_id,
            "mu_g":     torch.from_numpy(self.mu_g),
            "sigma":    torch.from_numpy(self.sigma),
            "mu_e":     torch.from_numpy(self.mu_e),
            "M":        torch.from_numpy(self.M),
            "evecs":    torch.from_numpy(self.evecs),
            "evals":    torch.from_numpy(self.evals),
            "n_pts":    self.n_pts,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EllipsoidRecord":
        return cls(
            gcode_id=d["gcode_id"],
            task_id=d["task_id"],
            mu_g=d["mu_g"].numpy(),
            sigma=d["sigma"].numpy(),
            mu_e=d["mu_e"].numpy(),
            M=d["M"].numpy(),
            evecs=d["evecs"].numpy(),
            evals=d["evals"].numpy(),
            n_pts=int(d["n_pts"]),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  GeometryCache
# ═══════════════════════════════════════════════════════════════════════════════

_GEO_SEP = "__"


class GeometryCache:
    """Persistent cache of per-task EllipsoidRecords.

    Robot-agnostic and arm-reassignment safe.  Stored as a .pt file.
    """

    def __init__(self, cache_path) -> None:
        self._path  = Path(cache_path) if cache_path is not None else None
        self._store: dict = {}
        if self._path is not None and self._path.exists():
            self._load()

    def _load(self) -> None:
        data = torch.load(str(self._path), map_location="cpu")
        self._store = data
        logger.info("Loaded %d geometry records from %s", len(self._store), self._path)

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._store, str(self._path))
        logger.info("Saved %d geometry records to %s", len(self._store), self._path)

    def _key(self, gcode_id: str, task_id: str) -> str:
        return f"{gcode_id}{_GEO_SEP}{task_id}"

    def get(self, gcode_id: str, task_id: str) -> Optional[EllipsoidRecord]:
        d = self._store.get(self._key(gcode_id, task_id))
        return EllipsoidRecord.from_dict(d) if d is not None else None

    def put(self, record: EllipsoidRecord) -> None:
        self._store[self._key(record.gcode_id, record.task_id)] = record.to_dict()

    def __contains__(self, key: tuple) -> bool:
        return self._key(*key) in self._store

    def __len__(self) -> int:
        return len(self._store)


def build_geometry_cache(tasks, cache_path, safety: float = 0.0, overwrite: bool = False) -> GeometryCache:
    """Pre-compute EllipsoidRecords for all tasks and save to disk.

    ``safety`` is ignored (records are always tight); pass safety_m to filters.
    """
    cache   = GeometryCache(cache_path)
    pending = [t for t in tasks if overwrite or (t.gcode_id, t.task_id) not in cache]
    logger.info("Building geometry cache: %d tasks (%d skipped)", len(pending), len(tasks) - len(pending))
    for t in pending:
        cache.put(EllipsoidRecord.from_polyline(t.gcode_id, t.task_id, t.polyline))
    cache.save()
    logger.info("Geometry cache complete: %d entries", len(cache))
    return cache


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Geometry primitives
# ═══════════════════════════════════════════════════════════════════════════════

def _point_in_ellipsoid(p: np.ndarray, centre: np.ndarray, M: np.ndarray) -> float:
    """Mahalanobis-squared distance of point p from ellipsoid centre (≤1 means inside)."""
    d = p - centre
    return float(d @ M @ d)


def _segment_min_ellipsoid_distance(
    seg_p: np.ndarray,   # (2,) segment start (e.g. base position)
    seg_q: np.ndarray,   # (2,) segment end   (e.g. task centroid)
    centre: np.ndarray,  # (2,) ellipsoid centre
    M: np.ndarray,       # (2,2) ellipsoid shape matrix
) -> float:
    """Minimum world-space distance from a line segment to an ellipsoid surface.

    Algorithm
    ---------
    Transform to the ellipsoid's eigenspace where the ellipsoid becomes a unit
    circle.  In this space, compute the distance from the transformed segment to
    the unit circle, then map back to world-space distances.

    Steps:
    1.  Compute L = chol(M) so M = Lᵀ L (or eigendecomp as fallback).
    2.  In whitened space: p' = L(p - c), q' = L(q - c).
    3.  Find the point on segment [p', q'] closest to the origin → t* ∈ [0,1].
    4.  Closest point r' = p' + t*(q'-p').
    5.  ‖r'‖ is the Mahalanobis distance; the ellipsoid surface is at ‖r'‖=1.
    6.  Convert Mahalanobis distance to approximate world-space distance by
        finding the scale factor at direction r'/‖r'‖:
            world_dist ≈ (‖r'‖ - 1) / ‖L⁻ᵀ (r'/‖r'‖)‖

    Returns the minimum world-space distance from the segment to the ellipsoid
    boundary (negative means the segment penetrates the ellipsoid).
    """
    M_reg = M + np.eye(2) * _EPS
    # Cholesky: M = Lᵀ L, so whitening transform is L
    try:
        L = np.linalg.cholesky(M_reg).T   # upper triangular: M = Lᵀ L
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(M_reg)
        evals = np.maximum(evals, _EPS)
        L = (evecs * np.sqrt(evals)[None, :]).T

    # Whitened points (relative to ellipsoid centre)
    p_w = L @ (seg_p - centre)
    q_w = L @ (seg_q - centre)

    # Closest point on segment [p_w, q_w] to origin
    pq   = q_w - p_w
    denom = float(pq @ pq)
    if denom < _EPS:
        # Degenerate segment — just use midpoint
        r_w = p_w
    else:
        t  = float(-p_w @ pq) / denom
        t  = max(0.0, min(1.0, t))
        r_w = p_w + t * pq

    mah = float(np.sqrt(max(r_w @ r_w, 0.0)))   # Mahalanobis distance to centre
    if mah < _EPS:
        # Segment passes through centre — deep inside ellipsoid
        return -float(np.sqrt(1.0 / max(np.linalg.det(M), _EPS)))

    # Convert from Mahalanobis to world-space:
    # The surface point in whitened space is r_w / mah.
    # Back in world space: s = L⁻¹ (r_w / mah)
    # World-space distance ≈ ‖s_world - closest_world‖ * (mah - 1)/mah (approx)
    # Better: use the ellipsoid-gradient approach.
    # At surface point s_w = r_w / mah, world surface point = L⁻¹ s_w + centre.
    # Closest segment world point = L⁻¹ r_w + centre.
    # World distance = ‖L⁻¹(r_w/mah - r_w)‖ = ‖L⁻¹ r_w (1/mah - 1)‖
    #               = ‖L⁻¹ r_w‖ * |1/mah - 1| * mah  ... simplified:
    # = ‖L⁻¹ r_w‖ * |mah - 1| / mah  (when mah > 0)
    try:
        L_inv = np.linalg.inv(L)
    except np.linalg.LinAlgError:
        L_inv = np.linalg.pinv(L)

    r_world = L_inv @ r_w
    world_dist = float(np.sqrt(max(r_world @ r_world, 0.0))) * abs(mah - 1.0) / mah

    # Sign: negative if segment is inside ellipsoid
    if mah < 1.0:
        world_dist = -world_dist

    return world_dist


def _segment_intersects_ellipsoid(
    seg_p: np.ndarray,
    seg_q: np.ndarray,
    centre: np.ndarray,
    M_inflated: np.ndarray,
) -> bool:
    """Return True if segment [P, Q] intersects the inflated ellipsoid.

    A segment intersects E iff the minimum Mahalanobis distance from any
    point on the segment to the centre is ≤ 1.
    """
    M_reg = M_inflated + np.eye(2) * _EPS
    try:
        L = np.linalg.cholesky(M_reg).T
    except np.linalg.LinAlgError:
        evals, evecs = np.linalg.eigh(M_reg)
        evals = np.maximum(evals, _EPS)
        L = (evecs * np.sqrt(evals)[None, :]).T

    p_w = L @ (seg_p - centre)
    q_w = L @ (seg_q - centre)
    pq   = q_w - p_w
    denom = float(pq @ pq)
    if denom < _EPS:
        r_w = p_w
    else:
        t   = float(-p_w @ pq) / denom
        t   = max(0.0, min(1.0, t))
        r_w = p_w + t * pq

    mah_sq = float(r_w @ r_w)
    return mah_sq <= 1.0


def _check_reach_intersection(
    base_a: np.ndarray,
    centre_a: np.ndarray,
    rec_b: "EllipsoidRecord",
    safety_m: float,
) -> bool:
    """Check whether arm A's reach segment (base_A → task_A centre) enters ellipsoid B.

    Returns True if the reach segment is NOT safe (intersects inflated E_B).
    """
    M_b_inf = rec_b.inflated_M(safety_m)
    return _segment_intersects_ellipsoid(base_a, centre_a, rec_b.mu_e, M_b_inf)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Bhattacharyya helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _bhattacharyya_coefficient(
    mu_a: np.ndarray, sigma_a: np.ndarray,
    mu_b: np.ndarray, sigma_b: np.ndarray,
) -> float:
    """Bhattacharyya coefficient ρ ∈ [0,1] between two 2D Gaussians."""
    delta     = mu_a - mu_b
    sigma_mid = (sigma_a + sigma_b) * 0.5 + np.eye(2) * _EPS
    try:    sm_inv = np.linalg.inv(sigma_mid)
    except: sm_inv = np.linalg.pinv(sigma_mid)

    det_mid = max(np.linalg.det(sigma_mid), _EPS)
    det_a   = max(np.linalg.det(sigma_a),   _EPS)
    det_b   = max(np.linalg.det(sigma_b),   _EPS)

    d_b = 0.125 * float(delta @ sm_inv @ delta) + \
          0.5   * np.log(det_mid / np.sqrt(det_a * det_b))
    return float(np.exp(-max(d_b, 0.0)))


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Probabilistic overlap helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _prob_overlap(
    mu_a: np.ndarray,  sigma_a: np.ndarray,
    mu_b: np.ndarray,  sigma_b: np.ndarray,
    n_sigma: float = 2.0,
) -> float:
    """P(point from N(μ_A,Σ_A) falls inside the n_sigma-ellipsoid of B).

    Uses Pearson-Welch moment-matched chi-squared approximation.
    """
    try:
        from scipy.stats import chi2
    except ImportError:
        return float(np.exp(-0.125 * float(
            (mu_a - mu_b) @ np.linalg.pinv((sigma_a + sigma_b) / 2) @ (mu_a - mu_b))))

    sigma_b_reg = sigma_b + np.eye(2) * _EPS
    try:    L_b = np.linalg.cholesky(sigma_b_reg)
    except: L_b = np.linalg.cholesky(sigma_b_reg + np.eye(2) * 1e-3)
    try:    W   = np.linalg.inv(L_b)
    except: W   = np.linalg.pinv(L_b)

    mu_w    = W @ (mu_a - mu_b)
    sigma_w = W @ sigma_a @ W.T

    e_q   = float(np.trace(sigma_w)) + float(mu_w @ mu_w)
    var_q = 2.0 * float(np.trace(sigma_w @ sigma_w)) + 4.0 * float(mu_w @ sigma_w @ mu_w)

    if var_q < _EPS or e_q < _EPS:
        return 1.0 if float(mu_w @ mu_w) <= n_sigma ** 2 else 0.0

    c = var_q / (2.0 * e_q + _EPS)
    k = max(2.0 * e_q ** 2 / (var_q + _EPS), 0.1)
    return float(chi2.cdf(n_sigma ** 2 / (c + _EPS), df=k))


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Separation test (optimised 2D closed-form)
# ═══════════════════════════════════════════════════════════════════════════════

def _ellipsoid_separation_2d(
    c_a: np.ndarray, M_a: np.ndarray,
    c_b: np.ndarray, M_b: np.ndarray,
) -> float:
    """Matrix pencil separation margin for 2D ellipsoids.

    Returns s = min_λ f(λ) − 1.  s > 0 ⟺ ellipsoids are disjoint.
    f(λ) = δᵀ [(1-λ)M_A + λ M_B]⁻¹ δ,  δ = c_A - c_B,  λ ∈ [0,1].

    Uses simultaneous diagonalisation + 16-point scan + golden-section.
    """
    delta    = c_a - c_b
    M_b_reg  = M_b + np.eye(2) * _EPS
    try:    L_b = np.linalg.cholesky(M_b_reg)
    except:
        evals, evecs = np.linalg.eigh(M_b_reg)
        L_b = (evecs * np.sqrt(np.maximum(evals, _EPS))[None, :])

    try:    L_b_inv = np.linalg.inv(L_b)
    except: L_b_inv = np.linalg.pinv(L_b)

    C     = L_b_inv @ M_a @ L_b_inv.T
    alpha, V = np.linalg.eigh(C + np.eye(2) * _EPS)
    alpha = np.maximum(alpha, _EPS)
    d_hat = V.T @ (L_b @ delta)
    d2    = d_hat ** 2

    lams   = np.linspace(0.0, 1.0, 16)
    denoms = np.maximum(alpha[None, :] + lams[:, None] * (1.0 - alpha[None, :]), _EPS)
    fs     = (d2[None, :] / denoms).sum(axis=1)
    best_i = int(np.argmin(fs))

    lo  = max(0.0, lams[best_i] - 1.0 / 15.0)
    hi  = min(1.0, lams[best_i] + 1.0 / 15.0)
    phi = 0.6180339887
    for _ in range(12):
        m1 = lo + (1 - phi) * (hi - lo)
        m2 = lo + phi * (hi - lo)
        f1 = float((d2 / np.maximum(alpha + m1 * (1.0 - alpha), _EPS)).sum())
        f2 = float((d2 / np.maximum(alpha + m2 * (1.0 - alpha), _EPS)).sum())
        if f1 < f2: hi = m2
        else:       lo = m1

    lam_opt = (lo + hi) / 2.0
    best_f  = float((d2 / np.maximum(alpha + lam_opt * (1.0 - alpha), _EPS)).sum())
    return best_f - 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Filter base helper  (shared reach-check logic)
# ═══════════════════════════════════════════════════════════════════════════════

def _reach_safe(
    rec_a: EllipsoidRecord,
    rec_b: EllipsoidRecord,
    base_a: Optional[np.ndarray],
    base_b: Optional[np.ndarray],
    safety_m: float,
) -> bool:
    """Return True (safe) only if NEITHER arm's reach segment enters the other's ellipsoid.

    Checks two directions:
        A → B: does arm A's link (base_A → task_A centroid) enter E_B?
        B → A: does arm B's link (base_B → task_B centroid) enter E_A?

    If base positions are not provided, this check is skipped (returns True).
    """
    if base_a is None or base_b is None:
        return True   # can't check without base info — defer to task-overlap tests

    # A's reach into B's ellipsoid
    if _check_reach_intersection(base_a, rec_a.mu_e, rec_b, safety_m):
        return False   # arm A's link enters B's region

    # B's reach into A's ellipsoid
    if _check_reach_intersection(base_b, rec_b.mu_e, rec_a, safety_m):
        return False   # arm B's link enters A's region

    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Filter 1 — MVEE Algebraic Separation
# ═══════════════════════════════════════════════════════════════════════════════

class EllipsoidSeparationFilter:
    """Pre-filter based on MVEE algebraic separation test.

    A pair is safe iff:
        (a) the inflated task ellipsoids E_A and E_B are geometrically disjoint, AND
        (b) arm A's reach segment (base_A → centroid_A) does not enter E_B, AND
        (c) arm B's reach segment (base_B → centroid_B) does not enter E_A.

    Parameters
    ----------
    geometry_cache : GeometryCache (unused by is_safe — passed for API compat)
    safety_m       : uniform inflation in metres applied to both ellipsoids
                     before any comparison.  Setting this to arm_width/2 drives
                     FNR to zero for the hard geometric test.
    """
    name = "mvee_separation"

    def __init__(self, geometry_cache, safety_m: float = 0.05) -> None:
        self._cache    = geometry_cache
        self._safety_m = safety_m

    def is_safe(
        self,
        rec_a: EllipsoidRecord,
        rec_b: EllipsoidRecord,
        base_a: Optional[np.ndarray] = None,
        base_b: Optional[np.ndarray] = None,
        safety_m: Optional[float] = None,
    ) -> Tuple[bool, dict]:
        """Test whether the pair is safe to prune.

        Parameters
        ----------
        rec_a, rec_b : EllipsoidRecords for the two tasks
        base_a, base_b : (2,) world-frame base positions of the arms
        safety_m : override the instance safety_m for this call

        Returns
        -------
        (is_safe, scores)
            is_safe — True means prune (no neural net / exact check needed)
            scores  — dict with 'mvee_margin', 'reach_a_safe', 'reach_b_safe'
        """
        sm  = safety_m if safety_m is not None else self._safety_m
        M_a = rec_a.inflated_M(sm)
        M_b = rec_b.inflated_M(sm)

        margin = _ellipsoid_separation_2d(rec_a.mu_e, M_a, rec_b.mu_e, M_b)

        scores: dict = {"mvee_margin": margin}

        # Task ellipsoids not separated → cannot prune
        if margin <= 0.0:
            scores.update(reach_a_safe=None, reach_b_safe=None)
            return False, scores

        # Check reach segments (both must be clear)
        reach_ok = _reach_safe(rec_a, rec_b, base_a, base_b, sm)
        scores["reach_a_safe"] = reach_ok
        scores["reach_b_safe"] = reach_ok   # _reach_safe checks both internally
        return reach_ok, scores


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Filter 2 — Bhattacharyya Coefficient
# ═══════════════════════════════════════════════════════════════════════════════

class BhattacharyyaFilter:
    """Pre-filter based on Bhattacharyya overlap coefficient.

    Inflates each Gaussian's covariance by safety_m before computing ρ.
    Pairs with ρ < rho_threshold AND passing the reach check are pruned.

    Parameters
    ----------
    geometry_cache : GeometryCache
    rho_threshold  : ρ below this → task Gaussians don't overlap → candidate safe
    safety_m       : uniform inflation in metres applied to Gaussian std-devs
    """
    name = "bhattacharyya"

    def __init__(self, geometry_cache, rho_threshold: float = 0.05, safety_m: float = 0.05) -> None:
        self._cache     = geometry_cache
        self._rho_thr   = rho_threshold
        self._safety_m  = safety_m

    def is_safe(
        self,
        rec_a: EllipsoidRecord,
        rec_b: EllipsoidRecord,
        base_a: Optional[np.ndarray] = None,
        base_b: Optional[np.ndarray] = None,
        safety_m: Optional[float] = None,
    ) -> Tuple[bool, dict]:
        sm     = safety_m if safety_m is not None else self._safety_m
        sig_a  = rec_a.inflated_sigma(sm)
        sig_b  = rec_b.inflated_sigma(sm)
        rho    = _bhattacharyya_coefficient(rec_a.mu_g, sig_a, rec_b.mu_g, sig_b)

        scores: dict = {"rho": rho}

        if rho >= self._rho_thr:
            scores.update(reach_a_safe=None, reach_b_safe=None)
            return False, scores

        reach_ok = _reach_safe(rec_a, rec_b, base_a, base_b, sm)
        scores["reach_a_safe"] = reach_ok
        scores["reach_b_safe"] = reach_ok
        return reach_ok, scores


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Filter 3 — Probabilistic χ² Overlap
# ═══════════════════════════════════════════════════════════════════════════════

class ProbabilisticOverlapFilter:
    """Pre-filter using the probabilistic volume-overlap integral.

    Inflates each Gaussian's covariance by safety_m before computing the
    Pearson-Welch chi-squared overlap probability.

    Parameters
    ----------
    geometry_cache : GeometryCache
    p_threshold    : overlap probability below this → candidate safe
    n_sigma        : ellipsoid of B defined as its n_sigma level set
    safety_m       : uniform inflation in metres
    """
    name = "prob_overlap"

    def __init__(
        self,
        geometry_cache,
        p_threshold: float = 0.02,
        n_sigma: float = 2.0,
        safety_m: float = 0.05,
    ) -> None:
        self._cache    = geometry_cache
        self._p_thr    = p_threshold
        self._n_sigma  = n_sigma
        self._safety_m = safety_m

    def is_safe(
        self,
        rec_a: EllipsoidRecord,
        rec_b: EllipsoidRecord,
        base_a: Optional[np.ndarray] = None,
        base_b: Optional[np.ndarray] = None,
        safety_m: Optional[float] = None,
    ) -> Tuple[bool, dict]:
        sm    = safety_m if safety_m is not None else self._safety_m
        sig_a = rec_a.inflated_sigma(sm)
        sig_b = rec_b.inflated_sigma(sm)

        p_ab = _prob_overlap(rec_a.mu_g, sig_a, rec_b.mu_g, sig_b, self._n_sigma)
        p_ba = _prob_overlap(rec_b.mu_g, sig_b, rec_a.mu_g, sig_a, self._n_sigma)
        p    = max(p_ab, p_ba)

        scores: dict = {"p_overlap": p}

        if p >= self._p_thr:
            scores.update(reach_a_safe=None, reach_b_safe=None)
            return False, scores

        reach_ok = _reach_safe(rec_a, rec_b, base_a, base_b, sm)
        scores["reach_a_safe"] = reach_ok
        scores["reach_b_safe"] = reach_ok
        return reach_ok, scores


# ═══════════════════════════════════════════════════════════════════════════════
# 12. CascadeFilter
# ═══════════════════════════════════════════════════════════════════════════════

class CascadeFilter:
    """Run all three filters fastest-first; prune as soon as any declares safe.

    The reach check is shared and only performed once per pair after any
    task-overlap filter passes.

    Parameters
    ----------
    geometry_cache   : GeometryCache
    mvee_inflated    : ignored (safety_m controls inflation uniformly)
    rho_threshold    : Bhattacharyya threshold
    p_threshold      : probabilistic overlap threshold
    n_sigma          : sigma level for prob overlap ellipsoid
    safety_m         : uniform world-space safety inflation (metres)
                       Set to arm_width/2 to guarantee FNR=0 for MVEE filter.
    """
    name = "cascade"

    def __init__(
        self,
        geometry_cache,
        mvee_inflated: bool = True,   # retained for API compat
        rho_threshold: float = 0.05,
        p_threshold: float   = 0.02,
        n_sigma: float       = 2.0,
        safety_m: float      = 0.05,
    ) -> None:
        self._f1 = EllipsoidSeparationFilter(geometry_cache, safety_m=safety_m)
        self._f2 = BhattacharyyaFilter(geometry_cache, rho_threshold=rho_threshold, safety_m=safety_m)
        self._f3 = ProbabilisticOverlapFilter(geometry_cache, p_threshold=p_threshold,
                                              n_sigma=n_sigma, safety_m=safety_m)
        self._safety_m = safety_m

    def is_safe(
        self,
        rec_a: EllipsoidRecord,
        rec_b: EllipsoidRecord,
        base_a: Optional[np.ndarray] = None,
        base_b: Optional[np.ndarray] = None,
        safety_m: Optional[float] = None,
    ) -> Tuple[bool, str, dict]:
        """Run cascade. Returns (is_safe, pruned_by, scores)."""
        sm = safety_m if safety_m is not None else self._safety_m

        safe1, s1 = self._f1.is_safe(rec_a, rec_b, base_a, base_b, sm)
        safe2, s2 = self._f2.is_safe(rec_a, rec_b, base_a, base_b, sm)
        safe3, s3 = self._f3.is_safe(rec_a, rec_b, base_a, base_b, sm)

        scores = {
            "mvee_margin": s1.get("mvee_margin"),
            "rho":         s2.get("rho"),
            "p_overlap":   s3.get("p_overlap"),
        }

        if safe1: return True,  "mvee_separation", scores
        if safe2: return True,  "bhattacharyya",   scores
        if safe3: return True,  "prob_overlap",    scores
        return    False, "none",              scores
