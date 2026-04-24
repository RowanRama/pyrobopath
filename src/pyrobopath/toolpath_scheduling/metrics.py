"""Planning metrics collection for toolpath scheduling.

:class:`PlanningMetrics` is an optional instrumentation object that can be
passed into the planner and collision-checking functions. When provided, it
accumulates timing and count data for every cache build, allocation attempt,
and collision check without altering any control flow.

Usage::

    metrics = PlanningMetrics()
    planner.plan(toolpath, dg, options, metrics=metrics)
    metrics.print_summary()
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PlanningMetrics:
    """Accumulates performance counters for a single planning run.

    All timing fields are in seconds. Counts are non-negative integers.

    Attributes
    ----------
    n_tasks : int
        Number of contours (tasks) in the toolpath.
    cache_build_time_s : float
        Total wall-clock time spent building the prefilter cache.
    cache_build_times_per_task_s : list of float
        Per-contour build time; one entry per contour processed during
        ``build_cache``. Populated only when a prefilter is active.
    allocation_time_s : float
        Total wall-clock time spent inside the planning loop (excluding
        cache build).
    allocation_times_per_task_s : list of float
        Wall-clock time for each individual task allocation attempt that
        succeeded (one entry per assigned contour event).
    n_collision_pairs_checked : int
        Total number of (agent, other_agent) pairs evaluated for collision.
    n_prefilter_pruned : int
        Pairs skipped after the prefilter declared them safe (subset of
        ``n_collision_pairs_checked``).
    n_fcl_checked : int
        Pairs that reached the full FCL trajectory collision query
        (``n_collision_pairs_checked - n_prefilter_pruned``).
    n_collisions_detected : int
        Pairs where the full FCL check detected an actual collision.
    """

    n_tasks: int = 0

    # cache build
    cache_build_time_s: float = 0.0
    cache_build_times_per_task_s: List[float] = field(default_factory=list)

    # allocation
    allocation_time_s: float = 0.0
    allocation_times_per_task_s: List[float] = field(default_factory=list)

    # total scheduling wall time (allocation loop + collision checks, excl. cache)
    total_planning_time_s: float = 0.0

    # collision checks
    n_collision_pairs_checked: int = 0
    n_prefilter_pruned: int = 0
    n_fcl_checked: int = 0
    n_collisions_detected: int = 0

    # ------------------------------------------------------------------ derived
    @property
    def avg_cache_time_per_task_s(self) -> Optional[float]:
        """Mean per-task cache build time, or None if no tasks were cached."""
        if not self.cache_build_times_per_task_s:
            return None
        return sum(self.cache_build_times_per_task_s) / len(
            self.cache_build_times_per_task_s
        )

    @property
    def avg_allocation_time_per_task_s(self) -> Optional[float]:
        """Mean per-task allocation time, or None if no tasks were allocated."""
        if not self.allocation_times_per_task_s:
            return None
        return sum(self.allocation_times_per_task_s) / len(
            self.allocation_times_per_task_s
        )

    @property
    def prefilter_prune_rate(self) -> Optional[float]:
        """Fraction of checked pairs pruned by the prefilter, or None if
        no pairs were checked."""
        if self.n_collision_pairs_checked == 0:
            return None
        return self.n_prefilter_pruned / self.n_collision_pairs_checked

    # ------------------------------------------------------------------ helpers
    @contextmanager
    def _time_cache_entry(self):
        """Context manager that records one per-task cache build time."""
        t0 = time.perf_counter()
        yield
        self.cache_build_times_per_task_s.append(time.perf_counter() - t0)

    def record_cache_build(self, elapsed_s: float) -> None:
        """Record total cache build time (called once per planning run)."""
        self.cache_build_time_s += elapsed_s

    def record_allocation(self, elapsed_s: float) -> None:
        """Record one successful task allocation."""
        self.allocation_times_per_task_s.append(elapsed_s)
        self.allocation_time_s += elapsed_s

    def record_pair_checked(self) -> None:
        """Increment the total pair-checked counter."""
        self.n_collision_pairs_checked += 1

    def record_prefilter_pruned(self) -> None:
        """Record one pair pruned by the prefilter (implies pair was checked)."""
        self.n_prefilter_pruned += 1

    def record_fcl_checked(self) -> None:
        """Record one pair that reached the FCL trajectory query."""
        self.n_fcl_checked += 1

    def record_collision_detected(self) -> None:
        """Record one pair where FCL detected a collision."""
        self.n_collisions_detected += 1

    def record_total_planning_time(self, elapsed_s: float) -> None:
        """Record total planning loop wall time (excl. cache build)."""
        self.total_planning_time_s = elapsed_s

    # ------------------------------------------------------------------ display
    def print_summary(self, label: str = "") -> None:
        """Print a human-readable summary to stdout."""
        header = f"PlanningMetrics — {label}" if label else "PlanningMetrics"
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  {header}")
        print(sep)
        print(f"  Tasks in toolpath          : {self.n_tasks}")

        print(f"\n  — Cache build —")
        if self.cache_build_times_per_task_s:
            print(f"  Total cache build time     : {self.cache_build_time_s*1e3:.2f} ms")
            print(
                f"  Avg per task               : "
                f"{self.avg_cache_time_per_task_s * 1e6:.1f} µs"
            )
            print(
                f"  Tasks cached               : "
                f"{len(self.cache_build_times_per_task_s)}"
            )
        else:
            print("  (no prefilter — cache not used)")

        print(f"\n  — Task allocation —")
        print(
            f"  Total planning time        : {self.total_planning_time_s*1e3:.2f} ms"
        )
        print(
            f"  Total allocation time      : {self.allocation_time_s*1e3:.2f} ms"
        )
        if self.avg_allocation_time_per_task_s is not None:
            print(
                f"  Avg per task               : "
                f"{self.avg_allocation_time_per_task_s*1e3:.3f} ms"
            )
        print(
            f"  Tasks allocated            : {len(self.allocation_times_per_task_s)}"
        )

        print(f"\n  — Collision checks —")
        print(f"  Pairs checked total        : {self.n_collision_pairs_checked}")
        print(f"  Pruned by prefilter        : {self.n_prefilter_pruned}")
        print(f"  Reached FCL check          : {self.n_fcl_checked}")
        print(f"  Collisions detected        : {self.n_collisions_detected}")
        if self.prefilter_prune_rate is not None:
            print(
                f"  Prefilter prune rate       : "
                f"{self.prefilter_prune_rate*100:.1f}%"
            )
        print(sep)
