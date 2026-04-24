"""Task priority benchmark.

Sweeps every available task-priority strategy under two collision-check
configurations — baseline (FCL only) and MVEE prefilter — and measures how
each strategy affects planning time, pair disposition (pruned vs FCL vs
collision), and the final prune rate.

Each (priority, prefilter) pair is run as one experiment.  Plans are verified
pairwise: the MVEE plan for a given priority must exactly match the baseline
plan for the same priority — the prefilter should never change the schedule,
only speed it up.

Usage::

    cd examples/
    python prefilter_benchmark.py
"""
import argparse
import time
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — swap to "TkAgg" / "Qt5Agg" for GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import replace

# ── resolve imports when run directly from examples/ ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from pyrobopath.process import AgentModel, create_dependency_graph_by_z
from pyrobopath.toolpath.preprocessing import *
from pyrobopath.collision_detection import (
    FCLRobotBBCollisionModel,
    MVEECachedModel,
    CachedCollisionModel,
)
from pyrobopath.toolpath_scheduling import (
    MultiAgentToolpathPlanner,
    PlanningOptions,
    PlanningMetrics,
)

from utilities import toolpath_from_gcode, print_schedule_info


# ─────────────────────────────────── constants ────────────────────────────────

GCODE_FILE = "../test/test_gcode/mona_lisa.gcode"
LAYER_RANGE = (0, 1)
SAFETY_M    = 25       # mm; at this value MVEE achieves FNR = 0

BASE_OPTIONS = PlanningOptions(
    retract_height=10.0,
    collision_offset=3.0,
    collision_gap_threshold=5.0,
    priority_weight=0.5,
)

BASE_RADIUS = 350.0   # mm — robots placed on a circle around the table centre
HOME_RADIUS = 250.0   # mm — home position on the same ray, inside the base
MAX_ROBOTS  = 16
COL_DIMS    = (200.0, 50.0, 300.0)


def _robot_positions(n_robots):
    """Return (bases, homes) — each a list of length n_robots of (3,) arrays.

    Robots are placed equally spaced around a circle of radius BASE_RADIUS
    starting at angle 0 (positive x-axis).  Homes sit on the same ray at
    HOME_RADIUS from the centre.
    """
    if not (1 <= n_robots <= MAX_ROBOTS):
        raise ValueError(f"n_robots must be in [1, {MAX_ROBOTS}], got {n_robots}")

    angles = np.linspace(0.0, 2.0 * np.pi, n_robots, endpoint=False)
    bases, homes = [], []
    for theta in angles:
        c, s = np.cos(theta), np.sin(theta)
        bases.append(np.array([BASE_RADIUS * c, BASE_RADIUS * s, 0.0]))
        homes.append(np.array([HOME_RADIUS * c, HOME_RADIUS * s, 0.0]))
    return bases, homes

# Task priority strategies to benchmark
PRIORITIES = [
    "out_degree",
    "farthest_centroid",
    "nearest_start",
    "nearest_base",
    "min_ellipsoid",
    "farthest_other_bases",
    "prune_aware",
    "hybrid_base_separation",
    "hybrid_travel_prune",
]


# ─────────────────────────────────── helpers ──────────────────────────────────

def get_toolpath():
    toolpath = toolpath_from_gcode(GCODE_FILE)
    preprocessor = ToolpathPreprocessor()
    preprocessor.add_step(MaxContourLengthStep(100.0))
    preprocessor.add_step(LayerRangeStep(*LAYER_RANGE))
    preprocessor.process(toolpath)
    return toolpath


def make_agent_models(n_robots, prefilter=None):
    """Build n_robots agent models placed equally around a circle.

    All agents share the same capabilities, kinematics, and (optional)
    prefilter instance — required by the scheduler.
    """
    bases, homes = _robot_positions(n_robots)
    agent_models = {}
    for i, (base, home) in enumerate(zip(bases, homes), start=1):
        agent_models[f"robot{i}"] = AgentModel(
            base_frame_position=base,
            home_position=home,
            capabilities=[0],
            velocity=50.0,
            travel_velocity=50.0,
            collision_model=FCLRobotBBCollisionModel(COL_DIMS, base),
            collision_prefilter=prefilter,
        )
    return agent_models


def run_experiment(label, prefilter, task_priority, n_robots):
    """Run one planning trial with the given priority and prefilter.

    Returns
    -------
    (metrics, schedule)
    """
    CachedCollisionModel.clear_cache()

    toolpath = get_toolpath()
    dg = create_dependency_graph_by_z(toolpath)
    agent_models = make_agent_models(n_robots, prefilter)
    planner = MultiAgentToolpathPlanner(agent_models)
    metrics = PlanningMetrics()
    options = replace(BASE_OPTIONS, task_priority=task_priority)

    print(f"\n{'#' * 80}")
    print(f"  Running: {label}")
    print(f"{'#' * 80}")

    t0 = time.perf_counter()
    sched = planner.plan(toolpath, dg, options, metrics=metrics)
    wall = time.perf_counter() - t0

    print(f"  Wall time: {wall * 1e3:.1f} ms")
    print_schedule_info(sched)
    metrics.print_summary(label)
    return metrics, sched


# ─────────────────────────────── plan verification ───────────────────────────

def _plan_signature(sched):
    """Return a comparable representation of a schedule's task assignments."""
    from pyrobopath.toolpath_scheduling import ContourEvent as _CE
    sig = {}
    for agent, s in sched.schedules.items():
        sig[agent] = [
            (e.contour.id, round(e.start, 3))
            for e in s._events
            if isinstance(e, _CE)
        ]
    return sig


def verify_plans_match(results):
    """For each priority, check MVEE plan == baseline plan (same priority).

    The prefilter should only accelerate planning — it must never change the
    task assignments produced.  Cross-priority differences are expected and
    not checked.

    Parameters
    ----------
    results : list of (priority, prefilter_label, metrics, sched)
    """
    print(f"\n{'=' * 60}")
    print("  Plan equivalence check (MVEE must match baseline per priority)")
    print(f"{'=' * 60}")

    # Group results by priority
    by_priority = {}
    for priority, pf_label, _, sched in results:
        by_priority.setdefault(priority, {})[pf_label] = sched

    all_pass = True
    for priority, schedules in by_priority.items():
        if "baseline" not in schedules or "MVEE" not in schedules:
            continue
        base_sig = _plan_signature(schedules["baseline"])
        mvee_sig = _plan_signature(schedules["MVEE"])

        if base_sig == mvee_sig:
            print(f"  PASS : {priority}")
            continue

        all_pass = False
        print(f"  FAIL : {priority}")
        for agent in base_sig:
            b_seq = base_sig[agent]
            m_seq = mvee_sig.get(agent, [])
            if b_seq == m_seq:
                continue
            print(f"    Agent '{agent}': {len(b_seq)} tasks (baseline) "
                  f"vs {len(m_seq)} tasks (MVEE)")
            for i, (b, m) in enumerate(zip(b_seq, m_seq)):
                if b != m:
                    print(f"    First diff at index {i}: "
                          f"baseline={b}, MVEE={m}")
                    break

    print(f"{'=' * 60}")
    print(f"  Result: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print(f"{'=' * 60}\n")
    return all_pass


# ─────────────────────────────── plotting ─────────────────────────────────────

COLOUR_BASELINE = "#8da0cb"
COLOUR_MVEE     = "#fc8d62"
COLOUR_PRUNE    = "#e78ac3"
COLOUR_FCL      = "#66c2a5"


def plot_results(priorities, baseline_metrics, mvee_metrics,
                 out_path="prefilter_benchmark.png"):
    """Grouped bar charts comparing baseline vs MVEE across all priorities."""

    n = len(priorities)
    x = np.arange(n)
    w = 0.38

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        "Task Priority Benchmark — Baseline (FCL) vs MVEE Prefilter\n"
        f"Gcode: {os.path.basename(GCODE_FILE)}  ·  "
        f"Layers {LAYER_RANGE[0]}–{LAYER_RANGE[1]}  ·  "
        f"safety_m = {SAFETY_M} mm",
        fontsize=11, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.22)

    # ── panel 1: total planning time (baseline vs MVEE) ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    b_times = [m.total_planning_time_s * 1e3 for m in baseline_metrics]
    m_times = [m.total_planning_time_s * 1e3 for m in mvee_metrics]
    ax1.bar(x - w/2, b_times, width=w, label="Baseline (FCL)",
            color=COLOUR_BASELINE, edgecolor="black", linewidth=0.6)
    ax1.bar(x + w/2, m_times, width=w, label="MVEE",
            color=COLOUR_MVEE, edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("ms", fontsize=9)
    ax1.set_title("Total Planning Time", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(priorities, rotation=35, ha="right", fontsize=7)
    ax1.legend(fontsize=8)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)

    # ── panel 2: MVEE speedup ratio ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    speedup = [
        (b / m) if m > 0 else 0.0
        for b, m in zip(b_times, m_times)
    ]
    ax2.bar(x, speedup, width=0.55, color=COLOUR_MVEE,
            edgecolor="black", linewidth=0.6)
    ax2.axhline(1.0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_ylabel("baseline / MVEE", fontsize=9)
    ax2.set_title("MVEE Speedup (higher = better)", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(priorities, rotation=35, ha="right", fontsize=7)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    for xi, s in zip(x, speedup):
        ax2.text(xi, s * 1.01, f"{s:.2f}×", ha="center", va="bottom", fontsize=7)

    # ── panel 3: prune rate (MVEE only) ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    prune_pct = [
        (m.prefilter_prune_rate * 100 if m.prefilter_prune_rate else 0.0)
        for m in mvee_metrics
    ]
    ax3.bar(x, prune_pct, width=0.55, color=COLOUR_PRUNE,
            edgecolor="black", linewidth=0.6)
    ax3.set_ylabel("%", fontsize=9)
    ax3.set_title("MVEE Prune Rate", fontsize=10, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(priorities, rotation=35, ha="right", fontsize=7)
    ax3.set_ylim(0, 105)
    ax3.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax3.set_axisbelow(True)
    ax3.spines[["top", "right"]].set_visible(False)
    for xi, p in zip(x, prune_pct):
        ax3.text(xi, p + 1, f"{p:.1f}%", ha="center", va="bottom", fontsize=7)

    # ── panel 4: pair disposition (MVEE only) ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    pruned = [m.n_prefilter_pruned for m in mvee_metrics]
    fcl    = [m.n_fcl_checked      for m in mvee_metrics]
    ax4.bar(x, pruned, width=0.55, label="Pruned",
            color=COLOUR_PRUNE, edgecolor="black", linewidth=0.6)
    ax4.bar(x, fcl, width=0.55, bottom=pruned, label="FCL",
            color=COLOUR_FCL, edgecolor="black", linewidth=0.6)
    ax4.set_ylabel("pair count", fontsize=9)
    ax4.set_title("Pair Disposition (MVEE)", fontsize=10, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(priorities, rotation=35, ha="right", fontsize=7)
    ax4.legend(fontsize=8, loc="upper right")
    ax4.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax4.set_axisbelow(True)
    ax4.spines[["top", "right"]].set_visible(False)

    # ── panel 5: collisions detected (MVEE) ──────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    collisions = [m.n_collisions_detected for m in mvee_metrics]
    ax5.bar(x, collisions, width=0.55, color=COLOUR_FCL,
            edgecolor="black", linewidth=0.6)
    ax5.set_ylabel("count", fontsize=9)
    ax5.set_title("Collisions Detected by FCL (MVEE run)",
                  fontsize=10, fontweight="bold")
    ax5.set_xticks(x)
    ax5.set_xticklabels(priorities, rotation=35, ha="right", fontsize=7)
    ax5.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax5.set_axisbelow(True)
    ax5.spines[["top", "right"]].set_visible(False)

    # ── panel 6: allocation time per task ────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    b_alloc = [
        (m.avg_allocation_time_per_task_s * 1e3
         if m.avg_allocation_time_per_task_s else 0.0)
        for m in baseline_metrics
    ]
    m_alloc = [
        (m.avg_allocation_time_per_task_s * 1e3
         if m.avg_allocation_time_per_task_s else 0.0)
        for m in mvee_metrics
    ]
    ax6.bar(x - w/2, b_alloc, width=w, label="Baseline (FCL)",
            color=COLOUR_BASELINE, edgecolor="black", linewidth=0.6)
    ax6.bar(x + w/2, m_alloc, width=w, label="MVEE",
            color=COLOUR_MVEE, edgecolor="black", linewidth=0.6)
    ax6.set_ylabel("ms / task", fontsize=9)
    ax6.set_title("Avg Allocation Time per Task",
                  fontsize=10, fontweight="bold")
    ax6.set_xticks(x)
    ax6.set_xticklabels(priorities, rotation=35, ha="right", fontsize=7)
    ax6.legend(fontsize=8)
    ax6.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax6.set_axisbelow(True)
    ax6.spines[["top", "right"]].set_visible(False)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved → {os.path.abspath(out_path)}")
    return fig


# ─────────────────────────────── main ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Task-priority × MVEE-prefilter benchmark."
    )
    parser.add_argument(
        "--n-robots", type=int, default=2,
        help=f"number of robots, placed equally spaced around a circle of "
             f"radius {BASE_RADIUS:.0f} mm (1–{MAX_ROBOTS}, default: 2)",
    )
    args = parser.parse_args()

    if not (1 <= args.n_robots <= MAX_ROBOTS):
        parser.error(f"--n-robots must be in [1, {MAX_ROBOTS}]")

    print(f"Running benchmark with {args.n_robots} robots "
          f"(base radius {BASE_RADIUS:.0f} mm)")

    experiments = []
    for p in PRIORITIES:
        experiments.append((p, "baseline", None))
        experiments.append((p, "MVEE", MVEECachedModel(safety_m=SAFETY_M)))

    results = []
    for priority, pf_label, prefilter in experiments:
        label = f"{priority} ({pf_label})"
        metrics, sched = run_experiment(label, prefilter, priority, args.n_robots)
        results.append((priority, pf_label, metrics, sched))

    # ── plan equivalence check ────────────────────────────────────────────────
    verify_plans_match(results)

    # ── split by prefilter variant for plotting/tables ────────────────────────
    baseline_metrics = [
        m for (_, pf, m, _) in results if pf == "baseline"
    ]
    mvee_metrics = [
        m for (_, pf, m, _) in results if pf == "MVEE"
    ]

    # ── tabular summary ───────────────────────────────────────────────────────
    col_w = 24
    header = (
        f"{'Priority':<{col_w}}"
        f"{'BaselineTime(ms)':>18}"
        f"{'MVEETime(ms)':>14}"
        f"{'Speedup':>10}"
        f"{'PairsChk':>10}"
        f"{'Pruned':>8}"
        f"{'FCL':>6}"
        f"{'PruneRate':>11}"
        f"{'Collide':>9}"
    )
    divider = "-" * len(header)
    print(f"\n\n{divider}\n{header}\n{divider}")

    for p, bm, mm in zip(PRIORITIES, baseline_metrics, mvee_metrics):
        b_t = bm.total_planning_time_s * 1e3
        m_t = mm.total_planning_time_s * 1e3
        speedup = (b_t / m_t) if m_t > 0 else 0.0
        pr = mm.prefilter_prune_rate or 0.0
        print(
            f"{p:<{col_w}}"
            f"{b_t:>18.2f}"
            f"{m_t:>14.2f}"
            f"{speedup:>9.2f}×"
            f"{mm.n_collision_pairs_checked:>10}"
            f"{mm.n_prefilter_pruned:>8}"
            f"{mm.n_fcl_checked:>6}"
            f"{pr*100:>10.1f}%"
            f"{mm.n_collisions_detected:>9}"
        )
    print(divider)

    # ── plots ─────────────────────────────────────────────────────────────────
    plot_results(
        PRIORITIES, baseline_metrics, mvee_metrics,
        out_path="prefilter_benchmark.png",
    )


if __name__ == "__main__":
    main()
