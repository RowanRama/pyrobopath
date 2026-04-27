"""Prefilter animation example.

Schedules an N-robot toolpath with the MVEE prefilter and produces an
animated visualisation of the resulting schedule.  Robots are placed equally
spaced around a circle of radius BASE_RADIUS.

Usage::

    cd examples/
    python prefilter_example.py                # default: 2 robots
    python prefilter_example.py --n-robots 4
    python prefilter_example.py --n-robots 16
"""
import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from pyrobopath.process import AgentModel, create_dependency_graph_by_z
from pyrobopath.toolpath.preprocessing import *
from pyrobopath.collision_detection import FCLRobotBBCollisionModel, MVEECachedModel
from pyrobopath.toolpath_scheduling import (
    MultiAgentToolpathPlanner,
    DepthBasedSequentialPlanner,
    DepthBasedParallelPlanner,
    BatchedSequentialPlanner,
    BatchedParallelPlanner,
    PlanningOptions,
    PlanningMetrics,
    animate_multi_agent_toolpath_full,
    extract_replan_context,
    replan as replan_schedule,
)

from utilities import toolpath_from_gcode, print_schedule_info


GCODE_FILE = "../test/test_gcode/aim.gcode"
LAYER_RANGE = (0, 1)
SAFETY_M    = 50

BASE_RADIUS = 350.0   # mm — robot base frames placed on this circle
HOME_RADIUS = 250.0   # mm — home positions on the same ray, inside the base
MAX_ROBOTS  = 16
COL_DIMS    = (200.0, 50.0, 300.0)


def _robot_positions(n_robots):
    """Return (bases, homes) — each a list of (3,) arrays for n_robots
    equally spaced around a circle of radius BASE_RADIUS, starting at θ=0.
    Homes sit on the same ray at HOME_RADIUS.
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


def make_agent_models(n_robots, prefilter=None):
    """Build n_robots agent models placed equally around the table."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Animated multi-robot MVEE-prefilter planning demo."
    )
    parser.add_argument(
        "--n-robots", type=int, default=2,
        help=f"number of robots (1–{MAX_ROBOTS}, default: 2), placed equally "
             f"spaced around a circle of radius {BASE_RADIUS:.0f} mm",
    )
    args = parser.parse_args()

    if not (1 <= args.n_robots <= MAX_ROBOTS):
        parser.error(f"--n-robots must be in [1, {MAX_ROBOTS}]")

    toolpath = toolpath_from_gcode(GCODE_FILE)

    preprocessor = ToolpathPreprocessor()
    preprocessor.add_step(MaxContourLengthStep(500.0))
    preprocessor.add_step(LayerRangeStep(*LAYER_RANGE))
    preprocessor.process(toolpath)
    dg = create_dependency_graph_by_z(toolpath)

    prefilter = MVEECachedModel(safety_m=SAFETY_M)
    agent_models = make_agent_models(args.n_robots)

    options = PlanningOptions(
        retract_height=10.0,
        collision_offset=3.0,
        collision_gap_threshold=5.0,
        # task_priority="farthest_centroid",
        task_priority="nearest_start",
        # task_priority="nearest_base",
    )

    metrics = PlanningMetrics()
    planner = MultiAgentToolpathPlanner(agent_models)

    print(f"Planning schedule with {args.n_robots} robots and MVEE prefilter…")
    sched = planner.plan(toolpath, dg, options, metrics=metrics)

    print_schedule_info(sched)
    metrics.print_summary("MVEE")

    # Closure capturing the current schedule so the replan UI can resume from
    # any time t against the same toolpath / dg / options.  Must be a closure
    # because each replan needs to see the most recent schedule state.
    current = {"sched": sched}

    def replan_fn(t_replan, disabled_agents):
        ctx = extract_replan_context(
            current["sched"], t_replan, disabled_agents=disabled_agents,
        )
        new_sched = replan_schedule(
            agent_models, toolpath, dg, options, ctx,
        )
        current["sched"] = new_sched
        return new_sched

    animate_multi_agent_toolpath_full(
        toolpath, sched, agent_models,
        limits=((-550, 550), (-550, 550)),
        replan_fn=replan_fn,
    )


if __name__ == "__main__":
    main()
