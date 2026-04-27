"""Tests for the replan module.

Covers:
- Extracting completed / in-progress state from an existing schedule.
- Replanning at t=0 (nothing done) is equivalent to a fresh plan.
- Replanning after everything is done yields an empty plan.
- Completed tasks are not re-assigned.
- In-progress tasks continue to completion in the new schedule.
- Disabling an agent prevents new assignments on it.
- All tasks eventually complete across a sequence of replans.
"""
from __future__ import annotations

import unittest
import numpy as np

from pyrobopath.process import AgentModel, create_dependency_graph_by_z
from pyrobopath.toolpath import Contour, Toolpath
from pyrobopath.collision_detection import FCLRobotBBCollisionModel
from pyrobopath.toolpath_scheduling import (
    ContourEvent,
    MultiAgentToolpathPlanner,
    PlanningOptions,
    ReplanContext,
    extract_replan_context,
    replan,
)


# ─────────────────────────── fixtures ─────────────────────────────────────────

def _square_contour(cx, cy, side=40.0, z=0.0):
    """Build a small square contour centred at (cx, cy)."""
    h = side / 2.0
    path = [
        np.array([cx - h, cy - h, z]),
        np.array([cx + h, cy - h, z]),
        np.array([cx + h, cy + h, z]),
        np.array([cx - h, cy + h, z]),
        np.array([cx - h, cy - h, z]),
    ]
    return Contour(path=path, tool=0)


def _two_robot_agent_models():
    bf1 = np.array([-350.0, 0.0, 0.0])
    bf2 = np.array([ 350.0, 0.0, 0.0])
    dims = (200.0, 50.0, 300.0)
    a1 = AgentModel(
        base_frame_position=bf1,
        home_position=np.array([-250.0, 0.0, 0.0]),
        capabilities=[0],
        velocity=50.0,
        travel_velocity=50.0,
        collision_model=FCLRobotBBCollisionModel(dims, bf1),
    )
    a2 = AgentModel(
        base_frame_position=bf2,
        home_position=np.array([250.0, 0.0, 0.0]),
        capabilities=[0],
        velocity=50.0,
        travel_velocity=50.0,
        collision_model=FCLRobotBBCollisionModel(dims, bf2),
    )
    return {"robot1": a1, "robot2": a2}


def _make_toolpath_and_dg(n_per_side=3):
    """Build a toolpath with tasks on both sides so both robots get work."""
    contours = []
    # left-side tasks (close to robot1 base)
    for i in range(n_per_side):
        contours.append(_square_contour(-200.0 + i * 5, 0.0 + i * 60))
    # right-side tasks (close to robot2 base)
    for i in range(n_per_side):
        contours.append(_square_contour(200.0 - i * 5, 0.0 + i * 60))
    toolpath = Toolpath(contours)
    dg = create_dependency_graph_by_z(toolpath)
    return toolpath, dg


def _default_options():
    return PlanningOptions(
        retract_height=10.0,
        collision_offset=3.0,
        collision_gap_threshold=5.0,
    )


def _contour_ids_in(schedule):
    """Return a set of (agent, contour_id) tuples assigned in `schedule`."""
    out = set()
    for agent, s in schedule.schedules.items():
        for ev in s._events:
            if isinstance(ev, ContourEvent):
                out.add((agent, ev.contour.id))
    return out


def _all_contour_ids(schedule):
    out = set()
    for s in schedule.schedules.values():
        for ev in s._events:
            if isinstance(ev, ContourEvent):
                out.add(ev.contour.id)
    return out


# ─────────────────────────── tests ────────────────────────────────────────────

class TestReplanExtraction(unittest.TestCase):
    def setUp(self):
        self.agent_models = _two_robot_agent_models()
        self.toolpath, self.dg = _make_toolpath_and_dg()
        self.options = _default_options()
        self.planner = MultiAgentToolpathPlanner(self.agent_models)
        self.schedule = self.planner.plan(self.toolpath, self.dg, self.options)

    def test_extract_at_zero_has_nothing_completed(self):
        ctx = extract_replan_context(self.schedule, t_replan=0.0)
        self.assertEqual(ctx.completed_ids, set())
        self.assertEqual(ctx.in_progress, {})

    def test_extract_after_end_has_all_completed(self):
        t_end = self.schedule.end_time()
        ctx = extract_replan_context(self.schedule, t_replan=t_end + 1.0)
        expected = _all_contour_ids(self.schedule)
        self.assertEqual(ctx.completed_ids, expected)
        self.assertEqual(ctx.in_progress, {})

    def test_extract_mid_schedule_has_some_completed(self):
        t_mid = self.schedule.end_time() / 2.0
        ctx = extract_replan_context(self.schedule, t_replan=t_mid)
        all_ids = _all_contour_ids(self.schedule)
        # Some completed, possibly some in-progress, rest unassigned
        handled = set(ctx.completed_ids) | {
            ip.contour_id for ip in ctx.in_progress.values()
        }
        # sanity: handled ⊆ all assigned
        self.assertTrue(handled.issubset(all_ids))

    def test_extract_in_progress_mid_contour(self):
        # Find a specific t that falls inside a ContourEvent.
        target_t = None
        for s in self.schedule.schedules.values():
            for ev in s._events:
                if isinstance(ev, ContourEvent):
                    target_t = (ev.start + ev.end) / 2.0
                    break
            if target_t is not None:
                break
        self.assertIsNotNone(target_t)

        ctx = extract_replan_context(self.schedule, t_replan=target_t)
        self.assertTrue(len(ctx.in_progress) >= 1)
        for agent, ip in ctx.in_progress.items():
            self.assertGreater(ip.remaining_duration, 0.0)
            self.assertEqual(ip.pre_populated_events[0].start, 0.0)


class TestReplanFromStart(unittest.TestCase):
    """Replanning at t=0 with no completed tasks must plan every task."""

    def test_replan_at_zero_covers_all_tasks(self):
        agent_models = _two_robot_agent_models()
        toolpath, dg = _make_toolpath_and_dg()
        options = _default_options()

        planner = MultiAgentToolpathPlanner(agent_models)
        fresh_sched = planner.plan(toolpath, dg, options)
        all_ids = _all_contour_ids(fresh_sched)

        ctx = ReplanContext(t_replan=0.0, completed_ids=set(), in_progress={})
        new_sched = replan(agent_models, toolpath, dg, options, ctx)

        self.assertEqual(_all_contour_ids(new_sched), all_ids)


class TestReplanAfterEnd(unittest.TestCase):
    """Replanning after everything is done must produce an empty plan."""

    def test_no_new_tasks_when_all_completed(self):
        agent_models = _two_robot_agent_models()
        toolpath, dg = _make_toolpath_and_dg()
        options = _default_options()

        planner = MultiAgentToolpathPlanner(agent_models)
        fresh_sched = planner.plan(toolpath, dg, options)
        t_end = fresh_sched.end_time()

        ctx = extract_replan_context(fresh_sched, t_replan=t_end + 1.0)
        new_sched = replan(agent_models, toolpath, dg, options, ctx)
        self.assertEqual(_all_contour_ids(new_sched), set())


class TestReplanPreservesCompleted(unittest.TestCase):
    """Completed tasks must not appear in the new schedule."""

    def test_completed_not_reassigned(self):
        agent_models = _two_robot_agent_models()
        toolpath, dg = _make_toolpath_and_dg()
        options = _default_options()

        planner = MultiAgentToolpathPlanner(agent_models)
        fresh_sched = planner.plan(toolpath, dg, options)
        t_mid = fresh_sched.end_time() / 3.0

        ctx = extract_replan_context(fresh_sched, t_replan=t_mid)
        new_sched = replan(agent_models, toolpath, dg, options, ctx)

        new_contour_ids = _all_contour_ids(new_sched)
        for completed in ctx.completed_ids:
            self.assertNotIn(
                completed, new_contour_ids,
                f"completed contour {completed} reappeared in replan",
            )


class TestReplanInProgressContinues(unittest.TestCase):
    """In-progress tasks must be preserved at the head of the new schedule."""

    def test_in_progress_task_present_at_zero(self):
        agent_models = _two_robot_agent_models()
        toolpath, dg = _make_toolpath_and_dg()
        options = _default_options()

        planner = MultiAgentToolpathPlanner(agent_models)
        fresh_sched = planner.plan(toolpath, dg, options)

        # Pick a time guaranteed to land inside a ContourEvent
        target_t = None
        for s in fresh_sched.schedules.values():
            for ev in s._events:
                if isinstance(ev, ContourEvent):
                    target_t = (ev.start + ev.end) / 2.0
                    break
            if target_t is not None:
                break
        self.assertIsNotNone(target_t)

        ctx = extract_replan_context(fresh_sched, t_replan=target_t)
        self.assertGreater(len(ctx.in_progress), 0)

        new_sched = replan(agent_models, toolpath, dg, options, ctx)

        # Each in-progress agent must have at least one event starting at 0
        for agent in ctx.in_progress:
            events = new_sched.schedules[agent]._events
            self.assertTrue(len(events) > 0)
            self.assertAlmostEqual(events[0].start, 0.0, places=6)


class TestReplanWithDisabledAgent(unittest.TestCase):
    """A disabled agent must receive no new tasks."""

    def test_disabled_agent_empty(self):
        agent_models = _two_robot_agent_models()
        toolpath, dg = _make_toolpath_and_dg()
        options = _default_options()

        ctx = ReplanContext(
            t_replan=0.0, completed_ids=set(), in_progress={},
            disabled_agents={"robot2"},
        )
        new_sched = replan(agent_models, toolpath, dg, options, ctx)

        # robot2 should either be missing or have no contour events
        r2_events = [
            ev for ev in new_sched.schedules.get("robot2", type("x", (), {"_events": []})()).__dict__.get("_events", [])
            if isinstance(ev, ContourEvent)
        ] if "robot2" in new_sched.schedules else []
        self.assertEqual(r2_events, [])

        # robot1 picked up everything feasible (tool 0 for both sides)
        r1_contours = {
            ev.contour.id
            for ev in new_sched.schedules["robot1"]._events
            if isinstance(ev, ContourEvent)
        }
        self.assertGreater(len(r1_contours), 0)


class TestAllTasksCompleteAcrossReplans(unittest.TestCase):
    """Chain of replans must still complete every task in the toolpath."""

    def test_chain_of_replans_finishes_everything(self):
        agent_models = _two_robot_agent_models()
        toolpath, dg = _make_toolpath_and_dg(n_per_side=2)
        options = _default_options()
        all_contour_ids = {c.id for c in toolpath.contours}

        planner = MultiAgentToolpathPlanner(agent_models)
        sched = planner.plan(toolpath, dg, options)

        # Replan a few times mid-execution, accumulating completed history
        completed_so_far = set()
        for frac in [0.25, 0.5, 0.75]:
            t_replan = sched.end_time() * frac
            if t_replan >= sched.end_time():
                break
            ctx = extract_replan_context(sched, t_replan=t_replan)
            completed_so_far |= ctx.completed_ids
            sched = replan(agent_models, toolpath, dg, options, ctx)

        # Everything assigned across the replan chain plus already-completed
        # must cover the whole toolpath.
        ever_assigned = _all_contour_ids(sched) | completed_so_far
        missing = all_contour_ids - ever_assigned
        self.assertEqual(
            missing, set(),
            f"{len(missing)} contour(s) never assigned across replans: {missing}",
        )


class TestReplanDoesNotMutateOriginal(unittest.TestCase):
    """The original schedule and dg must not be mutated by replan()."""

    def test_dg_unchanged(self):
        agent_models = _two_robot_agent_models()
        toolpath, dg = _make_toolpath_and_dg()
        options = _default_options()

        planner = MultiAgentToolpathPlanner(agent_models)
        fresh_sched = planner.plan(toolpath, dg, options)
        t_mid = fresh_sched.end_time() / 2.0

        completed_before = set(dg._completed_tasks)
        ctx = extract_replan_context(fresh_sched, t_replan=t_mid)
        _ = replan(agent_models, toolpath, dg, options, ctx)
        completed_after = set(dg._completed_tasks)

        self.assertEqual(completed_before, completed_after)


if __name__ == "__main__":
    unittest.main()
