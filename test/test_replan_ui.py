"""Headless smoke tests for the replan UI in animate_multi_agent_toolpath_full.

These tests verify the UI state transitions (agent toggling, replan at cursor,
reset) without actually rendering a GUI.  The ``apply_replan`` callable is
exposed in the controls dict so tests can trigger it directly.
"""
from __future__ import annotations

import unittest
import matplotlib
matplotlib.use("Agg")
import numpy as np

from pyrobopath.process import AgentModel, create_dependency_graph_by_z
from pyrobopath.toolpath import Contour, Toolpath
from pyrobopath.collision_detection import FCLRobotBBCollisionModel
from pyrobopath.toolpath_scheduling import (
    ContourEvent,
    MultiAgentToolpathPlanner,
    PlanningOptions,
    animate_multi_agent_toolpath_full,
    extract_replan_context,
    replan,
)


# ─────────────────────────── fixtures ─────────────────────────────────────────

def _sq(cx, cy, z=0.0, h=20.0):
    return Contour(path=[
        np.array([cx - h, cy - h, z]),
        np.array([cx + h, cy - h, z]),
        np.array([cx + h, cy + h, z]),
        np.array([cx - h, cy + h, z]),
        np.array([cx - h, cy - h, z]),
    ], tool=0)


def _build_setup():
    contours = [_sq(-200 + i * 5, i * 60) for i in range(3)] + \
               [_sq( 200 - i * 5, i * 60) for i in range(3)]
    tp = Toolpath(contours)
    dg = create_dependency_graph_by_z(tp)

    bf1 = np.array([-350.0, 0, 0])
    bf2 = np.array([ 350.0, 0, 0])
    am = {
        "robot1": AgentModel(
            base_frame_position=bf1,
            home_position=np.array([-250.0, 0, 0]),
            capabilities=[0], velocity=50, travel_velocity=50,
            collision_model=FCLRobotBBCollisionModel((200, 50, 300), bf1),
        ),
        "robot2": AgentModel(
            base_frame_position=bf2,
            home_position=np.array([250.0, 0, 0]),
            capabilities=[0], velocity=50, travel_velocity=50,
            collision_model=FCLRobotBBCollisionModel((200, 50, 300), bf2),
        ),
    }
    opts = PlanningOptions(
        retract_height=10.0, collision_offset=3.0, collision_gap_threshold=5.0,
    )
    sched = MultiAgentToolpathPlanner(am).plan(tp, dg, opts)

    # Closure-based replan_fn — mirrors how a user would wire it up.
    def replan_fn(t_replan, disabled):
        ctx = extract_replan_context(sched, t_replan, disabled_agents=disabled)
        return replan(am, tp, dg, opts, ctx)

    return tp, dg, am, opts, sched, replan_fn


def _contour_ids_in_schedule(schedule):
    return {
        ev.contour.id
        for s in schedule.schedules.values()
        for ev in s._events
        if isinstance(ev, ContourEvent)
    }


# ─────────────────────────── tests ────────────────────────────────────────────

class TestAnimateConstruction(unittest.TestCase):
    """Without replan_fn, the legacy UI still builds cleanly."""

    def test_construct_without_replan(self):
        tp, _, am, _, sched, _ = _build_setup()
        fig, ctrls = animate_multi_agent_toolpath_full(
            tp, sched, am, show=False,
        )
        self.assertNotIn("agent_checks", ctrls)
        self.assertNotIn("btn_replan", ctrls)
        self.assertIn("button", ctrls)   # play button always present


class TestAnimateWithReplan(unittest.TestCase):
    def setUp(self):
        self.tp, self.dg, self.am, self.opts, self.sched, self.rfn = _build_setup()
        self.fig, self.ctrls = animate_multi_agent_toolpath_full(
            self.tp, self.sched, self.am, show=False, replan_fn=self.rfn,
        )

    def test_replan_widgets_present(self):
        for key in ("agent_checks", "btn_replan", "btn_reset", "apply_replan"):
            self.assertIn(key, self.ctrls, f"missing {key}")

    def test_checkbox_toggle_updates_disabled_set(self):
        self.assertEqual(self.ctrls["state"]["disabled_agents"], set())
        self.ctrls["agent_checks"].set_active(1)   # toggle robot2
        self.assertEqual(self.ctrls["state"]["disabled_agents"], {"robot2"})
        self.ctrls["agent_checks"].set_active(1)   # toggle back
        self.assertEqual(self.ctrls["state"]["disabled_agents"], set())

    def test_replan_at_zero_preserves_all_tasks(self):
        """Replan at t=0 with both agents enabled should re-plan all tasks."""
        original_ids = _contour_ids_in_schedule(self.sched)
        self.ctrls["apply_replan"](0.0)
        new_ids = _contour_ids_in_schedule(self.ctrls["view"].schedule)
        self.assertEqual(new_ids, original_ids)
        self.assertEqual(self.ctrls["state"]["completed_override"], set())

    def test_replan_mid_schedule_moves_cursor_to_zero(self):
        t_mid = self.sched.end_time() / 2.0
        self.ctrls["state"]["t"] = t_mid
        self.ctrls["apply_replan"](t_mid)
        self.assertAlmostEqual(self.ctrls["state"]["t"], 0.0, places=6)
        self.assertEqual(self.ctrls["anim_slider"].valmin, 0.0)

    def test_replan_marks_completed_tasks(self):
        """After replanning mid-schedule, tasks that finished before the cursor
        are added to completed_override and won't appear in the new schedule."""
        t_mid = self.sched.end_time() / 2.0
        original_ids = _contour_ids_in_schedule(self.sched)

        self.ctrls["apply_replan"](t_mid)

        new_ids = _contour_ids_in_schedule(self.ctrls["view"].schedule)
        completed = self.ctrls["state"]["completed_override"]
        # All original contours are either in the new schedule or marked completed
        self.assertEqual(new_ids | completed, original_ids)
        self.assertEqual(new_ids & completed, set())

    def test_replan_with_disabled_agent_empties_its_schedule(self):
        self.ctrls["agent_checks"].set_active(1)   # disable robot2
        self.ctrls["apply_replan"](0.0)
        r2_sched = self.ctrls["view"].schedule.schedules.get("robot2")
        if r2_sched is not None:
            contour_events = [
                ev for ev in r2_sched._events if isinstance(ev, ContourEvent)
            ]
            self.assertEqual(contour_events, [])

    def test_reset_clears_completed_override(self):
        t_mid = self.sched.end_time() / 2.0
        self.ctrls["apply_replan"](t_mid)
        self.assertGreater(len(self.ctrls["state"]["completed_override"]), 0)

        # Reset mimics on_reset: clear + replan(0.0)
        self.ctrls["state"]["completed_override"] = set()
        self.ctrls["apply_replan"](0.0)
        self.assertEqual(self.ctrls["state"]["completed_override"], set())

    def test_anim_model_schedule_swapped_after_replan(self):
        """After replan, each agent's AnimationModel must point at the new
        per-agent schedule (so workspace playback reads from the right data)."""
        self.ctrls["apply_replan"](0.0)
        view = self.ctrls["view"]
        for i, agent in enumerate(("robot1", "robot2")):
            anim_m = view.anim_models[i]
            self.assertIs(anim_m.sched, view.schedule.schedules[agent])

    def test_all_tasks_remain_accounted_for(self):
        """Completed + newly-assigned must cover the original toolpath ids."""
        original_ids = {c.id for c in self.tp.contours}
        t_mid = self.sched.end_time() / 2.0
        self.ctrls["apply_replan"](t_mid)

        ever = _contour_ids_in_schedule(self.ctrls["view"].schedule) \
             | self.ctrls["state"]["completed_override"]
        self.assertEqual(original_ids, ever)


if __name__ == "__main__":
    unittest.main()
