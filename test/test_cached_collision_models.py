"""Tests for CachedCollisionModel subclasses and AgentModel prefilter field."""
from __future__ import annotations

import unittest
from unittest import mock
import numpy as np

from pyrobopath.toolpath import Contour
from pyrobopath.collision_detection import (
    CachedCollisionModel,
    MVEECachedModel,
    BhattacharyyaCachedModel,
    CascadeCachedModel,
    LearnedCachedModel,
)
from pyrobopath.collision_detection.cached_models import _contour_to_raw_task
from pyrobopath.process import AgentModel


def _box_contour(x0, y0, x1, y1, z=0.0, n=8):
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    path = [np.array([float(x), float(y), z]) for x, y in zip(xs, ys)]
    return Contour(path=path, tool=0)


class TestCachedCollisionModelBase(unittest.TestCase):
    def test_is_abstract(self):
        with self.assertRaises(TypeError):
            CachedCollisionModel()

    def test_build_cache_populates_entries(self):
        m = MVEECachedModel(safety_m=0.0)
        contours = [
            _box_contour(0.0, 0.0, 1.0, 0.0),
            _box_contour(10.0, 0.0, 11.0, 0.0),
            _box_contour(5.0, 5.0, 6.0, 6.0),
        ]
        m.build_cache(contours)
        self.assertEqual(len(m._cache), 3)
        for c in contours:
            self.assertIn(c.id, m._cache)

    def test_is_safe_pair_missing_ids_returns_false(self):
        m = MVEECachedModel(safety_m=0.0)
        c1 = _box_contour(0.0, 0.0, 1.0, 0.0)
        m.build_cache([c1])
        base_a = np.array([-1.0, 0.0, 0.0])
        base_b = np.array([1.0, 0.0, 0.0])
        self.assertFalse(m.is_safe_pair(c1.id, 9999, base_a, base_b))
        self.assertFalse(m.is_safe_pair(9998, c1.id, base_a, base_b))
        self.assertFalse(m.is_safe_pair(9998, 9999, base_a, base_b))

    def test_in_collision_returns_false(self):
        m = MVEECachedModel()
        self.assertFalse(m.in_collision(m))


class _SeparatedAndOverlappingMixin:
    """Mixin with the standard separated/overlapping assertions."""

    model_cls = None  # override in subclass

    def _make_model(self):
        return self.model_cls(safety_m=0.05)

    def test_clearly_separated_contours_safe(self):
        m = self._make_model()
        c_a = _box_contour(0.0, 0.0, 1.0, 0.0)
        c_b = _box_contour(10.0, 10.0, 11.0, 11.0)
        m.build_cache([c_a, c_b])
        base_a = np.array([-1.0, 0.0, 0.0])
        base_b = np.array([12.0, 12.0, 0.0])
        self.assertTrue(m.is_safe_pair(c_a.id, c_b.id, base_a, base_b))

    def test_clearly_overlapping_contours_not_safe(self):
        m = self._make_model()
        c_a = _box_contour(0.0, 0.0, 1.0, 1.0)
        c_b = _box_contour(0.0, 0.0, 1.0, 1.0)
        m.build_cache([c_a, c_b])
        base_a = np.array([-1.0, 0.0, 0.0])
        base_b = np.array([-1.0, 0.0, 0.0])
        self.assertFalse(m.is_safe_pair(c_a.id, c_b.id, base_a, base_b))

    def test_3d_bases_accepted(self):
        """Bases passed as 3D arrays (pyrobopath style) are accepted."""
        m = self._make_model()
        c_a = _box_contour(0.0, 0.0, 1.0, 0.0)
        c_b = _box_contour(10.0, 0.0, 11.0, 0.0)
        m.build_cache([c_a, c_b])
        base_a = np.array([-1.0, 0.0, 5.0])
        base_b = np.array([12.0, 0.0, 5.0])
        self.assertTrue(m.is_safe_pair(c_a.id, c_b.id, base_a, base_b))


class TestMVEECachedModel(_SeparatedAndOverlappingMixin, unittest.TestCase):
    model_cls = MVEECachedModel


class TestBhattacharyyaCachedModel(_SeparatedAndOverlappingMixin, unittest.TestCase):
    model_cls = BhattacharyyaCachedModel


class TestCascadeCachedModel(_SeparatedAndOverlappingMixin, unittest.TestCase):
    model_cls = CascadeCachedModel


class TestAgentModelRegression(unittest.TestCase):
    """Ensure AgentModel with collision_prefilter=None still works (no regression)."""

    def test_agent_model_without_prefilter(self):
        agent = AgentModel(
            capabilities=[0],
            collision_model=None,
            base_frame_position=np.array([0.0, 0.0, 0.0]),
            home_position=np.array([0.0, 0.0, 10.0]),
            velocity=5.0,
            travel_velocity=10.0,
        )
        self.assertIsNone(agent.collision_prefilter)

    def test_agent_model_with_prefilter(self):
        prefilter = MVEECachedModel(safety_m=0.05)
        agent = AgentModel(
            capabilities=[0],
            collision_model=None,
            base_frame_position=np.array([0.0, 0.0, 0.0]),
            home_position=np.array([0.0, 0.0, 10.0]),
            velocity=5.0,
            travel_velocity=10.0,
            collision_prefilter=prefilter,
        )
        self.assertIs(agent.collision_prefilter, prefilter)


class TestSharedCache(unittest.TestCase):
    """Shared class-level cache tests (spec v2 Change 1)."""

    def setUp(self):
        CachedCollisionModel.clear_cache()

    def tearDown(self):
        CachedCollisionModel.clear_cache()

    def test_cache_shared_across_instances_same_class(self):
        m1 = MVEECachedModel(safety_m=0.0)
        m2 = MVEECachedModel(safety_m=0.0)
        contours = [
            _box_contour(0.0, 0.0, 1.0, 0.0),
            _box_contour(10.0, 0.0, 11.0, 0.0),
        ]
        m1.build_cache(contours)
        # second instance sees the same entries without its own build
        self.assertEqual(len(m2._cache), 2)
        for c in contours:
            self.assertIn(c.id, m2._cache)
        # they are literally the same dict
        self.assertIs(m1._cache, m2._cache)

    def test_cache_isolated_between_subclasses(self):
        m_mvee = MVEECachedModel()
        m_bh = BhattacharyyaCachedModel()
        c1 = _box_contour(0.0, 0.0, 1.0, 0.0)
        m_mvee.build_cache([c1])
        self.assertIn(c1.id, m_mvee._cache)
        self.assertNotIn(c1.id, m_bh._cache)

    def test_clear_cache_all(self):
        m = MVEECachedModel()
        m.build_cache([_box_contour(0.0, 0.0, 1.0, 0.0)])
        self.assertEqual(len(m._cache), 1)
        CachedCollisionModel.clear_cache()
        # After clearing, accessing _cache re-creates empty slot
        m2 = MVEECachedModel()
        self.assertEqual(len(m2._cache), 0)

    def test_clear_cache_specific_class(self):
        m_mvee = MVEECachedModel()
        m_bh = BhattacharyyaCachedModel()
        c1 = _box_contour(0.0, 0.0, 1.0, 0.0)
        m_mvee.build_cache([c1])
        m_bh.build_cache([c1])
        CachedCollisionModel.clear_cache(MVEECachedModel)
        self.assertEqual(len(MVEECachedModel().  _cache), 0)
        self.assertEqual(len(m_bh._cache), 1)

    def test_build_cache_idempotent(self):
        """Second build_cache call with same IDs does NOT recompute."""
        m = MVEECachedModel()
        c1 = _box_contour(0.0, 0.0, 1.0, 0.0)
        m.build_cache([c1])
        first_entry = m._cache[c1.id]
        # Mutate the path to check that the old entry is preserved
        c1.path[0] = np.array([999.0, 999.0, 999.0])
        m.build_cache([c1])
        self.assertIs(m._cache[c1.id], first_entry)


class TestSchedulerPrefilterEnforcement(unittest.TestCase):
    """Scheduler raises ValueError when agents have different prefilter instances."""

    def setUp(self):
        CachedCollisionModel.clear_cache()

    def tearDown(self):
        CachedCollisionModel.clear_cache()

    def test_different_prefilter_instances_raise(self):
        from pyrobopath.toolpath_scheduling import (
            MultiAgentToolpathPlanner,
            PlanningOptions,
        )
        from pyrobopath.toolpath import Toolpath
        from pyrobopath.process import DependencyGraph

        pre_a = MVEECachedModel(safety_m=0.0)
        pre_b = MVEECachedModel(safety_m=0.0)
        self.assertIsNot(pre_a, pre_b)

        agents = {
            "a1": AgentModel(
                capabilities=[0], collision_model=None,
                base_frame_position=np.array([0.0, 0.0, 0.0]),
                home_position=np.array([0.0, 0.0, 10.0]),
                velocity=5.0, travel_velocity=10.0,
                collision_prefilter=pre_a,
            ),
            "a2": AgentModel(
                capabilities=[0], collision_model=None,
                base_frame_position=np.array([1.0, 0.0, 0.0]),
                home_position=np.array([1.0, 0.0, 10.0]),
                velocity=5.0, travel_velocity=10.0,
                collision_prefilter=pre_b,
            ),
        }
        planner = MultiAgentToolpathPlanner(agents)
        tp = Toolpath()
        tp.contours = [_box_contour(0.0, 0.0, 1.0, 0.0)]
        dg = DependencyGraph()
        for i, _c in enumerate(tp.contours):
            dg.add_node(i)
        with self.assertRaises(ValueError):
            planner.plan(tp, dg, PlanningOptions())

    def test_shared_prefilter_instance_ok(self):
        from pyrobopath.toolpath_scheduling import (
            MultiAgentToolpathPlanner,
            PlanningOptions,
        )
        from pyrobopath.toolpath import Toolpath
        from pyrobopath.process import DependencyGraph

        pre = MVEECachedModel(safety_m=0.0)
        agents = {
            "a1": AgentModel(
                capabilities=[0], collision_model=None,
                base_frame_position=np.array([0.0, 0.0, 0.0]),
                home_position=np.array([0.0, 0.0, 10.0]),
                velocity=5.0, travel_velocity=10.0,
                collision_prefilter=pre,
            ),
            "a2": AgentModel(
                capabilities=[0], collision_model=None,
                base_frame_position=np.array([5.0, 5.0, 0.0]),
                home_position=np.array([5.0, 5.0, 10.0]),
                velocity=5.0, travel_velocity=10.0,
                collision_prefilter=pre,
            ),
        }
        planner = MultiAgentToolpathPlanner(agents)
        tp = Toolpath()
        tp.contours = [_box_contour(0.0, 0.0, 1.0, 0.0)]
        dg = DependencyGraph()
        for i, _c in enumerate(tp.contours):
            dg.add_node(i)
        # Should not raise — both agents share the same instance.
        planner.plan(tp, dg, PlanningOptions())


class TestContourToRawTask(unittest.TestCase):
    def test_conversion_shape_and_types(self):
        c = _box_contour(0.0, 0.0, 1.0, 0.0, n=5)
        raw = _contour_to_raw_task(c)
        self.assertEqual(raw.gcode_id, "pyrobopath")
        self.assertEqual(raw.task_id, str(c.id))
        self.assertEqual(len(raw.polyline), 5)
        for i, p in enumerate(raw.polyline):
            self.assertEqual(len(p), 3)
            self.assertIsInstance(p[0], float)
            self.assertIsInstance(p[1], float)
            self.assertEqual(p[2], float(i))
        self.assertEqual(raw.duration, 4.0)


class TestLearnedCachedModel(unittest.TestCase):
    """Tests for LearnedCachedModel — mock-based to avoid needing a checkpoint."""

    def setUp(self):
        CachedCollisionModel.clear_cache()

    def tearDown(self):
        CachedCollisionModel.clear_cache()

    def test_build_cache_with_mocked_predictor(self):
        """LearnedCachedModel.build_cache populates _cache via mocked encode_task."""
        with mock.patch.object(
            LearnedCachedModel, "__init__", lambda self, safety_m=0.0: (
                CachedCollisionModel.__init__(self, safety_m=safety_m),
                setattr(self, "_predictor", mock.MagicMock()),
                setattr(self, "_arm_params", mock.MagicMock()),
                setattr(self, "_tmp_cache_path", "/tmp/_nonexistent_xyz"),
            )[0],
        ):
            m = LearnedCachedModel(safety_m=0.0)
            # encode_task returns a dummy tuple; we just need it to be called
            m._predictor.encode_task = mock.MagicMock(return_value=(object(), object()))

            contours = [
                _box_contour(0.0, 0.0, 1.0, 0.0),
                _box_contour(2.0, 2.0, 3.0, 3.0),
            ]
            m.build_cache(contours)
            self.assertEqual(len(m._cache), 2)
            for c in contours:
                self.assertIn(c.id, m._cache)
                # cache entry is just the contour.id sentinel (int)
                # the real data lives in _predictor._cache (EmbeddingCache)
                self.assertEqual(m._cache[c.id], c.id)
            # encode_task should have been called once per contour
            self.assertEqual(m._predictor.encode_task.call_count, 2)


if __name__ == '__main__':
    unittest.main()
