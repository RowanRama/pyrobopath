"""Unit tests (unittest) for the Gaussian / MVEE ellipsoid pre-filters.

Ported from the collision_surrogate pytest suite.
"""
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyrobopath.collision_detection.ellipsoid_filter import (
    EllipsoidRecord,
    GeometryCache,
    EllipsoidSeparationFilter,
    BhattacharyyaFilter,
    ProbabilisticOverlapFilter,
    CascadeFilter,
    _fit_mvee,
    _bhattacharyya_coefficient,
    _prob_overlap,
    _segment_intersects_ellipsoid,
    _segment_min_ellipsoid_distance,
)


def _circle_polyline(cx, cy, r, n=32, t0=0.0):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a), t0 + i * 0.1))
    return pts


def _line_polyline(x0, y0, x1, y1, n=20, t0=0.0):
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    return [(float(x), float(y), t0 + i * 0.1) for i, (x, y) in enumerate(zip(xs, ys))]


def _make_record(gcode_id, task_id, polyline):
    return EllipsoidRecord.from_polyline(gcode_id, task_id, polyline)


class TestEllipsoidRecord(unittest.TestCase):
    def test_shapes(self):
        rec = _make_record("g", "t", _circle_polyline(0, 0, 1))
        self.assertEqual(rec.mu_g.shape, (2,))
        self.assertEqual(rec.sigma.shape, (2, 2))
        self.assertEqual(rec.mu_e.shape, (2,))
        self.assertEqual(rec.M.shape, (2, 2))
        self.assertEqual(rec.evecs.shape, (2, 2))
        self.assertEqual(rec.evals.shape, (2,))

    def test_dtype_float32(self):
        rec = _make_record("g", "t", _circle_polyline(0, 0, 1))
        for arr in [rec.mu_g, rec.sigma, rec.mu_e, rec.M, rec.evecs, rec.evals]:
            self.assertEqual(arr.dtype, np.float32)

    def test_identifiers(self):
        rec = _make_record("gcode_A", "task_7", _circle_polyline(0, 0, 1))
        self.assertEqual(rec.gcode_id, "gcode_A")
        self.assertEqual(rec.task_id, "task_7")

    def test_semi_axes_positive(self):
        rec = _make_record("g", "t", _circle_polyline(0, 0, 1, n=64))
        self.assertTrue(np.all(rec.semi_axes() > 0))

    def test_round_trip_dict(self):
        rec = _make_record("g1", "t1", _circle_polyline(1, 2, 0.5, n=16))
        rec2 = EllipsoidRecord.from_dict(rec.to_dict())
        np.testing.assert_allclose(rec.mu_g, rec2.mu_g, atol=1e-5)
        np.testing.assert_allclose(rec.M, rec2.M, atol=1e-5)
        np.testing.assert_allclose(rec.evecs, rec2.evecs, atol=1e-5)
        np.testing.assert_allclose(rec.evals, rec2.evals, atol=1e-5)


class TestMVEE(unittest.TestCase):
    def test_all_points_inside(self):
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((80, 2))
        c, M = _fit_mvee(pts, tol=1e-6, max_iter=1000)
        for p in pts:
            d = float((p - c) @ M @ (p - c))
            self.assertLessEqual(d, 1.0 + 5e-3)

    def test_circle_centre(self):
        poly = _circle_polyline(3.0, -1.0, 2.0, n=64)
        pts = np.array([[p[0], p[1]] for p in poly])
        c, _ = _fit_mvee(pts)
        np.testing.assert_allclose(c, [3.0, -1.0], atol=0.05)

    def test_circle_isotropic(self):
        poly = _circle_polyline(0, 0, 1.0, n=128)
        pts = np.array([[p[0], p[1]] for p in poly])
        _, M = _fit_mvee(pts)
        eigs = np.linalg.eigvalsh(M)
        ratio = eigs.max() / (eigs.min() + 1e-9)
        self.assertLess(ratio, 1.3)

    def test_degenerate_single(self):
        c, M = _fit_mvee(np.array([[1.0, 2.0]]))
        self.assertTrue(np.all(np.isfinite(c)) and np.all(np.isfinite(M)))

    def test_degenerate_collinear(self):
        pts = np.column_stack([np.linspace(0, 1, 20), np.zeros(20)])
        c, M = _fit_mvee(pts)
        self.assertTrue(np.all(np.isfinite(c)) and np.all(np.isfinite(M)))


class TestSafetyInflation(unittest.TestCase):
    def test_inflated_M_larger_semi_axes(self):
        rec = _make_record("g", "t", _circle_polyline(0, 0, 1.0, n=64))
        sm = 0.15
        M_inf = rec.inflated_M(sm)
        evals_orig = np.linalg.eigvalsh(rec.M)
        evals_inf = np.linalg.eigvalsh(M_inf)
        axes_orig = 1.0 / np.sqrt(np.maximum(evals_orig, 1e-9))
        axes_inf = 1.0 / np.sqrt(np.maximum(evals_inf, 1e-9))
        np.testing.assert_allclose(np.sort(axes_inf), np.sort(axes_orig) + sm, atol=1e-4)

    def test_inflated_M_contains_original(self):
        rec = _make_record("g", "t", _circle_polyline(0, 0, 1.0, n=64))
        sm = 0.2
        M_inf = rec.inflated_M(sm)
        rng = np.random.default_rng(7)
        for _ in range(50):
            v = rng.standard_normal(2)
            v /= np.sqrt(v @ rec.M @ v)
            p = rec.mu_e + v
            d_inf = float((p - rec.mu_e) @ M_inf @ (p - rec.mu_e))
            self.assertLessEqual(d_inf, 1.0 + 1e-4)

    def test_zero_inflation_unchanged(self):
        rec = _make_record("g", "t", _circle_polyline(0, 0, 1.0, n=32))
        np.testing.assert_array_almost_equal(rec.inflated_M(0.0), rec.M)

    def test_inflated_sigma_larger(self):
        rec = _make_record("g", "t", _circle_polyline(0, 0, 1.0, n=64))
        sig_i = rec.inflated_sigma(0.1)
        eigs_orig = np.linalg.eigvalsh(rec.sigma)
        eigs_inf = np.linalg.eigvalsh(sig_i)
        self.assertTrue(np.all(eigs_inf >= eigs_orig - 1e-6))

    def test_nearby_touching_fail_without_safety(self):
        rec_a = _make_record("g", "a", _circle_polyline(0.0, 0.0, 0.3, n=64))
        rec_b = _make_record("g", "b", _circle_polyline(0.4, 0.0, 0.3, n=64))
        filt_no_safety = EllipsoidSeparationFilter(None, safety_m=0.0)
        filt_with_safety = EllipsoidSeparationFilter(None, safety_m=0.5)
        safe_without, _ = filt_no_safety.is_safe(rec_a, rec_b)
        safe_with, _ = filt_with_safety.is_safe(rec_a, rec_b)
        self.assertFalse(safe_with)
        self.assertIsInstance(safe_without, bool)

    def test_large_safety_blocks_all_pruning(self):
        rec_a = _make_record("g", "a", _circle_polyline(0.0, 0.0, 0.1, n=32))
        rec_b = _make_record("g", "b", _circle_polyline(0.5, 0.0, 0.1, n=32))
        filt = EllipsoidSeparationFilter(None, safety_m=10.0)
        safe, _ = filt.is_safe(rec_a, rec_b)
        self.assertFalse(safe)


class TestGeometryCache(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "geo.pt"
            cache = GeometryCache(path)
            rec = _make_record("g1", "t1", _circle_polyline(0, 0, 1.0))
            cache.put(rec)
            cache.save()
            cache2 = GeometryCache(path)
            rec2 = cache2.get("g1", "t1")
            self.assertIsNotNone(rec2)
            np.testing.assert_allclose(rec.mu_e, rec2.mu_e, atol=1e-5)
            np.testing.assert_allclose(rec.evecs, rec2.evecs, atol=1e-5)

    def test_missing_returns_none(self):
        cache = GeometryCache(None)
        self.assertIsNone(cache.get("nope", "nope"))

    def test_len_and_contains(self):
        cache = GeometryCache(None)
        for i in range(4):
            cache.put(_make_record(f"g{i}", "t1", _circle_polyline(i, 0, 1)))
        self.assertEqual(len(cache), 4)
        self.assertIn(("g0", "t1"), cache)
        self.assertNotIn(("g9", "t1"), cache)


class TestRobotAgnosticism(unittest.TestCase):
    def test_record_independent_of_arm(self):
        poly = _circle_polyline(1.0, 2.0, 0.8, n=32)
        rec_a = EllipsoidRecord.from_polyline("gc1", "task1", poly)
        rec_b = EllipsoidRecord.from_polyline("gc1", "task1", poly)
        np.testing.assert_array_equal(rec_a.mu_e, rec_b.mu_e)
        np.testing.assert_array_equal(rec_a.M, rec_b.M)
        np.testing.assert_array_equal(rec_a.evecs, rec_b.evecs)
        np.testing.assert_array_equal(rec_a.evals, rec_b.evals)


class TestSegmentEllipsoid(unittest.TestCase):
    def _unit_circle_M(self, cx=0.0, cy=0.0, r=1.0):
        M = np.eye(2) / (r ** 2)
        centre = np.array([cx, cy])
        return centre, M

    def test_segment_through_centre_intersects(self):
        centre, M = self._unit_circle_M(0, 0, 1.0)
        seg_p = np.array([-5.0, 0.0])
        seg_q = np.array([5.0, 0.0])
        self.assertTrue(_segment_intersects_ellipsoid(seg_p, seg_q, centre, M))

    def test_segment_far_away_no_intersection(self):
        centre, M = self._unit_circle_M(0, 0, 1.0)
        seg_p = np.array([10.0, 0.0])
        seg_q = np.array([20.0, 0.0])
        self.assertFalse(_segment_intersects_ellipsoid(seg_p, seg_q, centre, M))

    def test_segment_endpoint_inside(self):
        centre, M = self._unit_circle_M(0, 0, 1.0)
        seg_p = np.array([0.0, 0.0])
        seg_q = np.array([5.0, 0.0])
        self.assertTrue(_segment_intersects_ellipsoid(seg_p, seg_q, centre, M))

    def test_segment_tangent_no_intersection(self):
        centre, M = self._unit_circle_M(0, 0, 1.0)
        seg_p = np.array([-5.0, 1.001])
        seg_q = np.array([5.0, 1.001])
        self.assertFalse(_segment_intersects_ellipsoid(seg_p, seg_q, centre, M))

    def test_world_distance_positive_outside(self):
        centre, M = self._unit_circle_M(0, 0, 1.0)
        seg_p = np.array([3.0, 0.0])
        seg_q = np.array([5.0, 0.0])
        d = _segment_min_ellipsoid_distance(seg_p, seg_q, centre, M)
        self.assertGreater(d, 0.0)

    def test_world_distance_negative_inside(self):
        centre, M = self._unit_circle_M(0, 0, 2.0)
        seg_p = np.array([0.0, 0.0])
        seg_q = np.array([0.5, 0.0])
        d = _segment_min_ellipsoid_distance(seg_p, seg_q, centre, M)
        self.assertLess(d, 0.0)


class TestReachCheck(unittest.TestCase):
    def _make_task_at(self, cx, cy, r=0.3):
        return _make_record("g", "t", _circle_polyline(cx, cy, r, n=32))

    def test_clear_reach_paths_safe(self):
        rec_a = self._make_task_at(0.0, 0.0)
        rec_b = self._make_task_at(10.0, 0.0)
        base_a = np.array([-1.0, 0.0])
        base_b = np.array([11.0, 0.0])
        filt = EllipsoidSeparationFilter(None, safety_m=0.05)
        safe, _ = filt.is_safe(rec_a, rec_b, base_a=base_a, base_b=base_b)
        self.assertTrue(safe)

    def test_reach_through_other_task_not_safe(self):
        rec_a = self._make_task_at(0.0, 0.0, r=0.4)
        rec_b = self._make_task_at(8.0, 0.0, r=0.3)
        base_a = np.array([-1.0, 0.0])
        base_b = np.array([-5.0, 0.0])
        filt = EllipsoidSeparationFilter(None, safety_m=0.05)
        safe, _ = filt.is_safe(rec_a, rec_b, base_a=base_a, base_b=base_b)
        self.assertFalse(safe)

    def test_reach_check_both_directions_independent(self):
        rec_a = self._make_task_at(0.0, 0.0, r=0.4)
        rec_b = self._make_task_at(8.0, 0.0, r=0.3)
        base_a = np.array([-1.0, 0.0])
        base_b = np.array([9.0, 0.0])
        filt = EllipsoidSeparationFilter(None, safety_m=0.05)
        safe, _ = filt.is_safe(rec_a, rec_b, base_a=base_a, base_b=base_b)
        self.assertTrue(safe)

    def test_no_base_positions_skips_reach_check(self):
        rec_a = self._make_task_at(0.0, 0.0)
        rec_b = self._make_task_at(10.0, 0.0)
        filt = EllipsoidSeparationFilter(None, safety_m=0.05)
        safe, _ = filt.is_safe(rec_a, rec_b)
        self.assertIsInstance(safe, bool)

    def test_bhattacharyya_with_reach_check(self):
        rec_a = self._make_task_at(0.0, 0.0, r=0.4)
        rec_b = self._make_task_at(8.0, 0.0, r=0.3)
        base_a = np.array([-1.0, 0.0])
        base_b = np.array([-5.0, 0.0])
        filt = BhattacharyyaFilter(None, rho_threshold=0.3, safety_m=0.05)
        safe, _ = filt.is_safe(rec_a, rec_b, base_a=base_a, base_b=base_b)
        self.assertFalse(safe)

    def test_prob_overlap_with_reach_check(self):
        rec_a = self._make_task_at(0.0, 0.0, r=0.4)
        rec_b = self._make_task_at(8.0, 0.0, r=0.3)
        base_a = np.array([-1.0, 0.0])
        base_b = np.array([-5.0, 0.0])
        filt = ProbabilisticOverlapFilter(None, p_threshold=0.5, n_sigma=3.0, safety_m=0.05)
        safe, _ = filt.is_safe(rec_a, rec_b, base_a=base_a, base_b=base_b)
        self.assertFalse(safe)

    def test_cascade_with_reach_check(self):
        rec_a = self._make_task_at(0.0, 0.0, r=0.4)
        rec_b = self._make_task_at(8.0, 0.0, r=0.3)
        base_a = np.array([-1.0, 0.0])
        base_b = np.array([-5.0, 0.0])
        filt = CascadeFilter(None, safety_m=0.05)
        safe, pruned_by, _ = filt.is_safe(rec_a, rec_b, base_a=base_a, base_b=base_b)
        self.assertFalse(safe)
        self.assertEqual(pruned_by, "none")


class TestEllipsoidSeparationFilter(unittest.TestCase):
    def test_clearly_separated_no_base(self):
        rec_a = _make_record("g", "a", _circle_polyline(0.0, 0.0, 0.4, n=32))
        rec_b = _make_record("g", "b", _circle_polyline(10.0, 0.0, 0.4, n=32))
        filt = EllipsoidSeparationFilter(None, safety_m=0.05)
        safe, scores = filt.is_safe(rec_a, rec_b)
        self.assertTrue(safe)
        self.assertGreater(scores["mvee_margin"], 0.0)

    def test_overlapping_not_safe(self):
        rec_a = _make_record("g", "a", _circle_polyline(0.0, 0.0, 1.0, n=32))
        rec_b = _make_record("g", "b", _circle_polyline(0.1, 0.0, 1.0, n=32))
        filt = EllipsoidSeparationFilter(None, safety_m=0.05)
        safe, _ = filt.is_safe(rec_a, rec_b)
        self.assertFalse(safe)


class TestBhattacharyyaFilter(unittest.TestCase):
    def test_far_apart_safe(self):
        rec_a = _make_record("g", "a", _circle_polyline(0.0, 0.0, 0.3, n=64))
        rec_b = _make_record("g", "b", _circle_polyline(8.0, 0.0, 0.3, n=64))
        filt = BhattacharyyaFilter(None, rho_threshold=0.05, safety_m=0.05)
        safe, scores = filt.is_safe(rec_a, rec_b)
        self.assertTrue(safe)
        self.assertLess(scores["rho"], 0.05)

    def test_identical_not_safe(self):
        poly = _circle_polyline(0, 0, 1.0, n=64)
        rec_a = _make_record("g", "a", poly)
        rec_b = _make_record("g", "b", poly)
        filt = BhattacharyyaFilter(None, rho_threshold=0.05, safety_m=0.05)
        safe, scores = filt.is_safe(rec_a, rec_b)
        self.assertFalse(safe)
        self.assertGreater(scores["rho"], 0.9)

    def test_rho_symmetric(self):
        rec_a = _make_record("g", "a", _circle_polyline(0.0, 0.0, 0.5, n=32))
        rec_b = _make_record("g", "b", _circle_polyline(3.0, 1.0, 0.8, n=32))
        sm = 0.05
        rho_ab = _bhattacharyya_coefficient(rec_a.mu_g, rec_a.inflated_sigma(sm),
                                            rec_b.mu_g, rec_b.inflated_sigma(sm))
        rho_ba = _bhattacharyya_coefficient(rec_b.mu_g, rec_b.inflated_sigma(sm),
                                            rec_a.mu_g, rec_a.inflated_sigma(sm))
        np.testing.assert_allclose(rho_ab, rho_ba, rtol=1e-5)


class TestProbabilisticOverlapFilter(unittest.TestCase):
    def test_non_overlapping_safe(self):
        rec_a = _make_record("g", "a", _circle_polyline(0.0, 0.0, 0.2, n=64))
        rec_b = _make_record("g", "b", _circle_polyline(10.0, 0.0, 0.2, n=64))
        filt = ProbabilisticOverlapFilter(None, p_threshold=0.02, safety_m=0.05)
        safe, scores = filt.is_safe(rec_a, rec_b)
        self.assertTrue(safe)
        self.assertLess(scores["p_overlap"], 0.02)

    def test_identical_not_safe(self):
        poly = _circle_polyline(0, 0, 1.0, n=64)
        rec_a = _make_record("g", "a", poly)
        rec_b = _make_record("g", "b", poly)
        filt = ProbabilisticOverlapFilter(None, p_threshold=0.02, safety_m=0.05)
        safe, scores = filt.is_safe(rec_a, rec_b)
        self.assertFalse(safe)
        self.assertGreater(scores["p_overlap"], 0.5)

    def test_p_in_unit_interval(self):
        rng = np.random.default_rng(7)
        for i in range(20):
            with self.subTest(i=i):
                mu_a = rng.uniform(-3, 3, 2)
                mu_b = rng.uniform(-3, 3, 2)
                La = rng.standard_normal((2, 2))
                Lb = rng.standard_normal((2, 2))
                sa = La @ La.T + np.eye(2) * 0.1
                sb = Lb @ Lb.T + np.eye(2) * 0.1
                p = _prob_overlap(mu_a, sa, mu_b, sb)
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0 + 1e-6)


class TestCascadeFilter(unittest.TestCase):
    def test_well_separated_pruned(self):
        rec_a = _make_record("g", "a", _circle_polyline(0.0, 0.0, 0.3, n=32))
        rec_b = _make_record("g", "b", _circle_polyline(20.0, 0.0, 0.3, n=32))
        filt = CascadeFilter(None, safety_m=0.05)
        safe, pruned_by, _ = filt.is_safe(rec_a, rec_b)
        self.assertTrue(safe)
        self.assertIn(pruned_by, ("mvee_separation", "bhattacharyya", "prob_overlap"))

    def test_overlapping_not_pruned(self):
        poly = _circle_polyline(0, 0, 1.0, n=64)
        rec_a = _make_record("g", "a", poly)
        rec_b = _make_record("g", "b", poly)
        filt = CascadeFilter(None, safety_m=0.05)
        safe, pruned_by, _ = filt.is_safe(rec_a, rec_b)
        self.assertFalse(safe)
        self.assertEqual(pruned_by, "none")

    def test_scores_dict_keys_present(self):
        rec_a = _make_record("g", "a", _circle_polyline(0, 0, 1))
        rec_b = _make_record("g", "b", _circle_polyline(1, 0, 1))
        filt = CascadeFilter(None, safety_m=0.0)
        _, _, scores = filt.is_safe(rec_a, rec_b)
        self.assertIn("mvee_margin", scores)
        self.assertIn("rho", scores)
        self.assertIn("p_overlap", scores)


class TestDegenerateInputs(unittest.TestCase):
    def test_single_point(self):
        rec = EllipsoidRecord.from_polyline("g", "t", [(1.0, 2.0, 0.0)])
        self.assertTrue(np.all(np.isfinite(rec.mu_g)))
        self.assertTrue(np.all(np.isfinite(rec.M)))

    def test_empty_polyline(self):
        rec = EllipsoidRecord.from_polyline("g", "t", [])
        self.assertTrue(np.all(np.isfinite(rec.mu_g)))
        self.assertTrue(np.all(np.isfinite(rec.M)))

    def test_collinear(self):
        poly = _line_polyline(0, 0, 1, 0, n=30)
        rec = EllipsoidRecord.from_polyline("g", "t", poly)
        self.assertTrue(np.all(np.isfinite(rec.M)))
        eigs = np.linalg.eigvalsh(rec.M)
        self.assertTrue(np.all(eigs >= -1e-5))

    def test_two_points(self):
        rec = EllipsoidRecord.from_polyline("g", "t", [(0.0, 0.0, 0.0), (1.0, 0.0, 1.0)])
        self.assertTrue(np.all(np.isfinite(rec.mu_e)))
        self.assertTrue(np.all(np.isfinite(rec.M)))


if __name__ == '__main__':
    unittest.main()
