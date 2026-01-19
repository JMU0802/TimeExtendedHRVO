"""
test_geometry.py - 几何工具测试
"""
from time_extended_hrvo.core.vessel import VesselState
from time_extended_hrvo.utils.geometry import (
    normalize,
    angle_between,
    signed_angle,
    rotate_vector,
    perpendicular,
    point_to_line_distance,
    line_intersection,
    compute_dcpa_tcpa,
    classify_encounter
)
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNormalize(unittest.TestCase):
    """测试向量归一化"""

    def test_unit_vector_unchanged(self):
        """测试单位向量不变"""
        v = np.array([1, 0])
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_normalize_to_unit(self):
        """测试归一化到单位长度"""
        v = np.array([3, 4])
        result = normalize(v)

        self.assertAlmostEqual(np.linalg.norm(result), 1.0)
        np.testing.assert_array_almost_equal(result, [0.6, 0.8])

    def test_zero_vector(self):
        """测试零向量"""
        v = np.array([0, 0])
        result = normalize(v)

        np.testing.assert_array_equal(result, [0, 0])

    def test_very_small_vector(self):
        """测试极小向量"""
        v = np.array([1e-12, 0])
        result = normalize(v)

        # 应该返回零向量（避免除零）
        np.testing.assert_array_equal(result, [0, 0])


class TestAngleBetween(unittest.TestCase):
    """测试两向量夹角"""

    def test_parallel_vectors(self):
        """测试平行向量"""
        v1 = np.array([1, 0])
        v2 = np.array([2, 0])

        angle = angle_between(v1, v2)

        self.assertAlmostEqual(angle, 0)

    def test_perpendicular_vectors(self):
        """测试垂直向量"""
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])

        angle = angle_between(v1, v2)

        self.assertAlmostEqual(angle, np.pi / 2)

    def test_opposite_vectors(self):
        """测试相反向量"""
        v1 = np.array([1, 0])
        v2 = np.array([-1, 0])

        angle = angle_between(v1, v2)

        self.assertAlmostEqual(angle, np.pi)

    def test_45_degree_angle(self):
        """测试45度夹角"""
        v1 = np.array([1, 0])
        v2 = np.array([1, 1])

        angle = angle_between(v1, v2)

        self.assertAlmostEqual(angle, np.pi / 4)


class TestSignedAngle(unittest.TestCase):
    """测试有符号角度"""

    def test_counterclockwise(self):
        """测试逆时针角度"""
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])

        angle = signed_angle(v1, v2)

        self.assertAlmostEqual(angle, np.pi / 2)

    def test_clockwise(self):
        """测试顺时针角度"""
        v1 = np.array([1, 0])
        v2 = np.array([0, -1])

        angle = signed_angle(v1, v2)

        self.assertAlmostEqual(angle, -np.pi / 2)

    def test_range(self):
        """测试角度范围 [-π, π]"""
        v1 = np.array([1, 0])

        for deg in range(0, 360, 30):
            rad = np.deg2rad(deg)
            v2 = np.array([np.cos(rad), np.sin(rad)])

            angle = signed_angle(v1, v2)

            self.assertGreaterEqual(angle, -np.pi)
            self.assertLessEqual(angle, np.pi)


class TestRotateVector(unittest.TestCase):
    """测试向量旋转"""

    def test_rotate_90_counterclockwise(self):
        """测试逆时针旋转90度"""
        v = np.array([1, 0])
        result = rotate_vector(v, np.pi / 2)

        np.testing.assert_array_almost_equal(result, [0, 1])

    def test_rotate_90_clockwise(self):
        """测试顺时针旋转90度"""
        v = np.array([1, 0])
        result = rotate_vector(v, -np.pi / 2)

        np.testing.assert_array_almost_equal(result, [0, -1])

    def test_rotate_180(self):
        """测试旋转180度"""
        v = np.array([3, 4])
        result = rotate_vector(v, np.pi)

        np.testing.assert_array_almost_equal(result, [-3, -4])

    def test_rotate_zero(self):
        """测试旋转0度"""
        v = np.array([3, 4])
        result = rotate_vector(v, 0)

        np.testing.assert_array_almost_equal(result, v)

    def test_full_rotation(self):
        """测试完整旋转360度"""
        v = np.array([3, 4])
        result = rotate_vector(v, 2 * np.pi)

        np.testing.assert_array_almost_equal(result, v)


class TestPerpendicular(unittest.TestCase):
    """测试垂直向量"""

    def test_perpendicular_counterclockwise(self):
        """测试逆时针垂直"""
        v = np.array([1, 0])
        result = perpendicular(v, clockwise=False)

        np.testing.assert_array_equal(result, [0, 1])

    def test_perpendicular_clockwise(self):
        """测试顺时针垂直"""
        v = np.array([1, 0])
        result = perpendicular(v, clockwise=True)

        np.testing.assert_array_equal(result, [0, -1])

    def test_is_perpendicular(self):
        """测试结果确实垂直"""
        v = np.array([3, 4])
        perp = perpendicular(v)

        dot = np.dot(v, perp)
        self.assertAlmostEqual(dot, 0)


class TestPointToLineDistance(unittest.TestCase):
    """测试点到直线距离"""

    def test_point_on_line(self):
        """测试点在直线上"""
        point = np.array([5, 0])
        line_point = np.array([0, 0])
        line_dir = np.array([1, 0])

        dist = point_to_line_distance(point, line_point, line_dir)

        self.assertAlmostEqual(dist, 0)

    def test_point_above_line(self):
        """测试点在直线上方"""
        point = np.array([0, 5])
        line_point = np.array([0, 0])
        line_dir = np.array([1, 0])

        dist = point_to_line_distance(point, line_point, line_dir)

        self.assertAlmostEqual(dist, 5)

    def test_point_below_line(self):
        """测试点在直线下方"""
        point = np.array([0, -5])
        line_point = np.array([0, 0])
        line_dir = np.array([1, 0])

        dist = point_to_line_distance(point, line_point, line_dir)

        self.assertAlmostEqual(dist, -5)


class TestLineIntersection(unittest.TestCase):
    """测试直线交点"""

    def test_perpendicular_lines(self):
        """测试垂直线交点"""
        p1, d1 = np.array([0, 0]), np.array([1, 0])
        p2, d2 = np.array([5, 0]), np.array([0, 1])

        intersection = line_intersection(p1, d1, p2, d2)

        np.testing.assert_array_almost_equal(intersection, [5, 0])

    def test_parallel_lines(self):
        """测试平行线无交点"""
        p1, d1 = np.array([0, 0]), np.array([1, 0])
        p2, d2 = np.array([0, 5]), np.array([1, 0])

        intersection = line_intersection(p1, d1, p2, d2)

        self.assertIsNone(intersection)

    def test_diagonal_intersection(self):
        """测试斜线交点"""
        p1, d1 = np.array([0, 0]), np.array([1, 1])
        p2, d2 = np.array([0, 10]), np.array([1, -1])

        intersection = line_intersection(p1, d1, p2, d2)

        np.testing.assert_array_almost_equal(intersection, [5, 5])


class TestComputeDcpaTcpa(unittest.TestCase):
    """测试 DCPA/TCPA 计算"""

    def test_head_on_collision(self):
        """测试对遇碰撞"""
        dcpa, tcpa = compute_dcpa_tcpa(
            [0, 0], [5, 0],      # 本船向东
            [1000, 0], [-5, 0]   # 目标船向西
        )

        # DCPA 应该是 0（正面碰撞）
        self.assertAlmostEqual(dcpa, 0)

        # TCPA 应该是 100s (1000m / 10m/s)
        self.assertAlmostEqual(tcpa, 100)

    def test_safe_passing(self):
        """测试安全通过"""
        dcpa, tcpa = compute_dcpa_tcpa(
            [0, 0], [5, 0],        # 本船向东
            [500, 100], [-5, 0]   # 目标船向西，有偏移
        )

        # DCPA 应该是 100m
        self.assertAlmostEqual(dcpa, 100)

    def test_diverging_vessels(self):
        """测试远离的船舶"""
        dcpa, tcpa = compute_dcpa_tcpa(
            [0, 0], [5, 0],       # 本船向东
            [-500, 0], [-5, 0]    # 目标船在后方向西
        )

        # TCPA <= 0 表示已经远离
        self.assertEqual(tcpa, 0)

    def test_stationary_vessels(self):
        """测试静止船舶"""
        dcpa, tcpa = compute_dcpa_tcpa(
            [0, 0], [0, 0],
            [100, 0], [0, 0]
        )

        # DCPA 应该是当前距离
        self.assertAlmostEqual(dcpa, 100)
        self.assertEqual(tcpa, 0)


class TestClassifyEncounter(unittest.TestCase):
    """测试会遇类型分类"""

    def test_head_on(self):
        """测试对遇"""
        own = VesselState([0, 0], [0, 5], 50)      # 向北
        obs = VesselState([0, 1000], [0, -5], 50)  # 向南

        encounter = classify_encounter(own, obs)

        self.assertEqual(encounter, 'head-on')

    def test_overtaking(self):
        """测试追越"""
        own = VesselState([0, 0], [0, 8], 50)     # 快船向北
        obs = VesselState([0, 500], [0, 3], 50)   # 慢船向北

        encounter = classify_encounter(own, obs)

        self.assertEqual(encounter, 'overtaking')

    def test_being_overtaken(self):
        """测试被追越"""
        own = VesselState([0, 0], [0, 3], 50)        # 慢船向北
        obs = VesselState([0, -500], [0, 8], 50)     # 快船在后方向北

        encounter = classify_encounter(own, obs)

        self.assertEqual(encounter, 'being-overtaken')

    def test_crossing_give_way(self):
        """测试交叉相遇-让路"""
        own = VesselState([0, 0], [5, 0], 50)       # 向东
        obs = VesselState([300, -200], [0, 5], 50)  # 在右前方向北

        encounter = classify_encounter(own, obs)

        # 目标船在右舷，本船应让路
        self.assertIn('crossing', encounter)

    def test_crossing_stand_on(self):
        """测试交叉相遇-直航"""
        own = VesselState([0, 0], [5, 0], 50)      # 向东
        obs = VesselState([300, 200], [0, -5], 50)  # 在左前方向南

        encounter = classify_encounter(own, obs)

        # 目标船在左舷，本船应直航
        self.assertIn('crossing', encounter)


if __name__ == '__main__':
    unittest.main()
