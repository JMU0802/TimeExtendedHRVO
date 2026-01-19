"""
test_vessel.py - 船舶状态模型测试
"""
from time_extended_hrvo.core.vessel import VesselState
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVesselState(unittest.TestCase):
    """测试 VesselState 类"""

    def test_init_basic(self):
        """测试基本初始化"""
        vessel = VesselState([0, 0], [5, 0], 50)

        np.testing.assert_array_equal(vessel.p, [0, 0])
        np.testing.assert_array_equal(vessel.v, [5, 0])
        self.assertEqual(vessel.r, 50)

    def test_init_with_lists(self):
        """测试使用列表初始化"""
        vessel = VesselState([100, 200], [3, 4], 30)

        self.assertIsInstance(vessel.p, np.ndarray)
        self.assertIsInstance(vessel.v, np.ndarray)
        self.assertEqual(vessel.p[0], 100)
        self.assertEqual(vessel.p[1], 200)

    def test_init_with_numpy_arrays(self):
        """测试使用 numpy 数组初始化"""
        pos = np.array([10.5, 20.5])
        vel = np.array([1.0, 2.0])
        vessel = VesselState(pos, vel, 25.5)

        np.testing.assert_array_almost_equal(vessel.p, [10.5, 20.5])
        np.testing.assert_array_almost_equal(vessel.v, [1.0, 2.0])
        self.assertAlmostEqual(vessel.r, 25.5)

    def test_speed_property(self):
        """测试速度大小计算"""
        # 3-4-5 三角形
        vessel = VesselState([0, 0], [3, 4], 50)
        self.assertAlmostEqual(vessel.speed, 5.0)

        # 纯水平速度
        vessel2 = VesselState([0, 0], [10, 0], 50)
        self.assertAlmostEqual(vessel2.speed, 10.0)

        # 零速度
        vessel3 = VesselState([0, 0], [0, 0], 50)
        self.assertAlmostEqual(vessel3.speed, 0.0)

    def test_heading_property(self):
        """测试航向角计算"""
        # 向东
        vessel_east = VesselState([0, 0], [5, 0], 50)
        self.assertAlmostEqual(vessel_east.heading, 0.0)

        # 向北
        vessel_north = VesselState([0, 0], [0, 5], 50)
        self.assertAlmostEqual(vessel_north.heading, np.pi / 2)

        # 向西
        vessel_west = VesselState([0, 0], [-5, 0], 50)
        self.assertAlmostEqual(abs(vessel_west.heading), np.pi)

        # 向南
        vessel_south = VesselState([0, 0], [0, -5], 50)
        self.assertAlmostEqual(vessel_south.heading, -np.pi / 2)

        # 东北方向 (45度)
        vessel_ne = VesselState([0, 0], [5, 5], 50)
        self.assertAlmostEqual(vessel_ne.heading, np.pi / 4)

    def test_predict_position(self):
        """测试位置预测"""
        vessel = VesselState([100, 100], [5, 0], 50)

        # 10 秒后
        pos_10s = vessel.predict_position(10)
        np.testing.assert_array_almost_equal(pos_10s, [150, 100])

        # 0 秒（当前位置）
        pos_0s = vessel.predict_position(0)
        np.testing.assert_array_almost_equal(pos_0s, [100, 100])

        # 负时间（回溯）
        pos_neg = vessel.predict_position(-5)
        np.testing.assert_array_almost_equal(pos_neg, [75, 100])

    def test_predict_position_diagonal(self):
        """测试对角线运动的位置预测"""
        vessel = VesselState([0, 0], [3, 4], 50)

        pos_2s = vessel.predict_position(2)
        np.testing.assert_array_almost_equal(pos_2s, [6, 8])

    def test_copy(self):
        """测试深拷贝"""
        original = VesselState([100, 200], [5, 10], 50)
        copied = original.copy()

        # 验证值相等
        np.testing.assert_array_equal(copied.p, original.p)
        np.testing.assert_array_equal(copied.v, original.v)
        self.assertEqual(copied.r, original.r)

        # 验证是独立副本
        copied.p[0] = 999
        self.assertEqual(original.p[0], 100)

        copied.v[0] = 999
        self.assertEqual(original.v[0], 5)

    def test_repr(self):
        """测试字符串表示"""
        vessel = VesselState([0, 0], [5, 0], 50)
        repr_str = repr(vessel)

        self.assertIn("VesselState", repr_str)
        self.assertIn("p=", repr_str)
        self.assertIn("v=", repr_str)
        self.assertIn("r=", repr_str)


class TestVesselStateEdgeCases(unittest.TestCase):
    """边界情况测试"""

    def test_zero_velocity(self):
        """测试零速度"""
        vessel = VesselState([0, 0], [0, 0], 50)

        self.assertEqual(vessel.speed, 0)
        # heading 对于零向量的行为
        self.assertEqual(vessel.heading, 0)  # arctan2(0, 0) = 0

    def test_very_small_values(self):
        """测试极小值"""
        vessel = VesselState([1e-10, 1e-10], [1e-10, 1e-10], 1e-10)

        self.assertIsNotNone(vessel.speed)
        self.assertIsNotNone(vessel.heading)

    def test_large_values(self):
        """测试大值"""
        vessel = VesselState([1e6, 1e6], [100, 100], 1000)

        pos_1000s = vessel.predict_position(1000)
        expected = [1e6 + 100000, 1e6 + 100000]
        np.testing.assert_array_almost_equal(pos_1000s, expected)

    def test_negative_radius(self):
        """测试负半径（应该被允许，由调用者确保合理性）"""
        vessel = VesselState([0, 0], [5, 0], -50)
        self.assertEqual(vessel.r, -50)


if __name__ == '__main__':
    unittest.main()
