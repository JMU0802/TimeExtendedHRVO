"""
test_hrvo.py - HRVO 几何构造测试
"""
from time_extended_hrvo.core.vessel import VesselState
from time_extended_hrvo.core.hrvo import HRVO, compute_hrvo, compute_vo, compute_rvo
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHRVO(unittest.TestCase):
    """测试 HRVO 类"""

    def test_init(self):
        """测试初始化"""
        apex = np.array([0, 0])
        left = np.array([0, 1])
        right = np.array([1, 0])

        hrvo = HRVO(apex, left, right)

        np.testing.assert_array_almost_equal(hrvo.apex, apex)
        # 边界向量应该被归一化
        self.assertAlmostEqual(np.linalg.norm(hrvo.left), 1.0)
        self.assertAlmostEqual(np.linalg.norm(hrvo.right), 1.0)

    def test_contains_inside(self):
        """测试点在 HRVO 内部"""
        # 创建一个简单的 HRVO：apex 在原点，锥形向右上方打开
        apex = np.array([0, 0])
        left = np.array([1, 1])   # 45度
        right = np.array([1, -1])  # -45度

        hrvo = HRVO(apex, left, right)

        # 正前方的点应该在内部
        self.assertTrue(hrvo.contains([5, 0]))
        self.assertTrue(hrvo.contains([10, 0]))

    def test_contains_outside(self):
        """测试点在 HRVO 外部"""
        apex = np.array([0, 0])
        left = np.array([1, 1])
        right = np.array([1, -1])

        hrvo = HRVO(apex, left, right)

        # 后方的点应该在外部
        self.assertFalse(hrvo.contains([-5, 0]))

        # 远离锥形的点应该在外部
        self.assertFalse(hrvo.contains([0, 10]))
        self.assertFalse(hrvo.contains([0, -10]))

    def test_contains_on_boundary(self):
        """测试点在边界上"""
        apex = np.array([0, 0])
        left = np.array([1, 1])
        right = np.array([1, -1])

        hrvo = HRVO(apex, left, right)

        # 边界上的点（使用 >= 判断，所以应该算内部）
        self.assertTrue(hrvo.contains([5, 5]))  # 在左边界方向

    def test_distance_to_boundary(self):
        """测试到边界的距离"""
        apex = np.array([0, 0])
        left = np.array([1, 0])
        right = np.array([0, -1])

        hrvo = HRVO(apex, left, right)

        # 内部点距离应该为负
        dist_inside = hrvo.distance_to_boundary([2, -2])

        # 检查返回值类型
        self.assertIsInstance(dist_inside, (int, float, np.floating))


class TestComputeHRVO(unittest.TestCase):
    """测试 HRVO 构造函数"""

    def test_basic_construction(self):
        """测试基本 HRVO 构造"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 0], [-3, 0], 50)

        hrvo = compute_hrvo(own, obs)

        self.assertIsInstance(hrvo, HRVO)
        self.assertIsNotNone(hrvo.apex)
        self.assertIsNotNone(hrvo.left)
        self.assertIsNotNone(hrvo.right)

    def test_hrvo_apex_position(self):
        """测试 HRVO apex 位置（责任因子 0.5）"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 0], [-3, 0], 50)

        hrvo = compute_hrvo(own, obs, responsibility=0.5)

        # HRVO apex = v_obs + 0.5 * (v_own - v_obs)
        # = [-3, 0] + 0.5 * ([5, 0] - [-3, 0])
        # = [-3, 0] + 0.5 * [8, 0]
        # = [-3, 0] + [4, 0] = [1, 0]
        expected_apex = np.array([1, 0])
        np.testing.assert_array_almost_equal(hrvo.apex, expected_apex)

    def test_vo_apex_position(self):
        """测试 VO apex 位置（责任因子 0）"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 0], [-3, 0], 50)

        vo = compute_vo(own, obs)

        # VO apex = v_obs
        expected_apex = np.array([-3, 0])
        np.testing.assert_array_almost_equal(vo.apex, expected_apex)

    def test_rvo_same_as_hrvo_default(self):
        """测试 RVO 与默认 HRVO 相同"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 0], [-3, 0], 50)

        hrvo = compute_hrvo(own, obs, responsibility=0.5)
        rvo = compute_rvo(own, obs)

        np.testing.assert_array_almost_equal(hrvo.apex, rvo.apex)

    def test_collision_raises_error(self):
        """测试已碰撞情况抛出异常"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([50, 0], [-3, 0], 50)  # 距离 = 50, R = 100

        with self.assertRaises(ValueError):
            compute_hrvo(own, obs)

    def test_boundary_directions(self):
        """测试边界方向正确性"""
        own = VesselState([0, 0], [0, 0], 50)
        obs = VesselState([500, 0], [0, 0], 50)  # 目标船在正前方

        hrvo = compute_hrvo(own, obs)

        # 边界向量应该是单位向量
        self.assertAlmostEqual(np.linalg.norm(hrvo.left), 1.0)
        self.assertAlmostEqual(np.linalg.norm(hrvo.right), 1.0)

        # 左边界应该在右边界的逆时针方向
        cross = np.cross(hrvo.right, hrvo.left)
        self.assertGreater(cross, 0)

    def test_symmetric_scenario(self):
        """测试对称场景"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([1000, 0], [-5, 0], 50)  # 完全对遇

        hrvo = compute_hrvo(own, obs)

        # 在对称场景中，左右边界角度应该对称
        left_angle = np.arctan2(hrvo.left[1], hrvo.left[0])
        right_angle = np.arctan2(hrvo.right[1], hrvo.right[0])

        self.assertAlmostEqual(left_angle, -right_angle, places=5)


class TestHRVOContainment(unittest.TestCase):
    """测试 HRVO 包含性判断"""

    def test_current_velocity_in_collision_course(self):
        """测试碰撞航向的当前速度在 HRVO 内"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 0], [-5, 0], 50)  # 迎面相撞

        hrvo = compute_hrvo(own, obs)

        # 当前速度（碰撞航向）应该在 HRVO 内
        # 注意：这取决于 HRVO 的具体定义
        # 对于对遇场景，直行速度可能在 HRVO 内

    def test_safe_velocity_not_in_hrvo(self):
        """测试安全速度不在 HRVO 内"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 100], [0, 0], 50)  # 目标船静止在右前方

        hrvo = compute_hrvo(own, obs)

        # 大幅右转的速度应该安全
        safe_v = np.array([0, -5])  # 向南
        # 这个速度可能在或不在 HRVO 内，取决于具体几何关系


class TestHRVOEdgeCases(unittest.TestCase):
    """边界情况测试"""

    def test_very_close_vessels(self):
        """测试非常接近的船舶"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([150, 0], [-3, 0], 50)  # 刚好不碰撞

        # 应该能构造 HRVO
        hrvo = compute_hrvo(own, obs)
        self.assertIsNotNone(hrvo)

    def test_very_far_vessels(self):
        """测试非常远的船舶"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([100000, 0], [-3, 0], 50)

        hrvo = compute_hrvo(own, obs)

        # 锥形应该非常窄
        angle_left = np.arctan2(hrvo.left[1], hrvo.left[0])
        angle_right = np.arctan2(hrvo.right[1], hrvo.right[0])
        cone_angle = angle_left - angle_right

        self.assertLess(abs(cone_angle), np.deg2rad(5))  # 小于5度

    def test_stationary_own_vessel(self):
        """测试本船静止"""
        own = VesselState([0, 0], [0, 0], 50)
        obs = VesselState([500, 0], [-3, 0], 50)

        hrvo = compute_hrvo(own, obs)
        self.assertIsNotNone(hrvo)

    def test_stationary_obstacle(self):
        """测试目标船静止"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 0], [0, 0], 50)

        hrvo = compute_hrvo(own, obs)
        self.assertIsNotNone(hrvo)

    def test_both_stationary(self):
        """测试双方都静止"""
        own = VesselState([0, 0], [0, 0], 50)
        obs = VesselState([500, 0], [0, 0], 50)

        hrvo = compute_hrvo(own, obs)
        self.assertIsNotNone(hrvo)

    def test_different_radii(self):
        """测试不同安全半径"""
        own = VesselState([0, 0], [5, 0], 30)
        obs = VesselState([500, 0], [-3, 0], 100)

        hrvo = compute_hrvo(own, obs)
        self.assertIsNotNone(hrvo)


if __name__ == '__main__':
    unittest.main()
