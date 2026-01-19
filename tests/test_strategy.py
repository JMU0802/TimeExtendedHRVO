"""
test_strategy.py - 避让策略测试
"""
from time_extended_hrvo.core.strategy import AvoidanceStrategy, generate_strategy_space
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestAvoidanceStrategy(unittest.TestCase):
    """测试 AvoidanceStrategy 类"""

    def test_init(self):
        """测试初始化"""
        strategy = AvoidanceStrategy(np.deg2rad(10), 1.0)

        self.assertAlmostEqual(strategy.delta_psi, np.deg2rad(10))
        self.assertAlmostEqual(strategy.delta_speed, 1.0)

    def test_zero_strategy(self):
        """测试零策略（保持原状）"""
        strategy = AvoidanceStrategy(0, 0)
        v0 = np.array([5, 0])

        # t=0 时应该等于 v0
        v_t0 = strategy.velocity_profile(v0, 0)
        np.testing.assert_array_almost_equal(v_t0, v0)

        # t=无穷大时也应该等于 v0
        v_inf = strategy.velocity_profile(v0, 1000)
        np.testing.assert_array_almost_equal(v_inf, v0, decimal=3)

    def test_velocity_profile_initial(self):
        """测试初始时刻的速度曲线"""
        strategy = AvoidanceStrategy(np.deg2rad(30), 2.0)
        v0 = np.array([5, 0])

        # t=0 时应该接近 v0
        v_t0 = strategy.velocity_profile(v0, 0)
        np.testing.assert_array_almost_equal(v_t0, v0)

    def test_velocity_profile_convergence(self):
        """测试速度曲线收敛到最终值"""
        strategy = AvoidanceStrategy(np.deg2rad(30), 2.0)
        v0 = np.array([5, 0])
        tau = 10.0

        # 在 10*tau 后应该非常接近最终值
        v_final_approx = strategy.velocity_profile(v0, 10 * tau, tau)
        v_final = strategy.get_final_velocity(v0, tau)

        np.testing.assert_array_almost_equal(
            v_final_approx, v_final, decimal=1)

    def test_velocity_profile_monotonic(self):
        """测试速度变化的单调性（一阶指数响应）"""
        strategy = AvoidanceStrategy(np.deg2rad(20), 1.0)
        v0 = np.array([5, 0])

        # 速度大小应该单调增加
        speeds = []
        for t in [0, 5, 10, 15, 20, 30]:
            v_t = strategy.velocity_profile(v0, t)
            speeds.append(np.linalg.norm(v_t))

        # 检查单调递增（允许小误差）
        for i in range(len(speeds) - 1):
            self.assertLessEqual(speeds[i], speeds[i + 1] + 1e-6)

    def test_heading_change_positive(self):
        """测试正向航向改变（右转/顺时针）

        转向约定：正值=右转(Starboard)，负值=左转(Port)
        """
        strategy = AvoidanceStrategy(np.deg2rad(90), 0)  # 右转90度（正值）
        v0 = np.array([5, 0])  # 初始向东

        v_final = strategy.get_final_velocity(v0)

        # 右转90度应该指向南方（-Y方向）
        expected_heading = -np.pi / 2
        actual_heading = np.arctan2(v_final[1], v_final[0])
        self.assertAlmostEqual(actual_heading, expected_heading, places=5)

    def test_heading_change_negative(self):
        """测试负向航向改变（左转/逆时针）

        转向约定：正值=右转(Starboard)，负值=左转(Port)
        """
        strategy = AvoidanceStrategy(np.deg2rad(-90), 0)  # 左转90度（负值）
        v0 = np.array([5, 0])  # 初始向东

        v_final = strategy.get_final_velocity(v0)

        # 左转90度应该指向北方（+Y方向）
        expected_heading = np.pi / 2
        actual_heading = np.arctan2(v_final[1], v_final[0])
        self.assertAlmostEqual(actual_heading, expected_heading, places=5)

    def test_speed_increase(self):
        """测试速度增加"""
        strategy = AvoidanceStrategy(0, 3.0)
        v0 = np.array([5, 0])

        v_final = strategy.get_final_velocity(v0)

        self.assertAlmostEqual(np.linalg.norm(v_final), 8.0)

    def test_speed_decrease(self):
        """测试速度减少"""
        strategy = AvoidanceStrategy(0, -2.0)
        v0 = np.array([5, 0])

        v_final = strategy.get_final_velocity(v0)

        self.assertAlmostEqual(np.linalg.norm(v_final), 3.0)

    def test_speed_non_negative(self):
        """测试速度不会变为负值"""
        strategy = AvoidanceStrategy(0, -10.0)  # 减速超过当前速度
        v0 = np.array([5, 0])

        v_final = strategy.get_final_velocity(v0)

        # 速度大小应该是 0，不是负数
        self.assertGreaterEqual(np.linalg.norm(v_final), 0)

    def test_repr(self):
        """测试字符串表示"""
        strategy = AvoidanceStrategy(np.deg2rad(15), 1.5)
        repr_str = repr(strategy)

        self.assertIn("AvoidanceStrategy", repr_str)
        self.assertIn("15.0", repr_str)  # 角度
        self.assertIn("1.5", repr_str)   # 速度


class TestGenerateStrategySpace(unittest.TestCase):
    """测试策略空间生成"""

    def test_default_space(self):
        """测试默认策略空间"""
        strategies = generate_strategy_space()

        # 应该生成多个策略
        self.assertGreater(len(strategies), 0)

        # 所有元素应该是 AvoidanceStrategy
        for s in strategies:
            self.assertIsInstance(s, AvoidanceStrategy)

    def test_custom_space(self):
        """测试自定义策略空间"""
        strategies = generate_strategy_space(
            psi_range_deg=(-20, 20),
            psi_step_deg=10,
            speed_range=(-1, 1),
            speed_step=1.0
        )

        # 计算预期数量: (-20, -10, 0, 10, 20) x (-1, 0, 1) = 5 x 3 = 15
        self.assertEqual(len(strategies), 15)

    def test_contains_zero_strategy(self):
        """测试策略空间包含零策略"""
        strategies = generate_strategy_space()

        has_zero = any(
            abs(s.delta_psi) < 1e-6 and abs(s.delta_speed) < 1e-6
            for s in strategies
        )
        self.assertTrue(has_zero)

    def test_symmetric_heading(self):
        """测试航向变化对称性"""
        strategies = generate_strategy_space(
            psi_range_deg=(-30, 30),
            psi_step_deg=10,
            speed_range=(0, 0),
            speed_step=1.0
        )

        headings = [s.delta_psi for s in strategies]

        # 检查正负对称
        for h in headings:
            if abs(h) > 1e-6:
                self.assertIn(-h, headings)


class TestStrategyEdgeCases(unittest.TestCase):
    """边界情况测试"""

    def test_large_heading_change(self):
        """测试大角度航向改变"""
        strategy = AvoidanceStrategy(np.deg2rad(180), 0)
        v0 = np.array([5, 0])

        v_final = strategy.get_final_velocity(v0)

        # 180度转向应该反向
        self.assertAlmostEqual(v_final[0], -5, places=3)
        self.assertAlmostEqual(abs(v_final[1]), 0, places=3)

    def test_zero_initial_velocity(self):
        """测试零初始速度"""
        strategy = AvoidanceStrategy(np.deg2rad(30), 5.0)
        v0 = np.array([0, 0])

        v_final = strategy.get_final_velocity(v0)

        # 从零开始加速
        self.assertAlmostEqual(np.linalg.norm(v_final), 5.0)

    def test_very_small_tau(self):
        """测试极小的响应时间"""
        strategy = AvoidanceStrategy(np.deg2rad(30), 1.0)
        v0 = np.array([5, 0])

        # tau 很小意味着快速响应
        v_1s = strategy.velocity_profile(v0, 1.0, tau=0.1)
        v_final = strategy.get_final_velocity(v0)

        # 1秒后应该非常接近最终值
        np.testing.assert_array_almost_equal(v_1s, v_final, decimal=2)

    def test_very_large_tau(self):
        """测试极大的响应时间"""
        strategy = AvoidanceStrategy(np.deg2rad(30), 1.0)
        v0 = np.array([5, 0])

        # tau 很大意味着慢速响应
        v_10s = strategy.velocity_profile(v0, 10.0, tau=100.0)

        # 10秒后应该接近初始值（速度大小变化小于20%）
        speed_ratio = np.linalg.norm(v_10s) / np.linalg.norm(v0)
        self.assertGreater(speed_ratio, 0.8)
        self.assertLess(speed_ratio, 1.3)


if __name__ == '__main__':
    unittest.main()
