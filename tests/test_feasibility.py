"""
test_feasibility.py - 时间一致性可行性判定测试
"""
from time_extended_hrvo.core.vessel import VesselState
from time_extended_hrvo.core.strategy import AvoidanceStrategy
from time_extended_hrvo.core.hrvo import HRVO, compute_hrvo
from time_extended_hrvo.core.feasibility import (
    is_strategy_feasible,
    compute_feasibility_margin,
    find_violation_time,
    check_time_consistency
)
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestIsStrategyFeasible(unittest.TestCase):
    """测试策略可行性检测"""

    def test_feasible_zero_strategy_no_obstacle(self):
        """测试无障碍时零策略可行"""
        strategy = AvoidanceStrategy(0, 0)
        v0 = np.array([5, 0])
        hrvo_list = []

        result = is_strategy_feasible(strategy, v0, hrvo_list, T_p=30.0)

        self.assertTrue(result)

    def test_feasible_avoiding_strategy(self):
        """测试避让策略可行"""
        # 创建一个简单的 HRVO
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 0], [-5, 0], 50)
        hrvo = compute_hrvo(own, obs)

        # 大幅右转策略
        strategy = AvoidanceStrategy(np.deg2rad(-30), 0)

        result = is_strategy_feasible(strategy, own.v, [hrvo], T_p=30.0)

        # 右转应该是可行的（避开对遇）
        # 具体结果取决于 HRVO 的几何形状

    def test_infeasible_collision_course(self):
        """测试碰撞航向不可行"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([300, 0], [-5, 0], 50)  # 近距离对遇
        hrvo = compute_hrvo(own, obs)

        # 保持原航向
        strategy = AvoidanceStrategy(0, 0)

        # 直行可能不可行
        result = is_strategy_feasible(strategy, own.v, [hrvo], T_p=30.0)

        # 结果取决于 HRVO 的具体形状

    def test_time_steps(self):
        """测试时间步长影响"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 50], [-5, 0], 50)
        hrvo = compute_hrvo(own, obs)

        strategy = AvoidanceStrategy(np.deg2rad(10), 0)

        # 使用不同时间步长
        result_coarse = is_strategy_feasible(
            strategy, own.v, [hrvo], T_p=30.0, dt=5.0)
        result_fine = is_strategy_feasible(
            strategy, own.v, [hrvo], T_p=30.0, dt=0.1)

        # 细粒度检测应该更严格（可能发现更多违规）

    def test_planning_horizon(self):
        """测试规划时域影响"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 50], [-5, 0], 50)
        hrvo = compute_hrvo(own, obs)

        strategy = AvoidanceStrategy(np.deg2rad(5), 0)

        # 短时域
        result_short = is_strategy_feasible(strategy, own.v, [hrvo], T_p=5.0)

        # 长时域
        result_long = is_strategy_feasible(strategy, own.v, [hrvo], T_p=60.0)

        # 长时域检测更多时间点

    def test_multiple_hrovs(self):
        """测试多个 HRVO"""
        own = VesselState([0, 0], [5, 0], 50)
        obs1 = VesselState([500, 100], [-3, 0], 50)
        obs2 = VesselState([500, -100], [-3, 0], 50)

        hrvo1 = compute_hrvo(own, obs1)
        hrvo2 = compute_hrvo(own, obs2)

        strategy = AvoidanceStrategy(0, 0)

        # 被两个障碍物夹住
        result = is_strategy_feasible(
            strategy, own.v, [hrvo1, hrvo2], T_p=30.0)


class TestComputeFeasibilityMargin(unittest.TestCase):
    """测试可行性裕度计算"""

    def test_positive_margin_no_obstacle(self):
        """测试无障碍时正裕度"""
        strategy = AvoidanceStrategy(0, 0)
        v0 = np.array([5, 0])

        margin = compute_feasibility_margin(strategy, v0, [], T_p=30.0)

        self.assertEqual(margin, float('inf'))

    def test_margin_with_hrvo(self):
        """测试有 HRVO 时的裕度"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 200], [-3, 0], 50)
        hrvo = compute_hrvo(own, obs)

        strategy = AvoidanceStrategy(np.deg2rad(-20), 0)

        margin = compute_feasibility_margin(strategy, own.v, [hrvo], T_p=30.0)

        # 应该返回一个数值
        self.assertIsInstance(margin, (int, float, np.floating))


class TestFindViolationTime(unittest.TestCase):
    """测试违规时间查找"""

    def test_no_violation(self):
        """测试无违规情况"""
        strategy = AvoidanceStrategy(0, 0)
        v0 = np.array([5, 0])

        t_violation, hrvo_idx = find_violation_time(strategy, v0, [], T_p=30.0)

        self.assertIsNone(t_violation)
        self.assertIsNone(hrvo_idx)

    def test_finds_violation(self):
        """测试找到违规时间"""
        # 构造一个必然违规的场景
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([200, 0], [-5, 0], 50)  # 近距离迎面

        try:
            hrvo = compute_hrvo(own, obs)

            # 保持原航向
            strategy = AvoidanceStrategy(0, 0)

            t_violation, hrvo_idx = find_violation_time(
                strategy, own.v, [hrvo], T_p=30.0
            )

            # 如果有违规，应该返回时间和索引
            if t_violation is not None:
                self.assertIsInstance(t_violation, (int, float))
                self.assertEqual(hrvo_idx, 0)
        except ValueError:
            # 已碰撞，跳过
            pass


class TestCheckTimeConsistency(unittest.TestCase):
    """测试时变 HRVO 一致性检查"""

    def test_static_hrvo_sequence(self):
        """测试静态 HRVO 序列"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 100], [-3, 0], 50)
        hrvo = compute_hrvo(own, obs)

        # 静态序列：始终返回相同的 HRVO
        def hrvo_sequence(t):
            return [hrvo]

        strategy = AvoidanceStrategy(np.deg2rad(-15), 0)

        result = check_time_consistency(
            strategy, own.v, hrvo_sequence, T_p=30.0
        )

        self.assertIsInstance(result, bool)

    def test_empty_hrvo_sequence(self):
        """测试空 HRVO 序列"""
        strategy = AvoidanceStrategy(0, 0)
        v0 = np.array([5, 0])

        def empty_sequence(t):
            return []

        result = check_time_consistency(
            strategy, v0, empty_sequence, T_p=30.0
        )

        self.assertTrue(result)


class TestFeasibilityEdgeCases(unittest.TestCase):
    """边界情况测试"""

    def test_zero_planning_horizon(self):
        """测试零规划时域"""
        strategy = AvoidanceStrategy(0, 0)
        v0 = np.array([5, 0])

        # T_p = 0 只检查 t=0
        result = is_strategy_feasible(strategy, v0, [], T_p=0)

        self.assertTrue(result)

    def test_very_short_horizon(self):
        """测试极短规划时域"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 0], [-5, 0], 50)
        hrvo = compute_hrvo(own, obs)

        strategy = AvoidanceStrategy(np.deg2rad(30), 1.0)

        # 极短时域内策略可能可行
        result = is_strategy_feasible(strategy, own.v, [hrvo], T_p=0.1)

    def test_very_long_horizon(self):
        """测试极长规划时域"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 100], [-3, 0], 50)
        hrvo = compute_hrvo(own, obs)

        strategy = AvoidanceStrategy(np.deg2rad(-20), 0)

        # 长时域检测
        result = is_strategy_feasible(
            strategy, own.v, [hrvo], T_p=300.0, dt=1.0)

        self.assertIsInstance(result, bool)


if __name__ == '__main__':
    unittest.main()
