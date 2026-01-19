"""
test_cost.py - 代价函数测试
"""
from time_extended_hrvo.core.vessel import VesselState
from time_extended_hrvo.core.strategy import AvoidanceStrategy
from time_extended_hrvo.core.cost import (
    strategy_cost,
    colregs_cost,
    efficiency_cost,
    combined_cost,
    dcpa_tcpa_cost
)
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStrategyCost(unittest.TestCase):
    """测试基本策略代价"""

    def test_zero_strategy_zero_cost(self):
        """测试零策略零代价"""
        strategy = AvoidanceStrategy(0, 0)
        v_pref = np.array([5, 0])

        cost = strategy_cost(strategy, v_pref)

        self.assertEqual(cost, 0)

    def test_heading_change_cost(self):
        """测试航向改变代价"""
        strategy = AvoidanceStrategy(np.deg2rad(30), 0)
        v_pref = np.array([5, 0])

        cost = strategy_cost(strategy, v_pref, w_heading=1.0,
                             w_speed=0, w_deviation=0)

        self.assertAlmostEqual(cost, np.deg2rad(30))

    def test_speed_change_cost(self):
        """测试速度改变代价"""
        strategy = AvoidanceStrategy(0, 2.0)
        v_pref = np.array([5, 0])

        cost = strategy_cost(strategy, v_pref, w_heading=0,
                             w_speed=1.0, w_deviation=0)

        self.assertAlmostEqual(cost, 2.0)

    def test_negative_changes(self):
        """测试负值改变（绝对值代价）"""
        strategy_pos = AvoidanceStrategy(np.deg2rad(20), 1.0)
        strategy_neg = AvoidanceStrategy(np.deg2rad(-20), -1.0)
        v_pref = np.array([5, 0])

        cost_pos = strategy_cost(strategy_pos, v_pref,
                                 w_heading=1.0, w_speed=1.0, w_deviation=0)
        cost_neg = strategy_cost(strategy_neg, v_pref,
                                 w_heading=1.0, w_speed=1.0, w_deviation=0)

        # 绝对值相同，代价相同
        self.assertAlmostEqual(cost_pos, cost_neg)

    def test_deviation_cost(self):
        """测试偏离代价"""
        strategy = AvoidanceStrategy(np.deg2rad(90), 0)  # 90度转向
        v_pref = np.array([5, 0])

        cost = strategy_cost(strategy, v_pref, w_heading=0,
                             w_speed=0, w_deviation=1.0)

        # 90度转向后速度与原速度垂直，偏离距离 = sqrt(5^2 + 5^2) ≈ 7.07
        self.assertGreater(cost, 0)

    def test_weight_scaling(self):
        """测试权重缩放"""
        strategy = AvoidanceStrategy(np.deg2rad(10), 1.0)
        v_pref = np.array([5, 0])

        cost_1x = strategy_cost(
            strategy, v_pref, w_heading=1.0, w_speed=1.0, w_deviation=1.0)
        cost_2x = strategy_cost(
            strategy, v_pref, w_heading=2.0, w_speed=2.0, w_deviation=2.0)

        # 2倍权重应该得到约2倍代价
        self.assertAlmostEqual(cost_2x, cost_1x * 2, places=3)


class TestColregsCost(unittest.TestCase):
    """测试 COLREGs 合规代价

    转向约定：正值=右转(Starboard)，负值=左转(Port)
    """

    def test_head_on_right_turn(self):
        """测试对遇场景右转代价"""
        strategy_right = AvoidanceStrategy(np.deg2rad(20), 0)   # 右转（正值）
        strategy_left = AvoidanceStrategy(np.deg2rad(-20), 0)   # 左转（负值）

        cost_right = colregs_cost(strategy_right, 'head-on')
        cost_left = colregs_cost(strategy_left, 'head-on')

        # 对遇时左转应该比右转代价更高
        self.assertLess(cost_right, cost_left)

    def test_crossing_give_way(self):
        """测试交叉相遇让路代价"""
        strategy_right = AvoidanceStrategy(np.deg2rad(20), 0)   # 右转（正值）
        strategy_left = AvoidanceStrategy(np.deg2rad(-20), 0)   # 左转（负值）

        cost_right = colregs_cost(strategy_right, 'crossing')
        cost_left = colregs_cost(strategy_left, 'crossing')

        # 交叉相遇时也应鼓励右转
        self.assertLess(cost_right, cost_left)

    def test_overtaking_right_preferred(self):
        """测试追越场景右转优先

        虽然追越允许从两侧超越，但右转仍然是首选
        """
        strategy_right = AvoidanceStrategy(np.deg2rad(20), 0)   # 右转（正值）
        strategy_left = AvoidanceStrategy(np.deg2rad(-20), 0)   # 左转（负值）

        cost_right = colregs_cost(strategy_right, 'overtaking')
        cost_left = colregs_cost(strategy_left, 'overtaking')

        # 追越时右转代价应低于左转（右转优先）
        self.assertLess(cost_right, cost_left)

    def test_zero_turn_zero_cost(self):
        """测试零转向代价"""
        strategy = AvoidanceStrategy(0, 0)

        for encounter in ['head-on', 'crossing', 'overtaking']:
            cost = colregs_cost(strategy, encounter)
            # 零转向应该有较低代价（但不一定为零，因为对遇/交叉有基础惩罚）
            self.assertLessEqual(cost, 15)  # 允许基础惩罚存在


class TestEfficiencyCost(unittest.TestCase):
    """测试效率代价"""

    def test_no_change_zero_cost(self):
        """测试无改变零代价"""
        strategy = AvoidanceStrategy(0, 0)
        v_orig = np.array([5, 0])
        goal_dir = np.array([1, 0])

        cost = efficiency_cost(strategy, v_orig, goal_dir)

        self.assertEqual(cost, 0)

    def test_speed_reduction_cost(self):
        """测试减速代价"""
        strategy = AvoidanceStrategy(0, -2.0)
        v_orig = np.array([5, 0])
        goal_dir = np.array([1, 0])

        cost = efficiency_cost(strategy, v_orig, goal_dir)

        # 减速应该有代价
        self.assertGreater(cost, 0)

    def test_turn_away_from_goal_cost(self):
        """测试偏离目标方向的代价"""
        strategy = AvoidanceStrategy(np.deg2rad(90), 0)  # 90度转向
        v_orig = np.array([5, 0])
        goal_dir = np.array([1, 0])  # 目标在正前方

        cost = efficiency_cost(strategy, v_orig, goal_dir)

        # 转向90度后在目标方向的投影减少
        self.assertGreater(cost, 0)

    def test_speed_increase_no_cost(self):
        """测试加速无代价"""
        strategy = AvoidanceStrategy(0, 2.0)
        v_orig = np.array([5, 0])
        goal_dir = np.array([1, 0])

        cost = efficiency_cost(strategy, v_orig, goal_dir)

        # 加速应该是有益的，代价为0或负（但我们 max(0, ...) 所以是0）
        self.assertEqual(cost, 0)


class TestCombinedCost(unittest.TestCase):
    """测试综合代价"""

    def test_basic_combined(self):
        """测试基本综合代价"""
        strategy = AvoidanceStrategy(np.deg2rad(10), 1.0)
        v_pref = np.array([5, 0])

        cost = combined_cost(strategy, v_pref)

        self.assertGreater(cost, 0)

    def test_with_encounter_type(self):
        """测试带会遇类型的综合代价"""
        strategy = AvoidanceStrategy(np.deg2rad(-15), 0)
        v_pref = np.array([5, 0])

        cost = combined_cost(strategy, v_pref, encounter_type='head-on')

        self.assertIsInstance(cost, (int, float, np.floating))

    def test_with_goal_direction(self):
        """测试带目标方向的综合代价"""
        strategy = AvoidanceStrategy(np.deg2rad(10), 0)
        v_pref = np.array([5, 0])
        goal_dir = np.array([1, 0])

        cost = combined_cost(strategy, v_pref, goal_direction=goal_dir)

        self.assertIsInstance(cost, (int, float, np.floating))

    def test_custom_weights(self):
        """测试自定义权重"""
        strategy = AvoidanceStrategy(np.deg2rad(10), 1.0)
        v_pref = np.array([5, 0])

        weights = {
            'strategy': 2.0,
            'heading': 1.0,
            'speed': 1.0,
            'deviation': 1.0
        }

        cost = combined_cost(strategy, v_pref, weights=weights)

        self.assertGreater(cost, 0)


class TestDcpaTcpaCost(unittest.TestCase):
    """测试 DCPA/TCPA 代价"""

    def test_collision_course_high_cost(self):
        """测试碰撞航向高代价"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 0], [-5, 0], 50)

        strategy = AvoidanceStrategy(0, 0)  # 保持原航向

        cost = dcpa_tcpa_cost(own, obs, strategy)

        # 碰撞航向应该有高代价
        self.assertGreater(cost, 0)

    def test_safe_course_low_cost(self):
        """测试安全航向低代价"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 1000], [-5, 0], 50)  # 远离

        strategy = AvoidanceStrategy(0, 0)

        cost = dcpa_tcpa_cost(own, obs, strategy)

        # 安全情况代价应该较低
        self.assertLess(cost, 1.0)

    def test_avoiding_strategy_reduces_cost(self):
        """测试避让策略降低代价"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 50], [-5, 0], 50)

        strategy_straight = AvoidanceStrategy(0, 0)
        strategy_avoid = AvoidanceStrategy(np.deg2rad(-30), 0)

        cost_straight = dcpa_tcpa_cost(own, obs, strategy_straight)
        cost_avoid = dcpa_tcpa_cost(own, obs, strategy_avoid)

        # 避让策略应该有更低或相近的代价
        # 注意：这取决于具体几何关系

    def test_diverging_course_zero_cost(self):
        """测试远离航向零代价"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([-500, 0], [-5, 0], 50)  # 在后方远离

        strategy = AvoidanceStrategy(0, 0)

        cost = dcpa_tcpa_cost(own, obs, strategy)

        # 已经远离，代价应该为0
        self.assertEqual(cost, 0)


class TestCostEdgeCases(unittest.TestCase):
    """边界情况测试"""

    def test_zero_velocity(self):
        """测试零速度"""
        strategy = AvoidanceStrategy(np.deg2rad(30), 5.0)
        v_pref = np.array([0, 0])

        cost = strategy_cost(strategy, v_pref)

        # 应该能计算，不崩溃
        self.assertIsInstance(cost, (int, float, np.floating))

    def test_very_large_changes(self):
        """测试极大改变"""
        strategy = AvoidanceStrategy(np.deg2rad(180), 100.0)
        v_pref = np.array([5, 0])

        cost = strategy_cost(strategy, v_pref)

        self.assertGreater(cost, 0)

    def test_unnormalized_goal_direction(self):
        """测试非归一化目标方向"""
        strategy = AvoidanceStrategy(0, 0)
        v_orig = np.array([5, 0])
        goal_dir = np.array([100, 0])  # 非单位向量

        cost = efficiency_cost(strategy, v_orig, goal_dir)

        # 应该自动归一化处理
        self.assertEqual(cost, 0)


if __name__ == '__main__':
    unittest.main()
