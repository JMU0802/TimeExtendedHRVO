"""
test_planner.py - 主规划器测试
"""
from time_extended_hrvo.core.strategy import AvoidanceStrategy
from time_extended_hrvo.core.vessel import VesselState
from time_extended_hrvo.planner.te_hrvo_planner import (
    TimeExtendedHRVOPlanner,
    TraditionalHRVOPlanner
)
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTimeExtendedHRVOPlanner(unittest.TestCase):
    """测试 Time-Extended HRVO 规划器"""

    def test_init(self):
        """测试初始化"""
        planner = TimeExtendedHRVOPlanner(T_p=30.0, dt=0.5, tau=10.0)

        self.assertEqual(planner.T_p, 30.0)
        self.assertEqual(planner.dt, 0.5)
        self.assertEqual(planner.tau, 10.0)

    def test_plan_no_obstacle(self):
        """测试无障碍规划"""
        planner = TimeExtendedHRVOPlanner()

        own = VesselState([0, 0], [5, 0], 50)

        strategy = planner.plan(own, [])

        # 无障碍应该返回零策略或最小代价策略
        self.assertIsNotNone(strategy)
        self.assertIsInstance(strategy, AvoidanceStrategy)

    def test_plan_single_obstacle(self):
        """测试单障碍规划"""
        planner = TimeExtendedHRVOPlanner()

        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 100], [-3, 0], 50)

        strategy = planner.plan(own, [obs])

        self.assertIsNotNone(strategy)
        self.assertIsInstance(strategy, AvoidanceStrategy)

    def test_plan_multiple_obstacles(self):
        """测试多障碍规划"""
        planner = TimeExtendedHRVOPlanner()

        own = VesselState([0, 0], [5, 0], 50)
        obstacles = [
            VesselState([500, 100], [-3, 0], 50),
            VesselState([500, -100], [-3, 0], 50),
            VesselState([600, 0], [-4, 0], 50),
        ]

        strategy = planner.plan(own, obstacles)

        # 应该返回一个策略（可能是后备策略）
        self.assertIsInstance(strategy, (AvoidanceStrategy, type(None)))

    def test_plan_with_v_pref(self):
        """测试指定偏好速度"""
        planner = TimeExtendedHRVOPlanner()

        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 100], [-3, 0], 50)
        v_pref = np.array([6, 0])  # 偏好更快

        strategy = planner.plan(own, [obs], v_pref=v_pref)

        self.assertIsNotNone(strategy)

    def test_plan_with_encounter_type(self):
        """测试指定会遇类型"""
        planner = TimeExtendedHRVOPlanner()

        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 0], [-5, 0], 50)

        strategy = planner.plan(own, [obs], encounter_type='head-on')

        self.assertIsNotNone(strategy)

    def test_plan_with_details(self):
        """测试详细规划结果"""
        planner = TimeExtendedHRVOPlanner()

        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 100], [-3, 0], 50)

        result = planner.plan_with_details(own, [obs])

        # 检查返回的字典结构
        self.assertIn('best_strategy', result)
        self.assertIn('feasible_strategies', result)
        self.assertIn('all_strategies', result)
        self.assertIn('hrvo_list', result)
        self.assertIn('costs', result)

        # 检查类型
        self.assertIsInstance(result['feasible_strategies'], list)
        self.assertIsInstance(result['all_strategies'], list)
        self.assertIsInstance(result['hrvo_list'], list)

    def test_head_on_scenario(self):
        """测试对遇场景"""
        planner = TimeExtendedHRVOPlanner()

        own = VesselState([0, 0], [0, 5], 50)      # 向北
        obs = VesselState([0, 1000], [0, -5], 50)  # 向南

        strategy = planner.plan(own, [obs])

        self.assertIsNotNone(strategy)

        # 对遇应该右转（负航向角）
        # 但具体策略取决于可行策略空间

    def test_crossing_scenario(self):
        """测试交叉相遇场景"""
        planner = TimeExtendedHRVOPlanner()

        own = VesselState([0, 0], [5, 0], 50)      # 向东
        obs = VesselState([300, -200], [0, 5], 50)  # 向北（在右前方）

        strategy = planner.plan(own, [obs])

        self.assertIsNotNone(strategy)

    def test_overtaking_scenario(self):
        """测试追越场景"""
        planner = TimeExtendedHRVOPlanner()

        own = VesselState([0, 0], [8, 0], 50)    # 快船
        obs = VesselState([300, 0], [3, 0], 50)  # 慢船在前方

        strategy = planner.plan(own, [obs])

        self.assertIsNotNone(strategy)


class TestTraditionalHRVOPlanner(unittest.TestCase):
    """测试传统 HRVO 规划器"""

    def test_init(self):
        """测试初始化"""
        planner = TraditionalHRVOPlanner()
        self.assertIsNotNone(planner)

    def test_plan_no_obstacle(self):
        """测试无障碍规划"""
        planner = TraditionalHRVOPlanner()

        own = VesselState([0, 0], [5, 0], 50)

        strategy = planner.plan(own, [])

        self.assertIsNotNone(strategy)

    def test_plan_single_obstacle(self):
        """测试单障碍规划"""
        planner = TraditionalHRVOPlanner()

        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([500, 100], [-3, 0], 50)

        strategy = planner.plan(own, [obs])

        self.assertIsInstance(strategy, (AvoidanceStrategy, type(None)))


class TestPlannerComparison(unittest.TestCase):
    """对比测试两种规划器"""

    def test_both_find_strategy(self):
        """测试两种规划器都能找到策略"""
        te_planner = TimeExtendedHRVOPlanner()
        trad_planner = TraditionalHRVOPlanner()

        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([600, 150], [-3, 0], 50)

        te_strategy = te_planner.plan(own, [obs])
        trad_strategy = trad_planner.plan(own, [obs])

        # 两种方法都应该找到策略
        self.assertIsNotNone(te_strategy)
        self.assertIsNotNone(trad_strategy)

    def test_strategy_difference(self):
        """测试策略可能存在差异"""
        te_planner = TimeExtendedHRVOPlanner(T_p=30.0, tau=15.0)
        trad_planner = TraditionalHRVOPlanner()

        # 构造一个可能导致差异的场景
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([300, 30], [-4, 0], 50)

        te_strategy = te_planner.plan(own, [obs])
        trad_strategy = trad_planner.plan(own, [obs])

        # 两种方法可能选择不同的策略
        # 这里我们只验证它们都能工作


class TestPlannerEdgeCases(unittest.TestCase):
    """边界情况测试"""

    def test_collision_already_occurred(self):
        """测试已碰撞情况"""
        planner = TimeExtendedHRVOPlanner()

        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([50, 0], [-5, 0], 50)  # 已经碰撞

        # 应该能处理（跳过碰撞的障碍物）
        strategy = planner.plan(own, [obs])

    def test_stationary_vessels(self):
        """测试静止船舶"""
        planner = TimeExtendedHRVOPlanner()

        own = VesselState([0, 0], [0, 0], 50)
        obs = VesselState([500, 0], [0, 0], 50)

        strategy = planner.plan(own, [obs])

        self.assertIsNotNone(strategy)

    def test_many_obstacles(self):
        """测试大量障碍物"""
        planner = TimeExtendedHRVOPlanner()

        own = VesselState([0, 0], [5, 0], 50)

        # 创建10个障碍物
        obstacles = []
        for i in range(10):
            angle = i * 2 * np.pi / 10
            x = 500 + 100 * np.cos(angle)
            y = 100 * np.sin(angle)
            obs = VesselState([x, y], [-2, 0], 30)
            obstacles.append(obs)

        strategy = planner.plan(own, obstacles)

        # 应该能处理（可能返回后备策略）
        self.assertIsInstance(strategy, (AvoidanceStrategy, type(None)))

    def test_different_planning_horizons(self):
        """测试不同规划时域"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([400, 50], [-4, 0], 50)

        for T_p in [5.0, 15.0, 30.0, 60.0]:
            planner = TimeExtendedHRVOPlanner(T_p=T_p)
            strategy = planner.plan(own, [obs])

            self.assertIsInstance(strategy, (AvoidanceStrategy, type(None)))

    def test_different_tau_values(self):
        """测试不同响应时间"""
        own = VesselState([0, 0], [5, 0], 50)
        obs = VesselState([400, 50], [-4, 0], 50)

        for tau in [1.0, 5.0, 10.0, 20.0]:
            planner = TimeExtendedHRVOPlanner(tau=tau)
            strategy = planner.plan(own, [obs])

            self.assertIsInstance(strategy, (AvoidanceStrategy, type(None)))


if __name__ == '__main__':
    unittest.main()
