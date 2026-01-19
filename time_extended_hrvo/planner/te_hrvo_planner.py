"""
te_hrvo_planner.py - Time-Extended HRVO 主规划器

对应论文 Algorithm 1: Time-Extended HRVO Planning

COLREGs 右转优先原则:
    除非陷入紧迫局面，船舶应始终选择向右转向避让
"""
import numpy as np
from ..core.hrvo import compute_hrvo
from ..core.feasibility import is_strategy_feasible, compute_feasibility_margin
from ..core.cost import (
    strategy_cost, combined_cost, dcpa_tcpa_cost,
    marine_strategy_cost, marine_combined_cost, is_emergency_situation
)
from ..core.strategy import (
    AvoidanceStrategy, generate_strategy_space,
    generate_marine_strategy_space, generate_colregs_strategy_space,
    generate_pure_heading_strategies, generate_emergency_strategies
)
from ..utils.geometry import classify_encounter


class TimeExtendedHRVOPlanner:
    """
    时间扩展 HRVO 规划器

    核心算法流程 (Algorithm 1):
        1. HRVO 构造: 为每个目标船构造 HRVO
        2. 策略空间生成: 离散化策略空间 Θ（优先右转改向）
        3. 可行性检测: 时间扩展约束检验
        4. 代价优化: 选择最优可行策略（右转优先）

    COLREGs 合规:
        - 除非紧迫局面，始终选择右转避让
        - 左转仅在右转不可行时考虑

    Attributes:
        T_p (float): 规划时域 (s)
        dt (float): 可行性检测时间步长 (s)
        tau (float): 机动响应时间常数 (s)
        use_marine_cost (bool): 是否使用船舶专用代价函数
        starboard_first (bool): 是否强制右转优先
        avoidance_mode (str): 避让模式 ('heading_only', 'speed_only', 'combined')
    """

    def __init__(self, T_p=30.0, dt=0.5, tau=10.0, use_marine_cost=True,
                 starboard_first=True, avoidance_mode='heading_only'):
        """
        初始化规划器

        Args:
            T_p: 规划时域 (s), 默认 30s
            dt: 可行性检测时间步长 (s), 默认 0.5s
            tau: 机动响应时间常数 (s), 默认 10s
            use_marine_cost: 是否使用船舶专用代价函数（优先改向）
            starboard_first: 是否强制右转优先（除非紧急情况）
            avoidance_mode: 避让模式
                - 'heading_only': 仅航向避让（只改向）
                - 'speed_only': 仅航速避让（只改速）
                - 'combined': 组合避让（改向+改速）
        """
        self.T_p = T_p
        self.dt = dt
        self.tau = tau
        self.use_marine_cost = use_marine_cost
        self.starboard_first = starboard_first
        self.avoidance_mode = avoidance_mode

    def plan(self, own_state, obstacles, v_pref=None,
             encounter_type=None, weights=None):
        """
        执行 Time-Extended HRVO 规划（强化右转优先）

        对应论文 Algorithm 1

        Args:
            own_state: 本船状态 (VesselState)
            obstacles: 目标船状态列表 [VesselState, ...]
            v_pref: 偏好速度（默认为当前速度）
            encounter_type: 会遇类型（用于 COLREGs 代价）
            weights: 代价函数权重

        Returns:
            AvoidanceStrategy: 最优可行策略，或 None 如果无可行策略
        """
        if v_pref is None:
            v_pref = own_state.v.copy()

        # 自动检测会遇类型
        if encounter_type is None and obstacles:
            encounter_type = classify_encounter(own_state, obstacles[0])

        # 检测是否为紧急情况
        is_emergency = is_emergency_situation(
            own_state, obstacles) if obstacles else False

        # ============================================
        # Step 1: HRVO 构造
        # ============================================
        hrvo_list = []
        for obs in obstacles:
            try:
                hrvo = compute_hrvo(own_state, obs)
                hrvo_list.append(hrvo)
            except ValueError as e:
                # 已碰撞情况
                print(f"Warning: {e}")
                continue

        # ============================================
        # Step 2: 策略空间 Θ 生成（分层策略）
        # 通常情况只用纯改向，紧急情况才引入减速策略
        # ============================================
        strategies = self._generate_strategies(encounter_type, is_emergency)

        # ============================================
        # Step 3: 时间扩展可行性检测
        # ============================================
        feasible_strategies = []
        feasible_starboard = []  # 可行的右转策略
        feasible_port = []       # 可行的左转策略

        for strategy in strategies:
            if is_strategy_feasible(strategy, own_state.v,
                                    hrvo_list, self.T_p, self.dt):
                feasible_strategies.append(strategy)
                # 分类右转/左转策略
                if strategy.delta_psi > 0.01:  # 右转
                    feasible_starboard.append(strategy)
                elif strategy.delta_psi < -0.01:  # 左转
                    feasible_port.append(strategy)
                else:  # 保持或纯减速
                    feasible_starboard.append(strategy)  # 归入右转组

        if not feasible_strategies:
            # 无可行策略，尝试放宽约束
            return self._fallback_plan(own_state, obstacles, hrvo_list,
                                       v_pref, encounter_type, is_emergency)

        # ============================================
        # Step 4: 代价优化（强化右转优先）
        # ============================================

        # 如果启用右转优先且非紧急情况，优先从右转策略中选择
        if self.starboard_first and not is_emergency and feasible_starboard:
            # 先尝试只从右转策略中选择
            best_strategy = min(
                feasible_starboard,
                key=lambda s: self._compute_cost(s, own_state, obstacles,
                                                 v_pref, encounter_type, weights,
                                                 is_emergency)
            )
        else:
            # 紧急情况或无可行右转策略：从所有可行策略中选择
            best_strategy = min(
                feasible_strategies,
                key=lambda s: self._compute_cost(s, own_state, obstacles,
                                                 v_pref, encounter_type, weights,
                                                 is_emergency)
            )

        return best_strategy

    def _generate_strategies(self, encounter_type=None, is_emergency=False):
        """
        生成策略空间（分层策略）

        根据避让模式生成不同的策略空间：
        - heading_only: 仅航向避让（只改向）
        - speed_only: 仅航速避让（只改速）
        - combined: 组合避让（改向+改速）

        Args:
            encounter_type: 会遇类型
            is_emergency: 是否为紧急情况
        """
        if self.use_marine_cost:
            # 使用船舶专用策略空间，根据避让模式生成
            return generate_marine_strategy_space(
                include_speed_change=is_emergency,
                mode=self.avoidance_mode
            )
        else:
            # 传统策略空间
            return generate_strategy_space(
                psi_range_deg=(-30, 30),
                psi_step_deg=5,
                speed_range=(-2, 2),
                speed_step=0.5
            )

    def _compute_cost(self, strategy, own_state, obstacles,
                      v_pref, encounter_type, weights, is_emergency=False):
        """
        计算综合代价（强化右转优先）
        """
        if self.use_marine_cost:
            # 使用船舶专用代价函数（强化右转优先）
            return marine_combined_cost(
                strategy, v_pref, encounter_type,
                obstacles=obstacles, own_state=own_state,
                is_emergency=is_emergency
            )
        else:
            # 传统代价函数
            cost = combined_cost(
                strategy, v_pref, encounter_type, weights=weights)
            for obs in obstacles:
                cost += dcpa_tcpa_cost(own_state, obs, strategy)
            return cost

    def _fallback_plan(self, own_state, obstacles, hrvo_list, v_pref,
                       encounter_type, is_emergency=False):
        """
        后备规划：当无可行策略时（增强版 - 确保总是返回策略）

        多船场景下可能没有完全可行的解，此时选择"最小侵入"策略

        优先级：
        1. 大幅右转（纯改向）
        2. 右转 + 减速
        3. 左转（紧急情况）
        4. 全方位搜索（最小侵入）
        5. 保持当前速度（最后手段）

        转向约定: 正值=右转(Starboard)，负值=左转(Port)
        """
        # 第一优先级：尝试更大角度的纯右转
        starboard_strategies = []
        for angle in [60, 70, 80, 90, 100, 110, 120, 135, 150, 170, 180]:
            starboard_strategies.append(
                AvoidanceStrategy(np.deg2rad(angle), 0.0)
            )

        for strategy in starboard_strategies:
            margin = compute_feasibility_margin(
                strategy, own_state.v, hrvo_list, self.T_p, self.dt
            )
            if margin > 0:  # 可行
                return strategy

        # 第二优先级：大幅右转 + 减速
        starboard_with_speed = []
        for angle in [45, 60, 90, 120, 150]:
            for du in [-0.5, -1.0, -1.5, -2.0]:
                starboard_with_speed.append(
                    AvoidanceStrategy(np.deg2rad(angle), du)
                )

        for strategy in starboard_with_speed:
            margin = compute_feasibility_margin(
                strategy, own_state.v, hrvo_list, self.T_p, self.dt
            )
            if margin > 0:
                return strategy

        # 第三优先级：考虑左转（紧急情况下放宽）
        port_strategies = []
        for angle in [-45, -60, -90, -120, -150, -180]:
            port_strategies.append(
                AvoidanceStrategy(np.deg2rad(angle), 0.0)
            )
            # 左转 + 减速
            for du in [-0.5, -1.0, -1.5]:
                port_strategies.append(
                    AvoidanceStrategy(np.deg2rad(angle), du)
                )

        for strategy in port_strategies:
            margin = compute_feasibility_margin(
                strategy, own_state.v, hrvo_list, self.T_p, self.dt
            )
            if margin > 0:
                return strategy

        # 第四优先级：全方位搜索（最小侵入策略）
        # 当没有完全可行解时，选择侵入HRVO程度最小的策略
        all_directions = []
        for angle in range(0, 360, 15):  # 每15度一个方向
            for du in [0, -0.5, -1.0, -1.5, -2.0, 0.5]:
                all_directions.append(
                    AvoidanceStrategy(np.deg2rad(angle), du)
                )

        best_strategy = None
        best_margin = float('-inf')

        for strategy in all_directions:
            margin = compute_feasibility_margin(
                strategy, own_state.v, hrvo_list, self.T_p, self.dt
            )
            # 给右转策略加分（0-180度为右转）
            adjusted_margin = margin
            angle_deg = np.rad2deg(strategy.delta_psi) % 360
            if 0 < angle_deg <= 180:  # 右转
                adjusted_margin += 1.0

            # 给保持速度的策略加分
            if abs(strategy.delta_speed) < 0.1:
                adjusted_margin += 0.5

            if adjusted_margin > best_margin:
                best_margin = adjusted_margin
                best_strategy = strategy

        # 第五优先级：如果还是没有找到策略，返回保持当前状态或小幅右转
        if best_strategy is None:
            # 返回小幅右转作为默认策略
            best_strategy = AvoidanceStrategy(np.deg2rad(30), 0.0)

        return best_strategy

    def plan_with_details(self, own_state, obstacles, v_pref=None,
                          encounter_type=None, weights=None):
        """
        带详细信息的规划

        Returns:
            dict: {
                'best_strategy': AvoidanceStrategy,
                'feasible_strategies': list,
                'all_strategies': list,
                'hrvo_list': list,
                'costs': dict,
                'is_emergency': bool
            }
        """
        if v_pref is None:
            v_pref = own_state.v.copy()

        # 自动检测会遇类型
        if encounter_type is None and obstacles:
            encounter_type = classify_encounter(own_state, obstacles[0])

        # 检测是否为紧急情况
        is_emergency = is_emergency_situation(
            own_state, obstacles) if obstacles else False

        # HRVO 构造
        hrvo_list = []
        for obs in obstacles:
            try:
                hrvo_list.append(compute_hrvo(own_state, obs))
            except ValueError:
                continue

        # 策略空间（分层策略：通常情况只用纯改向，紧急情况才加入减速）
        all_strategies = self._generate_strategies(
            encounter_type, is_emergency)

        # 可行性检测
        feasible_strategies = []
        feasibility_results = {}

        for i, strategy in enumerate(all_strategies):
            is_feasible = is_strategy_feasible(
                strategy, own_state.v, hrvo_list, self.T_p, self.dt
            )
            feasibility_results[i] = is_feasible
            if is_feasible:
                feasible_strategies.append(strategy)

        # 代价计算
        costs = {}
        for i, strategy in enumerate(feasible_strategies):
            costs[i] = self._compute_cost(
                strategy, own_state, obstacles, v_pref, encounter_type,
                weights, is_emergency
            )

        # 最优策略
        best_strategy = None
        if feasible_strategies:
            # 右转优先
            feasible_starboard = [
                s for s in feasible_strategies if s.delta_psi >= 0]
            if self.starboard_first and not is_emergency and feasible_starboard:
                best_idx = min(
                    [i for i, s in enumerate(
                        feasible_strategies) if s.delta_psi >= 0],
                    key=lambda k: costs[k]
                )
            else:
                best_idx = min(costs.keys(), key=lambda k: costs[k])
            best_strategy = feasible_strategies[best_idx]
        else:
            best_strategy = self._fallback_plan(
                own_state, obstacles, hrvo_list, v_pref, encounter_type, is_emergency)

        return {
            'best_strategy': best_strategy,
            'feasible_strategies': feasible_strategies,
            'all_strategies': all_strategies,
            'hrvo_list': hrvo_list,
            'feasibility_results': feasibility_results,
            'costs': costs,
            'encounter_type': encounter_type,
            'is_emergency': is_emergency
        }


class TraditionalHRVOPlanner:
    """
    传统 HRVO 规划器（用于对比实验）

    与 TimeExtendedHRVOPlanner 的区别:
        - 仅检测当前时刻的速度是否在 HRVO 内
        - 不考虑时间一致性约束
    """

    def __init__(self, use_marine_cost=True, starboard_first=True):
        """
        Args:
            use_marine_cost: 是否使用船舶专用代价函数
            starboard_first: 是否强制右转优先
        """
        self.use_marine_cost = use_marine_cost
        self.starboard_first = starboard_first

    def plan(self, own_state, obstacles, v_pref=None):
        """
        传统 HRVO 规划（仅考虑当前时刻，右转优先）
        """
        if v_pref is None:
            v_pref = own_state.v.copy()

        # 检测会遇类型
        encounter_type = None
        if obstacles:
            encounter_type = classify_encounter(own_state, obstacles[0])

        # 检测是否为紧急情况
        is_emergency = is_emergency_situation(
            own_state, obstacles) if obstacles else False

        # HRVO 构造
        hrvo_list = []
        for obs in obstacles:
            try:
                hrvo_list.append(compute_hrvo(own_state, obs))
            except ValueError:
                continue

        # 策略空间（分层策略：通常情况只用纯改向，紧急情况才加入减速）
        if self.use_marine_cost:
            strategies = generate_marine_strategy_space(
                include_speed_change=is_emergency)
        else:
            strategies = generate_strategy_space()

        # 传统可行性检测：仅检测最终速度
        feasible = []
        feasible_starboard = []
        for strategy in strategies:
            v_new = strategy.get_final_velocity(own_state.v)
            is_feasible = True
            for hrvo in hrvo_list:
                if hrvo.contains(v_new):
                    is_feasible = False
                    break
            if is_feasible:
                feasible.append(strategy)
                if strategy.delta_psi >= 0:
                    feasible_starboard.append(strategy)

        if not feasible:
            # 无可行策略时，返回最小侵入策略
            return self._fallback_strategy(own_state, hrvo_list)

        # 选择代价最小的策略（右转优先）
        if self.use_marine_cost:
            # 非紧急情况且有可行右转策略：只从右转中选
            if self.starboard_first and not is_emergency and feasible_starboard:
                best = min(feasible_starboard,
                           key=lambda s: marine_strategy_cost(s, v_pref, encounter_type, is_emergency))
            else:
                best = min(feasible,
                           key=lambda s: marine_strategy_cost(s, v_pref, encounter_type, is_emergency))
        else:
            best = min(feasible, key=lambda s: strategy_cost(s, v_pref))

        return best

    def _fallback_strategy(self, own_state, hrvo_list):
        """
        传统规划器的后备策略：选择最小侵入HRVO的策略
        """
        best_strategy = None
        best_margin = float('-inf')

        # 全方位搜索
        for angle in range(0, 360, 15):
            for du in [0, -0.5, -1.0, -1.5]:
                strategy = AvoidanceStrategy(np.deg2rad(angle), du)
                v_new = strategy.get_final_velocity(own_state.v)

                # 计算到所有HRVO边界的最小距离
                min_dist = float('inf')
                for hrvo in hrvo_list:
                    dist = hrvo.distance_to_boundary(v_new)
                    min_dist = min(min_dist, dist)

                # 给右转加分
                adjusted_margin = min_dist
                if 0 < angle <= 180:
                    adjusted_margin += 0.5

                if adjusted_margin > best_margin:
                    best_margin = adjusted_margin
                    best_strategy = strategy

        # 如果还是没找到，返回默认右转策略
        if best_strategy is None:
            best_strategy = AvoidanceStrategy(np.deg2rad(30), 0.0)

        return best_strategy
