"""
te_hrvo_planner.py - Time-Extended HRVO 主规划器

对应论文 Algorithm 1: Time-Extended HRVO Planning

COLREGs 右转优先原则:
    除非陷入紧迫局面，船舶应始终选择向右转向避让

核心改进（v2.0）：
    采用速度空间搜索方法替代离散策略采样
    1. 计算所有HRVO的并集
    2. 在速度空间的"白色区域"（可行区域）中密集搜索
    3. 选择代价最小且符合规则（右转优先）的解
"""
import numpy as np
import logging
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
from ..core.velocity_space import (
    adaptive_velocity_search, search_optimal_velocity,
    visualize_velocity_space, is_in_hrvo_union
)
from ..utils.geometry import classify_encounter, compute_dcpa_tcpa

# 配置日志
logger = logging.getLogger('TE_HRVO')


def enable_debug_logging(log_file=None):
    """
    启用详细日志输出

    Args:
        log_file: 日志文件路径（可选），如果不指定则输出到控制台
    """
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    if log_file:
        handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)

    # 清除已有的handler
    logger.handlers.clear()
    logger.addHandler(handler)

    logger.info("=== TE-HRVO 调试日志已启用 ===")


def disable_debug_logging():
    """禁用日志输出"""
    logger.setLevel(logging.WARNING)
    logger.handlers.clear()


class TimeExtendedHRVOPlanner:
    """
    时间扩展 HRVO 规划器

    核心算法流程 (Algorithm 1 v2.0 - 速度空间搜索):
        1. HRVO 构造: 为每个目标船构造 HRVO
        2. 速度空间搜索: 在HRVO并集外的可行区域中搜索最优速度
        3. 代价优化: 选择代价最小的可行速度（右转优先）
        4. 后备机制: 如果搜索失败，使用渐进式约束放宽

    相比原版改进:
        - 不再使用离散策略采样
        - 直接在连续速度空间中搜索可行区域
        - 能够找到位于离散采样点之间的可行解

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
        use_velocity_space_search (bool): 是否使用速度空间搜索（新方法）
    """

    def __init__(self, T_p=30.0, dt=2.0, tau=10.0, use_marine_cost=True,
                 starboard_first=True, avoidance_mode='heading_only',
                 use_velocity_space_search=False):
        """
        初始化规划器

        Args:
            T_p: 规划时域 (s), 默认 30s
            dt: 可行性检测时间步长 (s), 默认 2.0s（优化性能）
            tau: 机动响应时间常数 (s), 默认 10s
            use_marine_cost: 是否使用船舶专用代价函数（优先改向）
            starboard_first: 是否强制右转优先（除非紧急情况）
            avoidance_mode: 避让模式
                - 'heading_only': 仅航向避让（只改向）
                - 'speed_only': 仅航速避让（只改速）
                - 'combined': 组合避让（改向+改速）
            use_velocity_space_search: 是否使用速度空间搜索
                - False（默认）: 使用传统策略空间搜索（快速）
                - True: 使用速度空间搜索（更全面但较慢）
        """
        self.T_p = T_p
        self.dt = dt
        self.tau = tau
        self.use_marine_cost = use_marine_cost
        self.starboard_first = starboard_first
        self.avoidance_mode = avoidance_mode
        self.use_velocity_space_search = use_velocity_space_search

    def plan(self, own_state, obstacles, v_pref=None,
             encounter_type=None, weights=None):
        """
        执行 Time-Extended HRVO 规划（速度空间搜索版本）

        对应论文 Algorithm 1 v2.0

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

        # ============ 日志: 规划开始 ============
        logger.debug("=" * 60)
        logger.debug("【规划开始】")
        logger.debug(f"  本船位置: ({own_state.p[0]:.1f}, {own_state.p[1]:.1f})")
        logger.debug(
            f"  本船速度: ({own_state.v[0]:.2f}, {own_state.v[1]:.2f}), 速率={np.linalg.norm(own_state.v):.2f} m/s")
        heading_deg = np.rad2deg(np.arctan2(own_state.v[1], own_state.v[0]))
        logger.debug(f"  本船航向: {heading_deg:.1f}°")
        logger.debug(f"  目标船数量: {len(obstacles)}")
        logger.debug(f"  紧急情况: {is_emergency}")
        logger.debug(f"  避让模式: {self.avoidance_mode}")
        logger.debug(f"  使用速度空间搜索: {self.use_velocity_space_search}")

        # ============================================
        # Step 1: HRVO 构造
        # ============================================
        hrvo_list = []
        logger.debug("-" * 40)
        logger.debug("【Step 1: HRVO构造】")

        for i, obs in enumerate(obstacles):
            try:
                hrvo = compute_hrvo(own_state, obs)
                hrvo_list.append(hrvo)

                # 计算DCPA/TCPA
                dcpa, tcpa = compute_dcpa_tcpa(
                    own_state.p, own_state.v, obs.p, obs.v
                )
                dist = np.linalg.norm(obs.p - own_state.p)

                logger.debug(f"  目标船{i+1}:")
                logger.debug(f"    位置: ({obs.p[0]:.1f}, {obs.p[1]:.1f})")
                logger.debug(
                    f"    距离: {dist:.1f}m, DCPA: {dcpa:.1f}m, TCPA: {tcpa:.1f}s")
                logger.debug(
                    f"    HRVO apex: ({hrvo.apex[0]:.2f}, {hrvo.apex[1]:.2f})")

            except ValueError as e:
                logger.warning(f"  目标船{i+1}: 碰撞! {e}")
                continue

        if not hrvo_list:
            logger.debug("  无有效HRVO，返回保持策略")
            return AvoidanceStrategy(0, 0)

        # ============================================
        # Step 2 & 3: 速度空间搜索 或 传统策略空间搜索
        # ============================================
        if self.use_velocity_space_search:
            # 使用新的速度空间搜索方法
            return self._plan_with_velocity_space_search(
                own_state, obstacles, hrvo_list, v_pref,
                encounter_type, weights, is_emergency
            )
        else:
            # 使用传统的策略空间搜索方法
            return self._plan_with_strategy_space(
                own_state, obstacles, hrvo_list, v_pref,
                encounter_type, weights, is_emergency
            )

    def _plan_with_velocity_space_search(self, own_state, obstacles, hrvo_list,
                                         v_pref, encounter_type, weights, is_emergency):
        """
        使用速度空间搜索的规划方法

        核心改进：直接在速度空间中搜索可行区域，而不是离散策略采样
        """
        logger.debug("-" * 40)
        logger.debug("【Step 2: 速度空间搜索】")

        # 使用自适应速度空间搜索
        best_strategy = adaptive_velocity_search(
            own_state.v, hrvo_list, self.T_p, self.dt, self.tau,
            v_pref, is_emergency
        )

        if best_strategy is not None:
            logger.debug("-" * 40)
            logger.debug("【Step 3: 结果】")
            logger.debug(f"  最优策略: Δψ={np.rad2deg(best_strategy.delta_psi):+.1f}°, "
                         f"Δu={best_strategy.delta_speed:+.1f}m/s")
            logger.debug("=" * 60)
            return best_strategy

        # 速度空间搜索失败，使用后备策略
        logger.warning("  速度空间搜索失败，进入fallback...")
        return self._fallback_plan(own_state, obstacles, hrvo_list,
                                   v_pref, encounter_type, is_emergency)

    def _plan_with_strategy_space(self, own_state, obstacles, hrvo_list,
                                  v_pref, encounter_type, weights, is_emergency):
        """
        使用传统策略空间搜索的规划方法（备选方法）

        流程：
        1. 生成离散策略空间
        2. 对每个策略进行时间扩展可行性检测
        3. 在可行策略中选择代价最小的
        """
        logger.debug("-" * 40)
        logger.debug("【Step 2: 策略空间生成（传统方法）】")

        # 生成策略空间
        strategies = self._generate_strategies(encounter_type, is_emergency)
        logger.debug(f"  策略总数: {len(strategies)}")

        # 时间扩展可行性检测
        logger.debug("-" * 40)
        logger.debug("【Step 3: 时间扩展可行性检测】")
        logger.debug(f"  规划时域 T_p = {self.T_p}s, 检测步长 dt = {self.dt}s")

        feasible_strategies = []
        feasible_starboard = []
        feasible_port = []

        for strategy in strategies:
            if is_strategy_feasible(strategy, own_state.v,
                                    hrvo_list, self.T_p, self.dt):
                feasible_strategies.append(strategy)
                if strategy.delta_psi > 0.01:
                    feasible_starboard.append(strategy)
                elif strategy.delta_psi < -0.01:
                    feasible_port.append(strategy)
                else:
                    feasible_starboard.append(strategy)

        logger.debug(f"  可行策略总数: {len(feasible_strategies)}")
        logger.debug(f"  可行右转策略: {len(feasible_starboard)}")
        logger.debug(f"  可行左转策略: {len(feasible_port)}")

        if feasible_strategies:
            logger.debug("  可行策略列表 (前10个):")
            for s in feasible_strategies[:10]:
                margin = compute_feasibility_margin(
                    s, own_state.v, hrvo_list, self.T_p, self.dt)
                logger.debug(
                    f"    Δψ={np.rad2deg(s.delta_psi):+6.1f}°, Δu={s.delta_speed:+.1f}m/s, margin={margin:.2f}")
        else:
            logger.warning("  *** 无可行策略! 进入fallback ***")

        if not feasible_strategies:
            return self._fallback_plan(own_state, obstacles, hrvo_list,
                                       v_pref, encounter_type, is_emergency)

        # 代价优化
        logger.debug("-" * 40)
        logger.debug("【Step 4: 代价优化】")

        if self.starboard_first and not is_emergency and feasible_starboard:
            best_strategy = min(
                feasible_starboard,
                key=lambda s: self._compute_cost(s, own_state, obstacles,
                                                 v_pref, encounter_type, weights,
                                                 is_emergency)
            )
            logger.debug(f"  选择范围: 仅右转策略 ({len(feasible_starboard)}个)")
        else:
            best_strategy = min(
                feasible_strategies,
                key=lambda s: self._compute_cost(s, own_state, obstacles,
                                                 v_pref, encounter_type, weights,
                                                 is_emergency)
            )
            logger.debug(f"  选择范围: 所有可行策略 ({len(feasible_strategies)}个)")

        best_cost = self._compute_cost(best_strategy, own_state, obstacles,
                                       v_pref, encounter_type, weights, is_emergency)

        logger.debug(f"  最优策略: Δψ={np.rad2deg(best_strategy.delta_psi):+.1f}°, "
                     f"Δu={best_strategy.delta_speed:+.1f}m/s")
        logger.debug(f"  策略代价: {best_cost:.2f}")
        logger.debug("=" * 60)

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
        后备规划：当无可行策略时（增强版 - 渐进式约束放宽）

        多船场景下可能没有完全可行的解，采用以下策略：
        1. 尝试缩短规划时域（渐进式放宽）
        2. 选择"最小侵入"策略（改向优先）
        3. 确保总是返回一个策略

        优先级（始终遵循改向优先原则）：
        1. 大幅右转（纯改向）- 最优先
        2. 左转（纯改向）
        3. 右转 + 减速
        4. 全方位搜索（最小侵入，改向优先）

        转向约定: 正值=右转(Starboard)，负值=左转(Port)
        """
        logger.debug("-" * 40)
        logger.debug("【Fallback: 渐进式约束放宽】")

        # ============================================
        # 阶段1：尝试缩短规划时域（渐进式放宽约束）
        # ============================================
        shorter_horizons = [self.T_p * 0.5,
                            self.T_p * 0.3, self.T_p * 0.2, 5.0]

        logger.debug("  阶段1: 缩短规划时域搜索（纯改向）")
        for T_short in shorter_horizons:
            # 纯改向策略（右转优先）
            for angle in [30, 45, 60, 75, 90, 120, 150, 180]:
                strategy = AvoidanceStrategy(np.deg2rad(angle), 0.0)
                margin = compute_feasibility_margin(
                    strategy, own_state.v, hrvo_list, T_short, self.dt
                )
                if margin > 0:
                    logger.debug(
                        f"    找到可行策略! T={T_short:.1f}s, 右转{angle}°, margin={margin:.2f}")
                    return strategy

            # 左转策略
            for angle in [-30, -45, -60, -90, -120]:
                strategy = AvoidanceStrategy(np.deg2rad(angle), 0.0)
                margin = compute_feasibility_margin(
                    strategy, own_state.v, hrvo_list, T_short, self.dt
                )
                if margin > 0:
                    logger.debug(
                        f"    找到可行策略! T={T_short:.1f}s, 左转{-angle}°, margin={margin:.2f}")
                    return strategy

        logger.debug("    阶段1未找到可行策略")

        # ============================================
        # 阶段2：改向+减速组合（仍然改向优先）
        # ============================================
        logger.debug("  阶段2: 改向+减速组合搜索")
        for angle in [45, 60, 90, 120, 150]:
            for du in [-0.5, -1.0, -1.5, -2.0, -2.5]:
                strategy = AvoidanceStrategy(np.deg2rad(angle), du)
                margin = compute_feasibility_margin(
                    strategy, own_state.v, hrvo_list, 10.0, self.dt
                )
                if margin > 0:
                    logger.debug(
                        f"    找到可行策略! 右转{angle}°+减速{du}m/s, margin={margin:.2f}")
                    return strategy

        for angle in [-45, -60, -90, -120]:
            for du in [-0.5, -1.0, -1.5, -2.0]:
                strategy = AvoidanceStrategy(np.deg2rad(angle), du)
                margin = compute_feasibility_margin(
                    strategy, own_state.v, hrvo_list, 10.0, self.dt
                )
                if margin > 0:
                    logger.debug(
                        f"    找到可行策略! 左转{-angle}°+减速{du}m/s, margin={margin:.2f}")
                    return strategy

        logger.debug("    阶段2未找到可行策略")

        # ============================================
        # 阶段3：全方位搜索（最小侵入策略，改向优先）
        # ============================================
        logger.debug("  阶段3: 全方位搜索（最小侵入策略）")
        best_heading_strategy = None
        best_heading_margin = float('-inf')

        best_combined_strategy = None
        best_combined_margin = float('-inf')

        # 3.1 纯改向策略搜索（最高优先级）
        for angle in range(5, 181, 5):  # 右转 5-180度
            strategy = AvoidanceStrategy(np.deg2rad(angle), 0.0)
            margin = compute_feasibility_margin(
                strategy, own_state.v, hrvo_list, 5.0, self.dt  # 短时域
            )
            # 给右转较小角度加分
            adjusted_margin = margin + 0.5 * (1 - angle / 180)
            if adjusted_margin > best_heading_margin:
                best_heading_margin = adjusted_margin
                best_heading_strategy = strategy

        for angle in range(-5, -181, -5):  # 左转 -5到-180度
            strategy = AvoidanceStrategy(np.deg2rad(angle), 0.0)
            margin = compute_feasibility_margin(
                strategy, own_state.v, hrvo_list, 5.0, self.dt
            )
            # 左转惩罚
            adjusted_margin = margin - 1.0
            if adjusted_margin > best_heading_margin:
                best_heading_margin = adjusted_margin
                best_heading_strategy = strategy

        # 3.2 组合策略搜索（次优先级）
        for angle in range(0, 360, 15):
            for du in [0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]:
                strategy = AvoidanceStrategy(np.deg2rad(angle), du)
                margin = compute_feasibility_margin(
                    strategy, own_state.v, hrvo_list, 5.0, self.dt
                )

                # 评分调整：改向优先，减速惩罚
                adjusted_margin = margin

                # 给右转加分（0-180度）
                if 0 < angle <= 180:
                    adjusted_margin += 1.0

                # 给纯改向（不减速）加分
                if abs(du) < 0.1:
                    adjusted_margin += 2.0
                else:
                    # 减速惩罚
                    adjusted_margin -= abs(du) * 0.5

                if adjusted_margin > best_combined_margin:
                    best_combined_margin = adjusted_margin
                    best_combined_strategy = strategy

        # ============================================
        # 阶段4：选择最佳策略（改向优先）
        # ============================================
        logger.debug("  阶段4: 选择最佳策略")
        if best_heading_strategy:
            logger.debug(
                f"    纯改向最佳: Δψ={np.rad2deg(best_heading_strategy.delta_psi):+.1f}°, margin={best_heading_margin:.2f}")
        if best_combined_strategy:
            logger.debug(f"    组合最佳: Δψ={np.rad2deg(best_combined_strategy.delta_psi):+.1f}°, "
                         f"Δu={best_combined_strategy.delta_speed:+.1f}m/s, margin={best_combined_margin:.2f}")

        # 如果纯改向策略的margin足够好，优先返回
        if best_heading_strategy and best_heading_margin > -5.0:
            logger.debug(f"    选择: 纯改向策略 (margin > -5.0)")
            logger.debug("=" * 60)
            return best_heading_strategy

        # 否则返回组合策略
        if best_combined_strategy:
            logger.debug(f"    选择: 组合策略")
            logger.debug("=" * 60)
            return best_combined_strategy

        if best_heading_strategy:
            logger.debug(f"    选择: 纯改向策略 (fallback)")
            logger.debug("=" * 60)
            return best_heading_strategy

        # ============================================
        # 阶段5：速度空间搜索（最后尝试）
        # ============================================
        logger.debug("  阶段5: 尝试速度空间搜索...")
        try:
            velocity_strategy = adaptive_velocity_search(
                own_state.v, hrvo_list, self.T_p, self.dt, self.tau,
                v_pref, True  # 紧急模式
            )
            if velocity_strategy is not None:
                logger.debug(f"    速度空间搜索成功!")
                logger.debug("=" * 60)
                return velocity_strategy
        except Exception as e:
            logger.warning(f"    速度空间搜索异常: {e}")

        # ============================================
        # 阶段6：最后保障 - 返回默认右转策略
        # ============================================
        logger.warning("  阶段6: 所有搜索失败，返回默认90度右转!")
        logger.debug("=" * 60)
        return AvoidanceStrategy(np.deg2rad(90), 0.0)

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
