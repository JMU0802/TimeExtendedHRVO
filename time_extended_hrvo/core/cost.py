"""
cost.py - 代价函数

对应论文优化目标:
    J(θ) = w_ψ |Δψ| + w_u |Δu| + |v(θ) - v_pref|

转向方向约定（符合航海惯例）：
    - Δψ > 0: 右转（顺时针，Starboard）
    - Δψ < 0: 左转（逆时针，Port）

COLREGs 右转优先原则:
    除非陷入紧迫局面，船舶应始终选择向右转向避让
"""
import numpy as np


def strategy_cost(strategy, v_pref, w_heading=1.0, w_speed=1.0, w_deviation=1.0):
    """
    避让策略代价函数

    对应论文公式:
        J(θ) = w_ψ |Δψ| + w_u |Δu| + w_d |v(θ) - v_pref|

    Args:
        strategy: 避让策略 (AvoidanceStrategy)
        v_pref: 偏好速度向量（通常为原始目标速度）
        w_heading: 航向偏离权重
        w_speed: 速度偏离权重
        w_deviation: 最终速度偏离权重

    Returns:
        float: 策略代价
    """
    v_pref = np.asarray(v_pref, dtype=float)

    # 航向改变代价
    heading_cost = w_heading * abs(strategy.delta_psi)

    # 速度改变代价
    speed_cost = w_speed * abs(strategy.delta_speed)

    # 最终速度偏离代价
    v_final = strategy.get_final_velocity(v_pref)
    deviation_cost = w_deviation * np.linalg.norm(v_final - v_pref)

    return heading_cost + speed_cost + deviation_cost


def colregs_cost(strategy, encounter_type, w_starboard=0.3, w_port=5.0):
    """
    COLREGs 合规代价（强化右转优先）

    根据会遇类型对左/右转向施加不同代价:
        - 对遇/交叉相遇（让路）: 强烈鼓励右转
        - 追越: 优先右转，但允许左转

    转向约定: 正值=右转(Starboard)，负值=左转(Port)

    Args:
        strategy: 避让策略
        encounter_type: 会遇类型 ('head-on', 'crossing', 'overtaking')
        w_starboard: 右转代价权重（极低以强烈鼓励右转）
        w_port: 左转代价权重（极高以惩罚左转）

    Returns:
        float: COLREGs 合规代价
    """
    delta_psi = strategy.delta_psi

    if encounter_type in ['head-on', 'crossing', 'crossing-give-way']:
        # 对遇和交叉相遇：强烈鼓励右转（正航向角表示右转）
        if delta_psi > 0:  # 右转 - 极低代价
            return w_starboard * abs(delta_psi)
        else:  # 左转 - 极高代价（除非紧急情况）
            return w_port * abs(delta_psi) + 10.0  # 基础惩罚 + 左转固定惩罚
    elif encounter_type == 'overtaking':
        # 追越场景：优先右转，但允许左转
        if delta_psi > 0:  # 右转
            return 0.5 * abs(delta_psi)
        else:  # 左转
            return 1.5 * abs(delta_psi)
    else:
        # 其他场景：仍然优先右转
        if delta_psi > 0:
            return 0.5 * abs(delta_psi)
        else:
            return 2.0 * abs(delta_psi)


def efficiency_cost(strategy, v_orig, goal_direction):
    """
    航行效率代价

    衡量策略对到达目标效率的影响

    Args:
        strategy: 避让策略
        v_orig: 原始速度向量
        goal_direction: 目标方向单位向量

    Returns:
        float: 效率代价
    """
    v_orig = np.asarray(v_orig, dtype=float)
    goal_direction = np.asarray(goal_direction, dtype=float)

    # 归一化目标方向
    goal_norm = np.linalg.norm(goal_direction)
    if goal_norm > 1e-9:
        goal_direction = goal_direction / goal_norm

    # 计算最终速度在目标方向的投影
    v_final = strategy.get_final_velocity(v_orig)
    projection = np.dot(v_final, goal_direction)

    # 效率代价 = 速度损失
    original_projection = np.dot(v_orig, goal_direction)
    return max(0, original_projection - projection)


def combined_cost(strategy, v_pref, encounter_type=None, goal_direction=None,
                  weights=None):
    """
    综合代价函数

    组合多个代价项:
        J_total = λ_1 * J_strategy + λ_2 * J_colregs + λ_3 * J_efficiency

    Args:
        strategy: 避让策略
        v_pref: 偏好速度
        encounter_type: 会遇类型（可选）
        goal_direction: 目标方向（可选）
        weights: 代价权重字典

    Returns:
        float: 综合代价
    """
    if weights is None:
        weights = {
            'strategy': 1.0,
            'colregs': 1.0,
            'efficiency': 0.5,
            'heading': 1.0,
            'speed': 1.0,
            'deviation': 1.0
        }

    # 基本策略代价
    total = weights.get('strategy', 1.0) * strategy_cost(
        strategy, v_pref,
        w_heading=weights.get('heading', 1.0),
        w_speed=weights.get('speed', 1.0),
        w_deviation=weights.get('deviation', 1.0)
    )

    # COLREGs 代价
    if encounter_type is not None:
        total += weights.get('colregs', 1.0) * \
            colregs_cost(strategy, encounter_type)

    # 效率代价
    if goal_direction is not None:
        total += weights.get('efficiency', 0.5) * efficiency_cost(
            strategy, v_pref, goal_direction
        )

    return total


def dcpa_tcpa_cost(own_state, obs_state, strategy, w_dcpa=1.0, w_tcpa=0.1,
                   dcpa_safe=500.0):
    """
    基于 DCPA/TCPA 的代价函数

    DCPA (Distance at Closest Point of Approach): 最近会遇距离
    TCPA (Time to Closest Point of Approach): 最近会遇时间

    Args:
        own_state: 本船状态
        obs_state: 目标船状态
        strategy: 避让策略
        w_dcpa: DCPA 权重
        w_tcpa: TCPA 权重
        dcpa_safe: 安全 DCPA 阈值 (m)

    Returns:
        float: DCPA/TCPA 代价
    """
    # 计算策略执行后的速度
    v_new = strategy.get_final_velocity(own_state.v)

    # 相对位置（从本船指向目标船）
    p_rel = obs_state.p - own_state.p
    # 相对速度（目标船相对于本船的速度）
    v_rel = obs_state.v - v_new

    # 计算 TCPA
    v_rel_sq = np.dot(v_rel, v_rel)
    if v_rel_sq < 1e-9:
        return 0.0  # 相对静止

    tcpa = -np.dot(p_rel, v_rel) / v_rel_sq

    if tcpa < 0:
        return 0.0  # 已经远离

    # 计算 DCPA
    p_cpa = p_rel + v_rel * tcpa
    dcpa = np.linalg.norm(p_cpa)

    # 代价计算
    dcpa_cost = w_dcpa * max(0, dcpa_safe - dcpa) / dcpa_safe
    tcpa_cost = w_tcpa / (1 + tcpa / 60.0)  # 归一化到分钟

    return dcpa_cost + tcpa_cost


def marine_strategy_cost(strategy, v_pref, encounter_type=None, is_emergency=False):
    """
    船舶避碰专用代价函数（强化改向优先、右转优先）

    设计原则（符合实际航海实践和COLREGs）：
    1. 通常情况只采用单独改向避让
    2. 除非紧迫局面，始终选择右转
    3. 左转仅在右转不可行或紧急情况下使用
    4. 减速仅在紧迫局面/紧迫危险时才考虑（代价极高）
    5. 停船代价极高（几乎不应选择）

    转向约定: 正值=右转(Starboard)，负值=左转(Port)

    Args:
        strategy: 避让策略
        v_pref: 偏好速度（原始速度）
        encounter_type: 会遇类型
        is_emergency: 是否为紧急情况

    Returns:
        float: 策略代价
    """
    v_pref = np.asarray(v_pref, dtype=float)
    original_speed = np.linalg.norm(v_pref)

    delta_psi = strategy.delta_psi
    delta_u = strategy.delta_speed

    # ========== 改向代价（强化右转优先）==========
    # COLREGs 规定：除紧迫局面外，应始终向右转向避让

    LEFT_TURN_BASE_PENALTY = 15.0  # 左转基础惩罚

    if delta_psi > 0:  # 右转（正值）- 符合COLREGs
        # 右转代价极低
        heading_cost = 0.2 * abs(delta_psi)
    elif delta_psi < 0:  # 左转（负值）- 违反常规
        if is_emergency:
            # 紧急情况：允许左转，但仍有一定惩罚
            heading_cost = 1.0 * abs(delta_psi) + 5.0
        else:
            # 非紧急情况：左转受到严重惩罚
            if encounter_type in ['head-on', 'crossing', 'crossing-give-way']:
                # 对遇/交叉：左转是严重违规
                heading_cost = 3.0 * abs(delta_psi) + LEFT_TURN_BASE_PENALTY
            elif encounter_type == 'overtaking':
                # 追越：左转有较高代价但不是完全禁止
                heading_cost = 2.0 * abs(delta_psi) + 5.0
            else:
                # 其他情况：中等惩罚
                heading_cost = 2.0 * abs(delta_psi) + 8.0
    else:
        heading_cost = 0.0

    # ========== 速度改变代价（通常情况下极高惩罚）==========
    # 通常情况只采用单独改向，减速仅在紧迫局面时考虑

    SPEED_CHANGE_BASE_PENALTY = 50.0  # 减速基础惩罚（通常情况）
    EMERGENCY_SPEED_PENALTY = 10.0    # 紧急情况减速惩罚（较低）

    if delta_u < 0:  # 减速
        speed_reduction_ratio = abs(delta_u) / max(original_speed, 1.0)

        if is_emergency:
            # 紧急情况：允许减速，但仍有惩罚
            if speed_reduction_ratio > 0.5:
                # 大幅减速
                speed_cost = EMERGENCY_SPEED_PENALTY + 5.0 * abs(delta_u)
            else:
                # 适度减速
                speed_cost = EMERGENCY_SPEED_PENALTY * 0.5 + 3.0 * abs(delta_u)
        else:
            # 通常情况：减速惩罚极高，几乎不应选择
            speed_cost = SPEED_CHANGE_BASE_PENALTY + 20.0 * abs(delta_u)

            if speed_reduction_ratio > 0.5:
                # 大幅减速额外惩罚
                speed_cost += 30.0 * speed_reduction_ratio

        # 如果减速到接近停船，额外惩罚
        final_speed = max(0, original_speed + delta_u)
        if final_speed < 0.5:  # 接近停船
            if is_emergency:
                speed_cost += 20.0  # 紧急情况停船惩罚
            else:
                speed_cost += 100.0  # 通常情况停船惩罚极高

    elif delta_u > 0:  # 加速
        # 加速代价较低（追越时可能需要）
        speed_cost = 0.5 * abs(delta_u)
    else:
        speed_cost = 0.0

    # ========== 偏离代价 ==========
    # 最终速度与原始速度的偏离
    v_final = strategy.get_final_velocity(v_pref)
    deviation = np.linalg.norm(v_final - v_pref)
    deviation_cost = 0.3 * deviation

    # ========== 总代价 ==========
    total_cost = heading_cost + speed_cost + deviation_cost

    return total_cost


def marine_combined_cost(strategy, v_pref, encounter_type=None,
                         obstacles=None, own_state=None, is_emergency=False):
    """
    船舶避碰综合代价函数

    综合考虑：
    1. 策略代价（优先改向，强化右转）
    2. 安全性代价（DCPA/TCPA）
    3. COLREGs 合规性（右转优先）

    Args:
        strategy: 避让策略
        v_pref: 偏好速度
        encounter_type: 会遇类型
        obstacles: 目标船状态列表（可选）
        own_state: 本船状态（可选）
        is_emergency: 是否为紧急情况

    Returns:
        float: 综合代价
    """
    # 基本策略代价（强化右转优先）
    total = marine_strategy_cost(
        strategy, v_pref, encounter_type, is_emergency)

    # DCPA/TCPA 安全代价
    if obstacles and own_state:
        for obs_state in obstacles:
            total += 0.5 * dcpa_tcpa_cost(own_state, obs_state, strategy)

    return total


def is_emergency_situation(own_state, obstacles, dcpa_threshold=150.0, tcpa_threshold=30.0):
    """
    判断是否处于紧迫局面

    紧迫局面定义：
    - DCPA < 阈值 且 TCPA < 阈值 且 TCPA > 0
    - 或距离已经很近

    Args:
        own_state: 本船状态
        obstacles: 目标船状态列表
        dcpa_threshold: DCPA紧急阈值(m)
        tcpa_threshold: TCPA紧急阈值(s)

    Returns:
        bool: 是否为紧急情况
    """
    from time_extended_hrvo.utils.geometry import compute_dcpa_tcpa

    for obs in obstacles:
        # 计算当前距离
        distance = np.linalg.norm(own_state.p - obs.p)
        min_safe = (own_state.r + obs.r) * 2

        # 距离已经很近
        if distance < min_safe * 1.5:
            return True

        # 计算DCPA/TCPA
        dcpa, tcpa = compute_dcpa_tcpa(
            own_state.p, own_state.v,
            obs.p, obs.v
        )

        # 紧迫局面判断
        if dcpa < dcpa_threshold and 0 < tcpa < tcpa_threshold:
            return True

    return False
