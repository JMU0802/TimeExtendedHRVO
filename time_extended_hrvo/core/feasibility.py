"""
feasibility.py - 时间一致性可行性判定

对应论文核心创新:
    时间扩展 HRVO 约束检测
    ∀t ∈ [t_start, T_p], v_o(t; θ) ∉ HRVO_i

注意：当前速度可能已经在HRVO内（碰撞航线），因此检测从响应50%时刻开始，
允许轨迹初期穿过HRVO，只要最终能够离开即可。
"""
import numpy as np


def is_strategy_feasible(strategy, v0, hrvo_list, T_p, dt=2.0, tau=10.0):
    """
    时间扩展 HRVO 可行性检测（宽松模式）

    核心创新: 检验策略在规划时域后半段的时间一致性
    从响应达到50%的时刻开始检测，允许初期穿过HRVO

    Args:
        strategy: 避让策略 (AvoidanceStrategy)
        v0: 初始速度向量
        hrvo_list: HRVO 列表
        T_p: 规划时域 (s)
        dt: 时间步长 (s), 默认 2.0s
        tau: 机动响应时间常数 (s), 默认 10.0s

    Returns:
        bool: True 如果策略在检测时域内可行
    """
    v0 = np.asarray(v0, dtype=float)

    # 首先检查最终速度是否可行（快速排除）
    v_final = strategy.get_final_velocity(v0, tau)
    for hrvo in hrvo_list:
        if hrvo.contains(v_final):
            return False

    # 从响应50%时刻开始检测（跳过初期）
    t_start = max(dt, 0.693 * tau)
    t = t_start

    while t <= T_p:
        v_t = strategy.velocity_profile(v0, t, tau)

        for hrvo in hrvo_list:
            if hrvo.contains(v_t):
                return False

        t += dt

    return True


def compute_feasibility_margin(strategy, v0, hrvo_list, T_p, dt=2.0, tau=10.0):
    """
    计算策略的可行性裕度（宽松模式）

    返回速度轨迹到 HRVO 边界的最小距离，
    正值表示可行，负值表示违反约束

    Args:
        strategy: 避让策略 (AvoidanceStrategy)
        v0: 初始速度向量
        hrvo_list: HRVO 列表
        T_p: 规划时域 (s)
        dt: 时间步长 (s)
        tau: 机动响应时间常数 (s)

    Returns:
        float: 可行性裕度
    """
    v0 = np.asarray(v0, dtype=float)

    min_margin = float('inf')

    # 从响应50%时刻开始检测
    t_start = max(dt, 0.693 * tau)
    t = t_start

    while t <= T_p:
        v_t = strategy.velocity_profile(v0, t, tau)

        for hrvo in hrvo_list:
            margin = hrvo.distance_to_boundary(v_t)
            min_margin = min(min_margin, margin)

        t += dt

    # 也检查最终速度
    v_final = strategy.get_final_velocity(v0, tau)
    for hrvo in hrvo_list:
        margin = hrvo.distance_to_boundary(v_final)
        min_margin = min(min_margin, margin)

    return min_margin


def find_violation_time(strategy, v0, hrvo_list, T_p, dt=0.5):
    """
    找到首次违反 HRVO 约束的时刻

    Args:
        strategy: 避让策略 (AvoidanceStrategy)
        v0: 初始速度向量
        hrvo_list: HRVO 列表
        T_p: 规划时域 (s)
        dt: 时间步长 (s)

    Returns:
        tuple: (violation_time, hrvo_index) 或 (None, None) 如果无违反
    """
    v0 = np.asarray(v0, dtype=float)

    t = 0.0
    while t <= T_p:
        v_t = strategy.velocity_profile(v0, t)

        for i, hrvo in enumerate(hrvo_list):
            if hrvo.contains(v_t):
                return t, i

        t += dt

    return None, None


def check_time_consistency(strategy, v0, hrvo_sequence, T_p, dt=0.5):
    """
    检查时变 HRVO 序列的时间一致性

    对于动态变化的 HRVO（考虑目标船运动），
    检验策略是否在每个时刻都避开对应的 HRVO

    Args:
        strategy: 避让策略 (AvoidanceStrategy)
        v0: 初始速度向量
        hrvo_sequence: 时变 HRVO 序列，callable(t) -> list[HRVO]
        T_p: 规划时域 (s)
        dt: 时间步长 (s)

    Returns:
        bool: True 如果满足时间一致性
    """
    v0 = np.asarray(v0, dtype=float)

    t = 0.0
    while t <= T_p:
        v_t = strategy.velocity_profile(v0, t)
        hrvo_list_t = hrvo_sequence(t)

        for hrvo in hrvo_list_t:
            if hrvo.contains(v_t):
                return False

        t += dt

    return True
