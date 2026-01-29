"""
velocity_space.py - 速度空间搜索算法

核心创新：
    不再使用离散策略采样，而是直接在速度空间中搜索可行区域
    1. 计算所有HRVO的并集
    2. 在速度空间的"白色区域"（可行区域）中搜索
    3. 选择代价最小且符合规则（右转优先）的解

转向方向约定（符合航海惯例）：
    - 右转（顺时针）: 航向角减小，delta_psi > 0
    - 左转（逆时针）: 航向角增大，delta_psi < 0
"""
import numpy as np
import logging
from .strategy import AvoidanceStrategy

logger = logging.getLogger('TE_HRVO')


def is_in_hrvo_union(v, hrvo_list):
    """
    检测速度点是否在HRVO并集内

    Args:
        v: 速度向量 [vx, vy]
        hrvo_list: HRVO列表

    Returns:
        bool: True 如果速度在任何一个HRVO内
    """
    for hrvo in hrvo_list:
        if hrvo.contains(v):
            return True
    return False


def is_velocity_feasible_time_extended(v_target, v0, hrvo_list, T_p, dt=2.0, tau=10.0,
                                       strict_mode=False):
    """
    时间扩展可行性检测（针对目标速度）- 优化版本

    检查从当前速度v0过渡到目标速度v_target的轨迹是否可行

    Args:
        v_target: 目标速度向量
        v0: 当前速度向量
        hrvo_list: HRVO列表
        T_p: 规划时域 (s)
        dt: 时间步长 (s)，默认2.0减少计算量
        tau: 机动响应时间常数 (s)
        strict_mode: 是否使用严格模式

    Returns:
        bool: True 如果轨迹可行
    """
    v0 = np.asarray(v0, dtype=float)
    v_target = np.asarray(v_target, dtype=float)

    # 首先检查最终目标速度是否可行（快速排除）
    if is_in_hrvo_union(v_target, hrvo_list):
        return False

    if strict_mode:
        t = dt
    else:
        # 宽松模式：从响应达到50%的时刻开始检测
        t_start = max(dt, 0.693 * tau)
        t = t_start

    while t <= T_p:
        response = 1.0 - np.exp(-t / tau)
        v_t = v0 + (v_target - v0) * response

        if is_in_hrvo_union(v_t, hrvo_list):
            return False

        t += dt

    return True


def sample_velocity_space(v0, max_speed=None, speed_resolution=10, angle_resolution=36):
    """
    在速度空间中进行采样（优化版本，减少计算量）

    使用极坐标采样，覆盖以原点为中心的整个速度空间

    Args:
        v0: 当前速度向量（用于确定采样范围）
        max_speed: 最大速度（默认为当前速度的1.3倍或+3m/s）
        speed_resolution: 速度分辨率（速度方向的采样点数），默认10
        angle_resolution: 角度分辨率（角度方向的采样点数），默认36（每10度）

    Returns:
        np.ndarray: 采样速度点数组，形状为 (N, 2)
    """
    v0 = np.asarray(v0, dtype=float)
    current_speed = np.linalg.norm(v0)

    if max_speed is None:
        # 采样范围适中
        max_speed = max(current_speed * 1.3, current_speed + 3.0, 6.0)

    samples = []

    # 采样不同速度大小
    min_speed = max(0.5, current_speed * 0.5)
    speed_values = np.linspace(min_speed, max_speed, speed_resolution)

    # 采样不同角度（360度全覆盖，每10度一个点）
    angle_values = np.linspace(0, 2 * np.pi, angle_resolution, endpoint=False)

    for speed in speed_values:
        for angle in angle_values:
            vx = speed * np.cos(angle)
            vy = speed * np.sin(angle)
            samples.append([vx, vy])

    return np.array(samples)


def find_feasible_velocities(v0, hrvo_list, T_p, dt=1.0, tau=10.0,
                             max_speed=None, speed_resolution=10, angle_resolution=36):
    """
    在速度空间中找到所有可行速度点

    Args:
        v0: 当前速度向量
        hrvo_list: HRVO列表
        T_p: 规划时域 (s)
        dt: 时间步长 (s)
        tau: 机动响应时间常数 (s)
        max_speed: 最大速度
        speed_resolution: 速度分辨率
        angle_resolution: 角度分辨率

    Returns:
        list: 可行速度点列表，每个元素为 (vx, vy)
    """
    # 采样速度空间
    samples = sample_velocity_space(
        v0, max_speed, speed_resolution, angle_resolution)

    feasible = []
    for v_sample in samples:
        if is_velocity_feasible_time_extended(v_sample, v0, hrvo_list, T_p, dt, tau):
            feasible.append(v_sample)

    return feasible


def velocity_to_strategy(v_target, v0):
    """
    将目标速度转换为避让策略

    Args:
        v_target: 目标速度向量
        v0: 当前速度向量

    Returns:
        AvoidanceStrategy: 对应的避让策略
    """
    v0 = np.asarray(v0, dtype=float)
    v_target = np.asarray(v_target, dtype=float)

    # 当前速度和航向
    speed_0 = np.linalg.norm(v0)
    heading_0 = np.arctan2(v0[1], v0[0])

    # 目标速度和航向
    speed_target = np.linalg.norm(v_target)
    heading_target = np.arctan2(v_target[1], v_target[0])

    # 计算速度变化量
    delta_speed = speed_target - speed_0

    # 计算航向变化量
    # delta_psi = heading_0 - heading_target（右转为正）
    delta_heading = heading_0 - heading_target

    # 归一化到 [-pi, pi]
    while delta_heading > np.pi:
        delta_heading -= 2 * np.pi
    while delta_heading < -np.pi:
        delta_heading += 2 * np.pi

    return AvoidanceStrategy(delta_heading, delta_speed)


def compute_velocity_cost(v_target, v0, v_pref=None, is_emergency=False):
    """
    计算目标速度的代价（右转优先）

    Args:
        v_target: 目标速度向量
        v0: 当前速度向量
        v_pref: 偏好速度（默认为当前速度）
        is_emergency: 是否为紧急情况

    Returns:
        float: 代价值
    """
    if v_pref is None:
        v_pref = v0

    v0 = np.asarray(v0, dtype=float)
    v_target = np.asarray(v_target, dtype=float)
    v_pref = np.asarray(v_pref, dtype=float)

    # 当前速度和航向
    speed_0 = np.linalg.norm(v0)
    heading_0 = np.arctan2(v0[1], v0[0])

    # 目标速度和航向
    speed_target = np.linalg.norm(v_target)
    heading_target = np.arctan2(v_target[1], v_target[0])

    # 计算航向变化量（右转为正）
    delta_heading = heading_0 - heading_target
    while delta_heading > np.pi:
        delta_heading -= 2 * np.pi
    while delta_heading < -np.pi:
        delta_heading += 2 * np.pi

    # 计算速度变化量
    delta_speed = speed_target - speed_0

    cost = 0.0

    # === 航向代价（右转优先）===
    if delta_heading > 0.01:  # 右转
        # 右转代价很低
        cost += 0.2 * abs(delta_heading)
    elif delta_heading < -0.01:  # 左转
        # 左转高惩罚
        cost += 3.0 * abs(delta_heading) + 15.0
    # 直行无航向代价

    # === 速度代价（改向优先于减速）===
    if abs(delta_speed) > 0.1:
        if is_emergency:
            # 紧急情况下减速代价降低
            cost += 10.0 + 3.0 * abs(delta_speed)
        else:
            # 通常情况减速高代价
            cost += 50.0 + 20.0 * abs(delta_speed)

    # === 与偏好速度的偏离代价 ===
    cost += 0.1 * np.linalg.norm(v_target - v_pref)

    return cost


def search_optimal_velocity(v0, hrvo_list, T_p, dt=1.0, tau=10.0,
                            v_pref=None, is_emergency=False,
                            max_speed=None, speed_resolution=10, angle_resolution=36):
    """
    在可行速度空间中搜索最优速度（优化版本）

    核心算法流程：
    1. 在速度空间中采样（约360点）
    2. 筛选出所有可行速度（时间扩展检测）
    3. 对可行速度计算代价
    4. 返回代价最小的速度对应的策略

    Args:
        v0: 当前速度向量
        hrvo_list: HRVO列表
        T_p: 规划时域 (s)
        dt: 时间步长 (s)，默认1.0（减少计算量）
        tau: 机动响应时间常数 (s)
        v_pref: 偏好速度
        is_emergency: 是否为紧急情况
        max_speed: 最大速度
        speed_resolution: 速度分辨率，默认10
        angle_resolution: 角度分辨率，默认36

    Returns:
        tuple: (best_strategy, feasible_count, all_count)
    """
    v0 = np.asarray(v0, dtype=float)
    if v_pref is None:
        v_pref = v0.copy()

    # Step 1: 速度空间采样
    samples = sample_velocity_space(
        v0, max_speed, speed_resolution, angle_resolution)
    all_count = len(samples)

    logger.debug(f"  速度空间搜索: 采样点数={all_count}")

    # Step 2: 找出所有可行速度
    feasible_velocities = []
    feasible_starboard = []  # 右转可行速度
    feasible_port = []       # 左转可行速度

    current_heading = np.arctan2(v0[1], v0[0])

    for v_sample in samples:
        if is_velocity_feasible_time_extended(v_sample, v0, hrvo_list, T_p, dt, tau):
            feasible_velocities.append(v_sample)

            # 判断转向方向
            target_heading = np.arctan2(v_sample[1], v_sample[0])
            delta_heading = current_heading - target_heading
            while delta_heading > np.pi:
                delta_heading -= 2 * np.pi
            while delta_heading < -np.pi:
                delta_heading += 2 * np.pi

            if delta_heading > 0.01:
                feasible_starboard.append(v_sample)
            elif delta_heading < -0.01:
                feasible_port.append(v_sample)
            else:
                feasible_starboard.append(v_sample)  # 直行归入右转

    feasible_count = len(feasible_velocities)
    logger.debug(
        f"  可行速度点: {feasible_count} (右转={len(feasible_starboard)}, 左转={len(feasible_port)})")

    if feasible_count == 0:
        logger.warning("  速度空间搜索: 无可行速度!")
        return None, 0, all_count

    # Step 3: 在可行速度中搜索最优（右转优先）
    best_velocity = None
    best_cost = float('inf')

    # 非紧急情况：只从右转中选择
    search_set = feasible_starboard if (not is_emergency and len(
        feasible_starboard) > 0) else feasible_velocities

    for v_sample in search_set:
        cost = compute_velocity_cost(v_sample, v0, v_pref, is_emergency)
        if cost < best_cost:
            best_cost = cost
            best_velocity = v_sample

    if best_velocity is None:
        logger.warning("  速度空间搜索: 无法找到最优速度!")
        return None, feasible_count, all_count

    # Step 4: 转换为策略
    best_strategy = velocity_to_strategy(best_velocity, v0)

    logger.debug(f"  最优速度: ({best_velocity[0]:.2f}, {best_velocity[1]:.2f})")
    logger.debug(
        f"  对应策略: Δψ={np.rad2deg(best_strategy.delta_psi):+.1f}°, Δu={best_strategy.delta_speed:+.1f}m/s")
    logger.debug(f"  策略代价: {best_cost:.2f}")

    return best_strategy, feasible_count, all_count


def adaptive_velocity_search(v0, hrvo_list, T_p, dt=1.0, tau=10.0,
                             v_pref=None, is_emergency=False):
    """
    自适应速度空间搜索（优化版本，减少计算量）

    根据场景复杂度自动调整搜索分辨率

    Args:
        v0: 当前速度向量
        hrvo_list: HRVO列表
        T_p: 规划时域 (s)
        dt: 时间步长 (s)，默认1.0
        tau: 机动响应时间常数 (s)
        v_pref: 偏好速度
        is_emergency: 是否为紧急情况

    Returns:
        AvoidanceStrategy: 最优避让策略，或 None 如果无解
    """
    v0 = np.asarray(v0, dtype=float)
    num_obstacles = len(hrvo_list)

    logger.debug("-" * 40)
    logger.debug("【自适应速度空间搜索】")
    logger.debug(f"  目标船数量: {num_obstacles}")

    # 统一使用适中的分辨率（约360-720点）
    speed_res, angle_res = 10, 36
    if num_obstacles > 3:
        speed_res, angle_res = 15, 48  # 复杂场景稍微增加

    logger.debug(f"  搜索分辨率: speed={speed_res}, angle={angle_res}")

    # 第一次搜索
    strategy, feasible_count, all_count = search_optimal_velocity(
        v0, hrvo_list, T_p, dt, tau, v_pref, is_emergency,
        speed_resolution=speed_res, angle_resolution=angle_res
    )

    if strategy is not None:
        return strategy

    # 如果没找到可行解，尝试缩短规划时域（只重试一次）
    logger.debug("  第一次搜索失败，尝试缩短规划时域...")
    T_short = max(T_p * 0.5, 10.0)
    strategy, feasible_count, all_count = search_optimal_velocity(
        v0, hrvo_list, T_short, dt, tau, v_pref, True,
        speed_resolution=15, angle_resolution=48
    )

    if strategy is not None:
        logger.debug(f"  在T={T_short:.1f}s下找到可行解")
        return strategy

    logger.warning("  自适应搜索失败，返回None")
    return None


def visualize_velocity_space(v0, hrvo_list, T_p=30.0, dt=1.0, tau=10.0,
                             speed_resolution=15, angle_resolution=48):
    """
    可视化速度空间的可行区域

    用于调试和理解HRVO并集的形状

    Args:
        v0: 当前速度向量
        hrvo_list: HRVO列表
        T_p: 规划时域 (s)
        dt: 时间步长 (s)
        tau: 机动响应时间常数 (s)
        speed_resolution: 速度分辨率
        angle_resolution: 角度分辨率

    Returns:
        dict: 包含可视化数据的字典
    """
    v0 = np.asarray(v0, dtype=float)

    samples = sample_velocity_space(v0, speed_resolution=speed_resolution,
                                    angle_resolution=angle_resolution)

    feasible = []
    infeasible = []

    for v_sample in samples:
        if is_velocity_feasible_time_extended(v_sample, v0, hrvo_list, T_p, dt, tau):
            feasible.append(v_sample)
        else:
            infeasible.append(v_sample)

    return {
        'samples': samples,
        'feasible': np.array(feasible) if feasible else np.array([]).reshape(0, 2),
        'infeasible': np.array(infeasible) if infeasible else np.array([]).reshape(0, 2),
        'v0': v0,
        'hrvo_list': hrvo_list,
        'feasible_ratio': len(feasible) / len(samples) if len(samples) > 0 else 0
    }
