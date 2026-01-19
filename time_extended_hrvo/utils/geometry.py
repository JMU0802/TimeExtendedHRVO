"""
geometry.py - 几何工具函数

提供向量运算、角度转换等基础几何操作
"""
import numpy as np


def normalize(v):
    """
    向量归一化

    Args:
        v: 输入向量

    Returns:
        np.ndarray: 单位向量，零向量返回零向量
    """
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return np.zeros_like(v)
    return v / norm


def angle_between(v1, v2):
    """
    计算两向量之间的夹角

    Args:
        v1, v2: 输入向量

    Returns:
        float: 夹角 (rad), 范围 [0, π]
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)


def signed_angle(v1, v2):
    """
    计算从 v1 到 v2 的有符号角度

    正值表示逆时针，负值表示顺时针

    Args:
        v1, v2: 输入向量

    Returns:
        float: 有符号角度 (rad), 范围 [-π, π]
    """
    angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
    # 归一化到 [-π, π]
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def rotate_vector(v, angle):
    """
    旋转向量

    Args:
        v: 输入向量
        angle: 旋转角度 (rad), 正值为逆时针

    Returns:
        np.ndarray: 旋转后的向量
    """
    v = np.asarray(v, dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return rotation_matrix @ v


def perpendicular(v, clockwise=False):
    """
    获取垂直向量

    Args:
        v: 输入向量
        clockwise: True 为顺时针方向，False 为逆时针方向

    Returns:
        np.ndarray: 垂直向量
    """
    v = np.asarray(v, dtype=float)
    if clockwise:
        return np.array([v[1], -v[0]])
    else:
        return np.array([-v[1], v[0]])


def point_to_line_distance(point, line_point, line_dir):
    """
    点到直线的距离

    Args:
        point: 点坐标
        line_point: 直线上的一点
        line_dir: 直线方向向量

    Returns:
        float: 距离（有符号，正值表示在方向向量左侧）
    """
    point = np.asarray(point, dtype=float)
    line_point = np.asarray(line_point, dtype=float)
    line_dir = normalize(line_dir)

    diff = point - line_point
    return np.cross(line_dir, diff)


def line_intersection(p1, d1, p2, d2):
    """
    计算两条直线的交点

    Args:
        p1, d1: 第一条直线的点和方向
        p2, d2: 第二条直线的点和方向

    Returns:
        np.ndarray: 交点坐标，或 None 如果平行
    """
    p1 = np.asarray(p1, dtype=float)
    d1 = np.asarray(d1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d2 = np.asarray(d2, dtype=float)

    cross = np.cross(d1, d2)
    if abs(cross) < 1e-9:
        return None  # 平行

    t = np.cross(p2 - p1, d2) / cross
    return p1 + t * d1


def compute_dcpa_tcpa(own_pos, own_vel, obs_pos, obs_vel):
    """
    计算 DCPA 和 TCPA

    DCPA: Distance at Closest Point of Approach
    TCPA: Time to Closest Point of Approach

    Args:
        own_pos: 本船位置
        own_vel: 本船速度
        obs_pos: 目标船位置
        obs_vel: 目标船速度

    Returns:
        tuple: (dcpa, tcpa)
    """
    own_pos = np.asarray(own_pos, dtype=float)
    own_vel = np.asarray(own_vel, dtype=float)
    obs_pos = np.asarray(obs_pos, dtype=float)
    obs_vel = np.asarray(obs_vel, dtype=float)

    # 相对位置（从本船指向目标船）
    p_rel = obs_pos - own_pos
    # 相对速度（目标船相对于本船的速度）
    v_rel = obs_vel - own_vel

    # 计算 TCPA
    v_rel_sq = np.dot(v_rel, v_rel)
    if v_rel_sq < 1e-9:
        # 相对静止
        return float(np.linalg.norm(p_rel)), 0.0

    # TCPA = -dot(p_rel, v_rel) / |v_rel|^2
    tcpa = -np.dot(p_rel, v_rel) / v_rel_sq

    if tcpa < 0:
        # 已经远离，DCPA 为当前距离
        return float(np.linalg.norm(p_rel)), 0.0

    # 计算 DCPA：CPA 时刻的相对位置
    p_cpa = p_rel + v_rel * tcpa
    dcpa = float(np.linalg.norm(p_cpa))

    return dcpa, float(tcpa)


def classify_encounter(own_state, obs_state, head_on_threshold=15.0,
                       crossing_threshold=112.5, overtaking_threshold=22.5):
    """
    根据 COLREGs 分类会遇类型

    Args:
        own_state: 本船状态
        obs_state: 目标船状态
        head_on_threshold: 对遇判定阈值 (deg)
        crossing_threshold: 交叉相遇阈值 (deg)
        overtaking_threshold: 追越阈值 (deg)

    Returns:
        str: 'head-on', 'crossing-give-way', 'crossing-stand-on', 'overtaking', 'being-overtaken'
    """
    # 相对位置
    p_rel = obs_state.p - own_state.p
    bearing = np.rad2deg(np.arctan2(p_rel[1], p_rel[0]))

    # 本船航向
    own_heading = np.rad2deg(own_state.heading)

    # 目标船航向
    obs_heading = np.rad2deg(obs_state.heading)

    # 相对舷角
    relative_bearing = bearing - own_heading
    while relative_bearing > 180:
        relative_bearing -= 360
    while relative_bearing < -180:
        relative_bearing += 360

    # 航向差
    heading_diff = obs_heading - own_heading
    while heading_diff > 180:
        heading_diff -= 360
    while heading_diff < -180:
        heading_diff += 360

    # 判断会遇类型
    if abs(heading_diff) > 180 - head_on_threshold:
        return 'head-on'
    elif abs(relative_bearing) > 180 - overtaking_threshold:
        return 'being-overtaken'
    elif abs(relative_bearing) < overtaking_threshold:
        return 'overtaking'
    elif relative_bearing > 0:
        return 'crossing-give-way'  # 目标船在右舷
    else:
        return 'crossing-stand-on'  # 目标船在左舷
