"""
hrvo.py - 混合互惠速度障碍 (Hybrid Reciprocal Velocity Obstacle) 几何构造

对应论文:
    HRVO 定义与构造方法
    apex: 速度空间顶点
    left_boundary, right_boundary: 边界方向向量
"""
import numpy as np


class HRVO:
    """
    混合互惠速度障碍类

    HRVO 在速度空间中表示为一个以 apex 为顶点的锥形区域，
    由左右边界向量定义。

    Attributes:
        apex (np.ndarray): 速度空间顶点
        left (np.ndarray): 左边界单位方向向量
        right (np.ndarray): 右边界单位方向向量
    """

    def __init__(self, apex, left_boundary, right_boundary):
        """
        初始化 HRVO

        Args:
            apex: 速度空间顶点
            left_boundary: 左边界方向向量
            right_boundary: 右边界方向向量
        """
        self.apex = np.asarray(apex, dtype=float)
        self.left = np.asarray(left_boundary, dtype=float)
        self.right = np.asarray(right_boundary, dtype=float)

        # 归一化边界向量
        left_norm = np.linalg.norm(self.left)
        right_norm = np.linalg.norm(self.right)

        if left_norm > 1e-9:
            self.left = self.left / left_norm
        if right_norm > 1e-9:
            self.right = self.right / right_norm

    def contains(self, v):
        """
        判断速度 v 是否落入 HRVO 区域

        HRVO 是一个锥形区域，由 apex、left 边界和 right 边界定义。
        点在锥内当且仅当它在 left 边界的顺时针侧（右侧）
        且在 right 边界的逆时针侧（左侧）。

        Args:
            v: 待检测的速度向量

        Returns:
            bool: True 如果速度在 HRVO 内
        """
        v = np.asarray(v, dtype=float)
        rel = v - self.apex

        # cross(a, b) > 0: b 在 a 的逆时针方向（左侧）
        # cross(a, b) < 0: b 在 a 的顺时针方向（右侧）
        cross_left = np.cross(self.left, rel)
        cross_right = np.cross(self.right, rel)

        # 在锥内：在 left 的右侧（顺时针）且在 right 的左侧（逆时针）
        return bool(cross_left <= 0 and cross_right >= 0)

    def distance_to_boundary(self, v):
        """
        计算速度 v 到 HRVO 边界的距离

        Args:
            v: 速度向量

        Returns:
            float: 到边界的距离（负值表示在内部）
        """
        v = np.asarray(v, dtype=float)
        rel = v - self.apex

        # 计算到左右边界的有符号距离
        # 正值表示在边界外侧，负值表示在边界内侧
        dist_left = float(np.cross(self.left, rel))
        dist_right = float(np.cross(self.right, rel))

        # 返回最小距离（最接近违规的边界）
        return min(dist_left, -dist_right)

    def __repr__(self):
        return f"HRVO(apex={self.apex}, left={self.left}, right={self.right})"


def compute_hrvo(own, obs, responsibility=0.5, min_safe_distance=1000.0):
    """
    构造 HRVO

    对应论文中的 HRVO 几何构造公式

    Args:
        own: 本船状态 (VesselState)
        obs: 目标船状态 (VesselState)
        responsibility: 避让责任分配因子 (0-1), 默认 0.5 表示平等分担
        min_safe_distance: 最小安全会遇距离(m)，默认1000米
            HRVO将以此为安全半径构造，确保DCPA >= min_safe_distance

    Returns:
        HRVO: 构造的 HRVO 对象

    Raises:
        ValueError: 如果两船已经发生碰撞
    """
    # 相对位置和速度
    p_rel = obs.p - own.p
    v_rel = own.v - obs.v

    # 距离和组合半径
    dist = np.linalg.norm(p_rel)

    # 使用最小安全会遇距离作为HRVO的安全半径
    # 这确保了选择的速度会使DCPA >= min_safe_distance
    R = max(own.r + obs.r, min_safe_distance)

    # 检查是否已碰撞（使用物理半径判断）
    physical_R = own.r + obs.r
    if dist <= physical_R:
        raise ValueError(
            f"Collision already occurred: distance={dist:.2f}, R={physical_R:.2f}")

    # 如果距离小于安全半径，需要特殊处理
    if dist <= R:
        # 距离太近，无法构造有效的锥形HRVO
        # 创建一个覆盖整个速度空间前方的HRVO
        base_angle = np.arctan2(p_rel[1], p_rel[0])
        # 使用接近180度的张角
        theta = np.pi * 0.45
        left_dir = np.array([np.cos(base_angle + theta),
                             np.sin(base_angle + theta)])
        right_dir = np.array([np.cos(base_angle - theta),
                              np.sin(base_angle - theta)])
        apex = obs.v + responsibility * v_rel
        return HRVO(apex, left_dir, right_dir)

    # 计算切线角度
    # theta = arcsin(R / dist) 是从相对位置向量到切线的角度
    sin_theta = R / dist
    sin_theta = np.clip(sin_theta, -1.0, 1.0)  # 数值稳定性
    theta = np.arcsin(sin_theta)

    # 基准角度（相对位置方向）
    base_angle = np.arctan2(p_rel[1], p_rel[0])

    # 左右边界方向
    left_dir = np.array([np.cos(base_angle + theta),
                         np.sin(base_angle + theta)])
    right_dir = np.array([np.cos(base_angle - theta),
                          np.sin(base_angle - theta)])

    # HRVO apex 定义
    # 传统 VO: apex = v_obs
    # RVO: apex = v_obs + 0.5 * v_rel = 0.5 * (v_own + v_obs)
    # HRVO: apex 根据责任因子调整
    apex = obs.v + responsibility * v_rel

    return HRVO(apex, left_dir, right_dir)


def compute_vo(own, obs):
    """
    构造传统 VO (Velocity Obstacle)

    Args:
        own: 本船状态 (VesselState)
        obs: 目标船状态 (VesselState)

    Returns:
        HRVO: VO 对象（责任因子为 0）
    """
    return compute_hrvo(own, obs, responsibility=0.0)


def compute_rvo(own, obs):
    """
    构造 RVO (Reciprocal Velocity Obstacle)

    Args:
        own: 本船状态 (VesselState)
        obs: 目标船状态 (VesselState)

    Returns:
        HRVO: RVO 对象（责任因子为 0.5）
    """
    return compute_hrvo(own, obs, responsibility=0.5)
