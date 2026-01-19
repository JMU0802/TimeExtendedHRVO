"""
strategy.py - 避让策略参数化

对应论文符号:
    θ = (Δψ, Δu): 策略参数
    v_o(t; θ): 时间连续速度模型

转向方向约定（符合航海惯例）：
    - Δψ > 0: 右转（顺时针，Starboard）
    - Δψ < 0: 左转（逆时针，Port）
"""
import numpy as np


class AvoidanceStrategy:
    """
    避让策略类 - 参数化表示

    策略空间 Θ 中的一个策略由 (Δψ, Δu) 定义:
        - Δψ: 航向改变量 (rad)，正值为右转，负值为左转
        - Δu: 速度改变量 (m/s)

    转向方向约定（符合航海惯例）:
        - Δψ > 0: 右转/顺时针 (Starboard)
        - Δψ < 0: 左转/逆时针 (Port)

    Attributes:
        delta_psi (float): 航向改变量（正=右转，负=左转）
        delta_speed (float): 速度改变量
    """

    def __init__(self, delta_psi, delta_speed):
        """
        初始化避让策略

        Args:
            delta_psi: 航向改变量 (rad)，正值=右转，负值=左转
            delta_speed: 速度改变量 (m/s)
        """
        self.delta_psi = float(delta_psi)
        self.delta_speed = float(delta_speed)

    def velocity_profile(self, v0, t, tau=10.0):
        """
        恒策略假设下的时间连续速度模型

        对应论文公式: v_o(t; θ)

        使用一阶指数响应模型模拟船舶机动特性:
            speed(t) = |v0| + Δu * (1 - exp(-t/τ))
            heading(t) = ψ0 - Δψ * (1 - exp(-t/τ))

        注：航向减去Δψ是因为数学上顺时针旋转（右转）对应角度减小

        Args:
            v0: 初始速度向量 [vx, vy]
            t: 时间 (s)
            tau: 机动响应时间常数 (s), 默认 10s

        Returns:
            np.ndarray: t 时刻的速度向量
        """
        v0 = np.asarray(v0, dtype=float)

        # 计算响应因子
        response = 1.0 - np.exp(-t / tau)

        # 初始速度和航向
        speed_0 = np.linalg.norm(v0)
        heading_0 = np.arctan2(v0[1], v0[0])

        # 时变速度和航向
        # 注：右转(+Δψ)使航向减小，左转(-Δψ)使航向增大
        speed_t = speed_0 + self.delta_speed * response
        heading_t = heading_0 - self.delta_psi * response

        # 确保速度非负
        speed_t = max(0.0, speed_t)

        return speed_t * np.array([np.cos(heading_t), np.sin(heading_t)])

    def get_final_velocity(self, v0, tau=10.0):
        """
        获取策略执行完成后的最终速度（t→∞）

        Args:
            v0: 初始速度向量
            tau: 机动响应时间常数

        Returns:
            np.ndarray: 最终速度向量
        """
        v0 = np.asarray(v0, dtype=float)

        speed_0 = np.linalg.norm(v0)
        heading_0 = np.arctan2(v0[1], v0[0])

        speed_final = max(0.0, speed_0 + self.delta_speed)
        # 右转(+Δψ)使航向减小，左转(-Δψ)使航向增大
        heading_final = heading_0 - self.delta_psi

        return speed_final * np.array([np.cos(heading_final), np.sin(heading_final)])

    def __repr__(self):
        turn_dir = "Stbd" if self.delta_psi > 0 else "Port" if self.delta_psi < 0 else "None"
        return f"AvoidanceStrategy(Δψ={np.rad2deg(self.delta_psi):+.1f}°({turn_dir}), Δu={self.delta_speed:+.1f}m/s)"


def generate_strategy_space(psi_range_deg=(-30, 30), psi_step_deg=5,
                            speed_range=(-2, 2), speed_step=0.5):
    """
    生成策略空间 Θ

    Args:
        psi_range_deg: 航向改变范围 (度)，正值=右转，负值=左转
        psi_step_deg: 航向步长 (度)
        speed_range: 速度改变范围 (m/s)
        speed_step: 速度步长 (m/s)

    Returns:
        list[AvoidanceStrategy]: 策略列表
    """
    strategies = []

    psi_values = np.arange(
        psi_range_deg[0], psi_range_deg[1] + psi_step_deg, psi_step_deg)
    speed_values = np.arange(
        speed_range[0], speed_range[1] + speed_step, speed_step)

    for dpsi_deg in psi_values:
        for du in speed_values:
            strategies.append(AvoidanceStrategy(np.deg2rad(dpsi_deg), du))

    return strategies


def generate_colregs_strategy_space(prefer_starboard=True):
    """
    生成符合 COLREGs 规则的策略空间

    船舶避碰优先级：
    1. 纯改向（保持航速）- 首选
    2. 改向 + 微调速度
    3. 减速（仅作为辅助）

    根据 COLREGs 规则，优先考虑右转避让

    转向约定: 正值=右转(Starboard)，负值=左转(Port)

    Args:
        prefer_starboard: 是否优先右转（符合 COLREGs）

    Returns:
        list[AvoidanceStrategy]: 按优先级排序的策略列表
    """
    strategies = []

    # 第一优先级：纯改向策略（保持航速）
    # 右转角度（COLREGs 首选）: +5° 到 +60°
    if prefer_starboard:
        heading_angles = [5, 10, 15, 20, 25, 30, 40, 50, 60,  # 右转（正值）
                          -5, -10, -15, -20, -25, -30]  # 左转（负值，次选）
    else:
        heading_angles = [5, 10, 15, 20, 25, 30,
                          -5, -10, -15, -20, -25, -30,
                          40, 50, -40, -50]

    for angle in heading_angles:
        strategies.append(AvoidanceStrategy(np.deg2rad(angle), 0.0))

    # 第二优先级：改向 + 轻微减速
    for angle in [15, 20, 30, 45, -15, -20, -30]:
        strategies.append(AvoidanceStrategy(np.deg2rad(angle), -0.5))

    # 第三优先级：改向 + 适度减速
    for angle in [20, 30, 45, 60, -20, -30, -45]:
        strategies.append(AvoidanceStrategy(np.deg2rad(angle), -1.0))

    # 第四优先级：大幅改向
    for angle in [70, 80, 90, -70, -80, -90]:
        strategies.append(AvoidanceStrategy(np.deg2rad(angle), 0.0))

    # 第五优先级：纯减速（最后手段，但不应停船）
    strategies.append(AvoidanceStrategy(0, -1.0))
    strategies.append(AvoidanceStrategy(0, -1.5))

    # 保持原状（如果安全）
    strategies.append(AvoidanceStrategy(0, 0))

    return strategies


def generate_pure_heading_strategies():
    """
    生成纯改向策略空间（通常情况使用）

    船舶避碰原则：通常情况下只采用单独改向的方式避让
    - 不包含任何速度改变
    - 符合COLREGs右转优先原则

    转向约定: 正值=右转(Starboard)，负值=左转(Port)

    Returns:
        list[AvoidanceStrategy]: 纯改向策略列表
    """
    strategies = []

    # 右转（COLREGs 首选）: +5° 到 +90°（正值）
    for angle in range(5, 95, 5):
        strategies.append(AvoidanceStrategy(np.deg2rad(angle), 0.0))

    # 左转: -5° 到 -60°（负值，仅在右转不可行时）
    for angle in range(-5, -65, -5):
        strategies.append(AvoidanceStrategy(np.deg2rad(angle), 0.0))

    # 保持原状（如果安全）
    strategies.append(AvoidanceStrategy(0, 0))

    return strategies


def generate_emergency_strategies():
    """
    生成紧急情况策略空间（仅在紧迫局面或紧迫危险时使用）

    紧急情况包含：
    - 改向 + 减速组合
    - 大幅减速
    - 大幅改向

    转向约定: 正值=右转(Starboard)，负值=左转(Port)

    Returns:
        list[AvoidanceStrategy]: 紧急策略列表
    """
    strategies = []

    # 大幅右转（优先）
    for angle in [100, 110, 120, 135, 150]:
        strategies.append(AvoidanceStrategy(np.deg2rad(angle), 0.0))

    # 大幅左转（紧急情况允许）
    for angle in [-70, -80, -90, -120]:
        strategies.append(AvoidanceStrategy(np.deg2rad(angle), 0.0))

    # 改向 + 减速组合（右转优先）
    for angle in [30, 45, 60, 90, 120]:
        for du in [-0.5, -1.0, -1.5]:
            strategies.append(AvoidanceStrategy(np.deg2rad(angle), du))

    # 改向 + 减速组合（左转）
    for angle in [-30, -45, -60, -90]:
        for du in [-0.5, -1.0]:
            strategies.append(AvoidanceStrategy(np.deg2rad(angle), du))

    # 纯减速（最后手段）
    strategies.append(AvoidanceStrategy(0, -1.0))
    strategies.append(AvoidanceStrategy(0, -1.5))
    strategies.append(AvoidanceStrategy(0, -2.0))

    return strategies


def generate_marine_strategy_space(include_speed_change=False):
    """
    生成适用于海上船舶的策略空间

    特点：
    - 优先改向而非减速
    - 符合 COLREGs 右转优先原则
    - 默认只包含纯改向策略
    - 仅在紧急情况下才包含速度改变策略

    转向约定: 正值=右转(Starboard)，负值=左转(Port)

    Args:
        include_speed_change: 是否包含速度改变策略（紧急情况时设为True）

    Returns:
        list[AvoidanceStrategy]: 策略列表
    """
    # 基础策略：纯改向
    strategies = generate_pure_heading_strategies()

    # 紧急情况时才添加速度改变策略
    if include_speed_change:
        strategies.extend(generate_emergency_strategies())

    return strategies
