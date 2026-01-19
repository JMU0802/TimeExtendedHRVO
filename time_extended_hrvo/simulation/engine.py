"""
engine.py - 仿真引擎

负责管理仿真状态、时间步进和船舶运动更新
支持航向恢复功能和随机多船场景生成
"""
from time_extended_hrvo.utils.geometry import compute_dcpa_tcpa, classify_encounter
from time_extended_hrvo.planner.te_hrvo_planner import TimeExtendedHRVOPlanner, TraditionalHRVOPlanner
from time_extended_hrvo.core.hrvo import compute_hrvo, HRVO
from time_extended_hrvo.core.strategy import AvoidanceStrategy
from time_extended_hrvo.core.vessel import VesselState
import numpy as np
from typing import List, Optional, Callable
from dataclasses import dataclass, field
import random

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


@dataclass
class SimulationVessel:
    """仿真中的船舶对象"""
    state: VesselState
    name: str
    color: str
    is_own_ship: bool = False
    trajectory: List[np.ndarray] = field(default_factory=list)
    current_strategy: Optional[AvoidanceStrategy] = None
    target_velocity: Optional[np.ndarray] = None
    # 航向恢复相关
    original_velocity: Optional[np.ndarray] = None  # 原始航向速度
    is_avoiding: bool = False  # 是否正在避让

    def __post_init__(self):
        self.trajectory = [self.state.p.copy()]
        self.target_velocity = self.state.v.copy()
        self.original_velocity = self.state.v.copy()  # 保存原始速度

    def update_position(self, dt: float, tau: float = 10.0):
        """更新船舶位置"""
        if self.current_strategy and self.is_own_ship:
            # 应用策略的速度变化
            self.target_velocity = self.current_strategy.get_final_velocity(
                self.trajectory[0] if len(
                    self.trajectory) == 1 else self.state.v
            )

        # 一阶指数响应模型
        alpha = 1.0 - np.exp(-dt / tau)
        self.state.v = self.state.v + alpha * \
            (self.target_velocity - self.state.v)

        # 更新位置
        self.state.p = self.state.p + self.state.v * dt
        self.trajectory.append(self.state.p.copy())

        # 限制轨迹长度
        if len(self.trajectory) > 500:
            self.trajectory = self.trajectory[-500:]

    def reset(self, initial_state: VesselState):
        """重置船舶状态"""
        self.state = initial_state.copy()
        self.trajectory = [self.state.p.copy()]
        self.target_velocity = self.state.v.copy()
        self.original_velocity = self.state.v.copy()
        self.current_strategy = None
        self.is_avoiding = False


class SimulationEngine:
    """
    仿真引擎

    管理多船仿真、规划器调用和状态更新
    支持航向恢复功能
    """

    def __init__(self, use_time_extended: bool = True):
        """
        初始化仿真引擎

        Args:
            use_time_extended: 是否使用 Time-Extended HRVO（否则使用传统 HRVO）
        """
        self.vessels: List[SimulationVessel] = []
        self.own_ship_idx: int = 0
        self.time: float = 0.0
        self.dt: float = 0.1  # 仿真时间步长
        self.tau: float = 10.0  # 机动响应时间
        self.T_p: float = 30.0  # 规划时域

        self.use_time_extended = use_time_extended
        self.planner = TimeExtendedHRVOPlanner(
            T_p=self.T_p) if use_time_extended else TraditionalHRVOPlanner()

        self.is_running: bool = False
        self.collision_occurred: bool = False
        self.collision_distance: float = float('inf')

        # 回调函数
        self.on_update: Optional[Callable] = None
        self.on_collision: Optional[Callable] = None

        # 当前 HRVO 列表（用于可视化）
        self.current_hrvos: List[HRVO] = []
        self.current_strategy: Optional[AvoidanceStrategy] = None

        # 初始状态备份
        self._initial_states: List[VesselState] = []

        # 航向恢复参数
        self.safe_dcpa_threshold: float = 300.0  # 安全DCPA阈值(m)
        self.safe_tcpa_threshold: float = -10.0  # TCPA<0表示已过CPA
        self.heading_recovery_enabled: bool = True  # 是否启用航向恢复

    def add_vessel(self, position, velocity, radius, name="Vessel",
                   color="blue", is_own_ship=False):
        """添加船舶"""
        state = VesselState(position, velocity, radius)
        vessel = SimulationVessel(
            state=state,
            name=name,
            color=color,
            is_own_ship=is_own_ship
        )
        self.vessels.append(vessel)
        self._initial_states.append(state.copy())

        if is_own_ship:
            self.own_ship_idx = len(self.vessels) - 1

        return len(self.vessels) - 1

    def clear_vessels(self):
        """清空所有船舶"""
        self.vessels.clear()
        self._initial_states.clear()
        self.own_ship_idx = 0

    def get_own_ship(self) -> Optional[SimulationVessel]:
        """获取本船"""
        if 0 <= self.own_ship_idx < len(self.vessels):
            return self.vessels[self.own_ship_idx]
        return None

    def get_obstacles(self) -> List[SimulationVessel]:
        """获取所有目标船"""
        return [v for i, v in enumerate(self.vessels) if i != self.own_ship_idx]

    def _check_collision_risk(self, own: SimulationVessel) -> bool:
        """
        检查是否存在碰撞风险

        Returns:
            True 如果存在需要避让的目标船
        """
        obstacles = self.get_obstacles()
        for obs in obstacles:
            dcpa, tcpa = compute_dcpa_tcpa(
                own.state.p, own.state.v,
                obs.state.p, obs.state.v
            )
            # 如果DCPA小于阈值且TCPA>0(还未到达CPA)，则存在风险
            min_safe_dist = (own.state.r + obs.state.r) * 3  # 3倍安全半径
            if dcpa < max(self.safe_dcpa_threshold, min_safe_dist) and tcpa > 0:
                return True
        return False

    def _should_recover_heading(self, own: SimulationVessel) -> bool:
        """
        判断是否应该恢复航向

        条件:
        1. 正在避让状态
        2. 所有目标船的TCPA < 0 (已过CPA) 或 DCPA > 安全阈值
        """
        if not own.is_avoiding:
            return False

        obstacles = self.get_obstacles()
        for obs in obstacles:
            dcpa, tcpa = compute_dcpa_tcpa(
                own.state.p, own.state.v,
                obs.state.p, obs.state.v
            )
            min_safe_dist = (own.state.r + obs.state.r) * 2
            # 如果还有目标船需要避让，不恢复
            if tcpa > 5.0 and dcpa < max(self.safe_dcpa_threshold, min_safe_dist):
                return False
        return True

    def step(self):
        """执行一个仿真步骤"""
        if not self.vessels:
            return

        own = self.get_own_ship()
        if not own:
            return

        obstacles = self.get_obstacles()
        obstacle_states = [obs.state for obs in obstacles]

        # 构造 HRVO（用于可视化）
        self.current_hrvos = []
        for obs_state in obstacle_states:
            try:
                hrvo = compute_hrvo(own.state, obs_state)
                self.current_hrvos.append(hrvo)
            except ValueError:
                pass  # 已碰撞

        # 检查是否需要避让或恢复航向
        has_collision_risk = self._check_collision_risk(own)

        if has_collision_risk:
            # 存在碰撞风险，执行避让规划
            own.is_avoiding = True
            if self.use_time_extended:
                self.current_strategy = self.planner.plan(
                    own.state, obstacle_states)
            else:
                self.current_strategy = self.planner.plan(
                    own.state, obstacle_states)

            # 应用策略
            if self.current_strategy:
                own.current_strategy = self.current_strategy
            else:
                # 规划器未能返回策略时的默认行为：小幅右转
                # 这是一个安全的默认策略，符合COLREGs
                default_strategy = AvoidanceStrategy(np.deg2rad(30), 0.0)
                own.current_strategy = default_strategy
                self.current_strategy = default_strategy

        elif self.heading_recovery_enabled and self._should_recover_heading(own):
            # 无碰撞风险，恢复原航向
            self.current_strategy = None
            own.current_strategy = None
            own.target_velocity = own.original_velocity.copy()

            # 检查是否已恢复到原航向
            current_heading = np.arctan2(own.state.v[1], own.state.v[0])
            original_heading = np.arctan2(
                own.original_velocity[1], own.original_velocity[0])
            heading_diff = abs(current_heading - original_heading)
            if heading_diff > np.pi:
                heading_diff = 2 * np.pi - heading_diff

            # 如果航向差小于5度，认为已恢复
            if heading_diff < np.deg2rad(5):
                own.is_avoiding = False
        else:
            # 保持当前状态
            self.current_strategy = None
            own.current_strategy = None

        # 更新所有船舶位置
        for vessel in self.vessels:
            vessel.update_position(self.dt, self.tau)

        # 检测碰撞
        self._check_collisions()

        # 更新时间
        self.time += self.dt

        # 触发回调
        if self.on_update:
            self.on_update()

    def _check_collisions(self):
        """检测碰撞"""
        own = self.get_own_ship()
        if not own:
            return

        for obs in self.get_obstacles():
            dist = np.linalg.norm(own.state.p - obs.state.p)
            min_dist = own.state.r + obs.state.r

            self.collision_distance = min(self.collision_distance, dist)

            if dist < min_dist:
                self.collision_occurred = True
                if self.on_collision:
                    self.on_collision(own, obs)

    def reset(self):
        """重置仿真"""
        self.time = 0.0
        self.collision_occurred = False
        self.collision_distance = float('inf')
        self.current_hrvos = []
        self.current_strategy = None

        for i, vessel in enumerate(self.vessels):
            if i < len(self._initial_states):
                vessel.reset(self._initial_states[i])

    def set_planner_type(self, use_time_extended: bool):
        """切换规划器类型"""
        self.use_time_extended = use_time_extended
        if use_time_extended:
            self.planner = TimeExtendedHRVOPlanner(T_p=self.T_p, tau=self.tau)
        else:
            self.planner = TraditionalHRVOPlanner()

    def get_encounter_info(self) -> dict:
        """获取会遇信息"""
        own = self.get_own_ship()
        if not own:
            return {}

        info = {
            'time': self.time,
            'own_position': own.state.p.tolist(),
            'own_velocity': own.state.v.tolist(),
            'own_speed': own.state.speed,
            'is_avoiding': own.is_avoiding,
            'obstacles': []
        }

        for obs in self.get_obstacles():
            dcpa, tcpa = compute_dcpa_tcpa(
                own.state.p, own.state.v,
                obs.state.p, obs.state.v
            )
            encounter_type = classify_encounter(own.state, obs.state)
            distance = np.linalg.norm(own.state.p - obs.state.p)

            info['obstacles'].append({
                'name': obs.name,
                'position': obs.state.p.tolist(),
                'velocity': obs.state.v.tolist(),
                'distance': distance,
                'dcpa': dcpa,
                'tcpa': tcpa,
                'encounter_type': encounter_type
            })

        return info


def create_head_on_scenario(engine: SimulationEngine):
    """创建对遇场景"""
    engine.clear_vessels()
    engine.add_vessel([0, 0], [0, 5], 30, "Own Ship", "blue", is_own_ship=True)
    engine.add_vessel([50, 800], [0, -5], 30, "Target 1", "red")


def create_crossing_scenario(engine: SimulationEngine):
    """创建交叉相遇场景"""
    engine.clear_vessels()
    engine.add_vessel([0, 0], [5, 0], 30, "Own Ship", "blue", is_own_ship=True)
    engine.add_vessel([400, -250], [0, 5], 30, "Target 1", "red")


def create_overtaking_scenario(engine: SimulationEngine):
    """创建追越场景"""
    engine.clear_vessels()
    engine.add_vessel([0, 0], [0, 8], 30, "Own Ship", "blue", is_own_ship=True)
    engine.add_vessel([30, 300], [0, 3], 30, "Target 1", "red")


def create_multi_vessel_scenario(engine: SimulationEngine):
    """创建多船场景（固定）"""
    engine.clear_vessels()
    engine.add_vessel([0, 0], [5, 0], 30, "Own Ship", "blue", is_own_ship=True)
    engine.add_vessel([500, 100], [-4, 0], 25, "Target 1", "red")
    engine.add_vessel([500, -100], [-4, 0], 25, "Target 2", "orange")
    engine.add_vessel([600, 0], [-3, 1], 25, "Target 3", "green")


def create_random_scenario(engine: SimulationEngine, num_targets: int = None):
    """
    创建随机多船场景

    Args:
        engine: 仿真引擎
        num_targets: 目标船数量，None则随机选择3-5艘
    """
    engine.clear_vessels()

    # 随机选择目标船数量
    if num_targets is None:
        num_targets = random.randint(3, 5)
    num_targets = max(1, min(num_targets, 8))  # 限制1-8艘

    # 本船参数
    own_speed = random.uniform(4, 8)  # 4-8 m/s
    own_heading = random.uniform(0, 2 * np.pi)  # 随机航向
    own_vx = own_speed * np.cos(own_heading)
    own_vy = own_speed * np.sin(own_heading)

    engine.add_vessel([0, 0], [own_vx, own_vy], 30,
                      "Own Ship", "blue", is_own_ship=True)

    # 目标船颜色列表
    colors = ["red", "orange", "green", "purple",
              "brown", "pink", "cyan", "magenta"]

    # 生成目标船
    for i in range(num_targets):
        # 在本船前方生成目标船（确保会遇）
        # 距离：300-800m
        distance = random.uniform(300, 800)

        # 相对角度：在本船航向前方 ±60度范围内
        rel_angle = random.uniform(-np.pi/3, np.pi/3)
        spawn_angle = own_heading + rel_angle

        # 目标船位置
        target_x = distance * np.cos(spawn_angle)
        target_y = distance * np.sin(spawn_angle)

        # 目标船速度：生成朝向本船方向或交叉的速度
        target_speed = random.uniform(3, 7)

        # 确定目标船航向（朝向本船附近）
        # 计算从目标船到本船的方向
        to_own_angle = np.arctan2(-target_y, -target_x)
        # 添加随机偏移
        heading_offset = random.uniform(-np.pi/4, np.pi/4)
        target_heading = to_own_angle + heading_offset

        target_vx = target_speed * np.cos(target_heading)
        target_vy = target_speed * np.sin(target_heading)

        # 目标船半径
        target_radius = random.uniform(20, 35)

        engine.add_vessel(
            [target_x, target_y],
            [target_vx, target_vy],
            target_radius,
            f"Target {i+1}",
            colors[i % len(colors)]
        )


def create_random_3_vessel_scenario(engine: SimulationEngine):
    """创建随机3船场景"""
    create_random_scenario(engine, num_targets=2)  # 本船+2目标船=3船


def create_random_4_vessel_scenario(engine: SimulationEngine):
    """创建随机4船场景"""
    create_random_scenario(engine, num_targets=3)  # 本船+3目标船=4船


def create_random_5_vessel_scenario(engine: SimulationEngine):
    """创建随机5船场景"""
    create_random_scenario(engine, num_targets=4)  # 本船+4目标船=5船
