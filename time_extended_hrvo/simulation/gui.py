"""
gui.py - Time-Extended HRVO 图形化仿真界面

使用 tkinter 实现可视化仿真系统，包含：
- 位置空间视图（船舶轨迹、相对运动线）
- 速度空间视图（HRVO 可视化，支持缩放）
- 相对运动图表（DCPA/TCPA/距离随时间变化）
- 控制面板
- 信息显示

转向方向约定（符合航海惯例）：
- 右转（顺时针）= 正值 (+)
- 左转（逆时针）= 负值 (-)
"""
from time_extended_hrvo.simulation.engine import (
    SimulationEngine, SimulationVessel,
    create_head_on_scenario, create_crossing_scenario,
    create_overtaking_scenario, create_multi_vessel_scenario,
    create_random_scenario, create_random_3_vessel_scenario,
    create_random_4_vessel_scenario, create_random_5_vessel_scenario
)
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import math
from typing import Optional, List, Tuple
from collections import deque

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


class PositionSpaceCanvas(tk.Canvas):
    """
    位置空间画布

    显示船舶位置、轨迹、安全区域和相对运动线
    """

    def __init__(self, parent, width=700, height=600, **kwargs):
        super().__init__(parent, width=width, height=height, bg='#FAFAFA', **kwargs)

        self.width = width
        self.height = height

        # 视图参数
        self.scale = 0.6  # 像素/米
        self.offset_x = width / 2
        self.offset_y = height / 2

        # 绑定鼠标事件
        self.bind('<MouseWheel>', self._on_mousewheel)
        self.bind('<Button-1>', self._on_click)
        self.bind('<B1-Motion>', self._on_drag)
        self.bind('<Configure>', self._on_resize)
        self._last_x = 0
        self._last_y = 0

        # 绘制网格
        self._draw_grid()

    def _on_resize(self, event):
        """窗口大小改变"""
        self.width = event.width
        self.height = event.height
        self._draw_grid()

    def _draw_grid(self):
        """绘制背景网格"""
        self.delete('grid')

        # 动态计算网格间距（米）
        base_spacing = 100
        pixel_spacing = base_spacing * self.scale

        if pixel_spacing < 30:
            grid_spacing = 200
        elif pixel_spacing < 50:
            grid_spacing = 100
        elif pixel_spacing > 150:
            grid_spacing = 50
        else:
            grid_spacing = 100

        pixel_spacing = grid_spacing * self.scale

        # 获取视图边界对应的世界坐标
        world_left, world_top = self.screen_to_world(0, 0)
        world_right, world_bottom = self.screen_to_world(
            self.width, self.height)

        # 计算网格线的起止位置（扩展范围确保完整覆盖）
        start_x = int(world_left / grid_spacing - 2) * grid_spacing
        end_x = int(world_right / grid_spacing + 2) * grid_spacing
        start_y = int(world_bottom / grid_spacing - 2) * grid_spacing
        end_y = int(world_top / grid_spacing + 2) * grid_spacing

        # 绘制垂直网格线
        x = start_x
        while x <= end_x:
            sx, _ = self.world_to_screen(x, 0)
            is_axis = abs(x) < 0.1
            color = '#A0A0A0' if is_axis else '#D8D8D8'
            width = 2 if is_axis else 1
            self.create_line(sx, 0, sx, self.height,
                             fill=color, width=width, tags='grid')
            # 标签（每隔一个标注）
            if not is_axis and int(x / grid_spacing) % 2 == 0:
                _, label_y = self.world_to_screen(0, 0)
                label_y = min(max(label_y + 15, 15), self.height - 5)
                self.create_text(sx, label_y,
                                 text=f'{int(x)}',
                                 fill='#606060', font=('Arial', 8), tags='grid')
            x += grid_spacing

        # 绘制水平网格线
        y = start_y
        while y <= end_y:
            _, sy = self.world_to_screen(0, y)
            is_axis = abs(y) < 0.1
            color = '#A0A0A0' if is_axis else '#D8D8D8'
            width = 2 if is_axis else 1
            self.create_line(0, sy, self.width, sy,
                             fill=color, width=width, tags='grid')
            # 标签
            if not is_axis and int(y / grid_spacing) % 2 == 0:
                label_x, _ = self.world_to_screen(0, 0)
                label_x = min(max(label_x + 20, 25), self.width - 20)
                self.create_text(label_x, sy,
                                 text=f'{int(y)}',
                                 fill='#606060', font=('Arial', 8), tags='grid')
            y += grid_spacing

        # 坐标轴标签
        self.create_text(self.width - 35, self.offset_y - 18,
                         text='X(E) [m]', fill='#404040',
                         font=('Arial', 10, 'bold'), tags='grid')
        self.create_text(self.offset_x + 35, 18,
                         text='Y(N) [m]', fill='#404040',
                         font=('Arial', 10, 'bold'), tags='grid')

        # 比例尺
        scale_length = 100  # 米
        scale_pixels = scale_length * self.scale
        scale_x = 20
        scale_y = self.height - 25
        self.create_line(scale_x, scale_y, scale_x + scale_pixels, scale_y,
                         fill='#404040', width=2, tags='grid')
        self.create_line(scale_x, scale_y - 5, scale_x, scale_y + 5,
                         fill='#404040', width=2, tags='grid')
        self.create_line(scale_x + scale_pixels, scale_y - 5,
                         scale_x + scale_pixels, scale_y + 5,
                         fill='#404040', width=2, tags='grid')
        self.create_text(scale_x + scale_pixels / 2, scale_y - 12,
                         text=f'{scale_length}m', fill='#404040',
                         font=('Arial', 9), tags='grid')

        self.tag_lower('grid')

    def world_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        """世界坐标转屏幕坐标"""
        sx = self.offset_x + x * self.scale
        sy = self.offset_y - y * self.scale  # Y轴翻转
        return sx, sy

    def screen_to_world(self, sx: float, sy: float) -> Tuple[float, float]:
        """屏幕坐标转世界坐标"""
        x = (sx - self.offset_x) / self.scale
        y = (self.offset_y - sy) / self.scale
        return x, y

    def draw_vessel(self, vessel: SimulationVessel):
        """绘制船舶

        船首方向与航向（速度方向）严格一致
        使用标准船形：尖头（船首）指向航行方向
        """
        x, y = vessel.state.p
        r = vessel.state.r
        heading = vessel.state.heading  # 航向角（弧度）

        sx, sy = self.world_to_screen(x, y)
        sr = r * self.scale

        tag = f'vessel_{vessel.name}'
        self.delete(tag)

        # 绘制安全区域（虚线圆）
        self.create_oval(
            sx - sr, sy - sr, sx + sr, sy + sr,
            outline=vessel.color, width=2, dash=(5, 3),
            tags=tag
        )

        # 船体尺寸（根据缩放比例调整）
        ship_length = max(18, sr * 0.7)
        ship_width = ship_length * 0.35

        # 船形顶点（船体坐标系：船首在+X方向）
        # 船首（尖头）、右舷前、右舷后、左舷后、左舷前
        hull_points_local = [
            (ship_length * 0.5, 0),                    # 船首（尖头）
            (ship_length * 0.2, -ship_width * 0.8),   # 右舷前
            (-ship_length * 0.45, -ship_width * 0.7),  # 右舷后
            (-ship_length * 0.5, 0),                   # 船尾中心
            (-ship_length * 0.45, ship_width * 0.7),  # 左舷后
            (ship_length * 0.2, ship_width * 0.8),    # 左舷前
        ]

        # 旋转并转换到屏幕坐标
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        hull_points = []
        for lx, ly in hull_points_local:
            # 旋转到世界坐标系
            wx = lx * cos_h - ly * sin_h
            wy = lx * sin_h + ly * cos_h
            # 转换到屏幕坐标（Y轴翻转）
            px = sx + wx
            py = sy - wy
            hull_points.extend([px, py])

        # 绘制船体
        self.create_polygon(
            hull_points, fill=vessel.color, outline='#202020', width=1,
            tags=tag
        )

        # 绘制船首中心线（显示精确航向）
        bow_length = ship_length * 0.6
        bow_end_x = sx + bow_length * cos_h
        bow_end_y = sy - bow_length * sin_h
        self.create_line(
            sx, sy, bow_end_x, bow_end_y,
            fill='#202020', width=2,
            tags=tag
        )

        # 绘制速度向量（绿色箭头）
        vx, vy = vessel.state.v
        speed = np.linalg.norm([vx, vy])
        if speed > 0.1:
            v_scale = 6  # 秒
            end_x, end_y = self.world_to_screen(
                x + vx * v_scale, y + vy * v_scale)
            self.create_line(
                sx, sy, end_x, end_y,
                fill='#00AA00', width=2, arrow=tk.LAST, arrowshape=(10, 12, 5),
                tags=tag
            )

        # 绘制名称
        self.create_text(
            sx, sy - sr - 20,
            text=vessel.name, fill=vessel.color,
            font=('Arial', 10, 'bold'),
            tags=tag
        )

        # 显示速度和航向信息
        heading_deg = np.rad2deg(heading)
        # 转换到航海惯例：北为0度，顺时针为正
        nav_heading = (90 - heading_deg) % 360
        info_text = f'{speed:.1f}m/s, {nav_heading:.0f}°'
        self.create_text(
            sx, sy + sr + 15,
            text=info_text, fill='#404040',
            font=('Arial', 8),
            tags=tag
        )

    def draw_relative_motion_line(self, own: SimulationVessel, target: SimulationVessel):
        """绘制相对运动线

        显示目标船相对于本船的运动轨迹预测
        """
        tag = 'rel_motion'

        # 相对位置
        rel_pos = target.state.p - own.state.p
        # 相对速度（目标相对于本船）
        rel_vel = target.state.v - own.state.v

        rel_speed = np.linalg.norm(rel_vel)
        if rel_speed < 0.1:
            return

        # 计算CPA点
        # TCPA = -dot(rel_pos, rel_vel) / |rel_vel|^2
        tcpa = -np.dot(rel_pos, rel_vel) / (rel_speed ** 2)

        # 当前相对位置（以本船为原点）
        own_pos = own.state.p
        sx_own, sy_own = self.world_to_screen(own_pos[0], own_pos[1])

        # 目标当前位置
        sx_target, sy_target = self.world_to_screen(
            target.state.p[0], target.state.p[1])

        # 绘制当前连线（距离线）
        self.create_line(
            sx_own, sy_own, sx_target, sy_target,
            fill='#FF6600', width=1, dash=(3, 3),
            tags=tag
        )

        # 计算并绘制相对运动线延长（预测轨迹）
        if tcpa > 0:
            # CPA点位置
            cpa_rel = rel_pos + rel_vel * tcpa
            cpa_world = own_pos + cpa_rel
            sx_cpa, sy_cpa = self.world_to_screen(cpa_world[0], cpa_world[1])

            # 从当前目标位置延伸到CPA点之后
            extend_time = max(tcpa * 1.5, 30)  # 延伸到1.5倍TCPA或至少30秒
            future_rel = rel_pos + rel_vel * extend_time
            future_world = own_pos + future_rel
            sx_future, sy_future = self.world_to_screen(
                future_world[0], future_world[1])

            # 相对运动线（橙色虚线）
            self.create_line(
                sx_target, sy_target, sx_future, sy_future,
                fill='#FF6600', width=2, dash=(8, 4),
                tags=tag
            )

            # CPA点标记
            r = 6
            self.create_oval(
                sx_cpa - r, sy_cpa - r, sx_cpa + r, sy_cpa + r,
                fill='#FF0000', outline='#800000', width=2,
                tags=tag
            )

            # DCPA标注
            dcpa = np.linalg.norm(cpa_rel)
            self.create_text(
                sx_cpa + 15, sy_cpa - 15,
                text=f'CPA\nDCPA:{dcpa:.0f}m\nTCPA:{tcpa:.0f}s',
                fill='#CC0000', font=('Arial', 8),
                justify=tk.LEFT, tags=tag
            )

    def draw_trajectory(self, vessel: SimulationVessel):
        """绘制轨迹"""
        if len(vessel.trajectory) < 2:
            return

        tag = f'traj_{vessel.name}'
        self.delete(tag)

        points = []
        for p in vessel.trajectory:
            sx, sy = self.world_to_screen(p[0], p[1])
            points.extend([sx, sy])

        if len(points) >= 4:
            self.create_line(
                points, fill=vessel.color, width=2,
                smooth=True, tags=tag
            )

    def draw_all(self, vessels: List[SimulationVessel], show_rel_motion=True):
        """绘制所有船舶"""
        # 清除相对运动线
        self.delete('rel_motion')

        # 找到本船和目标船
        own_ship = None
        target_ships = []
        for v in vessels:
            if v.is_own_ship:
                own_ship = v
            else:
                target_ships.append(v)

        # 绘制相对运动线
        if show_rel_motion and own_ship and target_ships:
            for target in target_ships:
                self.draw_relative_motion_line(own_ship, target)

        # 绘制轨迹
        for vessel in vessels:
            self.draw_trajectory(vessel)

        # 绘制船舶
        for vessel in vessels:
            self.draw_vessel(vessel)

    def clear_vessels(self):
        """清除所有船舶绘制"""
        self.delete('all')
        self._draw_grid()

    def _on_mousewheel(self, event):
        """鼠标滚轮缩放"""
        factor = 1.15 if event.delta > 0 else 0.87
        self.scale *= factor
        self.scale = max(0.05, min(5.0, self.scale))
        self._draw_grid()

    def _on_click(self, event):
        """鼠标点击"""
        self._last_x = event.x
        self._last_y = event.y

    def _on_drag(self, event):
        """鼠标拖拽平移"""
        dx = event.x - self._last_x
        dy = event.y - self._last_y
        self.offset_x += dx
        self.offset_y += dy
        self._last_x = event.x
        self._last_y = event.y
        self._draw_grid()


class VelocitySpaceCanvas(tk.Canvas):
    """
    速度空间画布

    显示 HRVO、当前速度和可行速度区域
    支持鼠标滚轮缩放
    """

    def __init__(self, parent, width=400, height=400, **kwargs):
        super().__init__(parent, width=width, height=height, bg='#F5F5F5', **kwargs)

        self.width = width
        self.height = height
        self.center_x = width / 2
        self.center_y = height / 2
        self.scale = 25  # 像素/(m/s)

        # 绑定鼠标事件
        self.bind('<MouseWheel>', self._on_mousewheel)
        self.bind('<Configure>', self._on_resize)

        self._draw_axes()

    def _on_resize(self, event):
        """窗口大小改变"""
        self.width = event.width
        self.height = event.height
        self.center_x = self.width / 2
        self.center_y = self.height / 2
        self._draw_axes()

    def _on_mousewheel(self, event):
        """鼠标滚轮缩放"""
        factor = 1.15 if event.delta > 0 else 0.87
        self.scale *= factor
        self.scale = max(5, min(100, self.scale))
        self._draw_axes()

    def _draw_axes(self):
        """绘制坐标轴和网格"""
        self.delete('axes')

        # 计算网格间距
        if self.scale < 15:
            grid_step = 2
        elif self.scale < 30:
            grid_step = 1
        else:
            grid_step = 0.5

        # 计算可见范围
        max_v = max(self.width, self.height) / (2 * self.scale)

        # 绘制网格
        v = -int(max_v / grid_step) * grid_step
        while v <= max_v:
            sx = self.center_x + v * self.scale
            sy = self.center_y - v * self.scale

            if 0 <= sx <= self.width:
                color = '#B0B0B0' if abs(v) < 0.01 else '#E0E0E0'
                width = 2 if abs(v) < 0.01 else 1
                self.create_line(sx, 0, sx, self.height,
                                 fill=color, width=width, tags='axes')
                if abs(v) > 0.01 and abs(v) % (grid_step * 2) < 0.01:
                    self.create_text(sx, self.center_y + 12,
                                     text=f'{v:.1f}', fill='#606060',
                                     font=('Arial', 8), tags='axes')

            if 0 <= sy <= self.height:
                color = '#B0B0B0' if abs(v) < 0.01 else '#E0E0E0'
                width = 2 if abs(v) < 0.01 else 1
                self.create_line(0, sy, self.width, sy,
                                 fill=color, width=width, tags='axes')
                if abs(v) > 0.01 and abs(v) % (grid_step * 2) < 0.01:
                    self.create_text(self.center_x + 15, sy,
                                     text=f'{v:.1f}', fill='#606060',
                                     font=('Arial', 8), tags='axes')
            v += grid_step

        # 坐标轴标签
        self.create_text(self.width - 25, self.center_y - 15,
                         text='Vx', fill='#404040',
                         font=('Arial', 11, 'bold'), tags='axes')
        self.create_text(self.center_x + 18, 15,
                         text='Vy', fill='#404040',
                         font=('Arial', 11, 'bold'), tags='axes')

        # 缩放信息
        self.create_text(10, self.height - 10,
                         text=f'Scale: {self.scale:.0f} px/(m/s)',
                         fill='#808080', font=('Arial', 8),
                         anchor=tk.SW, tags='axes')

        self.tag_lower('axes')

    def vel_to_screen(self, vx: float, vy: float) -> Tuple[float, float]:
        """速度坐标转屏幕坐标"""
        sx = self.center_x + vx * self.scale
        sy = self.center_y - vy * self.scale
        return sx, sy

    def draw_hrvo(self, hrvo, color='red', alpha=0.3):
        """绘制 HRVO 区域"""
        apex = hrvo.apex
        left = hrvo.left
        right = hrvo.right

        ax, ay = self.vel_to_screen(apex[0], apex[1])

        # 计算远端点（延伸边界）
        extend = 20  # 延伸长度 (m/s)
        left_end = apex + left * extend
        right_end = apex + right * extend

        lx, ly = self.vel_to_screen(left_end[0], left_end[1])
        rx, ry = self.vel_to_screen(right_end[0], right_end[1])

        # 绘制填充区域
        self.create_polygon(
            ax, ay, lx, ly, rx, ry,
            fill=color, stipple='gray50', outline=color, width=2,
            tags='hrvo'
        )

        # 标注apex点
        r = 4
        self.create_oval(ax - r, ay - r, ax + r, ay + r,
                         fill=color, outline='#400000', tags='hrvo')

    def draw_velocity(self, vx: float, vy: float, color='blue',
                      label='Current', is_preferred=False):
        """绘制速度向量"""
        sx, sy = self.vel_to_screen(vx, vy)

        # 从原点画箭头
        self.create_line(
            self.center_x, self.center_y, sx, sy,
            fill=color, width=3 if is_preferred else 2,
            arrow=tk.LAST, arrowshape=(10, 12, 5),
            tags='velocity'
        )

        # 标签
        speed = np.sqrt(vx**2 + vy**2)
        self.create_text(
            sx + 12, sy - 12,
            text=f'{label}\n({speed:.1f}m/s)', fill=color,
            font=('Arial', 8), justify=tk.LEFT,
            tags='velocity'
        )

        # 速度点
        r = 5
        self.create_oval(
            sx - r, sy - r, sx + r, sy + r,
            fill=color, outline='#000000', width=1,
            tags='velocity'
        )

    def draw_velocity_circle(self, max_speed: float = 10.0):
        """绘制最大速度圆"""
        r = max_speed * self.scale
        self.create_oval(
            self.center_x - r, self.center_y - r,
            self.center_x + r, self.center_y + r,
            outline='#888888', dash=(6, 4), width=2,
            tags='circle'
        )
        self.create_text(
            self.center_x + r + 5, self.center_y,
            text=f'{max_speed}m/s', fill='#888888',
            font=('Arial', 8), anchor=tk.W,
            tags='circle'
        )

    def clear(self):
        """清除所有绘制"""
        self.delete('hrvo')
        self.delete('velocity')
        self.delete('circle')

    def draw_all(self, engine: 'SimulationEngine'):
        """绘制所有内容"""
        self.clear()
        self.draw_velocity_circle(10.0)

        # 绘制 HRVO
        colors = ['#FF6666', '#FF9966', '#FFCC66', '#66CCFF']
        for i, hrvo in enumerate(engine.current_hrvos):
            color = colors[i % len(colors)]
            self.draw_hrvo(hrvo, color)

        # 绘制当前速度
        own = engine.get_own_ship()
        if own:
            self.draw_velocity(
                own.state.v[0], own.state.v[1],
                '#0066CC', 'Current'
            )

            # 绘制目标速度
            if own.target_velocity is not None:
                self.draw_velocity(
                    own.target_velocity[0], own.target_velocity[1],
                    '#00AA00', 'Target', is_preferred=True
                )


class RelativeMotionChart(tk.Canvas):
    """
    相对运动图表

    显示距离、DCPA、TCPA随时间的变化曲线
    """

    def __init__(self, parent, width=350, height=200, **kwargs):
        super().__init__(parent, width=width, height=height, bg='white', **kwargs)

        self.width = width
        self.height = height

        # 数据存储
        self.max_points = 200
        self.time_data = deque(maxlen=self.max_points)
        self.distance_data = deque(maxlen=self.max_points)
        self.dcpa_data = deque(maxlen=self.max_points)
        self.tcpa_data = deque(maxlen=self.max_points)

        # 绘图区域
        self.margin_left = 50
        self.margin_right = 15
        self.margin_top = 25
        self.margin_bottom = 30

        self._draw_axes()

    def _draw_axes(self):
        """绘制坐标轴"""
        self.delete('axes')

        plot_width = self.width - self.margin_left - self.margin_right
        plot_height = self.height - self.margin_top - self.margin_bottom

        # 边框
        self.create_rectangle(
            self.margin_left, self.margin_top,
            self.width - self.margin_right, self.height - self.margin_bottom,
            outline='#808080', tags='axes'
        )

        # Y轴标签
        self.create_text(
            15, self.height / 2,
            text='Distance (m)', fill='#404040',
            font=('Arial', 9), angle=90, tags='axes'
        )

        # X轴标签
        self.create_text(
            self.width / 2, self.height - 8,
            text='Time (s)', fill='#404040',
            font=('Arial', 9), tags='axes'
        )

        # 标题
        self.create_text(
            self.width / 2, 10,
            text='Relative Motion Analysis', fill='#202020',
            font=('Arial', 10, 'bold'), tags='axes'
        )

        # 图例
        legend_x = self.margin_left + 10
        legend_y = self.margin_top + 10

        self.create_line(legend_x, legend_y, legend_x + 20, legend_y,
                         fill='#0066CC', width=2, tags='axes')
        self.create_text(legend_x + 25, legend_y, text='Dist',
                         fill='#0066CC', font=('Arial', 8),
                         anchor=tk.W, tags='axes')

        self.create_line(legend_x + 55, legend_y, legend_x + 75, legend_y,
                         fill='#CC0000', width=2, dash=(4, 2), tags='axes')
        self.create_text(legend_x + 80, legend_y, text='DCPA',
                         fill='#CC0000', font=('Arial', 8),
                         anchor=tk.W, tags='axes')

    def clear_data(self):
        """清除数据"""
        self.time_data.clear()
        self.distance_data.clear()
        self.dcpa_data.clear()
        self.tcpa_data.clear()

    def add_data(self, time: float, distance: float, dcpa: float, tcpa: float):
        """添加数据点"""
        self.time_data.append(time)
        self.distance_data.append(distance)
        self.dcpa_data.append(dcpa)
        self.tcpa_data.append(tcpa)

    def update_chart(self):
        """更新图表"""
        self.delete('chart')

        if len(self.time_data) < 2:
            return

        plot_width = self.width - self.margin_left - self.margin_right
        plot_height = self.height - self.margin_top - self.margin_bottom

        # 计算数据范围
        time_min = min(self.time_data)
        time_max = max(self.time_data)
        time_range = max(time_max - time_min, 10)

        all_distances = list(self.distance_data) + list(self.dcpa_data)
        dist_max = max(all_distances) if all_distances else 1000
        dist_min = 0
        dist_range = max(dist_max - dist_min, 100)

        def to_screen(t, d):
            x = self.margin_left + (t - time_min) / time_range * plot_width
            y = self.height - self.margin_bottom - \
                (d - dist_min) / dist_range * plot_height
            return x, y

        # 绘制距离曲线
        if len(self.distance_data) >= 2:
            points = []
            for t, d in zip(self.time_data, self.distance_data):
                x, y = to_screen(t, d)
                points.extend([x, y])
            self.create_line(points, fill='#0066CC', width=2,
                             smooth=True, tags='chart')

        # 绘制DCPA曲线
        if len(self.dcpa_data) >= 2:
            points = []
            for t, d in zip(self.time_data, self.dcpa_data):
                x, y = to_screen(t, d)
                points.extend([x, y])
            self.create_line(points, fill='#CC0000', width=2,
                             dash=(4, 2), smooth=True, tags='chart')

        # Y轴刻度
        for i in range(5):
            d = dist_min + (dist_range * i / 4)
            _, y = to_screen(time_min, d)
            self.create_text(self.margin_left - 5, y,
                             text=f'{d:.0f}', fill='#606060',
                             font=('Arial', 8), anchor=tk.E, tags='chart')
            self.create_line(self.margin_left - 3, y, self.margin_left, y,
                             fill='#808080', tags='chart')

        # X轴刻度
        for i in range(5):
            t = time_min + (time_range * i / 4)
            x, _ = to_screen(t, dist_min)
            self.create_text(x, self.height - self.margin_bottom + 12,
                             text=f'{t:.0f}', fill='#606060',
                             font=('Arial', 8), tags='chart')
            self.create_line(x, self.height - self.margin_bottom,
                             x, self.height - self.margin_bottom + 3,
                             fill='#808080', tags='chart')

        # 当前值标注
        if self.distance_data and self.dcpa_data:
            current_dist = self.distance_data[-1]
            current_dcpa = self.dcpa_data[-1]
            current_tcpa = self.tcpa_data[-1] if self.tcpa_data else 0

            info_text = f'Dist: {current_dist:.0f}m\nDCPA: {current_dcpa:.0f}m\nTCPA: {current_tcpa:.1f}s'
            self.create_text(
                self.width - self.margin_right - 5, self.margin_top + 10,
                text=info_text, fill='#202020', font=('Arial', 9),
                anchor=tk.NE, justify=tk.RIGHT, tags='chart'
            )


class ControlPanel(ttk.Frame):
    """控制面板"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.engine: Optional[SimulationEngine] = None
        self.on_scenario_change = None

        self._create_widgets()

    def _create_widgets(self):
        """创建控件"""
        # 场景选择
        scenario_frame = ttk.LabelFrame(self, text="Scenario")
        scenario_frame.pack(fill=tk.X, padx=5, pady=5)

        self.scenario_var = tk.StringVar(value="head_on")
        scenarios = [
            ("Head-on", "head_on"),
            ("Crossing", "crossing"),
            ("Overtaking", "overtaking"),
            ("Multi-vessel (Fixed)", "multi"),
        ]

        for text, value in scenarios:
            ttk.Radiobutton(
                scenario_frame, text=text, value=value,
                variable=self.scenario_var,
                command=self._on_scenario_select
            ).pack(anchor=tk.W, padx=5)

        # 随机场景
        ttk.Separator(scenario_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=3)
        ttk.Label(scenario_frame, text="Random Scenarios:",
                  font=('Arial', 9, 'bold')).pack(anchor=tk.W, padx=5)

        random_scenarios = [
            ("Random 3 Ships", "random_3"),
            ("Random 4 Ships", "random_4"),
            ("Random 5 Ships", "random_5"),
        ]

        for text, value in random_scenarios:
            ttk.Radiobutton(
                scenario_frame, text=text, value=value,
                variable=self.scenario_var,
                command=self._on_scenario_select
            ).pack(anchor=tk.W, padx=5)

        # 重新生成随机场景按钮
        self.regenerate_btn = ttk.Button(
            scenario_frame, text="Regenerate Random",
            command=self._on_regenerate_random
        )
        self.regenerate_btn.pack(anchor=tk.W, padx=5, pady=3)

        # 规划器选择
        planner_frame = ttk.LabelFrame(self, text="Planner")
        planner_frame.pack(fill=tk.X, padx=5, pady=5)

        self.planner_var = tk.BooleanVar(value=True)
        ttk.Radiobutton(
            planner_frame, text="Time-Extended HRVO",
            variable=self.planner_var, value=True,
            command=self._on_planner_change
        ).pack(anchor=tk.W, padx=5)
        ttk.Radiobutton(
            planner_frame, text="Traditional HRVO",
            variable=self.planner_var, value=False,
            command=self._on_planner_change
        ).pack(anchor=tk.W, padx=5)

        # 航向恢复选项
        self.heading_recovery_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            planner_frame, text="Heading Recovery",
            variable=self.heading_recovery_var,
            command=self._on_heading_recovery_change
        ).pack(anchor=tk.W, padx=5)

        # 参数设置
        param_frame = ttk.LabelFrame(self, text="Parameters")
        param_frame.pack(fill=tk.X, padx=5, pady=5)

        # 规划时域
        ttk.Label(param_frame, text="T_p (s):").grid(
            row=0, column=0, sticky=tk.W, padx=5)
        self.tp_var = tk.DoubleVar(value=30.0)
        self.tp_scale = ttk.Scale(
            param_frame, from_=5, to=60, variable=self.tp_var,
            orient=tk.HORIZONTAL, command=self._on_param_change
        )
        self.tp_scale.grid(row=0, column=1, sticky=tk.EW, padx=5)
        self.tp_label = ttk.Label(param_frame, text="30.0")
        self.tp_label.grid(row=0, column=2, padx=5)

        # 响应时间
        ttk.Label(param_frame, text="tau (s):").grid(
            row=1, column=0, sticky=tk.W, padx=5)
        self.tau_var = tk.DoubleVar(value=10.0)
        self.tau_scale = ttk.Scale(
            param_frame, from_=1, to=30, variable=self.tau_var,
            orient=tk.HORIZONTAL, command=self._on_param_change
        )
        self.tau_scale.grid(row=1, column=1, sticky=tk.EW, padx=5)
        self.tau_label = ttk.Label(param_frame, text="10.0")
        self.tau_label.grid(row=1, column=2, padx=5)

        # 仿真速度
        ttk.Label(param_frame, text="Speed:").grid(
            row=2, column=0, sticky=tk.W, padx=5)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = ttk.Scale(
            param_frame, from_=0.1, to=5.0, variable=self.speed_var,
            orient=tk.HORIZONTAL
        )
        self.speed_scale.grid(row=2, column=1, sticky=tk.EW, padx=5)
        self.speed_label = ttk.Label(param_frame, text="1.0x")
        self.speed_label.grid(row=2, column=2, padx=5)

        param_frame.columnconfigure(1, weight=1)

        # 控制按钮
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)

        self.start_btn = ttk.Button(
            btn_frame, text="Start", command=self._on_start)
        self.start_btn.pack(side=tk.LEFT, padx=2)

        self.pause_btn = ttk.Button(
            btn_frame, text="Pause", command=self._on_pause, state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=2)

        self.reset_btn = ttk.Button(
            btn_frame, text="Reset", command=self._on_reset)
        self.reset_btn.pack(side=tk.LEFT, padx=2)

        self.step_btn = ttk.Button(
            btn_frame, text="Step", command=self._on_step)
        self.step_btn.pack(side=tk.LEFT, padx=2)

    def set_engine(self, engine: SimulationEngine):
        """设置仿真引擎"""
        self.engine = engine

    def _on_scenario_select(self):
        """场景选择回调"""
        if self.on_scenario_change:
            self.on_scenario_change(self.scenario_var.get())

    def _on_regenerate_random(self):
        """重新生成随机场景"""
        scenario = self.scenario_var.get()
        if scenario.startswith("random_"):
            if self.on_scenario_change:
                self.on_scenario_change(scenario)

    def _on_heading_recovery_change(self):
        """航向恢复开关"""
        if self.engine:
            self.engine.heading_recovery_enabled = self.heading_recovery_var.get()

    def _on_planner_change(self):
        """规划器切换回调"""
        if self.engine:
            self.engine.set_planner_type(self.planner_var.get())

    def _on_param_change(self, *args):
        """参数变化回调"""
        tp = self.tp_var.get()
        tau = self.tau_var.get()

        self.tp_label.config(text=f"{tp:.1f}")
        self.tau_label.config(text=f"{tau:.1f}")

        if self.engine:
            self.engine.T_p = tp
            self.engine.tau = tau
            if self.engine.use_time_extended:
                from time_extended_hrvo.planner.te_hrvo_planner import TimeExtendedHRVOPlanner
                self.engine.planner = TimeExtendedHRVOPlanner(T_p=tp, tau=tau)

    def _on_start(self):
        """开始仿真"""
        if self.engine:
            self.engine.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)

    def _on_pause(self):
        """暂停仿真"""
        if self.engine:
            self.engine.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)

    def _on_reset(self):
        """重置仿真"""
        if self.engine:
            self.engine.reset()
            self.engine.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)

    def _on_step(self):
        """单步执行"""
        if self.engine:
            self.engine.step()


class InfoPanel(ttk.Frame):
    """信息显示面板"""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self._create_widgets()

    def _create_widgets(self):
        """创建控件"""
        # 仿真信息
        info_frame = ttk.LabelFrame(self, text="Simulation Info")
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        self.time_label = ttk.Label(info_frame, text="Time: 0.0 s")
        self.time_label.pack(anchor=tk.W, padx=5)

        self.collision_label = ttk.Label(info_frame, text="Min Distance: -- m")
        self.collision_label.pack(anchor=tk.W, padx=5)

        self.status_label = ttk.Label(info_frame, text="Status: Ready")
        self.status_label.pack(anchor=tk.W, padx=5)

        # 会遇信息
        encounter_frame = ttk.LabelFrame(self, text="Encounter Info")
        encounter_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.encounter_text = tk.Text(encounter_frame, height=8, width=28,
                                      font=('Consolas', 9))
        self.encounter_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 策略信息
        strategy_frame = ttk.LabelFrame(self, text="Current Strategy")
        strategy_frame.pack(fill=tk.X, padx=5, pady=5)

        self.strategy_label = ttk.Label(strategy_frame, text="No strategy",
                                        font=('Arial', 10))
        self.strategy_label.pack(anchor=tk.W, padx=5, pady=5)

        # 转向方向说明
        ttk.Label(strategy_frame, text="(+: Starboard/Right, -: Port/Left)",
                  font=('Arial', 8), foreground='#606060').pack(anchor=tk.W, padx=5)

    def update_info(self, engine: SimulationEngine):
        """更新信息显示"""
        # 更新时间
        self.time_label.config(text=f"Time: {engine.time:.1f} s")

        # 更新最小距离
        if engine.collision_distance < float('inf'):
            self.collision_label.config(
                text=f"Min Distance: {engine.collision_distance:.1f} m")

        # 获取本船状态
        own = engine.get_own_ship()

        # 更新状态
        if engine.collision_occurred:
            self.status_label.config(
                text="Status: COLLISION!", foreground='red')
        elif own and own.is_avoiding:
            self.status_label.config(
                text="Status: AVOIDING", foreground='#CC6600')
        elif engine.is_running:
            self.status_label.config(
                text="Status: Running", foreground='green')
        else:
            self.status_label.config(
                text="Status: Paused", foreground='#666666')

        # 更新会遇信息
        info = engine.get_encounter_info()
        self.encounter_text.delete(1.0, tk.END)

        if info:
            text = f"Own Ship:\n"
            text += f"  Pos: ({info['own_position'][0]:.0f}, {info['own_position'][1]:.0f})\n"
            text += f"  Speed: {info['own_speed']:.1f} m/s\n"
            heading_deg = np.rad2deg(np.arctan2(info.get('own_velocity', [0, 1])[1],
                                                info.get('own_velocity', [1, 0])[0]))
            nav_heading = (90 - heading_deg) % 360
            text += f"  Heading: {nav_heading:.0f} deg\n"
            # 显示避让状态
            if info.get('is_avoiding', False):
                text += f"  Mode: AVOIDING\n\n"
            else:
                text += f"  Mode: Normal\n\n"

            for obs in info.get('obstacles', []):
                text += f"{obs['name']}:\n"
                text += f"  Distance: {obs.get('distance', 0):.0f} m\n"
                text += f"  DCPA: {obs['dcpa']:.0f} m\n"
                text += f"  TCPA: {obs['tcpa']:.1f} s\n"
                text += f"  Type: {obs['encounter_type']}\n\n"

            self.encounter_text.insert(1.0, text)

        # 更新策略（使用航海惯例：右转为正）
        if engine.current_strategy:
            s = engine.current_strategy
            # 原代码中负值是右转，需要转换为航海惯例（正值为右转）
            delta_psi_nav = -np.rad2deg(s.delta_psi)  # 转换方向
            turn_dir = "Stbd" if delta_psi_nav > 0 else "Port" if delta_psi_nav < 0 else "None"
            self.strategy_label.config(
                text=f"Turn: {delta_psi_nav:+.1f} deg ({turn_dir}), Speed: {s.delta_speed:+.1f} m/s"
            )
        else:
            self.strategy_label.config(text="No strategy")


class HRVOSimulatorApp:
    """
    Time-Extended HRVO 仿真器主应用
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(
            "Time-Extended HRVO Ship Collision Avoidance Simulator")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 700)

        # 创建仿真引擎
        self.engine = SimulationEngine(use_time_extended=True)

        # 创建界面
        self._create_ui()

        # 初始化场景
        self._load_scenario("head_on")

        # 设置回调
        self.engine.on_update = self._on_engine_update
        self.engine.on_collision = self._on_collision

        # 启动更新循环
        self._update_loop()

    def _create_ui(self):
        """创建用户界面"""
        # 主布局
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧：位置空间
        left_frame = ttk.LabelFrame(
            main_frame, text="Position Space (Scroll to zoom, Drag to pan)")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.pos_canvas = PositionSpaceCanvas(
            left_frame, width=700, height=600)
        self.pos_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 中间：速度空间和图表
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)

        # 速度空间
        vel_frame = ttk.LabelFrame(
            middle_frame, text="Velocity Space / HRVO (Scroll to zoom)")
        vel_frame.pack(fill=tk.BOTH, expand=True)

        self.vel_canvas = VelocitySpaceCanvas(vel_frame, width=380, height=380)
        self.vel_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 图例
        legend_frame = ttk.Frame(vel_frame)
        legend_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Label(legend_frame, text="*", foreground='#FF6666',
                  font=('Arial', 14)).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="HRVO", font=(
            'Arial', 9)).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(legend_frame, text="*", foreground='#0066CC',
                  font=('Arial', 14)).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="Current", font=(
            'Arial', 9)).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(legend_frame, text="*", foreground='#00AA00',
                  font=('Arial', 14)).pack(side=tk.LEFT)
        ttk.Label(legend_frame, text="Target",
                  font=('Arial', 9)).pack(side=tk.LEFT)

        # 相对运动图表
        chart_frame = ttk.LabelFrame(
            middle_frame, text="Relative Motion Chart")
        chart_frame.pack(fill=tk.X, pady=5)

        self.rel_chart = RelativeMotionChart(
            chart_frame, width=380, height=180)
        self.rel_chart.pack(fill=tk.X, padx=5, pady=5)

        # 右侧：控制和信息
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        # 控制面板
        self.control_panel = ControlPanel(right_frame)
        self.control_panel.pack(fill=tk.X)
        self.control_panel.set_engine(self.engine)
        self.control_panel.on_scenario_change = self._load_scenario

        # 信息面板
        self.info_panel = InfoPanel(right_frame)
        self.info_panel.pack(fill=tk.BOTH, expand=True, pady=10)

    def _load_scenario(self, scenario_name: str):
        """加载场景"""
        self.engine.is_running = False

        if scenario_name == "head_on":
            create_head_on_scenario(self.engine)
        elif scenario_name == "crossing":
            create_crossing_scenario(self.engine)
        elif scenario_name == "overtaking":
            create_overtaking_scenario(self.engine)
        elif scenario_name == "multi":
            create_multi_vessel_scenario(self.engine)
        elif scenario_name == "random_3":
            create_random_3_vessel_scenario(self.engine)
        elif scenario_name == "random_4":
            create_random_4_vessel_scenario(self.engine)
        elif scenario_name == "random_5":
            create_random_5_vessel_scenario(self.engine)

        self.engine.reset()
        self.rel_chart.clear_data()
        self._redraw()

    def _on_engine_update(self):
        """引擎更新回调"""
        self._update_chart_data()
        self._redraw()

    def _on_collision(self, own, obs):
        """碰撞回调"""
        self.engine.is_running = False
        self.control_panel.start_btn.config(state=tk.NORMAL)
        self.control_panel.pause_btn.config(state=tk.DISABLED)

    def _update_chart_data(self):
        """更新图表数据"""
        own = self.engine.get_own_ship()
        if not own:
            return

        # 获取第一个目标船
        targets = [v for v in self.engine.vessels if not v.is_own_ship]
        if not targets:
            return

        target = targets[0]

        # 计算距离
        rel_pos = target.state.p - own.state.p
        distance = np.linalg.norm(rel_pos)

        # 计算DCPA/TCPA
        rel_vel = target.state.v - own.state.v
        rel_speed_sq = np.dot(rel_vel, rel_vel)

        if rel_speed_sq > 0.01:
            tcpa = -np.dot(rel_pos, rel_vel) / rel_speed_sq
            tcpa = max(0, tcpa)
            cpa_pos = rel_pos + rel_vel * tcpa
            dcpa = np.linalg.norm(cpa_pos)
        else:
            tcpa = 0
            dcpa = distance

        # 添加数据
        self.rel_chart.add_data(self.engine.time, distance, dcpa, tcpa)

    def _redraw(self):
        """重绘界面"""
        # 更新位置空间
        self.pos_canvas.clear_vessels()
        self.pos_canvas.draw_all(self.engine.vessels, show_rel_motion=True)

        # 更新速度空间
        self.vel_canvas.draw_all(self.engine)

        # 更新图表
        self.rel_chart.update_chart()

        # 更新信息
        self.info_panel.update_info(self.engine)

    def _update_loop(self):
        """更新循环"""
        if self.engine.is_running and not self.engine.collision_occurred:
            # 根据仿真速度调整
            speed = self.control_panel.speed_var.get()
            self.engine.dt = 0.1 * speed

            self.engine.step()

        # 更新速度标签
        speed = self.control_panel.speed_var.get()
        self.control_panel.speed_label.config(text=f"{speed:.1f}x")

        # 继续循环
        self.root.after(50, self._update_loop)

    def run(self):
        """运行应用"""
        self.root.mainloop()


def main():
    """主函数"""
    app = HRVOSimulatorApp()
    app.run()


if __name__ == "__main__":
    main()
