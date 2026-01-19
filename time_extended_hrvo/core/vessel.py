"""
vessel.py - 船舶状态与运动模型

对应论文符号:
    p_i: 船舶位置向量
    v_i: 船舶速度向量
    r_i: 船舶安全半径
"""
import numpy as np


class VesselState:
    """
    船舶状态类
    
    Attributes:
        p (np.ndarray): 位置向量 [x, y]
        v (np.ndarray): 速度向量 [vx, vy]
        r (float): 安全半径
    """
    
    def __init__(self, position, velocity, radius):
        """
        初始化船舶状态
        
        Args:
            position: 位置坐标 [x, y]
            velocity: 速度向量 [vx, vy]
            radius: 安全半径 (m)
        """
        self.p = np.asarray(position, dtype=float)
        self.v = np.asarray(velocity, dtype=float)
        self.r = float(radius)
    
    @property
    def speed(self):
        """返回船舶速度大小"""
        return np.linalg.norm(self.v)
    
    @property
    def heading(self):
        """返回船舶航向角 (rad)"""
        return np.arctan2(self.v[1], self.v[0])
    
    def predict_position(self, dt):
        """
        预测 dt 秒后的位置（匀速假设）
        
        Args:
            dt: 预测时间 (s)
            
        Returns:
            np.ndarray: 预测位置
        """
        return self.p + self.v * dt
    
    def copy(self):
        """返回状态的深拷贝"""
        return VesselState(self.p.copy(), self.v.copy(), self.r)
    
    def __repr__(self):
        return f"VesselState(p={self.p}, v={self.v}, r={self.r})"
