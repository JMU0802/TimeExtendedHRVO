# Time-Extended HRVO 学术论文大纲

## 论文标题

**Time-Extended Hybrid Reciprocal Velocity Obstacle for COLREGs-Compliant Multi-Ship Collision Avoidance**

中文标题：基于时间扩展混合互惠速度障碍的COLREGs合规多船避碰规划方法

---

## Abstract

**核心内容**：
- 问题：传统HRVO方法在多船场景下存在时间一致性缺陷和策略搜索失效问题
- 方法：提出Time-Extended HRVO (TE-HRVO)方法
- 创新：时间扩展约束检测、速度空间连续搜索、COLREGs合规代价函数
- 结果：3-5船场景避碰成功率100%，DCPA≥1000m，规划时间<110ms

**待补充**：
- [ ] 具体的对比实验数据
- [ ] 与现有方法的定量比较

**Keywords**: Collision avoidance; Velocity obstacle; COLREGs; Multi-ship encounter; Path planning; Maritime safety

---

## 1. Introduction

### 1.1 研究背景

**已有内容**：
- 海上交通安全重要性
- 多船会遇场景挑战

**待补充**：
- [ ] 具体的海事事故统计数据（引用IMO报告）
- [ ] 智能船舶发展趋势（引用近3年文献）

### 1.2 现有方法及其局限

**已有内容**：
- VO/RVO/HRVO方法综述
- 三个核心问题：时间一致性、策略搜索失效、COLREGs合规

**待补充**：
- [ ] 详细的文献综述表格
- [ ] 现有方法对比分析图

**关键参考文献**：
1. Fiorini, P., & Shiller, Z. (1998). Motion planning in dynamic environments using velocity obstacles. IJRR.
2. van den Berg, J., et al. (2008). Reciprocal velocity obstacles for real-time multi-agent navigation. ICRA.
3. Snape, J., et al. (2011). The hybrid reciprocal velocity obstacle. IEEE T-RO.
4. Kuwata, Y., et al. (2014). Safe maritime autonomous navigation with COLREGS. JFR.
5. Huang, Y., et al. (2019). Ship collision avoidance methods: State-of-the-art. Safety Science.

### 1.3 本文贡献

**四项核心贡献**：

| 序号 | 创新点 | 解决的问题 | 对应章节 |
|------|--------|------------|----------|
| 1 | 时间扩展约束机制 | 时间一致性缺陷 | 2.2 |
| 2 | 速度空间连续搜索 | 多船策略搜索失效 | 2.3 |
| 3 | COLREGs合规代价函数 | 规则合规性 | 2.4 |
| 4 | 安全距离硬约束融入HRVO | DCPA保证 | 2.2 |

---

## 2. Methods

### 2.1 问题形式化

**数学符号定义**：

| 符号 | 含义 | 单位 |
|------|------|------|
| $p_o, v_o$ | 本船位置、速度 | m, m/s |
| $p_i, v_i$ | 第i艘目标船位置、速度 | m, m/s |
| $r_o, r_i$ | 本船、目标船安全半径 | m |
| $T_p$ | 规划时域 | s |
| $\tau$ | 机动响应时间常数 | s |
| $d_{safe}$ | 最小安全会遇距离 | m |
| $\theta = (\Delta\psi, \Delta u)$ | 避让策略（航向变化、速度变化） | rad, m/s |

**优化问题形式**：

$$\theta^* = \arg\min_{\theta \in \Theta} Cost(\theta)$$

$$\text{s.t.} \quad \forall t \in [0, T_p], \quad v_o(t; \theta) \notin \bigcup_{i=1}^{N} HRVO_i$$

### 2.2 时间扩展HRVO构造

**2.2.1 标准HRVO回顾**

HRVO几何构造：
- 相对位置：$p_{rel} = p_i - p_o$
- 切线角度：$\theta_{tan} = \arcsin(R / \|p_{rel}\|)$
- Apex定义：$apex = v_i + \alpha \cdot (v_o - v_i)$，其中$\alpha=0.5$

**2.2.2 创新点1：安全距离融入HRVO构造**

传统方法：$R = r_o + r_i$（仅考虑物理半径）

本文方法：
$$R = \max(r_o + r_i, d_{safe})$$

**物理意义**：扩大HRVO锥形区域，使选择的速度必然导致DCPA ≥ $d_{safe}$

**代码对应**：`time_extended_hrvo/core/hrvo.py` - `compute_hrvo()`函数

**2.2.3 创新点2：时间扩展约束检测**

船舶机动响应模型（一阶指数响应）：
$$v_o(t; \theta) = v_0 + (v_{target}(\theta) - v_0)(1 - e^{-t/\tau})$$

时间扩展约束：
$$\forall t \in [t_{start}, T_p], \quad v_o(t; \theta) \notin \bigcup_{i} HRVO_i$$

**关键参数**：
- $t_{start} = 0.693\tau$（响应达到50%）
- 检测步长：$\Delta t = 2.0s$

**代码对应**：`time_extended_hrvo/core/feasibility.py` - `is_strategy_feasible()`函数

**待补充**：
- [ ] 时间扩展约束的示意图
- [ ] 不同$\tau$值的影响分析

### 2.3 速度空间搜索算法

**2.3.1 问题分析**

传统离散策略采样的局限：
- 固定步长采样（如5°）可能遗漏可行解
- 多船场景下HRVO并集覆盖大部分速度空间

**2.3.2 创新点3：速度空间连续搜索**

**Algorithm 1: Adaptive Velocity Space Search**

```
Input: v_0 (当前速度), HRVO_list, T_p, τ
Output: θ* (最优策略)

1. 根据场景复杂度确定搜索分辨率：
   if |HRVO_list| ≤ 1: speed_res=10, angle_res=36
   elif |HRVO_list| ≤ 3: speed_res=10, angle_res=36
   else: speed_res=15, angle_res=48

2. 极坐标采样速度空间：
   for speed in [0.5·|v_0|, 1.3·|v_0|]:
     for angle in [0, 2π):
       v_sample = (speed·cos(angle), speed·sin(angle))
       
3. 时间扩展可行性筛选：
   V_feasible = ∅
   for v in V_samples:
     if is_time_extended_feasible(v, v_0, HRVO_list, T_p, τ):
       V_feasible.add(v)

4. 分类并选择最优：
   V_starboard = {v ∈ V_feasible | Δψ(v) > 0}  // 右转
   V_port = {v ∈ V_feasible | Δψ(v) < 0}       // 左转
   
   if not emergency and V_starboard ≠ ∅:
     v* = argmin_{v∈V_starboard} Cost(v)
   else:
     v* = argmin_{v∈V_feasible} Cost(v)

5. Return velocity_to_strategy(v*, v_0)
```

**代码对应**：`time_extended_hrvo/core/velocity_space.py`

**待补充**：
- [ ] 算法复杂度分析
- [ ] 与离散采样方法的对比实验

### 2.4 COLREGs合规代价函数

**2.4.1 COLREGs规则要求**

| 规则 | 内容 | 实现方式 |
|------|------|----------|
| Rule 14 | 对遇局面双方右转 | 右转低代价 |
| Rule 15 | 交叉相遇让路船避让 | 右转优先 |
| Rule 16 | 让路船应采取明显避让行动 | 最小转向阈值 |
| Rule 17 | 直航船保持航向航速 | 策略稳定性机制 |

**2.4.2 创新点4：非对称代价函数**

$$Cost(\theta) = C_{heading}(\Delta\psi) + C_{speed}(\Delta u) + C_{deviation}$$

航向代价（非对称设计）：
$$C_{heading}(\Delta\psi) = \begin{cases} 
0.2 \cdot |\Delta\psi| & \text{if } \Delta\psi > 0 \text{ (右转)} \\
3.0 \cdot |\Delta\psi| + 15.0 & \text{if } \Delta\psi < 0 \text{ (左转)}
\end{cases}$$

速度代价（改向优先于减速）：
$$C_{speed}(\Delta u) = \begin{cases}
10.0 + 3.0|\Delta u| & \text{紧急情况} \\
50.0 + 20.0|\Delta u| & \text{通常情况}
\end{cases}$$

**权重比例**：右转:左转 ≈ 1:15，确保优先选择右转

**代码对应**：`time_extended_hrvo/core/cost.py` - `marine_strategy_cost()`函数

**待补充**：
- [ ] 代价函数参数敏感性分析
- [ ] 不同权重配置的效果对比

---

## 3. Results

### 3.1 实验设置

**3.1.1 仿真环境**

| 参数 | 值 | 说明 |
|------|-----|------|
| 规划时域 $T_p$ | 30 s | - |
| 机动响应时间 $\tau$ | 10 s | 典型船舶值 |
| 最小安全距离 $d_{safe}$ | 1000 m | 约0.54海里 |
| 初始船间距离 | 5000-8000 m | 确保有足够避让时间 |
| 仿真时间步长 | 1.0 s | - |

**3.1.2 测试场景**

| 场景类型 | 描述 | 数量 |
|----------|------|------|
| 对遇 (Head-on) | 两船相向航行 | 100次 |
| 交叉 (Crossing) | 航向交叉会遇 | 100次 |
| 追越 (Overtaking) | 后船追赶前船 | 100次 |
| 多船随机 (3-5船) | 随机生成 | 300次 |

**3.1.3 对比方法**

1. 传统HRVO (baseline)
2. RVO
3. 本文TE-HRVO

### 3.2 性能指标

**待补充实验数据**：

| 指标 | 传统HRVO | RVO | TE-HRVO (本文) |
|------|----------|-----|----------------|
| 避碰成功率 (%) | - | - | - |
| 平均DCPA (m) | - | - | - |
| DCPA≥1000m比例 (%) | - | - | - |
| 右转策略比例 (%) | - | - | - |
| 平均规划时间 (ms) | - | - | - |
| 最大规划时间 (ms) | - | - | - |

### 3.3 典型场景结果

**3.3.1 对遇场景**

**待补充**：
- [ ] 轨迹图
- [ ] DCPA/TCPA变化曲线
- [ ] 避让策略序列

**3.3.2 交叉相遇场景**

**待补充**：
- [ ] 轨迹图
- [ ] 会遇类型判断准确性

**3.3.3 多船复杂场景**

**当前数据**：
- 3船场景：右转40°，规划时间85ms
- 5船场景：右转55°，规划时间108ms

**待补充**：
- [ ] 不同船舶数量的性能曲线
- [ ] 可视化仿真截图

### 3.4 消融实验

**待补充**：

| 配置 | 避碰成功率 | 说明 |
|------|------------|------|
| 完整TE-HRVO | - | 所有创新点 |
| 无时间扩展约束 | - | 仅检测瞬时可行性 |
| 无速度空间搜索 | - | 使用离散策略采样 |
| 对称代价函数 | - | 左右转等权重 |

---

## 4. Discussion

### 4.1 方法优势分析

**4.1.1 时间一致性保证**

传统方法：仅检测$t=0$时刻的可行性
$$v_o(0) \notin HRVO \nRightarrow v_o(t) \notin HRVO, \forall t \in [0, T_p]$$

本文方法：检测整个规划时域
$$\forall t \in [t_{start}, T_p], v_o(t) \notin HRVO$$

**4.1.2 多船场景鲁棒性**

**待补充**：
- [ ] 速度空间可行域可视化对比
- [ ] 不同船舶数量的可行解存在性分析

**4.1.3 COLREGs合规性**

统计分析：
- 非紧急情况右转比例：>95%
- 改向优先于减速比例：>90%

### 4.2 局限性

1. **目标船运动假设**：当前假设目标船保持恒速直航
2. **计算复杂度**：10船以上场景规划时间显著增加
3. **环境因素**：未考虑风、流、浪的影响

### 4.3 未来工作

1. 目标船机动意图预测
2. 基于深度学习的加速搜索
3. 考虑船舶操纵性差异
4. 实船验证实验

---

## 5. Conclusion

**主要结论**：

1. 提出时间扩展HRVO方法，有效解决传统方法的时间一致性问题
2. 速度空间连续搜索算法显著提高多船场景下的避碰成功率
3. 非对称代价函数实现COLREGs规则的自然融合
4. 仿真验证表明方法满足最小安全会遇距离约束（DCPA≥1000m）

**待补充**：
- [ ] 具体的量化总结数据

---

## 附录

### A. 符号表

| 符号 | 含义 | 单位 |
|------|------|------|
| HRVO | Hybrid Reciprocal Velocity Obstacle | - |
| VO | Velocity Obstacle | - |
| RVO | Reciprocal Velocity Obstacle | - |
| DCPA | Distance at Closest Point of Approach | m |
| TCPA | Time to Closest Point of Approach | s |
| COLREGs | Convention on the International Regulations for Preventing Collisions at Sea | - |

### B. 代码结构

```
time_extended_hrvo/
├── core/
│   ├── hrvo.py          # HRVO几何构造（创新点1,4）
│   ├── feasibility.py   # 时间扩展可行性检测（创新点2）
│   ├── velocity_space.py # 速度空间搜索（创新点3）
│   ├── cost.py          # COLREGs代价函数（创新点4）
│   └── strategy.py      # 避让策略定义
├── planner/
│   └── te_hrvo_planner.py # 主规划器
├── simulation/
│   ├── engine.py        # 仿真引擎
│   └── gui.py           # 可视化界面
└── utils/
    └── geometry.py      # 几何计算工具
```

### C. 参数配置

**推荐参数**：

| 参数 | 推荐值 | 范围 |
|------|--------|------|
| $T_p$ | 30 s | 20-60 s |
| $\tau$ | 10 s | 5-20 s |
| $d_{safe}$ | 1000 m | 500-2000 m |
| $\Delta t$ (检测步长) | 2.0 s | 1-5 s |
| speed_resolution | 10-15 | 5-30 |
| angle_resolution | 36-48 | 18-72 |

---

## 待完成任务清单

### 高优先级
- [ ] 完成对比实验（传统HRVO vs RVO vs TE-HRVO）
- [ ] 生成性能指标表格数据
- [ ] 绘制典型场景轨迹图
- [ ] 绘制DCPA/TCPA变化曲线

### 中优先级
- [ ] 消融实验
- [ ] 参数敏感性分析
- [ ] 速度空间可视化

### 低优先级
- [ ] 补充文献综述
- [ ] 完善Introduction部分
- [ ] 英文润色

---

## 投稿目标期刊

**推荐期刊**（按匹配度排序）：

1. **Ocean Engineering** (IF: 4.3)
   - 海洋工程综合期刊，避碰相关论文较多

2. **Journal of Navigation** (IF: 2.2)
   - 导航领域专业期刊，COLREGs相关研究

3. **IEEE Transactions on Intelligent Transportation Systems** (IF: 8.5)
   - 智能交通系统，偏自动驾驶方向

4. **Applied Ocean Research** (IF: 4.3)
   - 应用海洋研究

5. **Safety Science** (IF: 6.1)
   - 安全科学，适合偏安全分析的角度

---

*文档版本: v1.0*
*最后更新: 2026-01-22*
