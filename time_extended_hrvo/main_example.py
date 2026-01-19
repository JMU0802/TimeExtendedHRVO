"""
main_example.py - Time-Extended HRVO 示例

演示如何使用 Time-Extended HRVO 进行船舶避碰规划
"""
from time_extended_hrvo.utils.geometry import compute_dcpa_tcpa, classify_encounter
from time_extended_hrvo.planner.te_hrvo_planner import (
    TimeExtendedHRVOPlanner,
    TraditionalHRVOPlanner
)
from time_extended_hrvo.core.vessel import VesselState
import numpy as np
import sys
import os

# 添加包路径（支持从包内部或外部运行）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def example_head_on_encounter():
    """
    示例 1: 对遇场景

    两船相向航行，需要各自右转避让
    """
    print("=" * 60)
    print("示例 1: 对遇场景 (Head-on Encounter)")
    print("=" * 60)

    # 本船: 向北航行
    own_ship = VesselState(
        position=[0, 0],
        velocity=[0, 5],  # 5 m/s 向北
        radius=50
    )

    # 目标船: 向南航行
    target_ship = VesselState(
        position=[100, 1000],  # 在本船前方 1000m
        velocity=[0, -5],  # 5 m/s 向南
        radius=50
    )

    # 计算 DCPA/TCPA
    dcpa, tcpa = compute_dcpa_tcpa(
        own_ship.p, own_ship.v,
        target_ship.p, target_ship.v
    )
    print(f"初始 DCPA: {dcpa:.1f} m")
    print(f"初始 TCPA: {tcpa:.1f} s")

    # 会遇类型分类
    encounter = classify_encounter(own_ship, target_ship)
    print(f"会遇类型: {encounter}")

    # Time-Extended HRVO 规划
    te_planner = TimeExtendedHRVOPlanner(T_p=30.0)
    te_strategy = te_planner.plan(own_ship, [target_ship])

    if te_strategy:
        print(f"\nTime-Extended HRVO 策略: {te_strategy}")
        v_new = te_strategy.get_final_velocity(own_ship.v)
        print(f"新速度向量: [{v_new[0]:.2f}, {v_new[1]:.2f}] m/s")
    else:
        print("\nTime-Extended HRVO: 无可行策略")

    # 传统 HRVO 规划（对比）
    trad_planner = TraditionalHRVOPlanner()
    trad_strategy = trad_planner.plan(own_ship, [target_ship])

    if trad_strategy:
        print(f"\n传统 HRVO 策略: {trad_strategy}")
        v_trad = trad_strategy.get_final_velocity(own_ship.v)
        print(f"新速度向量: [{v_trad[0]:.2f}, {v_trad[1]:.2f}] m/s")
    else:
        print("\n传统 HRVO: 无可行策略")

    print()


def example_crossing_encounter():
    """
    示例 2: 交叉相遇场景

    目标船在本船右舷，本船为让路船
    """
    print("=" * 60)
    print("示例 2: 交叉相遇场景 (Crossing Encounter)")
    print("=" * 60)

    # 本船: 向东航行
    own_ship = VesselState(
        position=[0, 0],
        velocity=[5, 0],  # 5 m/s 向东
        radius=50
    )

    # 目标船: 向北航行（在本船右前方）
    target_ship = VesselState(
        position=[500, -300],  # 右前方
        velocity=[0, 5],  # 5 m/s 向北
        radius=50
    )

    # 计算 DCPA/TCPA
    dcpa, tcpa = compute_dcpa_tcpa(
        own_ship.p, own_ship.v,
        target_ship.p, target_ship.v
    )
    print(f"初始 DCPA: {dcpa:.1f} m")
    print(f"初始 TCPA: {tcpa:.1f} s")

    # 会遇类型
    encounter = classify_encounter(own_ship, target_ship)
    print(f"会遇类型: {encounter}")

    # Time-Extended HRVO 规划
    te_planner = TimeExtendedHRVOPlanner(T_p=30.0)
    result = te_planner.plan_with_details(
        own_ship, [target_ship], encounter_type=encounter
    )

    print(f"\n可行策略数量: {len(result['feasible_strategies'])}")
    print(f"总策略数量: {len(result['all_strategies'])}")

    if result['best_strategy']:
        print(f"最优策略: {result['best_strategy']}")

    print()


def example_multi_vessel():
    """
    示例 3: 多船会遇场景

    本船需要同时避让多个目标船
    """
    print("=" * 60)
    print("示例 3: 多船会遇场景 (Multi-Vessel Encounter)")
    print("=" * 60)

    # 本船
    own_ship = VesselState(
        position=[0, 0],
        velocity=[5, 0],  # 向东
        radius=50
    )

    # 多个目标船
    targets = [
        VesselState([600, 200], [0, -4], 50),   # 从北方来
        VesselState([700, -100], [-3, 3], 50),  # 从东南来
        VesselState([400, 0], [-4, 0], 50),     # 对遇
    ]

    print(f"目标船数量: {len(targets)}")

    for i, t in enumerate(targets):
        dcpa, tcpa = compute_dcpa_tcpa(own_ship.p, own_ship.v, t.p, t.v)
        print(f"  目标船 {i+1}: DCPA={dcpa:.1f}m, TCPA={tcpa:.1f}s")

    # Time-Extended HRVO 规划
    te_planner = TimeExtendedHRVOPlanner(T_p=30.0)
    result = te_planner.plan_with_details(own_ship, targets)

    print(f"\nHRVO 数量: {len(result['hrvo_list'])}")
    print(
        f"可行策略: {len(result['feasible_strategies'])}/{len(result['all_strategies'])}")

    if result['best_strategy']:
        print(f"最优策略: {result['best_strategy']}")
        v_new = result['best_strategy'].get_final_velocity(own_ship.v)
        print(f"新速度: [{v_new[0]:.2f}, {v_new[1]:.2f}] m/s")
    else:
        print("警告: 无可行策略!")

    print()


def example_time_consistency_comparison():
    """
    示例 4: 时间一致性对比

    展示 Time-Extended HRVO 与传统 HRVO 的区别
    """
    print("=" * 60)
    print("示例 4: 时间一致性对比")
    print("=" * 60)

    # 构造一个传统 HRVO 可能失效的场景
    # 本船机动响应较慢，传统方法可能选择的策略在机动过程中进入 HRVO

    own_ship = VesselState(
        position=[0, 0],
        velocity=[4, 0],
        radius=30
    )

    target_ship = VesselState(
        position=[200, 50],
        velocity=[-3, -1],
        radius=30
    )

    # 传统 HRVO
    trad_planner = TraditionalHRVOPlanner()
    trad_strategy = trad_planner.plan(own_ship, [target_ship])

    # Time-Extended HRVO
    te_planner = TimeExtendedHRVOPlanner(T_p=20.0, tau=8.0)  # 较慢的响应
    te_strategy = te_planner.plan(own_ship, [target_ship])

    print("传统 HRVO (仅检测最终速度):")
    if trad_strategy:
        print(f"  策略: {trad_strategy}")
    else:
        print("  无可行策略")

    print("\nTime-Extended HRVO (检测整个机动过程):")
    if te_strategy:
        print(f"  策略: {te_strategy}")
    else:
        print("  无可行策略")

    # 详细分析
    if trad_strategy and te_strategy:
        print("\n策略差异分析:")
        print(f"  传统 Δψ: {np.rad2deg(trad_strategy.delta_psi):.1f}°")
        print(f"  扩展 Δψ: {np.rad2deg(te_strategy.delta_psi):.1f}°")
        print(f"  传统 Δu: {trad_strategy.delta_speed:.1f} m/s")
        print(f"  扩展 Δu: {te_strategy.delta_speed:.1f} m/s")

    print()


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Time-Extended HRVO 算法演示")
    print("=" * 60 + "\n")

    example_head_on_encounter()
    example_crossing_encounter()
    example_multi_vessel()
    example_time_consistency_comparison()

    print("=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
