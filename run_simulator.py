"""
run_simulator.py - 启动 Time-Extended HRVO 仿真器

用法: python run_simulator.py
"""
from time_extended_hrvo.simulation.gui import HRVOSimulatorApp
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    """启动仿真器"""
    print("Starting Time-Extended HRVO Simulator...")
    print("=" * 50)
    print("Controls:")
    print("  - Select scenario from the control panel")
    print("  - Click 'Start' to begin simulation")
    print("  - Click 'Pause' to pause")
    print("  - Click 'Reset' to reset simulation")
    print("  - Click 'Step' for single step")
    print("  - Mouse wheel: zoom in/out")
    print("  - Mouse drag: pan view")
    print("=" * 50)

    app = HRVOSimulatorApp()
    app.run()


if __name__ == "__main__":
    main()
