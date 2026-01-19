"""
run_tests.py - 运行所有测试
"""
import unittest
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def run_all_tests():
    """运行所有测试"""
    # 发现并运行 tests 目录下的所有测试
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回是否全部通过
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
