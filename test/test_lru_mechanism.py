#!/usr/bin/env python3
"""
测试StreamStateManager的LRU机制
"""

import os
import sys
import torch
import gc

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stream_state_manager import StreamStateManager


def test_lru_mechanism():
    """测试LRU机制是否正常工作"""
    print("=" * 60)
    print("测试LRU机制")
    print("=" * 60)

    # 创建一个最大缓存为3个状态的管理器
    manager = StreamStateManager(max_cached_states=3)

    print(f"创建状态管理器，最大缓存: {manager.max_cached_states}")

    # 创建一些状态数据
    def create_test_state(name):
        return {
            'features': torch.randn(100),  # 小张量用于测试
            'metadata': {'name': name, 'size': 100}
        }

    # 添加5个状态，超过最大缓存限制
    print("\n添加5个状态 (超过最大缓存3)...")
    for i in range(5):
        state = create_test_state(f"state_{i}")
        manager.update_state(state, f"seq_{i}", frame_idx=0, reset_state=True)
        print(f"  添加 seq_{i}, 当前缓存大小: {len(manager.state_cache)}, 访问顺序: {manager.sequence_access_order}")

    # 检查结果
    print(f"\n最终缓存大小: {len(manager.state_cache)}")
    print(f"缓存中的序列: {list(manager.state_cache.keys())}")
    print(f"访问顺序: {manager.sequence_access_order}")

    # 应该只剩下最后3个状态 (seq_2, seq_3, seq_4)
    expected_sequences = ['seq_2', 'seq_3', 'seq_4']
    actual_sequences = list(manager.state_cache.keys())

    if set(expected_sequences) == set(actual_sequences):
        print("✅ LRU机制工作正常！")
        print(f"   正确移除了最早的2个状态，保留了最新的3个状态")
        return True
    else:
        print("❌ LRU机制有问题！")
        print(f"   期望: {expected_sequences}")
        print(f"   实际: {actual_sequences}")
        return False


def test_state_release():
    """测试状态释放功能"""
    print("\n" + "=" * 60)
    print("测试状态释放")
    print("=" * 60)

    manager = StreamStateManager(max_cached_states=2)

    # 创建大张量以观察内存变化
    def create_large_state(name, size_mb=5):
        num_elements = int(size_mb * 1024 * 1024 / 4)  # 假设float32
        return {
            'large_tensor': torch.randn(num_elements),
            'name': name
        }

    print("添加第一个大状态...")
    state1 = create_large_state("state_1", 5)
    manager.update_state(state1, "seq_A", frame_idx=0, reset_state=True)

    print("添加第二个大状态...")
    state2 = create_large_state("state_2", 5)
    manager.update_state(state2, "seq_B", frame_idx=0, reset_state=True)

    print("添加第三个大状态（应触发释放第一个）...")
    state3 = create_large_state("state_3", 5)
    manager.update_state(state3, "seq_C", frame_idx=0, reset_state=True)

    print(f"当前缓存: {list(manager.state_cache.keys())}")

    # 验证第一个状态已被移除
    if "seq_A" not in manager.state_cache and len(manager.state_cache) == 2:
        print("✅ 状态释放机制工作正常！")
        print(f"   正确移除了最早的状态 seq_A")
        return True
    else:
        print("❌ 状态释放机制有问题！")
        return False


def run_tests():
    """运行所有测试"""
    print("StreamStateManager LRU机制测试")
    print("=" * 60)

    results = []

    results.append(test_lru_mechanism())
    results.append(test_state_release())

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    exit_code = run_tests()
    exit(exit_code)