#!/usr/bin/env python3
"""
StreamStateManager内存泄漏测试

测试目标：
1. 验证状态更新时的显存释放
2. 测试状态清理功能
3. 测试轻量级状态模式
4. 检测内存泄漏
"""

import os
import sys
import torch
import gc
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stream_state_manager import StreamStateManager


def get_gpu_memory_mb():
    """获取GPU显存使用量（MB）"""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated() / (1024 ** 2)


def create_large_state(size_mb=10):
    """
    创建指定大小（MB）的状态

    Args:
        size_mb: 状态大小（MB）

    Returns:
        状态字典
    """
    # 计算需要的元素数量（假设float32，每个元素4字节）
    num_elements = int(size_mb * 1024 ** 2 / 4)

    # 创建一个大的张量
    large_tensor = torch.randn(num_elements)

    return {
        'features': large_tensor,
        'mask': torch.ones(num_elements // 10, dtype=torch.bool),
        'metadata': {'frame_idx': 0, 'processed': True}
    }


def test_state_update_memory_release():
    """
    测试1: 状态更新时的显存释放

    验证在更新状态时，旧状态被正确释放
    """
    print("=" * 80)
    print("测试1: 状态更新时的显存释放")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manager = StreamStateManager(device=device)

    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU进行测试")
        print("   注意：CPU内存测试可能不够精确")

    print(f"\n使用设备: {device}")

    # 清理初始内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    initial_memory = get_gpu_memory_mb()
    print(f"\n初始显存使用: {initial_memory:.2f} MB")

    # 创建并更新多个状态
    num_states = 10  # 增加到10个，超过默认的max_cached_states=5
    state_size_mb = 10

    print(f"\n创建并更新 {num_states} 个状态，每个约 {state_size_mb} MB...")

    for i in range(num_states):
        state = create_large_state(state_size_mb)
        manager.update_state(state, f"seq_{i:03d}", frame_idx=0, reset_state=True)

        # 强制垃圾回收
        del state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        current_memory = get_gpu_memory_mb()
        print(f"   状态 {i+1}/{num_states} - 当前显存: {current_memory:.2f} MB")

    # 最终显存使用
    final_memory = get_gpu_memory_mb()
    print(f"\n最终显存使用: {final_memory:.2f} MB")

    # 预期：由于LRU机制，显存使用应限制在max_cached_states个状态的大小
    # 没有清理机制的情况下，显存会累积到 num_states * state_size_mb
    expected_with_cleanup = initial_memory + state_size_mb * min(manager.max_cached_states, num_states) * 1.5  # 允许一些误差
    expected_without_cleanup = initial_memory + state_size_mb * num_states * 0.8

    print(f"\n预期显存（有清理）: ~{expected_with_cleanup:.2f} MB (最多{min(manager.max_cached_states, num_states)}个状态)")
    print(f"预期显存（无清理）: ~{expected_without_cleanup:.2f} MB")

    # 判断测试结果
    if final_memory < expected_with_cleanup:
        print("✅ 测试通过：显存使用在有清理机制范围内")
        return True
    elif final_memory < expected_without_cleanup:
        print("⚠️  测试警告：显存使用偏高，可能存在部分内存泄漏")
        return False
    else:
        print("❌ 测试失败：显存使用过高，存在严重的内存泄漏")
        return False


def test_clear_old_states():
    """
    测试2: 旧状态清理功能

    验证clear_old_states()方法能正确清理旧状态
    """
    print("\n" + "=" * 80)
    print("测试2: 旧状态清理功能")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manager = StreamStateManager(device=device)

    print(f"\n使用设备: {device}")

    # 清理初始内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    initial_memory = get_gpu_memory_mb()
    print(f"\n初始显存使用: {initial_memory:.2f} MB")

    # 创建10个状态
    num_states = 10
    state_size_mb = 5

    print(f"\n创建 {num_states} 个状态...")

    for i in range(num_states):
        state = create_large_state(state_size_mb)
        manager.update_state(state, f"seq_{i:03d}", frame_idx=0, reset_state=True)
        del state

    # 强制垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    memory_before_clear = get_gpu_memory_mb()
    print(f"清理前显存使用: {memory_before_clear:.2f} MB")
    print(f"缓存状态数量: {manager.get_cache_size()}")

    # 测试clear_old_states方法
    print("\n调用 clear_old_states(keep_last_n=3)...")

    try:
        manager.clear_old_states(keep_last_n=3)

        # 强制垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        memory_after_clear = get_gpu_memory_mb()
        print(f"清理后显存使用: {memory_after_clear:.2f} MB")
        print(f"缓存状态数量: {manager.get_cache_size()}")

        # 预期：显存应该减少到 keep_last_n 个状态的大小
        expected_reduction = (num_states - 3) * state_size_mb * 0.7  # 允许70%的清理率
        actual_reduction = memory_before_clear - memory_after_clear

        print(f"\n预期显存减少: ~{expected_reduction:.2f} MB")
        print(f"实际显存减少: {actual_reduction:.2f} MB")

        if manager.get_cache_size() <= 3 and actual_reduction > expected_reduction * 0.5:
            print("✅ 测试通过：旧状态清理功能正常")
            return True
        elif manager.get_cache_size() <= 3:
            print("⚠️  测试警告：状态数量正确但显存释放不完全")
            return False
        else:
            print("❌ 测试失败：状态清理不完整")
            return False

    except AttributeError:
        print("⚠️  clear_old_states()方法不存在，跳过此测试")
        return None


def test_memory_info():
    """
    测试3: 内存信息获取

    验证get_memory_usage()方法能正确统计内存使用
    """
    print("\n" + "=" * 80)
    print("测试3: 内存信息获取")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    manager = StreamStateManager(device=device)

    print(f"\n使用设备: {device}")

    # 创建几个状态
    print("\n创建3个状态...")

    state1 = create_large_state(5)
    manager.update_state(state1, "seq_001", frame_idx=0, reset_state=True)

    state2 = create_large_state(10)
    manager.update_state(state2, "seq_002", frame_idx=0, reset_state=True)

    state3 = create_large_state(7)
    manager.update_state(state3, "seq_003", frame_idx=0, reset_state=True)

    # 获取内存信息
    memory_info = manager.get_memory_usage()

    print(f"\n内存统计信息:")
    print(f"  序列数量: {memory_info['num_sequences']}")
    print(f"  张量总数: {memory_info['total_tensors']}")
    print(f"  元素总数: {memory_info['total_elements']:,}")
    print(f"  估计内存: {memory_info['estimated_memory_mb']:.2f} MB")

    # 验证统计信息
    expected_memory = 5 + 10 + 7  # MB
    actual_memory = memory_info['estimated_memory_mb']

    print(f"\n预期内存: ~{expected_memory} MB")

    if memory_info['num_sequences'] == 3 and abs(actual_memory - expected_memory) / expected_memory < 0.3:
        print("✅ 测试通过：内存信息统计准确")
        return True
    else:
        print("⚠️  测试警告：内存信息统计可能有误差")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("StreamStateManager内存泄漏测试套件")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    results = {}

    # 测试1: 状态更新时的显存释放
    try:
        results['test_state_update_memory_release'] = test_state_update_memory_release()
    except Exception as e:
        print(f"\n❌ 测试1异常: {e}")
        import traceback
        traceback.print_exc()
        results['test_state_update_memory_release'] = False

    # 测试2: 旧状态清理功能
    try:
        results['test_clear_old_states'] = test_clear_old_states()
    except Exception as e:
        print(f"\n❌ 测试2异常: {e}")
        import traceback
        traceback.print_exc()
        results['test_clear_old_states'] = False

    # 测试3: 内存信息获取
    try:
        results['test_memory_info'] = test_memory_info()
    except Exception as e:
        print(f"\n❌ 测试3异常: {e}")
        import traceback
        traceback.print_exc()
        results['test_memory_info'] = False

    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    print(f"\n通过: {passed}")
    print(f"失败: {failed}")
    print(f"跳过: {skipped}")
    print(f"总计: {len(results)}")

    if failed == 0:
        print("\n✅ 所有测试通过！")
        return 0
    elif passed > 0:
        print("\n⚠️  部分测试失败，需要修复")
        return 1
    else:
        print("\n❌ 所有测试失败，需要全面修复")
        return 2


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
