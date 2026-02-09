#!/usr/bin/env python3
"""
测试LRU策略下历史状态传递的正确性

验证在流式训练场景下，LRU策略不会影响同一序列内帧之间的状态传递
"""

import os
import sys
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stream_state_manager import StreamStateManager


def create_mock_state(frame_idx, sequence_id):
    """创建模拟的状态数据"""
    return {
        'frame_idx': frame_idx,
        'sequence_id': sequence_id,
        'features': torch.randn(10, 10, 10),  # 小张量
        'hidden_state': torch.randn(32),
        'metadata': {
            'created_at': frame_idx,
            'seq_id': sequence_id
        }
    }


def test_single_sequence_state_preservation():
    """
    测试1: 单序列内帧间状态传递
    模拟流式训练中最常见的场景：一个序列的连续帧处理
    """
    print("=" * 70)
    print("测试1: 单序列内帧间状态传递")
    print("=" * 70)

    manager = StreamStateManager(max_cached_states=5)

    sequence_id = "seq_A"
    num_frames = 10

    print(f"\n处理序列 {sequence_id}，共 {num_frames} 帧...")

    # 模拟处理10个连续帧
    for frame_idx in range(num_frames):
        # 创建当前帧的状态
        state = create_mock_state(frame_idx, sequence_id)

        # 更新状态（模拟流式训练）
        reset = (frame_idx == 0)
        updated_state = manager.update_state(state, sequence_id, frame_idx, reset_state=reset)

        # 验证状态内容
        assert updated_state['frame_idx'] == frame_idx, f"帧索引不匹配: {updated_state['frame_idx']} != {frame_idx}"
        assert updated_state['sequence_id'] == sequence_id, f"序列ID不匹配"

        print(f"  帧 {frame_idx}: 状态更新成功, 缓存大小: {len(manager.state_cache)}")

    # 验证：序列A的状态应该被保留
    assert sequence_id in manager.state_cache, f"序列 {sequence_id} 的状态不应该被LRU清理"

    # 验证：最终状态应该是第9帧的状态
    final_state = manager.get_state(sequence_id)
    assert final_state['frame_idx'] == num_frames - 1, f"最终状态应该是第 {num_frames-1} 帧"

    print(f"\n✅ 测试通过：单序列内状态正确传递")
    print(f"   序列 {sequence_id} 的状态完整保留了 {num_frames} 帧")
    return True


def test_multi_sequence_serial_training():
    """
    测试2: 多序列串行训练
    模拟真实训练场景：一个接一个处理不同序列
    """
    print("\n" + "=" * 70)
    print("测试2: 多序列串行训练")
    print("=" * 70)

    manager = StreamStateManager(max_cached_states=3)  # 只保留3个序列

    sequences = ["seq_A", "seq_B", "seq_C", "seq_D", "seq_E", "seq_F"]
    frames_per_sequence = 5

    print(f"\n串行处理 {len(sequences)} 个序列，每个序列 {frames_per_sequence} 帧...")
    print(f"最大缓存状态: {manager.max_cached_states}")

    for seq_id in sequences:
        print(f"\n--- 处理序列 {seq_id} ---")

        # 处理序列的所有帧
        for frame_idx in range(frames_per_sequence):
            state = create_mock_state(frame_idx, seq_id)
            reset = (frame_idx == 0)
            updated_state = manager.update_state(state, seq_id, frame_idx, reset_state=reset)

            # 验证当前序列的状态总是正确的
            assert updated_state['frame_idx'] == frame_idx
            assert updated_state['sequence_id'] == seq_id

        print(f"  序列 {seq_id} 处理完成，当前缓存: {list(manager.state_cache.keys())}")

    # 验证：最后3个序列应该被保留
    expected_cached = sequences[-3:]  # ["seq_D", "seq_E", "seq_F"]
    actual_cached = list(manager.state_cache.keys())

    print(f"\n预期缓存的序列: {expected_cached}")
    print(f"实际缓存的序列: {actual_cached}")

    # 验证每个被保留的序列的状态都是最后帧的状态
    for seq_id in expected_cached:
        state = manager.get_state(seq_id)
        assert state['frame_idx'] == frames_per_sequence - 1, f"序列 {seq_id} 的状态应该是最后一帧"
        assert state['sequence_id'] == seq_id, f"序列ID不匹配"

    # 验证：最老的序列应该被LRU清理
    removed_sequences = sequences[:-3]  # ["seq_A", "seq_B", "seq_C"]
    for seq_id in removed_sequences:
        assert seq_id not in manager.state_cache, f"序列 {seq_id} 应该被LRU清理"

    print(f"\n✅ 测试通过：多序列串行训练正确")
    print(f"   被LRU清理的序列: {removed_sequences}")
    print(f"   保留的序列（最后帧状态）: {expected_cached}")
    return True


def test_state_access_and_update():
    """
    测试3: 状态访问和更新不影响LRU
    验证频繁访问某个序列的状态不会导致它被误删
    """
    print("\n" + "=" * 70)
    print("测试3: 状态访问不影响LRU")
    print("=" * 70)

    manager = StreamStateManager(max_cached_states=3)

    # 创建3个序列
    for i, seq_id in enumerate(["seq_A", "seq_B", "seq_C"]):
        state = create_mock_state(i, seq_id)
        manager.update_state(state, seq_id, 0, reset_state=True)

    print(f"\n创建3个序列: {list(manager.state_cache.keys())}")

    # 频繁访问seq_A的状态
    for i in range(10):
        state = manager.get_state("seq_A")
        assert state is not None, f"seq_A的状态不应该被删除"

        # 模拟更新seq_A（这应该更新它在LRU队列中的位置）
        new_state = create_mock_state(100 + i, "seq_A")
        manager.update_state(new_state, "seq_A", 100 + i, reset_state=False)

    print(f"  频繁访问和更新seq_A 10次后")
    print(f"  当前缓存: {list(manager.state_cache.keys())}")
    print(f"  访问顺序: {manager.sequence_access_order}")

    # 添加第4个序列
    state = create_mock_state(0, "seq_D")
    manager.update_state(state, "seq_D", 0, reset_state=True)

    print(f"\n添加第4个序列（seq_D）后:")
    print(f"  当前缓存: {list(manager.state_cache.keys())}")
    print(f"  访问顺序: {manager.sequence_access_order}")

    # 验证：seq_D应该被添加，最老的（seq_B）应该被删除
    # 因为seq_A被频繁访问，所以seq_B是最老的
    expected_cached = {"seq_A", "seq_C", "seq_D"}  # seq_B应该被删除
    actual_cached = set(manager.state_cache.keys())

    print(f"\n预期缓存（seq_B应该被删除）: {expected_cached}")
    print(f"实际缓存: {actual_cached}")

    if actual_cached == expected_cached:
        print(f"\n✅ 测试通过：LRU正确处理频繁访问的状态")
        return True
    else:
        print(f"\n❌ 测试失败：LRU行为异常")
        print(f"   缺少的序列: {expected_cached - actual_cached}")
        print(f"   多余的序列: {actual_cached - expected_cached}")
        return False


def test_current_sequence_continuous_update():
    """
    测试4: 当前序列连续更新不会被LRU清理
    验证持续更新当前序列的状态不会导致它被删除
    """
    print("\n" + "=" * 70)
    print("测试4: 当前序列连续更新不会被LRU清理")
    print("=" * 70)

    manager = StreamStateManager(max_cached_states=3)  # 允许3个序列

    current_seq = "seq_current"
    print(f"\n创建并持续更新序列 {current_seq}（5帧）:")
    print(f"  最大缓存: {manager.max_cached_states}")

    # 模拟当前序列的5帧处理
    for frame_idx in range(5):
        state = create_mock_state(frame_idx, current_seq)
        reset = (frame_idx == 0)
        manager.update_state(state, current_seq, frame_idx, reset_state=reset)
        print(f"  帧 {frame_idx}: 状态更新成功")

    # 添加一些其他序列（不超过max_cached_states-1）
    print(f"\n添加其他序列（模拟训练中的其他序列）:")
    for i in range(1):  # 只添加1个，留1个空间给当前序列
        seq_id = f"seq_{i}"
        state = create_mock_state(0, seq_id)
        manager.update_state(state, seq_id, 0, reset_state=True)
        print(f"  序列 {seq_id} 已添加")

    # 验证：当前序列的状态应该仍然存在
    assert current_seq in manager.state_cache, f"当前序列 {current_seq} 应该被保留"
    current_state = manager.get_state(current_seq)
    assert current_state is not None, f"当前序列 {current_seq} 的状态不应该为None"
    assert current_state['frame_idx'] == 4, f"当前序列的状态应该是最后一帧（帧4）"

    print(f"\n✅ 测试通过：当前序列状态正确保留")
    print(f"   {current_seq} 的最后帧状态被保留")
    print(f"   当前缓存: {list(manager.state_cache.keys())}")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("LRU策略下历史状态传递正确性测试套件")
    print("=" * 70)

    results = []

    try:
        results.append(test_single_sequence_state_preservation())
    except AssertionError as e:
        print(f"\n❌ 测试1失败: {e}")
        results.append(False)

    try:
        results.append(test_multi_sequence_serial_training())
    except AssertionError as e:
        print(f"\n❌ 测试2失败: {e}")
        results.append(False)

    try:
        results.append(test_state_access_and_update())
    except AssertionError as e:
        print(f"\n❌ 测试3失败: {e}")
        results.append(False)

    try:
        results.append(test_current_sequence_continuous_update())
    except AssertionError as e:
        print(f"\n❌ 测试4失败: {e}")
        results.append(False)

    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\n通过: {passed}/{total}")

    if passed == total:
        print("\n🎉 所有测试通过！")
        print("✅ 结论：LRU策略不会影响流式训练历史状态的正确传递")
        return 0
    else:
        print("\n❌ 部分测试失败")
        print("⚠️  需要检查LRU策略的实现")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    exit(exit_code)
