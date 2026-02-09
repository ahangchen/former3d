#!/usr/bin/env python3
"""
测试内存管理器
验证MemoryManager的显存清理功能
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_manager import MemoryManager

def test_memory_manager():
    """测试内存管理器功能"""

    print("="*70)
    print("测试内存管理器")
    print("="*70)

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过显存测试")
        return

    device = torch.device("cuda:0")

    # 创建内存管理器
    memory_manager = MemoryManager(cleanup_frequency=5, memory_threshold_gb=0.1)  # 低阈值用于测试

    print(f"\n初始显存状态：")
    initial_info = memory_manager.get_memory_info()
    for key, value in initial_info.items():
        print(f"  {key}: {value}")

    # 分配一些张量来增加显存使用
    print(f"\n分配张量以增加显存使用...")
    tensors = []
    for i in range(10):
        # 创建大张量
        t = torch.randn(1000, 1000, device=device)
        tensors.append(t)
        
        # 每5个张量执行一次定期清理
        if (i + 1) % 5 == 0:
            print(f"  步骤 {i+1}: 执行定期清理")
            memory_manager.step()

    print(f"\n分配张量后的显存状态：")
    after_alloc_info = memory_manager.get_memory_info()
    for key, value in after_alloc_info.items():
        print(f"  {key}: {value}")

    # 删除张量
    print(f"\n删除张量...")
    del tensors
    torch.cuda.empty_cache()  # 手动清空

    print(f"\n删除张量后的显存状态：")
    after_delete_info = memory_manager.get_memory_info()
    for key, value in after_delete_info.items():
        print(f"  {key}: {value}")

    # 测试按需清理
    print(f"\n测试按需清理功能...")
    
    # 创建一些张量
    large_tensors = [torch.randn(2000, 2000, device=device) for _ in range(3)]
    
    print(f"  创建大张量后的显存状态：")
    before_cleanup_info = memory_manager.get_memory_info()
    for key, value in before_cleanup_info.items():
        print(f"    {key}: {value}")

    # 执行按需清理（阈值设得很低，肯定会触发）
    cleaned = memory_manager.cleanup_if_needed(threshold_gb=0.1)
    print(f"  按需清理执行: {cleaned}")

    print(f"  按需清理后的显存状态：")
    after_cleanup_info = memory_manager.get_memory_info()
    for key, value in after_cleanup_info.items():
        print(f"    {key}: {value}")

    # 清理资源
    del large_tensors
    memory_manager.cleanup()

    print(f"\n最终显存状态：")
    final_info = memory_manager.get_memory_info()
    for key, value in final_info.items():
        print(f"  {key}: {value}")

    print(f"\n{'='*70}")
    print("✅ 测试完成！")
    print(f"{'='*70}")

def test_memory_manager_cleanup():
    """测试内存清理功能的详细效果"""

    print("\n" + "="*70)
    print("测试内存清理功能详细效果")
    print("="*70)

    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过显存测试")
        return

    device = torch.device("cuda:0")

    # 创建内存管理器
    memory_manager = MemoryManager(cleanup_frequency=1, memory_threshold_gb=1.0)

    # 创建一些大张量
    print(f"创建大张量...")
    big_tensors = [torch.randn(3000, 3000, device=device) for _ in range(2)]

    print(f"清理前显存状态：")
    before_cleanup = memory_manager.get_memory_info()
    for key, value in before_cleanup.items():
        if 'gb' in key.lower():
            print(f"  {key}: {value:.3f}GB")

    # 执行清理
    print(f"执行显存清理...")
    memory_manager.cleanup(verbose=True)

    print(f"清理后显存状态：")
    after_cleanup = memory_manager.get_memory_info()
    for key, value in after_cleanup.items():
        if 'gb' in key.lower():
            print(f"  {key}: {value:.3f}GB")

    # 删除张量
    del big_tensors
    print(f"删除张量后显存状态：")
    after_deletion = memory_manager.get_memory_info()
    for key, value in after_deletion.items():
        if 'gb' in key.lower():
            print(f"  {key}: {value:.3f}GB")

    print(f"\n{'='*70}")
    print("✅ 详细清理测试完成！")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_memory_manager()
    test_memory_manager_cleanup()