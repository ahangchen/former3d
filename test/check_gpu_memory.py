#!/usr/bin/env python3
"""
训练配置测试脚本
测试不同batch size下的显存使用情况
"""

import torch

def check_gpu_memory():
    """检查GPU显存"""
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return None

    device = torch.device("cuda:0")

    # 获取GPU信息
    total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    free_memory = total_memory - reserved

    print("="*60)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("="*60)
    print(f"总显存: {total_memory:.2f} GB")
    print(f"已分配: {allocated:.2f} GB")
    print(f"已保留: {reserved:.2f} GB")
    print(f"可用显存: {free_memory:.2f} GB")
    print("="*60)

    return {
        'total': total_memory,
        'allocated': allocated,
        'reserved': reserved,
        'free': free_memory
    }

if __name__ == "__main__":
    check_gpu_memory()
