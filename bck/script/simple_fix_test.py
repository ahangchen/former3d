#!/usr/bin/env python3
"""
简单测试修复
"""

import torch

print("="*80)
print("测试intrinsics形状修复")
print("="*80)

# 测试1: intrinsics形状修复
print("\n1. 测试intrinsics形状修复...")
batch = {
    'rgb_images': torch.randn(2, 3, 3, 64, 64),  # [B, F, C, H, W]
    'intrinsics': torch.eye(3),                  # (3, 3) - 问题所在
}

print(f"原始intrinsics形状: {batch['intrinsics'].shape}")

# 应用修复
intrinsics = batch['intrinsics']
if intrinsics.dim() == 2:
    B = batch['rgb_images'].shape[0]
    intrinsics = intrinsics.unsqueeze(0).repeat(B, 1, 1)

print(f"修复后intrinsics形状: {intrinsics.shape}")
print(f"期望形状: [{B}, 3, 3]")

if intrinsics.shape == torch.Size([B, 3, 3]):
    print("✅ intrinsics形状修复正确")
else:
    print("❌ intrinsics形状修复失败")

# 测试2: 检查修复后的训练脚本
print("\n2. 检查修复后的训练脚本...")
try:
    with open("optimized_online_training.py", "r") as f:
        content = f.read()
    
    # 检查是否添加了修复代码
    if "intrinsics.dim() == 2" in content:
        print("✅ 训练脚本已包含intrinsics形状修复")
    else:
        print("❌ 训练脚本未包含intrinsics形状修复")
        
except Exception as e:
    print(f"❌ 检查失败: {e}")

print("\n" + "="*80)
print("测试完成")
print("="*80)
print("建议:")
print("1. 如果测试通过，运行修复后的训练脚本")
print("2. 命令: python optimized_online_training.py")
print("\n🚀 现在可以尝试运行训练了!")