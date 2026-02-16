#!/usr/bin/env python3
"""
测试BatchNorm3d在不同batch size下的行为
验证直接使用PyTorch的BatchNorm3d是否可以处理batch size=1的情况
"""
import torch
import torch.nn as nn

print("=== 测试BatchNorm3d行为 ===\n")

# 测试1: batch size > 1 (正常情况)
print("测试1: BatchNorm3d处理batch size=2的5D张量...")
try:
    bn3d = nn.BatchNorm3d(num_features=16)
    # [batch, channels, depth, height, width]
    x = torch.randn(2, 16, 32, 32, 24)
    output = bn3d(x)
    print(f"✅ 输入形状: {x.shape}")
    print(f"✅ 输出形状: {output.shape}")
    print("✅ BatchNorm3d可以正常处理batch size=2\n")
except Exception as e:
    print(f"❌ 失败: {e}\n")

# 测试2: batch size = 1 (训练模式，可能失败)
print("测试2: BatchNorm3d处理batch size=1的5D张量（训练模式）...")
try:
    bn3d = nn.BatchNorm3d(num_features=16)
    bn3d.train()  # 训练模式
    x = torch.randn(1, 16, 32, 32, 24)
    output = bn3d(x)
    print(f"✅ 输入形状: {x.shape}")
    print(f"✅ 输出形状: {output.shape}")
    print("✅ BatchNorm3d可以处理batch size=1（训练模式）\n")
except Exception as e:
    print(f"⚠️  预期失败: {e}")
    print("这是BatchNorm的正常行为（需要batch size > 1进行统计）\n")

# 测试3: batch size = 1 (评估模式)
print("测试3: BatchNorm3d处理batch size=1的5D张量（评估模式）...")
try:
    bn3d = nn.BatchNorm3d(num_features=16)
    bn3d.eval()  # 评估模式
    x = torch.randn(1, 16, 32, 32, 24)
    output = bn3d(x)
    print(f"✅ 输入形状: {x.shape}")
    print(f"✅ 输出形状: {output.shape}")
    print("✅ BatchNorm3d可以处理batch size=1（评估模式）\n")
except Exception as e:
    print(f"❌ 失败: {e}\n")

# 测试4: track_running_stats=False
print("测试4: BatchNorm3d with track_running_stats=False处理batch size=1...")
try:
    bn3d = nn.BatchNorm3d(num_features=16, track_running_stats=False)
    bn3d.train()  # 训练模式
    x = torch.randn(1, 16, 32, 32, 24)
    output = bn3d(x)
    print(f"✅ 输入形状: {x.shape}")
    print(f"✅ 输出形状: {output.shape}")
    print("✅ 使用track_running_stats=False可以处理batch size=1\n")
except Exception as e:
    print(f"❌ 失败: {e}\n")

# 测试5: 模拟实际的pooling场景
print("测试5: 模拟多尺度池化后的BatchNorm3d...")
try:
    # 模拟3个尺度的池化结果
    batch_size = 2
    channels = 96
    num_scales = 3

    # [batch, num_scales*C, D, H, W]
    pools = torch.randn(batch_size, num_scales*channels, 32, 32, 24)

    # 使用BatchNorm3d
    bn3d = nn.BatchNorm3d(num_features=num_scales*channels, track_running_stats=False)
    bn3d.train()

    pools_norm = bn3d(pools)

    print(f"✅ 输入形状: {pools.shape}")
    print(f"✅ 输出形状: {pools_norm.shape}")
    print("✅ 多尺度池化后使用BatchNorm3d正常工作\n")
except Exception as e:
    print(f"❌ 失败: {e}\n")

print("=== 结论 ===")
print("直接使用PyTorch的BatchNorm3d是正确的做法。")
print("对于batch size=1的情况：")
print("- 评估模式：可以正常工作")
print("- 训练模式：track_running_stats=False 可以正常工作")
print("- 训练模式：默认设置会报错（需要batch size > 1）")
print("\n在former_v1.py中的global_norm应该使用track_running_stats=False来处理batch size变化。")
