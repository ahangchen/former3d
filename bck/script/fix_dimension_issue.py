#!/usr/bin/env python3
"""
修复维度不匹配问题
"""

import os
import sys
import torch
import time

print("=" * 60)
print("修复维度不匹配问题")
print("=" * 60)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")
print()

# 创建数据集
dataset = MultiSequenceTartanAirDataset(
    data_root="/home/cwh/Study/dataset/tartanair",
    n_view=5,
    stride=2,
    crop_size=(48, 48, 32),  # (height, width, depth)
    voxel_size=0.04,
    target_image_size=(256, 256),
    max_sequences=1,
    shuffle=True
)

# 获取一个样本
sample = dataset[0]
print("原始样本形状:")
print(f"  rgb_images: {sample['rgb_images'].shape}")      # [5, 3, 256, 256]
print(f"  tsdf: {sample['tsdf'].shape}")                  # [1, 48, 48, 32]
print(f"  occupancy: {sample['occupancy'].shape}")        # [1, 48, 48, 32]
print()

# 问题分析
print("问题分析:")
print("TSDF形状是 [1, 48, 48, 32] 表示 [channels, height, width, depth]")
print("但PyTorch期望的3D卷积输入是 [batch, channels, depth, height, width]")
print()

# 解决方案：调整维度顺序
tsdf_tensor = sample['tsdf']  # [1, 48, 48, 32]
print(f"原始TSDF形状: {tsdf_tensor.shape}")

# 方法1: 使用permute调整维度
tsdf_permuted = tsdf_tensor.permute(0, 3, 1, 2)  # [1, 32, 48, 48]
print(f"调整后形状(permute): {tsdf_permuted.shape}")

# 方法2: 使用unsqueeze和permute得到完整5D张量
tsdf_5d = tsdf_tensor.unsqueeze(0)  # [1, 1, 48, 48, 32]
tsdf_5d_correct = tsdf_5d.permute(0, 1, 4, 2, 3)  # [1, 1, 32, 48, 48]
print(f"5D张量形状: {tsdf_5d_correct.shape}")
print()

# 测试模型兼容性
print("测试模型兼容性...")
model = torch.nn.Conv3d(3, 1, 3, padding=1).to(device)

# 准备输入数据
rgb_images = sample['rgb_images'].unsqueeze(0).to(device)  # [1, 5, 3, 256, 256]

# 使用第一帧图像
current_images = rgb_images[:, 0]  # [1, 3, 256, 256]

# 调整图像尺寸以匹配TSDF
current_images_resized = torch.nn.functional.interpolate(
    current_images,
    size=(48, 48),
    mode='bilinear',
    align_corners=False
)  # [1, 3, 48, 48]

# 扩展为3D：重复深度维度
input_3d = current_images_resized.unsqueeze(2).repeat(1, 1, 32, 1, 1)  # [1, 3, 32, 48, 48]
print(f"输入形状: {input_3d.shape}")

# 准备目标数据（正确维度）
target_3d = tsdf_5d_correct.to(device)  # [1, 1, 32, 48, 48]
print(f"目标形状: {target_3d.shape}")
print()

# 验证维度匹配
print("维度验证:")
print(f"输入: batch={input_3d.shape[0]}, channels={input_3d.shape[1]}, depth={input_3d.shape[2]}, height={input_3d.shape[3]}, width={input_3d.shape[4]}")
print(f"目标: batch={target_3d.shape[0]}, channels={target_3d.shape[1]}, depth={target_3d.shape[2]}, height={target_3d.shape[3]}, width={target_3d.shape[4]}")
print()

# 测试训练
print("测试训练...")
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for i in range(3):
    # 前向传播
    output = model(input_3d)
    
    # 检查输出形状
    if i == 0:
        print(f"模型输出形状: {output.shape}")
        print(f"目标形状: {target_3d.shape}")
    
    # 计算损失
    loss = criterion(output, target_3d)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"迭代 {i+1}: 损失 = {loss.item():.6f}")

print()
print("✅ 维度问题修复成功!")
print()

# 更新数据集类的建议
print("💡 修复建议:")
print("1. 在数据集中将TSDF维度调整为标准格式:")
print("   原始: tsdf.shape = [1, height, width, depth]")
print("   调整: tsdf = tsdf.permute(0, 3, 1, 2)  # [1, depth, height, width]")
print()
print("2. 或者在训练脚本中调整:")
print("   tsdf_gt = batch['tsdf'].permute(0, 1, 4, 2, 3)")
print()

print("=" * 60)
print("修复完成 - 现在可以运行完整训练了!")
print("=" * 60)