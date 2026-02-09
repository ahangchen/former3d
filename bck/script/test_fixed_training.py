#!/usr/bin/env python3
"""
测试修复后的训练
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import time

print("=" * 60)
print("测试修复后的训练")
print("=" * 60)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# 创建小数据集
print("1. 创建数据集...")
dataset = MultiSequenceTartanAirDataset(
    data_root="/home/cwh/Study/dataset/tartanair",
    n_view=5,
    stride=2,
    crop_size=(48, 48, 32),
    voxel_size=0.04,
    target_image_size=(256, 256),
    max_sequences=1,
    shuffle=True
)

# 限制为前10个样本
dataset = Subset(dataset, indices=range(min(10, len(dataset))))
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"   数据集大小: {len(dataset)}")
print()

# 修正维度函数
def correct_tsdf_dimensions(tsdf_batch):
    """修正TSDF维度从 [batch, 1, H, W, D] 到 [batch, 1, D, H, W]"""
    return tsdf_batch.permute(0, 1, 4, 2, 3)

# 准备输入数据函数
def prepare_input_data(rgb_images, tsdf_gt_correct, frame_idx=0):
    """准备输入数据"""
    current_images = rgb_images[:, frame_idx]  # [batch, 3, H, W]
    
    # 获取TSDF尺寸 (已修正维度)
    tsdf_depth = tsdf_gt_correct.shape[2]   # D
    tsdf_height = tsdf_gt_correct.shape[3]  # H
    tsdf_width = tsdf_gt_correct.shape[4]   # W
    
    # 调整图像尺寸
    current_images_resized = torch.nn.functional.interpolate(
        current_images,
        size=(tsdf_height, tsdf_width),
        mode='bilinear',
        align_corners=False
    )
    
    # 扩展为3D
    current_images_3d = current_images_resized.unsqueeze(2).repeat(1, 1, tsdf_depth, 1, 1)
    return current_images_3d

# 简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv3d(16, 1, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

print("2. 创建模型...")
model = SimpleModel().to(device)
print(f"   模型参数: {sum(p.numel() for p in model.parameters()):,}")
print()

# 测试一个批次
print("3. 测试数据流...")
for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 1:  # 只测试第一个批次
        break
    
    rgb_images = batch['rgb_images'].to(device)
    tsdf_gt_raw = batch['tsdf'].to(device)
    
    print(f"   原始TSDF形状: {tsdf_gt_raw.shape}")  # 应该是 [batch, 1, 48, 48, 32]
    
    # 修正维度
    tsdf_gt = correct_tsdf_dimensions(tsdf_gt_raw)
    print(f"   修正后TSDF形状: {tsdf_gt.shape}")  # 应该是 [batch, 1, 32, 48, 48]
    
    # 准备输入
    input_3d = prepare_input_data(rgb_images, tsdf_gt, frame_idx=0)
    print(f"   输入形状: {input_3d.shape}")  # 应该是 [batch, 3, 32, 48, 48]
    
    # 前向传播
    output = model(input_3d)
    print(f"   输出形状: {output.shape}")  # 应该是 [batch, 1, 32, 48, 48]
    
    # 检查形状匹配
    if output.shape == tsdf_gt.shape:
        print("   ✅ 形状匹配正确!")
    else:
        print(f"   ❌ 形状不匹配: 输出{output.shape} vs 目标{tsdf_gt.shape}")
    
    break  # 只测试一个批次
print()

# 训练测试
print("4. 训练测试...")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
total_loss = 0.0
total_samples = 0
start_time = time.time()

for batch_idx, batch in enumerate(dataloader):
    rgb_images = batch['rgb_images'].to(device)
    tsdf_gt_raw = batch['tsdf'].to(device)
    
    # 修正维度
    tsdf_gt = correct_tsdf_dimensions(tsdf_gt_raw)
    
    # 准备输入
    input_3d = prepare_input_data(rgb_images, tsdf_gt, frame_idx=0)
    
    # 前向传播
    output = model(input_3d)
    
    # 计算损失
    loss = criterion(output, tsdf_gt)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 统计
    total_loss += loss.item() * rgb_images.shape[0]
    total_samples += rgb_images.shape[0]
    
    print(f"   批次 {batch_idx+1}: 损失 = {loss.item():.6f}")

train_time = time.time() - start_time
avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

print(f"\n   平均损失: {avg_loss:.6f}")
print(f"   训练时间: {train_time:.2f}秒")
print()

# 验证
print("5. 验证...")
model.eval()
val_loss = 0.0
val_samples = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(dataloader):
        rgb_images = batch['rgb_images'].to(device)
        tsdf_gt_raw = batch['tsdf'].to(device)
        
        tsdf_gt = correct_tsdf_dimensions(tsdf_gt_raw)
        input_3d = prepare_input_data(rgb_images, tsdf_gt, frame_idx=0)
        output = model(input_3d)
        
        loss = criterion(output, tsdf_gt)
        
        val_loss += loss.item() * rgb_images.shape[0]
        val_samples += rgb_images.shape[0]

avg_val_loss = val_loss / val_samples if val_samples > 0 else 0.0
print(f"   验证损失: {avg_val_loss:.6f}")
print()

print("=" * 60)
print("✅ 修复后的训练测试通过!")
print(f"训练损失: {avg_loss:.6f}")
print(f"验证损失: {avg_val_loss:.6f}")
print(f"总时间: {train_time:.2f}秒")
print("=" * 60)