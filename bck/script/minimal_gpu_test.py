#!/usr/bin/env python3
"""
极简GPU训练测试
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time

print("=" * 60)
print("极简GPU训练测试")
print("=" * 60)

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# 创建简单的模拟数据
print("1. 创建模拟数据...")
batch_size = 2
channels = 3
depth, height, width = 32, 48, 48

# 模拟RGB图像 (batch, n_view, channels, H, W)
n_view = 5
rgb_images = torch.randn(batch_size, n_view, channels, 256, 256).to(device)

# 模拟TSDF真值 (batch, 1, D, H, W)
tsdf_gt = torch.randn(batch_size, 1, depth, height, width).to(device)

print(f"   RGB图像形状: {rgb_images.shape}")
print(f"   TSDF GT形状: {tsdf_gt.shape}")
print()

# 简单的数据准备函数
def prepare_input(rgb_images, tsdf_gt, frame_idx=0):
    current_images = rgb_images[:, frame_idx]  # [batch, channels, H, W]
    
    # 调整尺寸
    current_images_resized = torch.nn.functional.interpolate(
        current_images,
        size=(height, width),
        mode='bilinear',
        align_corners=False
    )
    
    # 扩展为3D
    current_images_3d = current_images_resized.unsqueeze(2).repeat(1, 1, depth, 1, 1)
    return current_images_3d

# 创建简单模型
print("2. 创建模型...")
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 8, 3, padding=1)  # 减少通道数以加快速度
        self.conv2 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv3d(16, 1, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

model = SimpleModel().to(device)
print(f"   模型参数: {sum(p.numel() for p in model.parameters()):,}")
print()

# 训练
print("3. 开始训练...")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
start_time = time.time()

for epoch in range(2):  # 2个epoch
    epoch_loss = 0.0
    
    # 模拟3个批次
    for batch_idx in range(3):
        # 准备输入（使用第一帧）
        input_3d = prepare_input(rgb_images, tsdf_gt, frame_idx=0)
        
        # 前向传播
        output = model(input_3d)
        
        # 计算损失
        loss = criterion(output, tsdf_gt)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        print(f"   Epoch {epoch+1}, Batch {batch_idx+1}: 损失 = {loss.item():.6f}")
    
    avg_loss = epoch_loss / 3
    print(f"   Epoch {epoch+1} 平均损失: {avg_loss:.6f}")
    print()

training_time = time.time() - start_time
print(f"训练时间: {training_time:.2f}秒")

# 验证
print("4. 验证...")
model.eval()
with torch.no_grad():
    input_3d = prepare_input(rgb_images, tsdf_gt, frame_idx=0)
    output = model(input_3d)
    val_loss = criterion(output, tsdf_gt)
    print(f"   验证损失: {val_loss.item():.6f}")

print()
print("=" * 60)
print("极简GPU训练测试完成!")
print(f"最终验证损失: {val_loss.item():.6f}")
print(f"总训练时间: {training_time:.2f}秒")
print("=" * 60)