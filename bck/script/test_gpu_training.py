#!/usr/bin/env python3
"""
测试GPU训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time

print("=" * 60)
print("GPU训练测试")
print("=" * 60)

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if device.type == 'cuda':
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print()

# 创建简单的3D模型
class Simple3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# 创建模型并移动到GPU
model = Simple3DModel().to(device)
print(f"模型已移动到 {device}")

# 创建模拟数据
batch_size = 2
channels = 3
depth, height, width = 32, 48, 48

# 在GPU上创建数据
input_data = torch.randn(batch_size, channels, depth, height, width).to(device)
target_data = torch.randn(batch_size, 1, depth, height, width).to(device)

print(f"输入数据形状: {input_data.shape}")
print(f"目标数据形状: {target_data.shape}")
print(f"输入数据设备: {input_data.device}")
print(f"目标数据设备: {target_data.device}")
print()

# 测试前向传播
print("测试前向传播...")
start_time = time.time()
with torch.no_grad():
    output = model(input_data)
forward_time = time.time() - start_time
print(f"输出形状: {output.shape}")
print(f"前向传播时间: {forward_time:.4f}秒")
print()

# 测试训练循环
print("测试训练循环...")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model.train()
total_time = 0.0

for i in range(5):
    start_time = time.time()
    
    # 前向传播
    output = model(input_data)
    loss = criterion(output, target_data)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    iter_time = time.time() - start_time
    total_time += iter_time
    
    print(f"迭代 {i+1}: 损失 = {loss.item():.6f}, 时间 = {iter_time:.4f}秒")

print()
print(f"平均迭代时间: {total_time/5:.4f}秒")
print(f"总训练时间: {total_time:.4f}秒")
print()

# 检查GPU内存使用
if device.type == 'cuda':
    print("GPU内存使用:")
    print(f"  已分配: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
    print(f"  缓存: {torch.cuda.memory_reserved(0) / 1e6:.2f} MB")
    print(f"  最大已分配: {torch.cuda.max_memory_allocated(0) / 1e6:.2f} MB")

print()
print("=" * 60)
print("GPU训练测试完成!")
print("=" * 60)