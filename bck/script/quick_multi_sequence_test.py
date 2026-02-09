#!/usr/bin/env python3
"""
快速多序列测试 - 只运行一个epoch的少量批次
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入数据集
from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset

# 使用之前的简单模型
class Simple3DModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def prepare_input(rgb_images, tsdf_gt, frame_idx=0):
    """准备输入数据"""
    current_images = rgb_images[:, frame_idx]
    
    # 调整图像尺寸以匹配TSDF
    tsdf_height = tsdf_gt.shape[3]
    tsdf_width = tsdf_gt.shape[4]
    
    current_images_resized = torch.nn.functional.interpolate(
        current_images,
        size=(tsdf_height, tsdf_width),
        mode='bilinear',
        align_corners=False
    )
    
    # 扩展为3D
    tsdf_depth = tsdf_gt.shape[2]
    current_images_3d = current_images_resized.unsqueeze(2).repeat(1, 1, tsdf_depth, 1, 1)
    
    return current_images_3d

def main():
    print("快速多序列测试")
    print("=" * 50)
    
    # 配置
    data_root = "/home/cwh/Study/dataset/tartanair"
    batch_size = 2
    n_view = 5
    stride = 2
    crop_size = (48, 48, 32)
    voxel_size = 0.04
    target_image_size = (256, 256)
    max_sequences = 2
    max_batches = 5  # 只测试5个批次
    
    print(f"配置:")
    print(f"  批次大小: {batch_size}")
    print(f"  视图数: {n_view}")
    print(f"  最大序列数: {max_sequences}")
    print(f"  测试批次: {max_batches}")
    print()
    
    # 创建数据集
    print("1. 创建数据集...")
    dataset = MultiSequenceTartanAirDataset(
        data_root=data_root,
        n_view=n_view,
        stride=stride,
        crop_size=crop_size,
        voxel_size=voxel_size,
        target_image_size=target_image_size,
        max_sequences=max_sequences,
        shuffle=True
    )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"   数据集大小: {len(dataset)}")
    print()
    
    # 创建模型
    print("2. 创建模型...")
    device = torch.device('cpu')
    model = Simple3DModel(in_channels=3, out_channels=1)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环（简化）
    print("3. 开始训练...")
    print()
    
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        
        # 获取数据
        rgb_images = batch['rgb_images'].to(device)
        tsdf_gt = batch['tsdf'].to(device)
        
        batch_size = rgb_images.shape[0]
        
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
        
        # 统计
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        print(f"   批次 {batch_idx+1}: 损失 = {loss.item():.6f}")
        
        # 检查梯度
        if batch_idx == 0:
            has_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and param.grad.norm() > 0:
                    has_grad = True
                    break
            
            if has_grad:
                print(f"     ✅ 梯度计算正常")
            else:
                print(f"     ⚠️  没有检测到梯度")
            print()
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    print(f"   平均损失: {avg_loss:.6f}")
    print()
    
    # 验证
    print("4. 验证...")
    model.eval()
    
    with torch.no_grad():
        val_loss = 0.0
        val_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:  # 只验证2个批次
                break
            
            rgb_images = batch['rgb_images'].to(device)
            tsdf_gt = batch['tsdf'].to(device)
            
            input_3d = prepare_input(rgb_images, tsdf_gt, frame_idx=0)
            output = model(input_3d)
            
            loss = criterion(output, tsdf_gt)
            
            val_loss += loss.item() * rgb_images.shape[0]
            val_samples += rgb_images.shape[0]
        
        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0.0
        print(f"   验证损失: {avg_val_loss:.6f}")
    
    print()
    print("=" * 50)
    print("测试完成!")
    print(f"训练损失: {avg_loss:.6f}")
    print(f"验证损失: {avg_val_loss:.6f}")
    print("=" * 50)

if __name__ == "__main__":
    main()