#!/usr/bin/env python3
"""
简单的3D训练脚本 - 专注于数据流验证
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入数据集
try:
    from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
    DATASET_AVAILABLE = True
except ImportError as e:
    print(f"❌ 无法导入MultiSequenceTartanAirDataset: {e}")
    sys.exit(1)

# 简单的3D卷积模型，输出尺寸匹配输入
class Simple3DModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # 使用3D卷积，保持空间尺寸
        self.conv1 = nn.Conv3d(in_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: [batch, channels, depth, height, width]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)  # 输出: [batch, 1, depth, height, width]
        return x

def test_data_pipeline():
    """测试数据流水线"""
    print("============================================================")
    print("测试3D数据流水线")
    print("============================================================")
    
    # 配置
    data_root = "/home/cwh/Study/dataset/tartanair"
    batch_size = 2
    n_view = 5
    stride = 2
    crop_size = (48, 48, 32)  # (height, width, depth)
    voxel_size = 0.04
    target_image_size = (256, 256)
    max_sequences = 2
    
    print(f"配置:")
    print(f"  数据根目录: {data_root}")
    print(f"  批次大小: {batch_size}")
    print(f"  视图数: {n_view}")
    print(f"  裁剪尺寸: {crop_size}")
    print(f"  体素大小: {voxel_size}")
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
    
    print(f"   数据集大小: {len(dataset)}")
    print()
    
    # 创建数据加载器
    print("2. 创建数据加载器...")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # 获取一个批次
    print("3. 获取批次数据...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"   批次 {batch_idx}:")
        
        rgb_images = batch['rgb_images']  # [batch, n_view, 3, H, W]
        tsdf_gt = batch['tsdf']           # [batch, 1, D, H, W]
        poses = batch['poses']            # [batch, n_view, 4, 4]
        
        print(f"     RGB图像形状: {rgb_images.shape}")
        print(f"     TSDF GT形状: {tsdf_gt.shape}")
        print(f"     位姿形状: {poses.shape}")
        print(f"     序列名称: {batch['sequence_name']}")
        print()
        
        # 测试数据转换
        print("4. 测试数据转换...")
        
        # 将2D图像转换为3D体素网格
        batch_size = rgb_images.shape[0]
        n_view = rgb_images.shape[1]
        img_channels = rgb_images.shape[2]
        img_height = rgb_images.shape[3]
        img_width = rgb_images.shape[4]
        
        tsdf_depth = tsdf_gt.shape[2]
        tsdf_height = tsdf_gt.shape[3]
        tsdf_width = tsdf_gt.shape[4]
        
        print(f"     图像尺寸: {img_height}x{img_width}")
        print(f"     TSDF尺寸: {tsdf_depth}x{tsdf_height}x{tsdf_width}")
        print()
        
        # 方法1: 调整图像尺寸以匹配TSDF
        print("   方法1: 调整图像尺寸以匹配TSDF")
        current_images = rgb_images[:, 0]  # 第一帧 [batch, 3, H, W]
        
        # 调整图像尺寸
        current_images_resized = torch.nn.functional.interpolate(
            current_images, 
            size=(tsdf_height, tsdf_width), 
            mode='bilinear', 
            align_corners=False
        )
        
        # 扩展为3D
        current_images_3d = current_images_resized.unsqueeze(2).repeat(1, 1, tsdf_depth, 1, 1)
        
        print(f"     调整后图像形状: {current_images_resized.shape}")
        print(f"     3D扩展后形状: {current_images_3d.shape}")
        print(f"     目标TSDF形状: {tsdf_gt.shape}")
        print()
        
        # 方法2: 调整TSDF尺寸以匹配图像（需要处理5D张量）
        print("   方法2: 调整TSDF尺寸以匹配图像")
        # 对于5D张量，我们需要分别处理每个维度
        # tsdf_gt形状: [batch, channel, depth, height, width]
        # 我们想要调整height和width维度
        
        # 首先移除batch维度以便使用interpolate
        tsdf_sample = tsdf_gt[0]  # [channel, depth, height, width]
        
        # 调整高度和宽度
        tsdf_resized = torch.nn.functional.interpolate(
            tsdf_sample.unsqueeze(0),  # 添加batch维度 [1, channel, depth, height, width]
            size=(tsdf_depth, img_height, img_width),
            mode='trilinear',
            align_corners=False
        )
        
        print(f"     调整后TSDF形状: {tsdf_resized.shape}")
        print(f"     原始图像形状: {current_images.shape}")
        print()
        
        # 测试模型
        print("5. 测试模型...")
        device = torch.device('cpu')
        
        # 使用方法1的数据
        model = Simple3DModel(in_channels=3, out_channels=1)
        model = model.to(device)
        
        # 前向传播
        input_data = current_images_3d.to(device)
        target_data = tsdf_gt.to(device)
        
        output = model(input_data)
        
        print(f"     输入形状: {input_data.shape}")
        print(f"     输出形状: {output.shape}")
        print(f"     目标形状: {target_data.shape}")
        print()
        
        # 计算损失
        criterion = nn.MSELoss()
        loss = criterion(output, target_data)
        print(f"     损失值: {loss.item():.6f}")
        print()
        
        # 测试梯度
        print("6. 测试梯度...")
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        optimizer.zero_grad()
        loss.backward()
        
        # 检查梯度
        has_gradients = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 0:
                    has_gradients = True
                    print(f"     {name}: 梯度范数 = {grad_norm:.6f}")
        
        if has_gradients:
            print("     ✅ 梯度计算正常")
        else:
            print("     ⚠️ 没有检测到梯度")
        
        print()
        print("✅ 数据流水线测试完成!")
        
        break  # 只测试一个批次

def main():
    test_data_pipeline()

if __name__ == "__main__":
    main()