#!/usr/bin/env python3
"""
修复的多序列训练脚本 - 处理3D TSDF数据
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from datetime import datetime
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入数据集
try:
    from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
    DATASET_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入MultiSequenceTartanAirDataset: {e}")
    print("将使用简化的数据集进行演示")
    DATASET_AVAILABLE = False

# 简化的3D SDF预测模型
class Simple3DSDFFormer(nn.Module):
    """简化的3D Stream-SDFFormer模型"""
    
    def __init__(self, input_channels=3, hidden_dim=128, output_dim=1):
        super().__init__()
        
        # 3D卷积编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # 3D卷积解码器
        self.decoder = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, output_dim, kernel_size=3, padding=1),
        )
        
        self.hidden_state = None
        
    def forward(self, x, reset_state=False):
        """
        前向传播
        
        Args:
            x: 输入图像 [batch, channels, depth, height, width]
            reset_state: 是否重置隐藏状态
        """
        if reset_state:
            self.hidden_state = None
        
        # 编码
        features = self.encoder(x)
        
        # 更新状态（简单平均）
        if self.hidden_state is None:
            self.hidden_state = features
        else:
            # 确保尺寸匹配
            if features.shape != self.hidden_state.shape:
                features = torch.nn.functional.interpolate(
                    features, size=self.hidden_state.shape[-3:], mode='trilinear', align_corners=False
                )
            self.hidden_state = 0.5 * self.hidden_state + 0.5 * features
        
        # 解码
        sdf_pred = self.decoder(self.hidden_state)
        
        return sdf_pred
    
    def reset_state(self, batch_size, device, grid_size=(32, 48, 48)):
        """重置隐藏状态"""
        self.hidden_state = None
        # 预初始化隐藏状态尺寸
        d, h, w = grid_size
        self.hidden_state = torch.zeros(batch_size, 64, d, h, w, device=device)

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 获取数据
        rgb_images = batch['rgb_images'].to(device)  # [batch, n_view, 3, H, W]
        tsdf_gt = batch['tsdf'].to(device)           # [batch, 1, D, H, W]
        
        batch_size = rgb_images.shape[0]
        n_view = rgb_images.shape[1]
        
        # 重置模型状态
        model.reset_state(batch_size, device, grid_size=tsdf_gt.shape[-3:])
        
        batch_loss = 0.0
        
        # 处理每个视图
        for frame_idx in range(n_view):
            # 获取当前帧
            current_images = rgb_images[:, frame_idx]  # [batch, 3, H, W]
            
            # 添加深度维度以匹配3D卷积
            # 将2D图像扩展为3D体素网格（沿深度维度复制）
            current_images_3d = current_images.unsqueeze(2).repeat(1, 1, tsdf_gt.shape[2], 1, 1)
            
            # 前向传播
            sdf_pred = model(current_images_3d, reset_state=(frame_idx == 0))
            
            # 计算损失（使用所有视图的累积预测）
            if frame_idx == n_view - 1:  # 只在最后一帧计算损失
                loss = criterion(sdf_pred, tsdf_gt)
                batch_loss = loss
        
        # 反向传播
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size
        
        # 每10个批次打印一次
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / total_samples
            print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.6f}")
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss

def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 获取数据
            rgb_images = batch['rgb_images'].to(device)
            tsdf_gt = batch['tsdf'].to(device)
            
            batch_size = rgb_images.shape[0]
            n_view = rgb_images.shape[1]
            
            # 重置模型状态
            model.reset_state(batch_size, device, grid_size=tsdf_gt.shape[-3:])
            
            # 处理每个视图
            for frame_idx in range(n_view):
                current_images = rgb_images[:, frame_idx]
                current_images_3d = current_images.unsqueeze(2).repeat(1, 1, tsdf_gt.shape[2], 1, 1)
                
                sdf_pred = model(current_images_3d, reset_state=(frame_idx == 0))
                
                # 只在最后一帧计算损失
                if frame_idx == n_view - 1:
                    loss = criterion(sdf_pred, tsdf_gt)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='多序列TartanAir训练')
    parser.add_argument('--test-only', action='store_true', help='只测试不训练')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小')
    parser.add_argument('--max-sequences', type=int, default=2, help='最大序列数')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cpu/cuda)')
    args = parser.parse_args()
    
    print("============================================================")
    print("修复的多序列TartanAir训练脚本")
    print("============================================================")
    
    # 配置
    data_root = "/home/cwh/Study/dataset/tartanair"
    n_view = 5
    stride = 2
    crop_size = (48, 48, 32)  # (height, width, depth)
    voxel_size = 0.04
    target_image_size = (256, 256)
    learning_rate = 0.0001
    
    print(f"配置:")
    print(f"  data_root: {data_root}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  n_view: {n_view}")
    print(f"  stride: {stride}")
    print(f"  crop_size: {crop_size}")
    print(f"  voxel_size: {voxel_size}")
    print(f"  target_image_size: {target_image_size}")
    print(f"  max_sequences: {args.max_sequences}")
    print(f"  num_epochs: {args.epochs}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  device: {args.device}")
    print()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    print()
    
    # 创建数据加载器
    print("创建数据加载器...")
    if DATASET_AVAILABLE:
        dataset = MultiSequenceTartanAirDataset(
            data_root=data_root,
            n_view=n_view,
            stride=stride,
            crop_size=crop_size,
            voxel_size=voxel_size,
            target_image_size=target_image_size,
            max_sequences=args.max_sequences,
            shuffle=True
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        print(f"✅ 使用真实数据，数据集大小: {len(dataset)}")
    else:
        print("❌ 无法加载真实数据集，使用模拟数据")
        # 创建模拟数据集
        class MockDataset:
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    'rgb_images': torch.randn(5, 3, 256, 256),
                    'tsdf': torch.randn(1, 32, 48, 48),
                    'poses': torch.randn(5, 4, 4),
                    'intrinsics': torch.randn(3, 3),
                    'sequence_name': f'mock_seq_{idx}',
                    'segment_idx': idx
                }
        
        dataset = MockDataset(size=50)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print(f"✅ 使用模拟数据，数据集大小: {len(dataset)}")
    
    print()
    
    # 创建模型
    print("创建模型...")
    model = Simple3DSDFFormer(input_channels=3, hidden_dim=64, output_dim=1)
    model = model.to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    print()
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if args.test_only:
        print("测试模式 - 只运行一个批次验证数据流")
        print()
        
        # 测试一个批次
        for batch_idx, batch in enumerate(dataloader):
            print(f"测试批次 {batch_idx}:")
            
            # 检查数据形状
            rgb_images = batch['rgb_images']
            tsdf_gt = batch['tsdf']
            
            print(f"  RGB图像形状: {rgb_images.shape}")
            print(f"  TSDF GT形状: {tsdf_gt.shape}")
            print(f"  位姿形状: {batch['poses'].shape if 'poses' in batch else 'N/A'}")
            
            # 测试模型前向传播
            if batch_idx == 0:
                print("\n测试模型前向传播...")
                
                # 重置模型状态
                batch_size = rgb_images.shape[0]
                model.reset_state(batch_size, device, grid_size=tsdf_gt.shape[-3:])
                
                # 处理第一帧
                current_images = rgb_images[:, 0].to(device)
                current_images_3d = current_images.unsqueeze(2).repeat(1, 1, tsdf_gt.shape[2], 1, 1).to(device)
                
                sdf_pred = model(current_images_3d, reset_state=True)
                print(f"  输入形状: {current_images_3d.shape}")
                print(f"  输出形状: {sdf_pred.shape}")
                print(f"  目标形状: {tsdf_gt.shape}")
                
                # 计算损失
                tsdf_gt_device = tsdf_gt.to(device)
                loss = criterion(sdf_pred, tsdf_gt_device)
                print(f"  损失值: {loss.item():.6f}")
                
                print("\n✅ 数据流测试通过!")
            
            break  # 只测试一个批次
        
        return
    
    # 训练
    print("开始训练...")
    print(f"总epoch数: {args.epochs}")
    print()
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        
        # 验证
        val_loss = validate(model, dataloader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  训练损失: {train_loss:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"  时间: {epoch_time:.2f}秒")
        print()
    
    print("训练完成!")
    
    # 保存模型
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"multi_sequence_model_{timestamp}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'data_root': data_root,
            'n_view': n_view,
            'crop_size': crop_size,
            'voxel_size': voxel_size,
        }
    }, model_path)
    
    print(f"模型已保存到: {model_path}")

if __name__ == "__main__":
    main()