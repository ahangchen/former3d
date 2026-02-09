#!/usr/bin/env python3
"""
最终的多序列训练脚本
使用3D卷积模型处理TartanAir多序列数据
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
    print(f"❌ 无法导入MultiSequenceTartanAirDataset: {e}")
    sys.exit(1)

# 3D SDF预测模型
class SDF3DModel(nn.Module):
    """3D SDF预测模型"""
    
    def __init__(self, in_channels=3, hidden_dims=[16, 32, 64], out_channels=1):
        super().__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dims[0], hidden_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dims[1], hidden_dims[2], kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dims[2]),
            nn.ReLU(inplace=True),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv3d(hidden_dims[2], hidden_dims[1], kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dims[1], hidden_dims[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dims[0], out_channels, kernel_size=3, padding=1),
        )
        
        # 隐藏状态（用于流式处理）
        self.hidden_state = None
        
    def forward(self, x, reset_state=False):
        """
        前向传播
        
        Args:
            x: 输入体素网格 [batch, channels, depth, height, width]
            reset_state: 是否重置隐藏状态
        """
        if reset_state:
            self.hidden_state = None
        
        # 编码
        features = self.encoder(x)
        
        # 流式处理：更新隐藏状态
        if self.hidden_state is None:
            self.hidden_state = features
        else:
            # 确保尺寸匹配
            if features.shape != self.hidden_state.shape:
                features = torch.nn.functional.interpolate(
                    features, size=self.hidden_state.shape[-3:], mode='trilinear', align_corners=False
                )
            # 加权平均更新
            self.hidden_state = 0.7 * self.hidden_state + 0.3 * features
        
        # 解码
        sdf_pred = self.decoder(self.hidden_state)
        
        return sdf_pred
    
    def reset_state(self, batch_size, device, grid_size):
        """重置隐藏状态"""
        self.hidden_state = None
        # 预分配内存
        d, h, w = grid_size
        self.hidden_state = torch.zeros(batch_size, 64, d, h, w, device=device)

def prepare_input_data(rgb_images, tsdf_gt, frame_idx=0):
    """
    准备输入数据：将2D图像转换为3D体素网格
    
    Args:
        rgb_images: RGB图像 [batch, n_view, 3, H, W]
        tsdf_gt: TSDF真值 [batch, 1, D, H, W]
        frame_idx: 帧索引
    """
    batch_size = rgb_images.shape[0]
    current_images = rgb_images[:, frame_idx]  # [batch, 3, H, W]
    
    # 获取TSDF尺寸
    tsdf_depth = tsdf_gt.shape[2]
    tsdf_height = tsdf_gt.shape[3]
    tsdf_width = tsdf_gt.shape[4]
    
    # 调整图像尺寸以匹配TSDF空间尺寸
    current_images_resized = torch.nn.functional.interpolate(
        current_images,
        size=(tsdf_height, tsdf_width),
        mode='bilinear',
        align_corners=False
    )
    
    # 扩展为3D体素网格（沿深度维度复制）
    current_images_3d = current_images_resized.unsqueeze(2).repeat(1, 1, tsdf_depth, 1, 1)
    
    return current_images_3d

def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 获取数据
        rgb_images = batch['rgb_images'].to(device)
        tsdf_gt = batch['tsdf'].to(device)
        
        batch_size = rgb_images.shape[0]
        n_view = rgb_images.shape[1]
        
        # 重置模型状态（每个片段开始时）
        model.reset_state(batch_size, device, grid_size=tsdf_gt.shape[-3:])
        
        batch_loss = 0.0
        
        # 处理每个视图（流式处理）
        for frame_idx in range(n_view):
            # 准备输入数据
            input_3d = prepare_input_data(rgb_images, tsdf_gt, frame_idx)
            
            # 前向传播
            reset_state = (frame_idx == 0)  # 第一帧重置状态
            sdf_pred = model(input_3d, reset_state=reset_state)
            
            # 计算损失（累积所有视图）
            loss = criterion(sdf_pred, tsdf_gt)
            batch_loss = loss  # 使用最后一帧的损失
        
        # 反向传播
        optimizer.zero_grad()
        batch_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size
        
        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / total_samples
            print(f"  Epoch {epoch+1:03d}, Batch {batch_idx+1:04d}/{len(dataloader):04d}, "
                  f"Loss: {avg_loss:.6f}")
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss

def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            rgb_images = batch['rgb_images'].to(device)
            tsdf_gt = batch['tsdf'].to(device)
            
            batch_size = rgb_images.shape[0]
            n_view = rgb_images.shape[1]
            
            # 重置模型状态
            model.reset_state(batch_size, device, grid_size=tsdf_gt.shape[-3:])
            
            # 处理每个视图
            for frame_idx in range(n_view):
                input_3d = prepare_input_data(rgb_images, tsdf_gt, frame_idx)
                reset_state = (frame_idx == 0)
                sdf_pred = model(input_3d, reset_state=reset_state)
                
                # 使用最后一帧的预测计算损失
                if frame_idx == n_view - 1:
                    loss = criterion(sdf_pred, tsdf_gt)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='多序列TartanAir训练')
    parser.add_argument('--test-only', action='store_true', help='只测试不训练')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小')
    parser.add_argument('--max-sequences', type=int, default=3, help='最大序列数')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cpu/cuda)')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='保存目录')
    args = parser.parse_args()
    
    print("=" * 60)
    print("多序列TartanAir训练脚本")
    print("=" * 60)
    
    # 配置
    data_root = "/home/cwh/Study/dataset/tartanair"
    n_view = 5
    stride = 2
    crop_size = (48, 48, 32)  # (height, width, depth)
    voxel_size = 0.04
    target_image_size = (256, 256)
    
    print(f"配置:")
    print(f"  数据根目录: {data_root}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  视图数: {n_view}")
    print(f"  裁剪尺寸: {crop_size}")
    print(f"  体素大小: {voxel_size}")
    print(f"  最大序列数: {args.max_sequences}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  设备: {args.device}")
    print()
    
    # 设置设备
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"⚠️  使用CPU")
    print()
    
    # 创建数据加载器
    print("创建数据加载器...")
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
    
    # 分割训练集和验证集
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"  总数据集大小: {dataset_size}")
    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  验证集大小: {len(val_dataset)}")
    print()
    
    if args.test_only:
        print("测试模式 - 验证数据流...")
        print()
        
        # 测试一个训练批次
        for batch_idx, batch in enumerate(train_dataloader):
            print(f"训练批次 {batch_idx}:")
            print(f"  RGB图像形状: {batch['rgb_images'].shape}")
            print(f"  TSDF GT形状: {batch['tsdf'].shape}")
            print(f"  序列: {batch['sequence_name'][:2]}...")
            print()
            
            # 测试数据准备
            rgb_images = batch['rgb_images'].to(device)
            tsdf_gt = batch['tsdf'].to(device)
            
            input_3d = prepare_input_data(rgb_images, tsdf_gt, frame_idx=0)
            print(f"  输入3D形状: {input_3d.shape}")
            print(f"  目标形状: {tsdf_gt.shape}")
            print()
            
            # 测试模型
            model = SDF3DModel(in_channels=3, hidden_dims=[16, 32, 64], out_channels=1)
            model = model.to(device)
            
            model.reset_state(args.batch_size, device, grid_size=tsdf_gt.shape[-3:])
            output = model(input_3d, reset_state=True)
            
            print(f"  模型输出形状: {output.shape}")
            
            # 计算参数
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  模型参数: {total_params:,} (可训练: {trainable_params:,})")
            print()
            
            print("✅ 数据流测试通过!")
            break
        
        return
    
    # 创建模型
    print("创建模型...")
    model = SDF3DModel(in_channels=3, hidden_dims=[16, 32, 64], out_channels=1)
    model = model.to(device)
    
    # 计算参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数: {total_params:,} (可训练: {trainable_params:,})")
    print()
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 训练循环
    print("开始训练...")
    print(f"总epoch数: {args.epochs}")
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, device, epoch)
        
        # 验证
        val_loss = validate(model, val_dataloader, criterion, device)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:03d}/{args.epochs:03d}:")
        print(f"  训练损失: {train_loss:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"  时间: {epoch_time:.2f}秒")
        print(f"  学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(args.save_dir, f"best_model_epoch{epoch+1:03d}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'data_root': data_root,
                    'n_view': n_view,
                    'crop_size': crop_size,
                    'voxel_size': voxel_size,
                    'hidden_dims': [16, 32, 64],
                }
            }, model_path)
            print(f"  💾 保存最佳模型到: {model_path}")
            print()
    
    # 保存最终模型
    final_model_path = os.path.join(args.save_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'data_root': data_root,
            'n_view': n_view,
            'crop_size': crop_size,
            'voxel_size': voxel_size,
        }
    }, final_model_path)
    
    print("=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"最终模型保存到: {final_model_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()