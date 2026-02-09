#!/usr/bin/env python3
"""
修复后的多序列训练脚本 - 修复维度不匹配问题
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
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
        
    def forward(self, x):
        """前向传播"""
        features = self.encoder(x)
        output = self.decoder(features)
        return output

def prepare_input_data(rgb_images, tsdf_gt_correct, frame_idx=0):
    """
    准备输入数据：将2D图像转换为3D体素网格
    
    Args:
        rgb_images: RGB图像 [batch, n_view, 3, H, W]
        tsdf_gt_correct: TSDF真值 [batch, 1, D, H, W] (已修正维度)
        frame_idx: 帧索引
    """
    batch_size = rgb_images.shape[0]
    current_images = rgb_images[:, frame_idx]  # [batch, 3, H, W]
    
    # 获取TSDF尺寸 (已修正维度)
    tsdf_depth = tsdf_gt_correct.shape[2]   # D
    tsdf_height = tsdf_gt_correct.shape[3]  # H
    tsdf_width = tsdf_gt_correct.shape[4]   # W
    
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

def correct_tsdf_dimensions(tsdf_batch):
    """
    修正TSDF维度从 [batch, 1, H, W, D] 到 [batch, 1, D, H, W]
    
    Args:
        tsdf_batch: 原始TSDF批次 [batch, 1, height, width, depth]
    
    Returns:
        修正后的TSDF [batch, 1, depth, height, width]
    """
    # 原始: [batch, 1, height, width, depth]
    # 目标: [batch, 1, depth, height, width]
    return tsdf_batch.permute(0, 1, 4, 2, 3)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 获取数据
        rgb_images = batch['rgb_images'].to(device)
        tsdf_gt_raw = batch['tsdf'].to(device)  # [batch, 1, H, W, D]
        
        # 修正TSDF维度
        tsdf_gt = correct_tsdf_dimensions(tsdf_gt_raw)  # [batch, 1, D, H, W]
        
        batch_size = rgb_images.shape[0]
        n_view = rgb_images.shape[1]
        
        # 使用第一帧进行训练（简化）
        input_3d = prepare_input_data(rgb_images, tsdf_gt, frame_idx=0)
        
        # 前向传播
        sdf_pred = model(input_3d)
        
        # 计算损失
        loss = criterion(sdf_pred, tsdf_gt)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 统计
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # 打印进度
        if (batch_idx + 1) % 5 == 0 or batch_idx == 0:
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            print(f"  Batch {batch_idx+1:03d}/{len(dataloader):03d}, Loss: {avg_loss:.6f}")
    
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
            tsdf_gt_raw = batch['tsdf'].to(device)
            
            # 修正TSDF维度
            tsdf_gt = correct_tsdf_dimensions(tsdf_gt_raw)
            
            batch_size = rgb_images.shape[0]
            
            # 使用第一帧进行验证
            input_3d = prepare_input_data(rgb_images, tsdf_gt, frame_idx=0)
            sdf_pred = model(input_3d)
            
            loss = criterion(sdf_pred, tsdf_gt)
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description='修复后的多序列TartanAir训练')
    parser.add_argument('--test-only', action='store_true', help='只测试不训练')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小')
    parser.add_argument('--max-sequences', type=int, default=2, help='最大序列数')
    parser.add_argument('--max-samples', type=int, default=50, help='最大样本数（测试用）')
    parser.add_argument('--lr', type=float, default=0.0001, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cpu/cuda)')
    parser.add_argument('--save-dir', type=str, default='fixed_checkpoints', help='保存目录')
    args = parser.parse_args()
    
    print("=" * 60)
    print("修复后的多序列TartanAir训练脚本")
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
    print(f"  最大样本数: {args.max_samples}")
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
    
    # 创建数据集
    print("1. 创建数据集...")
    start_time = time.time()
    
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
        
        # 限制数据集大小以加快测试
        dataset_size = min(args.max_samples, len(dataset))
        dataset = Subset(dataset, indices=range(dataset_size))
        
        # 分割训练集和验证集
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size
        
        train_dataset = Subset(dataset, indices=range(train_size))
        val_dataset = Subset(dataset, indices=range(train_size, train_size + val_size))
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        dataset_load_time = time.time() - start_time
        print(f"   数据集大小: {dataset_size}")
        print(f"   训练集: {len(train_dataset)}")
        print(f"   验证集: {len(val_dataset)}")
        print(f"   加载时间: {dataset_load_time:.2f}秒")
    else:
        print("❌ 数据集不可用")
        return
    print()
    
    # 创建模型
    print("2. 创建模型...")
    model = SDF3DModel().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   模型参数: {total_params:,} (可训练: {trainable_params:,})")
    print()
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 如果只测试，验证后退出
    if args.test_only:
        print("3. 测试模式...")
        val_loss = validate(model, val_dataloader, criterion, device)
        print(f"   测试损失: {val_loss:.6f}")
        return
    
    # 训练循环
    print("3. 开始训练...")
    print()
    
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        train_history.append(train_loss)
        
        # 验证
        val_loss = validate(model, val_dataloader, criterion, device)
        val_history.append(val_loss)
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n   Epoch {epoch+1:02d}/{args.epochs:02d} 结果:")
        print(f"     训练损失: {train_loss:.6f}")
        print(f"     验证损失: {val_loss:.6f}")
        print(f"     学习率: {scheduler.get_last_lr()[0]:.6f}")
        print(f"     时间: {epoch_time:.2f}秒")
        print()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(args.save_dir, exist_ok=True)
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }, best_model_path)
            print(f"   💾 最佳模型保存到: {best_model_path}")
            print()
    
    # 保存最终模型
    print("4. 保存最终模型...")
    os.makedirs(args.save_dir, exist_ok=True)
    final_model_path = os.path.join(args.save_dir, 'final_model.pth')
    
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_history': train_history,
        'val_history': val_history,
        'best_val_loss': best_val_loss,
        'config': {
            'data_root': data_root,
            'n_view': n_view,
            'crop_size': crop_size,
            'voxel_size': voxel_size,
            'batch_size': args.batch_size,
            'max_sequences': args.max_sequences,
            'learning_rate': args.lr,
        }
    }, final_model_path)
    
    print(f"   最终模型保存到: {final_model_path}")
    print()
    
    # GPU内存信息
    if device.type == 'cuda':
        print("GPU内存使用:")
        print(f"   已分配: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
        print(f"   缓存: {torch.cuda.memory_reserved(0) / 1e6:.2f} MB")
        print(f"   最大已分配: {torch.cuda.max_memory_allocated(0) / 1e6:.2f} MB")
        print()
    
    print("=" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"最终验证损失: {val_history[-1]:.6f}")
    print(f"模型保存到: {args.save_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()