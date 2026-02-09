#!/usr/bin/env python3
"""
快速完整训练 - 测试完整的多序列训练流水线
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

# 简化版3D模型
class Quick3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 1, 3, padding=1)
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
    parser = argparse.ArgumentParser(description='快速完整训练')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小')
    parser.add_argument('--max-samples', type=int, default=50, help='最大样本数')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    args = parser.parse_args()
    
    print("=" * 60)
    print("快速完整训练 - 多序列TartanAir")
    print("=" * 60)
    
    # 配置
    data_root = "/home/cwh/Study/dataset/tartanair"
    n_view = 5
    stride = 2
    crop_size = (48, 48, 32)
    voxel_size = 0.04
    target_image_size = (256, 256)
    max_sequences = 2
    
    print(f"配置:")
    print(f"  数据根目录: {data_root}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  视图数: {n_view}")
    print(f"  最大样本数: {args.max_samples}")
    print(f"  训练轮数: {args.epochs}")
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
    
    # 限制数据集大小
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
    print()
    
    # 创建模型
    print("2. 创建模型...")
    model = Quick3DModel().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   模型参数: {total_params:,} (可训练: {trainable_params:,})")
    print()
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 训练循环
    print("3. 开始训练...")
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # 训练
        model.train()
        train_loss = 0.0
        train_samples = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            batch_start = time.time()
            
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
            train_loss += loss.item() * batch_size
            train_samples += batch_size
            
            batch_time = time.time() - batch_start
            
            # 打印进度
            if (batch_idx + 1) % 2 == 0 or batch_idx == 0:
                avg_loss = train_loss / train_samples if train_samples > 0 else 0.0
                print(f"   Epoch {epoch+1:02d}, Batch {batch_idx+1:03d}/{len(train_dataloader):03d}, "
                      f"Loss: {avg_loss:.6f}, Time: {batch_time:.2f}s")
        
        avg_train_loss = train_loss / train_samples if train_samples > 0 else 0.0
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                rgb_images = batch['rgb_images'].to(device)
                tsdf_gt = batch['tsdf'].to(device)
                
                input_3d = prepare_input(rgb_images, tsdf_gt, frame_idx=0)
                output = model(input_3d)
                
                loss = criterion(output, tsdf_gt)
                
                val_loss += loss.item() * rgb_images.shape[0]
                val_samples += rgb_images.shape[0]
        
        avg_val_loss = val_loss / val_samples if val_samples > 0 else 0.0
        
        epoch_time = time.time() - epoch_start
        
        print(f"\n   Epoch {epoch+1:02d} 结果:")
        print(f"     训练损失: {avg_train_loss:.6f}")
        print(f"     验证损失: {avg_val_loss:.6f}")
        print(f"     时间: {epoch_time:.2f}秒")
        print()
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
    
    # 最终测试
    print("4. 最终测试...")
    model.eval()
    
    test_loss = 0.0
    test_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            rgb_images = batch['rgb_images'].to(device)
            tsdf_gt = batch['tsdf'].to(device)
            
            input_3d = prepare_input(rgb_images, tsdf_gt, frame_idx=0)
            output = model(input_3d)
            
            loss = criterion(output, tsdf_gt)
            
            test_loss += loss.item() * rgb_images.shape[0]
            test_samples += rgb_images.shape[0]
            
            if batch_idx == 0:
                print(f"   测试批次 {batch_idx+1}:")
                print(f"     输入形状: {input_3d.shape}")
                print(f"     输出形状: {output.shape}")
                print(f"     目标形状: {tsdf_gt.shape}")
                print(f"     损失: {loss.item():.6f}")
    
    avg_test_loss = test_loss / test_samples if test_samples > 0 else 0.0
    print(f"\n   最终测试损失: {avg_test_loss:.6f}")
    print()
    
    # 保存模型
    print("5. 保存模型...")
    os.makedirs("quick_checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"quick_checkpoints/model_{timestamp}.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
        'test_loss': avg_test_loss,
        'config': {
            'data_root': data_root,
            'n_view': n_view,
            'crop_size': crop_size,
            'voxel_size': voxel_size,
        }
    }, model_path)
    
    print(f"   模型已保存到: {model_path}")
    print()
    
    # GPU内存信息
    if device.type == 'cuda':
        print("GPU内存使用:")
        print(f"   已分配: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
        print(f"   缓存: {torch.cuda.memory_reserved(0) / 1e6:.2f} MB")
        print(f"   最大已分配: {torch.cuda.max_memory_allocated(0) / 1e6:.2f} MB")
        print()
    
    print("=" * 60)
    print("快速完整训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    print(f"最终测试损失: {avg_test_loss:.6f}")
    print(f"模型保存到: {model_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()