#!/usr/bin/env python3
"""
多序列TartanAir训练脚本
使用MultiSequenceTartanAirDataset进行训练
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import logging
from datetime import datetime
import json
from torch.utils.data import DataLoader

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_sequence_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("="*80)
print("多序列TartanAir训练")
print("="*80)

# 检查环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    
    # 检查GPU内存
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU内存: {total_memory:.1f}GB")
else:
    device = torch.device("cpu")
    print("⚠️  CUDA不可用，使用CPU")

# 导入模型和数据集
try:
    from former3d.model import StreamSDFFormer
    from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
    print("✅ 成功导入模型和数据集")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def create_model():
    """创建StreamSDFFormer模型"""
    model = StreamSDFFormer(
        hidden_dim=128,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        voxel_size=0.04,
        grid_size=(48, 48, 32),
        use_sparse=True
    )
    
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: {total_params:,} 总参数, {trainable_params:,} 可训练参数")
    
    return model


def create_dataset():
    """创建多序列数据集"""
    # 使用样本数据目录
    data_root = "/home/cwh/coding/former3d/files/sample_tartanair"
    
    if not os.path.exists(data_root):
        print(f"⚠️  数据目录不存在: {data_root}")
        print("创建模拟数据集...")
        # 这里可以添加模拟数据集的创建逻辑
        return None
    
    dataset = MultiSequenceTartanAirDataset(
        data_root=data_root,
        n_view=5,  # 每个片段5帧
        stride=2,  # 片段步长
        crop_size=(48, 48, 32),
        voxel_size=0.04,
        target_image_size=(256, 256),
        max_depth=10.0,
        truncation_margin=0.2,
        augment=False,
        max_sequences=3,  # 限制序列数量用于测试
        shuffle=True
    )
    
    return dataset


def train_epoch(model, dataloader, optimizer, criterion, epoch, total_epochs):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_batches = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # 将数据移动到设备
        rgb_images = batch['rgb_images'].to(device)  # (batch_size, n_view, 3, H, W)
        poses = batch['poses'].to(device)            # (batch_size, n_view, 4, 4)
        tsdf_target = batch['tsdf'].to(device)       # (batch_size, 1, D, H, W)
        intrinsics = batch['intrinsics'].to(device)  # (3, 3) 或 (batch_size, 3, 3)
        
        batch_size, n_view = rgb_images.shape[:2]
        
        # 重置模型状态（每个片段开始时重置）
        model.reset_state(batch_size=batch_size)
        
        # 遍历每个时刻进行前向传播
        total_frame_loss = 0.0
        
        for frame_idx in range(n_view):
            # 提取当前帧
            current_images = rgb_images[:, frame_idx]  # (batch_size, 3, H, W)
            current_poses = poses[:, frame_idx]        # (batch_size, 4, 4)
            
            # 前向传播
            tsdf_pred = model(
                images=current_images,
                poses=current_poses,
                intrinsics=intrinsics
            )
            
            # 计算损失（只对最后一个帧的预测计算损失）
            if frame_idx == n_view - 1:
                loss = criterion(tsdf_pred, tsdf_target.squeeze(1))  # 移除通道维度
                total_frame_loss += loss
        
        # 平均损失
        avg_loss = total_frame_loss / n_view
        
        # 反向传播
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        
        total_loss += avg_loss.item()
        total_batches += 1
        
        # 每10个批次打印一次进度
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_batch_time = elapsed / (batch_idx + 1)
            remaining_batches = len(dataloader) - (batch_idx + 1)
            remaining_time = avg_batch_time * remaining_batches
            
            print(f"  Epoch {epoch}/{total_epochs} | "
                  f"Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss.item():.6f} | "
                  f"Time: {elapsed:.1f}s | "
                  f"ETA: {remaining_time:.1f}s")
    
    avg_loss = total_loss / max(total_batches, 1)
    epoch_time = time.time() - start_time
    
    return avg_loss, epoch_time


def validate(model, dataloader, criterion):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            rgb_images = batch['rgb_images'].to(device)
            poses = batch['poses'].to(device)
            tsdf_target = batch['tsdf'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            
            batch_size, n_view = rgb_images.shape[:2]
            
            # 重置模型状态
            model.reset_state(batch_size=batch_size)
            
            total_frame_loss = 0.0
            
            for frame_idx in range(n_view):
                current_images = rgb_images[:, frame_idx]
                current_poses = poses[:, frame_idx]
                
                tsdf_pred = model(
                    images=current_images,
                    poses=current_poses,
                    intrinsics=intrinsics
                )
                
                if frame_idx == n_view - 1:
                    loss = criterion(tsdf_pred, tsdf_target.squeeze(1))
                    total_frame_loss += loss
            
            avg_loss = total_frame_loss / n_view
            total_loss += avg_loss.item()
            total_batches += 1
            
            # 只验证部分批次
            if batch_idx >= 10:  # 验证10个批次
                break
    
    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss


def main():
    """主训练函数"""
    print("\n" + "="*80)
    print("开始多序列训练")
    print("="*80)
    
    # 创建数据集
    print("\n创建数据集...")
    dataset = create_dataset()
    
    if dataset is None:
        print("❌ 数据集创建失败")
        return
    
    print(f"数据集大小: {len(dataset)} 个片段")
    
    # 创建数据加载器
    batch_size = 2  # 小批量用于测试
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0表示不使用多进程
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"数据加载器: batch_size={batch_size}, 总批次={len(dataloader)}")
    
    # 创建模型
    print("\n创建模型...")
    model = create_model()
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=10,  # 10个epoch
        eta_min=1e-6
    )
    
    # 训练配置
    num_epochs = 10
    best_loss = float('inf')
    
    print(f"\n开始训练 ({num_epochs}个epoch)...")
    print("-"*80)
    
    # 训练循环
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-"*40)
        
        # 训练
        train_loss, epoch_time = train_epoch(
            model, dataloader, optimizer, criterion, epoch, num_epochs
        )
        
        # 验证
        val_loss = validate(model, dataloader, criterion)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印结果
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  LR:         {current_lr:.2e}")
        print(f"  Time:       {epoch_time:.1f}s")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': best_loss,
            }, 'best_multi_sequence_model.pth')
            print(f"  ✅ 保存最佳模型 (val_loss: {val_loss:.6f})")
        
        # 每5个epoch保存一次检查点
        if epoch % 5 == 0:
            checkpoint_path = f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  💾 保存检查点: {checkpoint_path}")
    
    print("\n" + "="*80)
    print("训练完成!")
    print(f"最佳验证损失: {best_loss:.6f}")
    print("="*80)
    
    # 保存最终模型
    final_model_path = 'final_multi_sequence_model.pth'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss,
    }, final_model_path)
    print(f"最终模型保存到: {final_model_path}")
    
    # 打印GPU内存使用情况
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU内存使用: {allocated:.2f}GB 已分配, {cached:.2f}GB 已缓存")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()