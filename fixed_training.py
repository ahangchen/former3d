#!/usr/bin/env python3
"""
修复版训练脚本 - 使用简单的MLP模型进行SDF学习
"""

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from online_tartanair_dataset import OnlineTartanAirDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fixed_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleSDFModel(nn.Module):
    """简单的SDF模型，用于验证训练流程"""
    
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=1):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化权重
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: 输入坐标 [batch, 3] 或 [batch, num_points, 3]
        
        Returns:
            SDF值 [batch, 1] 或 [batch, num_points, 1]
        """
        # 处理不同形状的输入
        if x.dim() == 3:
            batch_size, num_points, _ = x.shape
            x = x.view(-1, 3)  # [batch*num_points, 3]
            sdf = self.network(x)
            sdf = sdf.view(batch_size, num_points, -1)  # [batch, num_points, 1]
        else:
            sdf = self.network(x)  # [batch, 1]
        
        return sdf

def sample_points_from_tsdf(tsdf_grid, voxel_coords, num_points=1024):
    """从TSDF网格中采样点
    
    Args:
        tsdf_grid: TSDF值 [D, H, W]
        voxel_coords: 体素坐标 [D, H, W, 3]
        num_points: 采样点数
    
    Returns:
        points: 采样点坐标 [num_points, 3]
        sdf_values: 对应的SDF值 [num_points, 1]
    """
    # 展平网格
    tsdf_flat = tsdf_grid.reshape(-1)
    coords_flat = voxel_coords.reshape(-1, 3)
    
    # 过滤无效值（TSDF接近1表示未知区域）
    valid_mask = torch.abs(tsdf_flat) < 0.999
    valid_tsdf = tsdf_flat[valid_mask]
    valid_coords = coords_flat[valid_mask]
    
    if len(valid_tsdf) == 0:
        # 如果没有有效点，返回随机点
        return torch.randn(num_points, 3), torch.zeros(num_points, 1)
    
    # 根据TSDF值的重要性采样
    # 表面附近的点更重要（|TSDF| < 0.1）
    surface_mask = torch.abs(valid_tsdf) < 0.1
    surface_coords = valid_coords[surface_mask]
    surface_tsdf = valid_tsdf[surface_mask]
    
    # 采样策略：优先采样表面点
    if len(surface_coords) > num_points // 2:
        # 有足够的表面点
        surface_indices = torch.randperm(len(surface_coords))[:num_points // 2]
        other_indices = torch.randperm(len(valid_coords))[:num_points // 2]
        
        points = torch.cat([
            surface_coords[surface_indices],
            valid_coords[other_indices]
        ], dim=0)
        
        sdf_values = torch.cat([
            surface_tsdf[surface_indices].unsqueeze(1),
            valid_tsdf[other_indices].unsqueeze(1)
        ], dim=0)
    else:
        # 表面点不足，全部使用
        if len(surface_coords) > 0:
            points = surface_coords
            sdf_values = surface_tsdf.unsqueeze(1)
            
            # 补充随机点
            if len(points) < num_points:
                remaining = num_points - len(points)
                random_indices = torch.randperm(len(valid_coords))[:remaining]
                points = torch.cat([points, valid_coords[random_indices]], dim=0)
                sdf_values = torch.cat([
                    sdf_values, 
                    valid_tsdf[random_indices].unsqueeze(1)
                ], dim=0)
        else:
            # 完全没有表面点，随机采样
            indices = torch.randperm(len(valid_coords))[:num_points]
            points = valid_coords[indices]
            sdf_values = valid_tsdf[indices].unsqueeze(1)
    
    return points, sdf_values

def train_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 将数据移动到设备
        tsdf_grid = batch['tsdf'].to(device)  # [batch, D, H, W]
        voxel_coords = batch['voxel_coords'].to(device)  # [batch, D, H, W, 3]
        
        # 采样训练点
        points_list = []
        sdf_list = []
        
        for i in range(tsdf_grid.shape[0]):
            points, sdf_values = sample_points_from_tsdf(
                tsdf_grid[i], voxel_coords[i], num_points=1024
            )
            points_list.append(points)
            sdf_list.append(sdf_values)
        
        # 堆叠批次
        points = torch.stack(points_list, dim=0)  # [batch, num_points, 3]
        sdf_gt = torch.stack(sdf_list, dim=0)  # [batch, num_points, 1]
        
        # 前向传播
        optimizer.zero_grad()
        sdf_pred = model(points)  # [batch, num_points, 1]
        
        # 计算损失（Huber损失，对异常值更鲁棒）
        loss = F.huber_loss(sdf_pred, sdf_gt, delta=0.1)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 优化器步进
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss = {loss.item():.6f}")
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss

def validate(model, dataloader, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            tsdf_grid = batch['tsdf'].to(device)
            voxel_coords = batch['voxel_coords'].to(device)
            
            # 采样验证点
            points_list = []
            sdf_list = []
            
            for i in range(tsdf_grid.shape[0]):
                points, sdf_values = sample_points_from_tsdf(
                    tsdf_grid[i], voxel_coords[i], num_points=512
                )
                points_list.append(points)
                sdf_list.append(sdf_values)
            
            points = torch.stack(points_list, dim=0)
            sdf_gt = torch.stack(sdf_list, dim=0)
            
            # 前向传播
            sdf_pred = model(points)
            
            # 计算损失
            loss = F.huber_loss(sdf_pred, sdf_gt, delta=0.1)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss

def main():
    """主训练函数"""
    logger.info("开始修复版训练...")
    
    # 配置参数
    config = {
        'data_root': '/home/cwh/Study/dataset/tartanair',
        'sequence_name': 'abandonedfactory_sample_P001',
        'batch_size': 1,
        'n_frames': 4,
        'crop_size': (32, 32, 24),
        'voxel_size': 0.08,
        'image_size': (128, 128),
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'checkpoint_dir': 'fixed_checkpoints',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info("训练配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 创建输出目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # 设置设备
    device = torch.device(config['device'])
    logger.info(f"使用设备: {device}")
    
    # 创建数据集
    logger.info("创建数据集...")
    dataset = OnlineTartanAirDataset(
        data_root=config['data_root'],
        sequence_name=config['sequence_name'],
        n_frames=config['n_frames'],
        crop_size=config['crop_size'],
        voxel_size=config['voxel_size'],
        target_image_size=config['image_size'],
        max_depth=10.0,
        truncation_margin=0.2,
        augment=False
    )
    
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # 创建模型
    logger.info("创建模型...")
    model = SimpleSDFModel(input_dim=3, hidden_dim=256, output_dim=1)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: {total_params:,} (可训练: {trainable_params:,})")
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    # 训练循环
    logger.info(f"开始训练，共{config['num_epochs']}个epoch...")
    
    best_loss = float('inf')
    train_history = []
    
    for epoch in range(1, config['num_epochs'] + 1):
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch(model, dataloader, optimizer, device, epoch)
        
        # 验证（使用相同的数据集）
        val_loss = validate(model, dataloader, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        epoch_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch}/{config['num_epochs']}: "
                   f"Train Loss = {train_loss:.6f}, "
                   f"Val Loss = {val_loss:.6f}, "
                   f"LR = {current_lr:.2e}, "
                   f"Time = {epoch_time:.1f}s")
        
        train_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'time': epoch_time
        })
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = os.path.join(config['checkpoint_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"保存最佳模型到 {checkpoint_path} (损失 = {val_loss:.6f})")
        
        # 定期保存检查点
        if epoch % 2 == 0:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"保存检查点到 {checkpoint_path}")
    
    # 保存最终模型
    final_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_history': train_history
    }, final_path)
    logger.info(f"保存最终模型到 {final_path}")
    
    # 打印训练总结
    logger.info("训练完成!")
    logger.info(f"最佳验证损失: {best_loss:.6f}")
    logger.info(f"总训练时间: {sum(h['time'] for h in train_history):.1f}s")
    
    # 保存训练历史
    history_path = os.path.join(config['checkpoint_dir'], 'training_history.npy')
    np.save(history_path, train_history)
    logger.info(f"训练历史保存到 {history_path}")

if __name__ == '__main__':
    main()