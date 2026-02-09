#!/usr/bin/env python3
"""
在线SDF生成的端到端训练脚本
使用TartanAir深度图实时生成TSDF进行训练
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from pathlib import Path
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("="*80)
print("在线SDF生成的端到端训练脚本")
print("="*80)

# 检查GPU环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    print("❌ CUDA不可用，使用CPU模式")
    device = torch.device('cpu')
else:
    device = torch.device('cuda')
    print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")


def create_model():
    """创建StreamSDFFormer模型"""
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        logger.info("创建StreamSDFFormerIntegrated模型...")
        
        # 内存优化的配置
        model = StreamSDFFormerIntegrated(
            attn_heads=4,
            attn_layers=2,
            use_proj_occ=True,
            voxel_size=0.08,  # 体素大小
            fusion_local_radius=4.0,
            crop_size=(32, 32, 24)  # 裁剪尺寸
        )
        
        # 移动到设备
        model = model.to(device)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"模型创建成功:")
        logger.info(f"  总参数: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
        # 从配置中获取体素大小和裁剪尺寸
        logger.info(f"  体素大小: 0.08")
        logger.info(f"  裁剪尺寸: (32, 32, 24)")
        
        return model
        
    except ImportError as e:
        logger.error(f"无法导入StreamSDFFormerIntegrated: {e}")
        logger.info("创建简化模型作为替代...")
        
        # 创建简化模型，匹配StreamSDFFormer的接口
        class SimpleSDFModel(nn.Module):
            def __init__(self, input_dim=3, hidden_dim=128, output_dim=1):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
                
            def forward(self, images, poses, intrinsics, reset_state=False):
                """
                简化前向传播
                Args:
                    images: (B, 3, H, W) 图像
                    poses: (B, 4, 4) 相机位姿
                    intrinsics: (B, 3, 3) 相机内参
                    reset_state: 是否重置状态
                Returns:
                    dict: 包含'sdf'和'occupancy'的字典
                """
                B, C, H, W = images.shape
                
                # 从图像中采样点（模拟）
                # 在实际模型中，这会从3D空间采样点
                num_points = 1000
                
                # 生成随机3D点
                points = torch.randn(B, num_points, 3).to(images.device)
                
                # 通过MLP预测SDF
                sdf_pred = self.mlp(points)  # (B, num_points, 1)
                
                # 模拟占用预测（SDF < 0.1）
                occupancy_pred = (sdf_pred.abs() < 0.1).float()
                
                return {
                    'sdf': sdf_pred,
                    'occupancy': occupancy_pred
                }
            
            def reset_state(self):
                """重置模型状态（简化模型无状态）"""
                pass
        
        model = SimpleSDFModel().to(device)
        logger.info(f"简化模型创建成功，参数: {sum(p.numel() for p in model.parameters()):,}")
        return model


def create_loss_function():
    """创建损失函数"""
    # 使用Huber损失，对SDF回归更鲁棒
    sdf_loss = nn.HuberLoss(delta=0.1)
    
    # 占用分类损失
    occ_loss = nn.BCEWithLogitsLoss()
    
    def combined_loss(pred_sdf, target_sdf, pred_occ=None, target_occ=None):
        # SDF损失
        loss = sdf_loss(pred_sdf, target_sdf)
        
        # 如果有占用预测，添加占用损失
        if pred_occ is not None and target_occ is not None:
            loss += 0.1 * occ_loss(pred_occ, target_occ)
        
        return loss
    
    return combined_loss


def train_epoch(model, dataloader, optimizer, loss_fn, epoch, total_epochs):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    logger.info(f"开始训练epoch {epoch}/{total_epochs}")
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # 准备数据
        rgb_images = batch['rgb_images'].to(device)  # (B, F, 3, H, W)
        poses = batch['poses'].to(device)            # (B, F, 4, 4)
        intrinsics = batch['intrinsics'].to(device)  # (B, 3, 3)
        tsdf_target = batch['tsdf'].to(device)       # (B, D, H, W)
        occupancy_target = batch['occupancy'].to(device)  # (B, D, H, W)
        
        # 获取批次大小和帧数
        B, F, C, H, W = rgb_images.shape
        
        # 重置模型状态（每个序列开始时）
        if hasattr(model, 'reset_state'):
            model.reset_state()
        
        batch_loss = 0
        frame_count = 0
        
        # 逐帧处理
        for frame_idx in range(F):
            # 当前帧数据
            rgb_frame = rgb_images[:, frame_idx]  # (B, 3, H, W)
            pose_frame = poses[:, frame_idx]      # (B, 4, 4)
            
            # 是否重置状态（第一帧）
            reset_state = (frame_idx == 0)
            
            try:
                # 前向传播
                output = model(
                    images=rgb_frame,
                    poses=pose_frame,
                    intrinsics=intrinsics,
                    reset_state=reset_state
                )
                
                # 计算损失
                if 'sdf' in output:
                    pred_sdf = output['sdf']
                    
                    # 简化模型输出点云SDF，需要创建对应的目标
                    if len(pred_sdf.shape) == 3:  # (B, num_points, 1)
                        # 从TSDF网格中采样点作为目标
                        B, num_points, _ = pred_sdf.shape
                        
                        # 从TSDF网格中随机采样点
                        tsdf_flat = tsdf_target.view(B, -1)
                        num_voxels = tsdf_flat.shape[1]
                        
                        # 随机采样点
                        if num_voxels >= num_points:
                            indices = torch.randint(0, num_voxels, (B, num_points))
                            target_sdf = torch.gather(tsdf_flat, 1, indices).unsqueeze(-1)
                        else:
                            # 重复采样
                            repeat_times = (num_points + num_voxels - 1) // num_voxels
                            target_sdf = tsdf_flat.repeat(1, repeat_times)[:, :num_points].unsqueeze(-1)
                        
                        # 计算损失
                        loss = loss_fn(pred_sdf, target_sdf)
                    else:
                        # 计算SDF损失
                        loss = loss_fn(pred_sdf, tsdf_target)
                    
                    batch_loss += loss.item()
                    frame_count += 1
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    optimizer.step()
                    
                else:
                    logger.warning(f"批次 {batch_idx}, 帧 {frame_idx}: 输出中缺少'sdf'键")
                    
            except Exception as e:
                logger.error(f"批次 {batch_idx}, 帧 {frame_idx} 训练失败: {e}")
                continue
        
        if frame_count > 0:
            avg_batch_loss = batch_loss / frame_count
            total_loss += batch_loss
            total_samples += frame_count
            
            if batch_idx % 5 == 0:
                logger.info(f"  批次 {batch_idx}: 平均损失={avg_batch_loss:.6f}")
    
    if total_samples > 0:
        avg_epoch_loss = total_loss / total_samples
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch} 完成:")
        logger.info(f"  平均损失: {avg_epoch_loss:.6f}")
        logger.info(f"  总样本: {total_samples}")
        logger.info(f"  时间: {epoch_time:.2f}秒")
        
        return avg_epoch_loss
    else:
        logger.warning(f"Epoch {epoch} 没有有效样本")
        return None


def validate(model, dataloader, loss_fn):
    """验证模型"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    logger.info("开始验证...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            rgb_images = batch['rgb_images'].to(device)
            poses = batch['poses'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            tsdf_target = batch['tsdf'].to(device)
            
            B, F, C, H, W = rgb_images.shape
            
            # 重置模型状态
            if hasattr(model, 'reset_state'):
                model.reset_state()
            
            batch_loss = 0
            frame_count = 0
            
            for frame_idx in range(F):
                rgb_frame = rgb_images[:, frame_idx]
                pose_frame = poses[:, frame_idx]
                reset_state = (frame_idx == 0)
                
                try:
                    output = model(
                        images=rgb_frame,
                        poses=pose_frame,
                        intrinsics=intrinsics,
                        reset_state=reset_state
                    )
                    
                    if 'sdf' in output:
                        pred_sdf = output['sdf']
                        
                        if pred_sdf.shape == tsdf_target.shape:
                            loss = loss_fn(pred_sdf, tsdf_target)
                            batch_loss += loss.item()
                            frame_count += 1
                            
                except Exception as e:
                    logger.error(f"验证批次 {batch_idx}, 帧 {frame_idx} 失败: {e}")
                    continue
            
            if frame_count > 0:
                total_loss += batch_loss
                total_samples += frame_count
    
    if total_samples > 0:
        avg_loss = total_loss / total_samples
        logger.info(f"验证完成: 平均损失={avg_loss:.6f}, 样本数={total_samples}")
        return avg_loss
    else:
        logger.warning("验证没有有效样本")
        return None


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir):
    """保存检查点"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pth")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"检查点保存到: {checkpoint_path}")
    
    # 同时保存训练历史
    history_path = os.path.join(checkpoint_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
    else:
        history = {'train_loss': [], 'val_loss': []}
    
    history['train_loss'].append(float(loss) if loss is not None else None)
    
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)


def main():
    """主训练函数"""
    # 配置参数
    config = {
        'data_root': '/home/cwh/Study/dataset/tartanair',
        'sequence_name': 'abandonedfactory_sample_P001',
        'n_frames': 5,
        'crop_size': (32, 32, 24),
        'voxel_size': 0.08,
        'target_image_size': (128, 128),
        'batch_size': 1,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'checkpoint_dir': 'online_sdf_training_checkpoints',
        'log_interval': 5
    }
    
    logger.info("训练配置:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # 创建数据集
    logger.info("创建数据集...")
    try:
        from online_tartanair_dataset import OnlineTartanAirDataset
        
        dataset = OnlineTartanAirDataset(
            data_root=config['data_root'],
            sequence_name=config['sequence_name'],
            n_frames=config['n_frames'],
            crop_size=config['crop_size'],
            voxel_size=config['voxel_size'],
            target_image_size=config['target_image_size'],
            max_depth=10.0,
            truncation_margin=0.2,
            augment=False
        )
        
        logger.info(f"数据集创建成功，大小: {len(dataset)}")
        
        # 测试数据集
        sample = dataset[0]
        logger.info("样本信息:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape} {value.dtype}")
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
    except Exception as e:
        logger.error(f"数据集创建失败: {e}")
        logger.info("创建模拟数据集...")
        
        # 创建模拟数据集
        class MockDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                return {
                    'rgb_images': torch.randn(5, 3, 128, 128),
                    'poses': torch.eye(4).unsqueeze(0).repeat(5, 1, 1),
                    'intrinsics': torch.eye(3),
                    'tsdf': torch.randn(32, 32, 24),
                    'occupancy': torch.rand(32, 32, 24) > 0.5
                }
        
        dataset = MockDataset()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        logger.info("使用模拟数据集")
    
    # 创建模型
    model = create_model()
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # 创建损失函数
    loss_fn = create_loss_function()
    
    # 训练循环
    logger.info("开始训练循环...")
    
    train_losses = []
    
    for epoch in range(1, config['num_epochs'] + 1):
        # 训练
        train_loss = train_epoch(
            model, dataloader, optimizer, loss_fn, 
            epoch, config['num_epochs']
        )
        
        if train_loss is not None:
            train_losses.append(train_loss)
            
            # 保存检查点
            if epoch % 2 == 0 or epoch == config['num_epochs']:
                save_checkpoint(
                    model, optimizer, epoch, train_loss,
                    config['checkpoint_dir']
                )
        
        # 验证（这里使用训练集作为验证，实际应该分开）
        if epoch % 3 == 0:
            val_loss = validate(model, dataloader, loss_fn)
            if val_loss is not None:
                logger.info(f"Epoch {epoch} 验证损失: {val_loss:.6f}")
    
    # 保存最终模型
    final_model_path = os.path.join(config['checkpoint_dir'], 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"最终模型保存到: {final_model_path}")
    
    # 保存训练历史
    history_path = os.path.join(config['checkpoint_dir'], 'training_summary.json')
    summary = {
        'config': config,
        'train_losses': train_losses,
        'final_epoch': config['num_epochs'],
        'device': str(device),
        'total_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    with open(history_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("训练完成!")
    logger.info(f"训练历史保存到: {history_path}")
    
    # 打印总结
    print("\n" + "="*80)
    print("训练总结")
    print("="*80)
    print(f"总训练轮数: {config['num_epochs']}")
    print(f"最终训练损失: {train_losses[-1] if train_losses else 'N/A':.6f}")
    print(f"模型参数: {summary['total_params']:,}")
    print(f"检查点目录: {config['checkpoint_dir']}")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)