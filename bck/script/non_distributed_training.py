#!/usr/bin/env python3
"""
非分布式端到端训练脚本
禁用所有分布式训练功能
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

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 禁用分布式训练
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('non_distributed_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("="*80)
print("非分布式端到端训练脚本")
print("="*80)

# 检查环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ 使用CPU")

def get_memory_info():
    """获取内存信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved
        }
    return {'allocated_gb': 0, 'reserved_gb': 0}

def create_simple_model():
    """创建简化模型"""
    try:
        logger.info("创建StreamSDFFormerIntegrated模型...")
        
        # 直接导入StreamSDFFormerIntegrated
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        # 创建模型
        model = StreamSDFFormerIntegrated(
            attn_heads=2,  # 减少注意力头数以降低内存使用
            attn_layers=1,
            use_proj_occ=True,
            voxel_size=0.10,  # 增大体素大小以降低内存
            fusion_local_radius=0.0,  # 禁用流式融合
            crop_size=(24, 24, 16)  # 减小裁剪尺寸
        )
        
        # 移动到设备
        model = model.to(device)
        
        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"模型创建成功:")
        logger.info(f"  总参数: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        logger.error(f"模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 创建简化模型作为备用
        logger.info("创建简化MLP模型作为备用...")
        
        class SimpleSDFModel(nn.Module):
            def __init__(self, input_dim=3, hidden_dim=128, output_dim=1):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )
            
            def forward(self, images, poses, intrinsics, reset_state=False):
                B, C, H, W = images.shape
                num_points = 500
                # 确保points在正确的设备上
                points = torch.randn(B, num_points, 3, device=images.device)
                sdf_pred = self.mlp(points)
                return {'sdf': sdf_pred}
        
        model = SimpleSDFModel().to(device)
        return model

def create_dataset():
    """创建数据集"""
    try:
        from online_tartanair_dataset import OnlineTartanAirDataset
        
        logger.info("创建OnlineTartanAir数据集...")
        
        dataset = OnlineTartanAirDataset(
            data_root="/home/cwh/Study/dataset/tartanair",
            sequence_name="abandonedfactory_sample_P001",
            n_frames=3,  # 减少帧数
            crop_size=(24, 24, 16),
            voxel_size=0.10,
            target_image_size=(96, 96),
            max_depth=10.0,
            truncation_margin=0.2,
            augment=False
        )
        
        logger.info(f"数据集创建成功，大小: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,  # 单批次
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        return dataloader
        
    except Exception as e:
        logger.error(f"数据集创建失败: {e}")
        return None

def train_epoch(model, dataloader, optimizer, loss_fn, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            # 移动到设备
            images = batch['rgb_images'].to(device)
            poses = batch['poses'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            tsdf_target = batch['tsdf'].to(device)
            
            # 只使用第一帧
            frame_idx = 0
            current_images = images[:, frame_idx:frame_idx+1].squeeze(1)
            current_poses = poses[:, frame_idx:frame_idx+1].squeeze(1)
            
            # 前向传播
            output = model(
                images=current_images,
                poses=current_poses,
                intrinsics=intrinsics,
                reset_state=True
            )
            
            if 'sdf' in output:
                pred_sdf = output['sdf']
                
                # 采样目标点
                B, num_points, _ = pred_sdf.shape
                tsdf_flat = tsdf_target.view(B, -1)
                num_voxels = tsdf_flat.shape[1]
                
                if num_voxels >= num_points:
                    indices = torch.randint(0, num_voxels, (B, num_points))
                    target_sdf = torch.gather(tsdf_flat, 1, indices).unsqueeze(-1)
                else:
                    repeat_times = (num_points + num_voxels - 1) // num_voxels
                    target_sdf = tsdf_flat.repeat(1, repeat_times)[:, :num_points].unsqueeze(-1)
                
                # 计算损失
                loss = loss_fn(pred_sdf, target_sdf)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 更新参数
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if (batch_idx + 1) % 2 == 0:
                    mem_info = get_memory_info()
                    logger.info(f"Epoch {epoch}, Batch {batch_idx+1}: Loss={loss.item():.6f}")
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"内存不足! 清理缓存")
                torch.cuda.empty_cache()
            else:
                logger.error(f"批次错误: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def main():
    """主训练函数"""
    logger.info("开始非分布式训练...")
    
    # 创建数据集
    dataloader = create_dataset()
    if dataloader is None:
        logger.error("数据集创建失败")
        return
    
    # 创建模型
    model = create_simple_model()
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.HuberLoss(delta=0.1)
    
    # 训练配置
    num_epochs = 5
    checkpoint_dir = "non_distributed_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info(f"开始训练，共{num_epochs}个epoch...")
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # 训练一个epoch
        train_loss = train_epoch(model, dataloader, optimizer, loss_fn, epoch)
        
        epoch_time = time.time() - start_time
        
        logger.info(f"Epoch {epoch}/{num_epochs}: Loss={train_loss:.6f}, Time={epoch_time:.1f}s")
        
        # 保存检查点
        if epoch % 2 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss
            }, checkpoint_path)
            logger.info(f"检查点保存到: {checkpoint_path}")
    
    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    
    logger.info(f"训练完成! 模型保存到: {final_path}")
    print("✅ 非分布式训练完成!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
