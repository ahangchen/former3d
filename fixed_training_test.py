#!/usr/bin/env python3
"""
修复设备一致性后的训练测试
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

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fixed_training_test.log')
    ]
)
logger = logging.getLogger(__name__)

def get_memory_info():
    """获取GPU内存信息"""
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
            attn_heads=2,
            attn_layers=1,
            use_proj_occ=True,
            voxel_size=0.10,
            fusion_local_radius=0.0,
            crop_size=(24, 24, 16)
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
        
        # 创建备用简化模型
        class SimpleSDFModel(nn.Module):
            def __init__(self, input_dim=3, hidden_dim=64, output_dim=1):
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
        logger.info("创建简化MLP模型作为备用...")
        return model

def create_mock_dataloader():
    """创建模拟数据加载器"""
    class MockDataset:
        def __init__(self, size=10):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # 创建模拟数据，确保所有张量都在CPU上
            # 注意：StreamSDFFormerIntegrated期望的形状：
            # images: [3, H, W]  # 单帧，无batch维度
            # poses: [4, 4]      # 单帧，无batch维度
            # intrinsics: [3, 3] # 单帧，无batch维度
            batch = {
                'rgb_images': torch.randn(3, 96, 96),  # [3, H, W]
                'poses': torch.randn(4, 4),            # [4, 4]
                'intrinsics': torch.randn(3, 3),       # [3, 3]
                'tsdf': torch.randn(24, 24, 16)        # [H, W, D]
            }
            return batch
    
    from torch.utils.data import DataLoader
    dataset = MockDataset(size=5)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    return dataloader

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            # 将所有数据移动到设备
            images = batch['rgb_images'].to(device)
            poses = batch['poses'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            tsdf_target = batch['tsdf'].to(device)
            
            # 前向传播 - 使用forward_single_frame
            print(f"DEBUG: 输入形状 - images: {images.shape}, poses: {poses.shape}, intrinsics: {intrinsics.shape}")
            output, state = model.forward_single_frame(images, poses, intrinsics, reset_state=(batch_idx == 0))
            
            print(f"DEBUG: 输出类型: {type(output)}, 输出keys: {list(output.keys()) if isinstance(output, dict) else 'N/A'}")
            
            if 'sdf' in output:
                pred_sdf = output['sdf']
                print(f"DEBUG: pred_sdf形状: {pred_sdf.shape}")
                
                # 模型输出形状是 [num_points, 1]，我们需要采样相同数量的目标点
                num_points = pred_sdf.shape[0]
                
                # 将tsdf_target展平并采样
                tsdf_flat = tsdf_target.view(-1)  # 展平为1D
                num_voxels = tsdf_flat.shape[0]
                
                if num_voxels >= num_points:
                    indices = torch.randint(0, num_voxels, (num_points,), device=device)
                    target_sdf = torch.gather(tsdf_flat, 0, indices).unsqueeze(-1)
                else:
                    # 如果体素数量不足，重复采样
                    repeat_times = (num_points + num_voxels - 1) // num_voxels
                    target_sdf = tsdf_flat.repeat(repeat_times)[:num_points].unsqueeze(-1)
                
                # 确保目标张量在正确设备上
                target_sdf = target_sdf.to(device)
                
                # 计算损失 - 现在两个张量都是 [num_points, 1]
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
                
                logger.info(f"Epoch {epoch}, Batch {batch_idx+1}: Loss={loss.item():.6f}")
        
        except RuntimeError as e:
            logger.error(f"批次错误: {e}")
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def main():
    """主训练函数"""
    logger.info("开始设备一致性修复测试...")
    
    # 设置设备
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("❌ 使用CPU")
    
    # 创建模拟数据加载器
    dataloader = create_mock_dataloader()
    logger.info(f"模拟数据加载器创建成功，批次: {len(dataloader)}")
    
    # 创建模型
    model = create_simple_model()
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.HuberLoss(delta=0.1)
    
    # 训练循环
    num_epochs = 3
    logger.info(f"开始训练，共{num_epochs}个epoch...")
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # 训练一个epoch
        avg_loss = train_one_epoch(model, dataloader, optimizer, loss_fn, device, epoch)
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch}/{num_epochs}: 平均损失={avg_loss:.6f}, 时间={epoch_time:.1f}s")
        
        # 保存检查点
        if epoch % 2 == 0:
            checkpoint_path = f"fixed_checkpoints/epoch_{epoch}.pth"
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"检查点保存到: {checkpoint_path}")
    
    # 保存最终模型
    final_path = "fixed_checkpoints/final_model.pth"
    torch.save(model.state_dict(), final_path)
    logger.info(f"训练完成! 模型保存到: {final_path}")
    
    # 内存使用报告
    mem_info = get_memory_info()
    logger.info(f"最终内存使用: 分配={mem_info['allocated_gb']:.2f}GB, 保留={mem_info['reserved_gb']:.2f}GB")

if __name__ == "__main__":
    main()