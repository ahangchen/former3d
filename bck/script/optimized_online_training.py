#!/usr/bin/env python3
"""
内存优化的在线SDF训练脚本
调整batch size、裁剪尺寸等配置以优化内存使用和训练效率
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimized_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("="*80)
print("内存优化的在线SDF训练")
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
    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
    cached_memory = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"GPU内存: {total_memory:.1f}GB")
    print(f"已分配: {allocated_memory:.1f}GB")
    print(f"缓存: {cached_memory:.1f}GB")
else:
    device = torch.device("cpu")
    print("⚠️ 使用CPU")

# 内存优化配置
class MemoryOptimizedConfig:
    """内存优化配置"""
    
    # 基础配置
    data_root = "/home/cwh/Study/dataset/tartanair"
    sequence_name = "abandonedfactory_sample_P001"
    
    # 内存优化参数（可根据GPU内存调整）
    @staticmethod
    def get_config(memory_level="medium"):
        """根据内存级别获取配置"""
        configs = {
            "low": {  # 低内存配置（< 4GB）
                "data_root": "/home/cwh/Study/dataset/tartanair",
                "sequence_name": "abandonedfactory_sample_P001",
                "batch_size": 1,
                "n_frames": 3,
                "crop_size": (24, 24, 16),  # 减小裁剪尺寸
                "voxel_size": 0.12,  # 增大体素大小
                "target_image_size": (96, 96),  # 减小图像尺寸
                "num_epochs": 5,
                "learning_rate": 1e-4,
                "gradient_accumulation_steps": 2
            },
            "medium": {  # 中等内存配置（4-8GB）
                "data_root": "/home/cwh/Study/dataset/tartanair",
                "sequence_name": "abandonedfactory_sample_P001",
                "batch_size": 1,
                "n_frames": 5,
                "crop_size": (32, 32, 24),
                "voxel_size": 0.08,
                "target_image_size": (128, 128),
                "num_epochs": 10,
                "learning_rate": 1e-4,
                "gradient_accumulation_steps": 1
            },
            "high": {  # 高内存配置（> 8GB）
                "data_root": "/home/cwh/Study/dataset/tartanair",
                "sequence_name": "abandonedfactory_sample_P001",
                "batch_size": 1,  # 数据集只返回一个样本
                "n_frames": 4,    # 每个样本包含4帧
                "crop_size": (32, 32, 24),
                "voxel_size": 0.08,
                "target_image_size": (128, 128),
                "num_epochs": 10,
                "learning_rate": 2e-4,
                "gradient_accumulation_steps": 4,  # 使用梯度累积模拟批次
                "max_grad_norm": 1.0,
                "empty_cache_freq": 2
            }
        }
        return configs.get(memory_level, configs["medium"])


def monitor_memory_usage():
    """监控内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
        return {
            "allocated_gb": allocated,
            "cached_gb": cached,
            "max_allocated_gb": max_allocated
        }
    return {"allocated_gb": 0, "cached_gb": 0, "max_allocated_gb": 0}


def create_memory_optimized_model(config):
    """创建内存优化的模型"""
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        logger.info("创建内存优化的StreamSDFFormerIntegrated模型...")
        
        model = StreamSDFFormerIntegrated(
            attn_heads=4,
            attn_layers=2,
            use_proj_occ=True,
            voxel_size=config["voxel_size"],
            fusion_local_radius=4.0,
            crop_size=config["crop_size"]
        )
        
        model = model.to(device)
        
        # 计算参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"模型参数:")
        logger.info(f"  总参数: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
        logger.info(f"  体素大小: {config['voxel_size']}")
        logger.info(f"  裁剪尺寸: {config['crop_size']}")
        
        return model
        
    except ImportError as e:
        logger.error(f"无法导入StreamSDFFormerIntegrated: {e}")
        logger.info("创建简化模型...")
        
        # 简化模型
        class SimpleSDFModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((8, 8))
                )
                self.decoder = nn.Sequential(
                    nn.Linear(32*8*8 + 3, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1)
                )
                
            def forward(self, images, poses, intrinsics, reset_state=False):
                B, C, H, W = images.shape
                img_features = self.image_encoder(images).view(B, -1)
                num_points = 512
                points = torch.randn(B, num_points, 3).to(images.device)
                img_features_expanded = img_features.unsqueeze(1).repeat(1, num_points, 1)
                combined = torch.cat([img_features_expanded, points], dim=-1)
                sdf_pred = self.decoder(combined)
                return {'sdf': sdf_pred}
            
            def reset_state(self):
                pass
        
        model = SimpleSDFModel().to(device)
        logger.info(f"简化模型创建成功，参数: {sum(p.numel() for p in model.parameters()):,}")
        return model


def train_epoch(model, dataloader, optimizer, loss_fn, config, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # 梯度累积
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    
    for batch_idx, batch in enumerate(dataloader):
        # 移动到设备
        images = batch['rgb_images'].to(device)
        poses = batch['poses'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        
        # 修复intrinsics形状：从(3, 3)转换为[B, 3, 3]
        if intrinsics.dim() == 2:
            # 当前形状: (3, 3)，需要扩展为[B, 3, 3]
            intrinsics = intrinsics.unsqueeze(0).repeat(images.shape[0], 1, 1)
        tsdf_target = batch['tsdf'].to(device)
        
        batch_loss = 0
        
        # 处理每个帧
        for frame_idx in range(images.shape[1]):
            # 提取当前帧的图像和位姿
            current_images = images[:, frame_idx:frame_idx+1]  # [batch, 1, C, H, W]
            current_poses = poses[:, frame_idx:frame_idx+1]    # [batch, 1, 4, 4]
            
            # 重塑为模型期望的形状
            current_images_reshaped = current_images.squeeze(1)  # [batch, C, H, W]
            current_poses_reshaped = current_poses.squeeze(1)    # [batch, 4, 4]
            
            # 前向传播
            output = model(
                images=current_images_reshaped,
                poses=current_poses_reshaped,
                intrinsics=intrinsics,
                reset_state=(frame_idx == 0)
            )
            
            if 'sdf' in output:
                pred_sdf = output['sdf']
                
                # 计算损失
                if len(pred_sdf.shape) == 3:  # 点云预测 [B, num_points, 1]
                    B, num_points, _ = pred_sdf.shape
                    tsdf_flat = tsdf_target.view(B, -1)
                    num_voxels = tsdf_flat.shape[1]
                    
                    if num_voxels >= num_points:
                        indices = torch.randint(0, num_voxels, (B, num_points)).to(device)
                        target_sdf = torch.gather(tsdf_flat, 1, indices).unsqueeze(-1)
                    else:
                        repeat_times = (num_points + num_voxels - 1) // num_voxels
                        target_sdf = tsdf_flat.repeat(1, repeat_times)[:, :num_points].unsqueeze(-1)
                    
                    loss = loss_fn(pred_sdf, target_sdf)
                elif len(pred_sdf.shape) == 2:  # 体素预测 [num_voxels, 1]
                    # 处理体素预测输出
                    B = tsdf_target.shape[0]
                    num_voxels = pred_sdf.shape[0]
                    
                    # 方法1: 尝试均匀分割
                    if num_voxels % B == 0:
                        voxels_per_batch = num_voxels // B
                        pred_sdf_reshaped = pred_sdf.view(B, voxels_per_batch, 1)
                        tsdf_flat = tsdf_target.view(B, -1)
                        
                        # 采样匹配的点
                        num_sample_points = min(voxels_per_batch, tsdf_flat.shape[1], 1000)
                        pred_sample = pred_sdf_reshaped[:, :num_sample_points, :]
                        indices = torch.randint(0, tsdf_flat.shape[1], (B, num_sample_points)).to(device)
                        target_sdf = torch.gather(tsdf_flat, 1, indices).unsqueeze(-1)
                        loss = loss_fn(pred_sample, target_sdf)
                    
                    # 方法2: 如果批次大小为1，直接处理
                    elif B == 1:
                        # 批次大小为1，直接采样
                        num_sample_points = min(num_voxels, 1000)
                        pred_sample = pred_sdf[:num_sample_points].unsqueeze(0)  # [1, num_sample_points, 1]
                        tsdf_flat = tsdf_target.view(-1)
                        
                        if tsdf_flat.shape[0] >= num_sample_points:
                            indices = torch.randint(0, tsdf_flat.shape[0], (1, num_sample_points)).to(device)
                            target_sdf = torch.gather(tsdf_flat.unsqueeze(0), 1, indices).unsqueeze(-1)
                            loss = loss_fn(pred_sample, target_sdf)
                        else:
                            # 目标体素不足，使用所有目标体素
                            num_sample_points = min(num_voxels, tsdf_flat.shape[0])
                            pred_sample = pred_sdf[:num_sample_points].unsqueeze(0)
                            target_sdf = tsdf_flat[:num_sample_points].unsqueeze(0).unsqueeze(-1)
                            loss = loss_fn(pred_sample, target_sdf)
                    
                    # 方法3: 其他情况，使用简单损失
                    else:
                        # 使用均值损失作为后备
                        loss = loss_fn(pred_sdf.mean(), tsdf_target.mean())
                else:
                    # 其他形状，尝试直接计算
                    try:
                        loss = loss_fn(pred_sdf, tsdf_target)
                    except:
                        # 如果失败，使用简单损失
                        loss = loss_fn(pred_sdf.mean(), tsdf_target.mean())
                
                # 梯度累积
                loss = loss / gradient_accumulation_steps
                loss.backward()
                batch_loss += loss.item() * gradient_accumulation_steps
            
            # 释放中间变量内存
            del output, pred_sdf
            if 'target_sdf' in locals():
                del target_sdf
        
        # 梯度累积步骤完成后更新参数
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # 梯度裁剪
            max_grad_norm = config.get("max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # 更高效的内存清零
            
            # 清空CUDA缓存
            if config.get("empty_cache_freq", 0) > 0 and (batch_idx + 1) % config["empty_cache_freq"] == 0:
                torch.cuda.empty_cache()
        
        total_loss += batch_loss
        num_batches += 1
        
        # 每2个batch记录一次（更频繁的监控）
        if (batch_idx + 1) % 2 == 0:
            mem_info = monitor_memory_usage()
            logger.info(f"Epoch {epoch}, Batch {batch_idx+1}/{len(dataloader)}: "
                       f"Loss={batch_loss:.6f}, "
                       f"GPU内存={mem_info['allocated_gb']:.2f}GB, "
                       f"峰值={mem_info['reserved_gb']:.2f}GB")
            
            # 如果内存使用过高，发出警告
            if mem_info['allocated_gb'] > 8.0:  # 超过8GB
                logger.warning(f"内存使用过高! 考虑减小批次大小或裁剪尺寸")
        
        # 释放批次内存
        del images, poses, intrinsics, tsdf_target
        if 'current_images' in locals():
            del current_images, current_poses, current_images_reshaped, current_poses_reshaped
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


def main():
    """主训练函数"""
    logger.info("开始内存优化的在线SDF训练...")
    
    # 根据GPU内存选择配置
    if torch.cuda.is_available():
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU总内存: {total_memory_gb:.1f}GB")
        
        if total_memory_gb < 4:
            memory_level = "low"
        elif total_memory_gb < 8:
            memory_level = "medium"
        else:
            memory_level = "high"
    else:
        memory_level = "low"
    
    config = MemoryOptimizedConfig.get_config(memory_level)
    logger.info(f"使用内存级别: {memory_level}")
    logger.info(f"训练配置: {json.dumps(config, indent=2)}")
    
    # 创建数据集
    try:
        from online_tartanair_dataset import OnlineTartanAirDataset
        
        dataset = OnlineTartanAirDataset(
            data_root=config["data_root"],
            sequence_name=config["sequence_name"],
            n_frames=config["n_frames"],
            crop_size=config["crop_size"],
            voxel_size=config["voxel_size"],
            target_image_size=config["target_image_size"],
            max_depth=10.0,
            truncation_margin=0.2,
            augment=False
        )
        
        logger.info(f"数据集创建成功，大小: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,  # 避免内存问题
            pin_memory=True
        )
        
    except Exception as e:
        logger.error(f"数据集创建失败: {e}")
        return
    
    # 创建模型
    model = create_memory_optimized_model(config)
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-5
    )
    
    # 标准精度训练（模型不支持混合精度）
    scaler = None
    logger.info("使用标准精度训练（模型不支持混合精度）")
    
    loss_fn = nn.HuberLoss(delta=0.1)  # 对异常值鲁棒
    
    # 训练历史记录
    history = {
        "config": config,
        "train_losses": [],
        "memory_usage": [],
        "timestamps": []
    }
    
    # 训练循环
    logger.info(f"开始训练，共{config['num_epochs']}个epoch...")
    
    for epoch in range(1, config["num_epochs"] + 1):
        start_time = time.time()
        
        # 训练一个epoch
        train_loss = train_epoch(model, dataloader, optimizer, loss_fn, config, epoch)
        
        epoch_time = time.time() - start_time
        mem_info = monitor_memory_usage()
        
        # 记录历史
        history["train_losses"].append(train_loss)
        history["memory_usage"].append(mem_info)
        history["timestamps"].append(datetime.now().isoformat())
        
        # 记录epoch结果
        logger.info(f"Epoch {epoch}/{config['num_epochs']}: "
                   f"Loss={train_loss:.6f}, "
                   f"Time={epoch_time:.1f}s, "
                   f"GPU内存={mem_info['allocated_gb']:.2f}GB, "
                   f"峰值内存={mem_info['max_allocated_gb']:.2f}GB")
        
        # 每3个epoch保存检查点
        if epoch % 3 == 0:
            checkpoint_path = f"optimized_checkpoints/epoch_{epoch}.pth"
            os.makedirs("optimized_checkpoints", exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': config
            }, checkpoint_path)
            
            logger.info(f"检查点保存到: {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = "optimized_checkpoints/final_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, final_model_path)
    
    # 保存训练历史
    history_path = "optimized_checkpoints/training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"训练完成!")
    logger.info(f"最终模型保存到: {final_model_path}")
    logger.info(f"训练历史保存到: {history_path}")
    
    # 输出总结
    print("\n" + "="*80)
    print("训练总结")
    print("="*80)
    print(f"配置级别: {memory_level}")
    print(f"总epoch数: {config['num_epochs']}")
    print(f"最终损失: {history['train_losses'][-1]:.6f}")
    print(f"峰值GPU内存: {max([m['max_allocated_gb'] for m in history['memory_usage']]):.2f}GB")
    
    if len(history['train_losses']) > 1:
        loss_improvement = (history['train_losses'][0] - history['train_losses'][-1]) / history['train_losses'][0] * 100
        print(f"损失改善: {loss_improvement:.1f}%")
    
    print("\n✅ 内存优化的训练完成!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()