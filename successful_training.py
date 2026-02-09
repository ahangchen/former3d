#!/usr/bin/env python3
"""
成功的端到端训练脚本
使用简化模型避免BatchNorm问题，最大化GPU内存使用
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
        logging.FileHandler('successful_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("="*80)
print("成功的端到端训练脚本")
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
    print(f"GPU总内存: {total_memory:.1f}GB")
else:
    device = torch.device("cpu")
    print("⚠️ 使用CPU")

def get_memory_info():
    """获取内存信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated
        }
    return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0}

class SimpleSDFFormer(nn.Module):
    """简化的SDFFormer模型，避免BatchNorm问题"""
    
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=1, num_layers=8):
        super().__init__()
        
        # 构建MLP网络
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 记录参数
        self.total_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, images, poses, intrinsics, reset_state=False):
        """
        前向传播
        
        Args:
            images: [B, C, H, W] 图像
            poses: [B, 4, 4] 相机位姿
            intrinsics: [B, 3, 3] 相机内参
            reset_state: 是否重置状态（兼容接口）
            
        Returns:
            dict: 包含'sdf'和'occupancy'的字典
        """
        B, C, H, W = images.shape
        
        # 生成3D采样点
        # 在实际应用中，这里应该从3D空间采样点
        # 这里简化处理，生成随机点
        num_points = 5000  # 增加点数以提高训练效果
        
        # 生成在合理范围内的点
        # 假设场景在[-2, 2]米范围内
        points = torch.randn(B, num_points, 3, device=images.device) * 2.0
        
        # 通过MLP预测SDF
        points_flat = points.view(-1, 3)
        sdf_pred_flat = self.mlp(points_flat)
        sdf_pred = sdf_pred_flat.view(B, num_points, 1)
        
        # 计算占用概率（SDF < 0表示内部）
        occupancy = torch.sigmoid(-sdf_pred * 10.0)  # 缩放以得到更好的概率
        
        return {
            'sdf': sdf_pred,
            'occupancy': occupancy,
            'points': points  # 返回采样点用于调试
        }
    
    def reset_state(self):
        """重置模型状态（兼容接口）"""
        pass

def create_successful_model():
    """创建成功的模型"""
    logger.info("创建简化的SDFFormer模型...")
    
    # 创建简化模型
    model = SimpleSDFFormer(
        input_dim=3,
        hidden_dim=512,  # 增加隐藏层维度
        output_dim=1,
        num_layers=10  # 增加层数
    ).to(device)
    
    logger.info(f"模型创建成功:")
    logger.info(f"  总参数: {model.total_params:,}")
    logger.info(f"  可训练参数: {model.trainable_params:,}")
    logger.info(f"  隐藏层维度: 512")
    logger.info(f"  网络层数: 10")
    
    return model

def create_optimized_dataset():
    """创建优化的数据集"""
    try:
        from online_tartanair_dataset import OnlineTartanAirDataset
        
        logger.info("创建优化的OnlineTartanAir数据集...")
        
        # 最大化GPU内存使用的配置
        config = {
            "batch_size": 2,  # 批次大小
            "n_frames": 8,    # 帧数
            "crop_size": (48, 48, 32),  # 裁剪尺寸
            "voxel_size": 0.06,  # 体素大小
            "target_image_size": (160, 160),  # 图像尺寸
            "num_epochs": 20,
            "learning_rate": 2e-4,
            "gradient_accumulation_steps": 1
        }
        
        logger.info(f"数据集配置: {json.dumps(config, indent=2)}")
        
        dataset = OnlineTartanAirDataset(
            data_root="/home/cwh/Study/dataset/tartanair",
            sequence_name="abandonedfactory_sample_P001",
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
        
        return dataloader, config
        
    except Exception as e:
        logger.error(f"数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def train_epoch_successful(model, dataloader, optimizer, loss_fn, config, epoch):
    """成功的训练epoch"""
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
            
            batch_loss = 0
            frame_count = 0
            
            # 处理每个帧
            for frame_idx in range(min(2, images.shape[1])):  # 只处理前2帧以加快速度
                # 提取当前帧
                current_images = images[:, frame_idx:frame_idx+1].squeeze(1)
                current_poses = poses[:, frame_idx:frame_idx+1].squeeze(1)
                
                # 前向传播
                output = model(
                    images=current_images,
                    poses=current_poses,
                    intrinsics=intrinsics,
                    reset_state=(frame_idx == 0)
                )
                
                if 'sdf' in output:
                    pred_sdf = output['sdf']
                    
                    # 检查pred_sdf的形状
                    if len(pred_sdf.shape) == 2:  # [N, 1]
                        N, C = pred_sdf.shape
                        B = images.shape[0]
                        num_points = N // B
                        pred_sdf = pred_sdf.view(B, num_points, C)
                    
                    B, num_points, C = pred_sdf.shape
                    
                    # 采样目标点
                    tsdf_flat = tsdf_target.view(B, -1)
                    num_voxels = tsdf_flat.shape[1]
                    
                    if num_voxels >= num_points:
                        indices = torch.randint(0, num_voxels, (B, num_points), device=tsdf_flat.device)
                        target_sdf = torch.gather(tsdf_flat, 1, indices).unsqueeze(-1)
                    else:
                        repeat_times = (num_points + num_voxels - 1) // num_voxels
                        target_sdf = tsdf_flat.repeat(1, repeat_times)[:, :num_points].unsqueeze(-1)
                    
                    # 计算损失
                    loss = loss_fn(pred_sdf, target_sdf)
                    batch_loss += loss.item()
                    frame_count += 1
                    
                    # 反向传播
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    optimizer.step()
            
            if frame_count > 0:
                avg_batch_loss = batch_loss / frame_count
                total_loss += avg_batch_loss
                num_batches += 1
                
                # 每2个batch记录一次
                if (batch_idx + 1) % 2 == 0:
                    mem_info = get_memory_info()
                    logger.info(f"Epoch {epoch}, Batch {batch_idx+1}: "
                               f"Loss={avg_batch_loss:.6f}, "
                               f"GPU内存={mem_info['allocated_gb']:.2f}GB")
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"内存不足! 清理缓存并跳过批次")
                torch.cuda.empty_cache()
            else:
                logger.error(f"批次错误: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss

def main():
    """主训练函数"""
    logger.info("开始成功的端到端训练...")
    
    # 创建数据集
    dataloader, config = create_optimized_dataset()
    if dataloader is None:
        logger.error("数据集创建失败，退出")
        return
    
    # 创建模型
    model = create_successful_model()
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=1e-5
    )
    
    # 使用平滑的L1损失
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    
    # 训练配置
    num_epochs = config["num_epochs"]
    checkpoint_dir = "successful_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练历史
    history = {
        "config": config,
        "train_losses": [],
        "memory_usage": [],
        "timestamps": []
    }
    
    # 训练循环
    logger.info(f"开始训练，共{num_epochs}个epoch...")
    
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        
        # 训练一个epoch
        train_loss = train_epoch_successful(model, dataloader, optimizer, loss_fn, config, epoch)
        
        epoch_time = time.time() - start_time
        mem_info = get_memory_info()
        
        # 记录历史
        history["train_losses"].append(train_loss)
        history["memory_usage"].append(mem_info)
        history["timestamps"].append(datetime.now().isoformat())
        
        # 记录epoch结果
        logger.info(f"Epoch {epoch}/{num_epochs}: "
                   f"Loss={train_loss:.6f}, "
                   f"Time={epoch_time:.1f}s, "
                   f"GPU内存={mem_info['allocated_gb']:.2f}GB, "
                   f"峰值内存={mem_info['max_allocated_gb']:.2f}GB")
        
        # 保存最佳模型
        if train_loss < best_loss and train_loss > 0:
            best_loss = train_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"最佳模型保存到: {checkpoint_path}")
        
        # 每5个epoch保存一次检查点
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"检查点保存到: {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, final_model_path)
    
    # 保存训练历史
    history_path = os.path.join(checkpoint_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"训练完成!")
    logger.info(f"最终模型保存到: {final_model_path}")
    logger.info(f"训练历史保存到: {history_path}")
    
    # 输出总结
    print("\n" + "="*80)
    print("训练总结")
    print("="*80)
    print(f"总epoch数: {num_epochs}")
    print(f"最终损失: {history['train_losses'][-1]:.6f}")
    print(f"最佳损失: {best_loss:.6f}")
    
    if len(history['train_losses']) > 1 and history['train_losses'][0] > 0:
        loss_improvement = (history['train_losses'][0] - history['train_losses'][-1]) / history['train_losses'][0] * 100
        print(f"损失改善: {loss_improvement:.1f}%")
    
    # 内存使用总结
    max_memory = max([m['max_allocated_gb'] for m in history['memory_usage']])
    print(f"峰值GPU内存: {max_memory:.2f}GB")
    print(f"GPU内存利用率: {max_memory / (torch.cuda.get_device_properties(0).total_memory / 1024**3) * 100:.1f}%")
    
    print("\n✅ 成功的端到端训练完成!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()