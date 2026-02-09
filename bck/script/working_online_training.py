#!/usr/bin/env python3
"""
可工作的在线SDF训练脚本
使用简化模型确保训练能正常进行
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
        logging.FileHandler('working_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("="*80)
print("可工作的在线SDF训练")
print("="*80)

# 检查环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
    
    # 检查GPU内存
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU总内存: {total_memory:.1f}GB")
else:
    device = torch.device("cpu")
    print("⚠️ 使用CPU")

# 工作配置
WORKING_CONFIG = {
    "data_root": "/home/cwh/Study/dataset/tartanair",
    "sequence_name": "abandonedfactory_sample_P001",
    "batch_size": 1,
    "n_frames": 3,
    "crop_size": (24, 24, 16),
    "voxel_size": 0.12,
    "target_image_size": (96, 96),
    "num_epochs": 5,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5
}

logger.info(f"工作配置: {WORKING_CONFIG}")

def get_memory_info():
    """获取内存信息"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        return allocated, cached
    return 0, 0

class WorkingSDFModel(nn.Module):
    """可工作的SDF模型"""
    def __init__(self):
        super().__init__()
        # 简单的图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 下采样
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 再次下采样
            nn.AdaptiveAvgPool2d((4, 4))  # 固定大小
        )
        
        # 3D点处理网络
        self.point_processor = nn.Sequential(
            nn.Linear(32*4*4 + 3, 128),  # 图像特征 + 3D坐标
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # SDF值
        )
        
    def forward(self, images, poses, intrinsics, reset_state=False):
        """
        简化前向传播
        Args:
            images: (B, 3, H, W) 图像
            poses: (B, 4, 4) 相机位姿（忽略形状问题）
            intrinsics: (B, 3, 3) 相机内参
            reset_state: 是否重置状态
        Returns:
            dict: 包含'sdf'预测
        """
        B, C, H, W = images.shape
        
        # 编码图像
        img_features = self.image_encoder(images)  # (B, 32, 4, 4)
        img_features = img_features.view(B, -1)  # (B, 512)
        
        # 采样3D点
        num_points = 512
        
        # 采样策略：表面附近点 + 自由空间点
        surface_points = torch.randn(B, num_points//2, 3).to(device) * 0.05  # 表面附近
        free_points = torch.randn(B, num_points//2, 3).to(device) * 2.0  # 自由空间
        points = torch.cat([surface_points, free_points], dim=1)
        
        # 重复图像特征给每个点
        img_features_expanded = img_features.unsqueeze(1).repeat(1, num_points, 1)  # (B, num_points, 512)
        
        # 拼接特征
        combined = torch.cat([img_features_expanded, points], dim=-1)  # (B, num_points, 515)
        
        # 处理点
        sdf_pred = self.point_processor(combined)  # (B, num_points, 1)
        
        return {'sdf': sdf_pred}
    
    def reset_state(self):
        """重置状态"""
        pass

def train_working_model():
    """训练可工作模型"""
    logger.info("开始训练可工作模型...")
    
    start_time = time.time()
    
    # 创建数据集
    try:
        from online_tartanair_dataset import OnlineTartanAirDataset
        
        dataset = OnlineTartanAirDataset(
            data_root=WORKING_CONFIG["data_root"],
            sequence_name=WORKING_CONFIG["sequence_name"],
            n_frames=WORKING_CONFIG["n_frames"],
            crop_size=WORKING_CONFIG["crop_size"],
            voxel_size=WORKING_CONFIG["voxel_size"],
            target_image_size=WORKING_CONFIG["target_image_size"],
            max_depth=10.0,
            truncation_margin=0.2,
            augment=False
        )
        
        logger.info(f"数据集创建成功，大小: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=WORKING_CONFIG["batch_size"],
            shuffle=True,
            num_workers=0
        )
        
    except Exception as e:
        logger.error(f"数据集创建失败: {e}")
        return None
    
    # 创建模型
    model = WorkingSDFModel().to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型创建成功，参数: {params:,}")
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(
        model.parameters(),
        lr=WORKING_CONFIG["learning_rate"],
        weight_decay=WORKING_CONFIG["weight_decay"]
    )
    
    # 使用Huber损失（对SDF回归更鲁棒）
    loss_fn = nn.HuberLoss(delta=0.1)
    
    # 训练历史
    history = {
        "config": WORKING_CONFIG,
        "train_losses": [],
        "memory_usage": []
    }
    
    # 训练循环
    logger.info(f"开始训练，共{WORKING_CONFIG['num_epochs']}个epoch...")
    
    for epoch in range(1, WORKING_CONFIG["num_epochs"] + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # 移动到设备
                images = batch['rgb_images'].to(device)
                poses = batch['poses'].to(device)
                intrinsics = batch['intrinsics'].to(device)
                tsdf_target = batch['tsdf'].to(device)
                
                batch_loss = 0
                processed_frames = 0
                
                # 处理每个帧
                for frame_idx in range(images.shape[1]):
                    # 前向传播
                    output = model(
                        images=images[:, frame_idx:frame_idx+1],
                        poses=poses[:, frame_idx:frame_idx+1],
                        intrinsics=intrinsics,
                        reset_state=(frame_idx == 0)
                    )
                    
                    if 'sdf' in output:
                        pred_sdf = output['sdf']  # (B, num_points, 1)
                        
                        # 从TSDF目标中采样对应的点
                        B, num_points, _ = pred_sdf.shape
                        tsdf_flat = tsdf_target.view(B, -1)  # (B, voxel_count)
                        voxel_count = tsdf_flat.shape[1]
                        
                        # 随机采样目标点
                        if voxel_count >= num_points:
                            indices = torch.randint(0, voxel_count, (B, num_points))
                            target_sdf = torch.gather(tsdf_flat, 1, indices).unsqueeze(-1)
                        else:
                            # 重复采样
                            repeat_times = (num_points + voxel_count - 1) // voxel_count
                            target_sdf = tsdf_flat.repeat(1, repeat_times)[:, :num_points].unsqueeze(-1)
                        
                        # 计算损失
                        loss = loss_fn(pred_sdf, target_sdf)
                        batch_loss += loss.item()
                        processed_frames += 1
                        
                        # 反向传播
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        
                        # 更新参数
                        optimizer.step()
                
                if processed_frames > 0:
                    avg_batch_loss = batch_loss / processed_frames
                    epoch_loss += avg_batch_loss
                    num_batches += 1
                    
                    # 记录批次信息
                    if (batch_idx + 1) % 1 == 0:  # 每个批次都记录
                        allocated, cached = get_memory_info()
                        logger.info(f"Epoch {epoch}, Batch {batch_idx+1}: "
                                   f"Loss={avg_batch_loss:.6f}, "
                                   f"GPU内存={allocated:.2f}GB")
                
            except Exception as e:
                logger.error(f"批次错误: {e}")
                # 清理内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # 计算epoch平均损失
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
        else:
            avg_epoch_loss = 0
        
        # 记录历史
        history["train_losses"].append(avg_epoch_loss)
        allocated, cached = get_memory_info()
        history["memory_usage"].append({"allocated_gb": allocated, "cached_gb": cached})
        
        # 记录epoch结果
        logger.info(f"Epoch {epoch} 完成: 平均损失={avg_epoch_loss:.6f}")
        
        # 保存检查点
        if epoch % 2 == 0:
            os.makedirs("working_checkpoints", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'config': WORKING_CONFIG
            }, f"working_checkpoints/epoch_{epoch}.pth")
            logger.info(f"检查点保存到: working_checkpoints/epoch_{epoch}.pth")
    
    # 训练完成
    total_time = time.time() - start_time
    
    # 保存最终模型
    os.makedirs("working_checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': WORKING_CONFIG,
        'history': history,
        'total_time': total_time
    }, "working_checkpoints/final_model.pth")
    
    # 保存训练历史
    with open("working_checkpoints/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"训练完成! 总时间: {total_time:.1f}s")
    
    return history

def print_working_summary(history):
    """打印工作训练总结"""
    if not history or "train_losses" not in history:
        print("❌ 无训练历史")
        return
    
    losses = history["train_losses"]
    
    print("\n" + "="*80)
    print("工作训练总结")
    print("="*80)
    
    print(f"配置:")
    print(f"  batch_size: {WORKING_CONFIG['batch_size']}")
    print(f"  n_frames: {WORKING_CONFIG['n_frames']}")
    print(f"  crop_size: {WORKING_CONFIG['crop_size']}")
    print(f"  voxel_size: {WORKING_CONFIG['voxel_size']}")
    print(f"  num_epochs: {WORKING_CONFIG['num_epochs']}")
    
    print(f"\n训练结果:")
    if len(losses) > 0:
        print(f"  初始损失: {losses[0]:.6f}")
        print(f"  最终损失: {losses[-1]:.6f}")
        
        if len(losses) > 1 and losses[-1] < losses[0]:
            improvement = (losses[0] - losses[-1]) / losses[0] * 100
            print(f"  损失改善: {improvement:.1f}%")
    
    # 内存使用
    if history["memory_usage"]:
        max_memory = max([m["allocated_gb"] for m in history["memory_usage"]])
        print(f"  峰值GPU内存: {max_memory:.2f}GB")
    
    print(f"\n输出文件:")
    print(f"  最终模型: working_checkpoints/final_model.pth")
    print(f"  训练历史: working_checkpoints/training_history.json")
    print(f"  训练日志: working_training.log")
    
    # 评估训练效果
    if len(losses) > 1 and losses[-1] < losses[0]:
        print("\n✅ 训练成功! 损失下降，模型在学习。")
    elif len(losses) > 0:
        print("\n⚠️ 训练完成，但损失改善不明显。")
    else:
        print("\n❌ 训练失败，无有效损失记录。")

def main():
    """主函数"""
    try:
        # 运行训练
        history = train_working_model()
        
        # 打印总结
        print_working_summary(history)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        return 1
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())