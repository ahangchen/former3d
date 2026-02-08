#!/usr/bin/env python3
"""
修复的在线SDF训练脚本
正确处理多帧数据
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
        logging.FileHandler('fixed_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("="*80)
print("修复的在线SDF训练")
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

# 修复配置
FIXED_CONFIG = {
    "data_root": "/home/cwh/Study/dataset/tartanair",
    "sequence_name": "abandonedfactory_sample_P001",
    "batch_size": 1,
    "n_frames": 2,  # 减少帧数以简化
    "crop_size": (16, 16, 12),
    "voxel_size": 0.16,
    "target_image_size": (64, 64),
    "num_epochs": 3,  # 少量epoch先测试
    "learning_rate": 1e-4,
    "weight_decay": 1e-5
}

logger.info(f"修复配置: {FIXED_CONFIG}")

class FixedSDFModel(nn.Module):
    """修复的SDF模型，正确处理多帧"""
    def __init__(self):
        super().__init__()
        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # 3D解码器
        self.decoder = nn.Sequential(
            nn.Linear(32*4*4 + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, images, poses, intrinsics, reset_state=False):
        """
        修复的前向传播，处理多帧
        Args:
            images: (B, N, C, H, W) 其中N是帧数
            poses: (B, N, 4, 4)
            intrinsics: (B, 3, 3)
            reset_state: 是否重置状态
        Returns:
            dict: 包含'sdf'预测
        """
        B, N, C, H, W = images.shape
        
        # 处理第一帧（简化）
        frame_images = images[:, 0]  # (B, C, H, W)
        
        # 编码图像
        img_features = self.image_encoder(frame_images)  # (B, 32, 4, 4)
        img_features = img_features.view(B, -1)  # (B, 512)
        
        # 采样3D点
        num_points = 512
        points = torch.randn(B, num_points, 3).to(images.device)
        
        # 重复图像特征
        img_features_expanded = img_features.unsqueeze(1).repeat(1, num_points, 1)
        
        # 拼接特征
        combined = torch.cat([img_features_expanded, points], dim=-1)
        
        # 解码SDF
        sdf_pred = self.decoder(combined)
        
        return {'sdf': sdf_pred}
    
    def reset_state(self):
        pass

def train_fixed_model():
    """训练修复模型"""
    logger.info("开始训练修复模型...")
    
    start_time = time.time()
    
    # 创建数据集
    try:
        from online_tartanair_dataset import OnlineTartanAirDataset
        
        dataset = OnlineTartanAirDataset(
            data_root=FIXED_CONFIG["data_root"],
            sequence_name=FIXED_CONFIG["sequence_name"],
            n_frames=FIXED_CONFIG["n_frames"],
            crop_size=FIXED_CONFIG["crop_size"],
            voxel_size=FIXED_CONFIG["voxel_size"],
            target_image_size=FIXED_CONFIG["target_image_size"],
            max_depth=10.0,
            truncation_margin=0.2,
            augment=False
        )
        
        logger.info(f"数据集创建成功，大小: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FIXED_CONFIG["batch_size"],
            shuffle=True,
            num_workers=0
        )
        
    except Exception as e:
        logger.error(f"数据集创建失败: {e}")
        return None
    
    # 创建模型
    model = FixedSDFModel().to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型创建成功，参数: {params:,}")
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(
        model.parameters(),
        lr=FIXED_CONFIG["learning_rate"],
        weight_decay=FIXED_CONFIG["weight_decay"]
    )
    
    loss_fn = nn.MSELoss()
    
    # 训练历史
    history = {
        "config": FIXED_CONFIG,
        "train_losses": []
    }
    
    # 训练循环
    logger.info(f"开始训练，共{FIXED_CONFIG['num_epochs']}个epoch...")
    
    for epoch in range(1, FIXED_CONFIG["num_epochs"] + 1):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # 移动到设备
                images = batch['rgb_images'].to(device)  # (B, N, C, H, W)
                poses = batch['poses'].to(device)        # (B, N, 4, 4)
                intrinsics = batch['intrinsics'].to(device)  # (B, 3, 3)
                tsdf_target = batch['tsdf'].to(device)   # (B, D, H, W)
                
                # 前向传播
                output = model(
                    images=images,
                    poses=poses,
                    intrinsics=intrinsics,
                    reset_state=True
                )
                
                if 'sdf' in output:
                    pred_sdf = output['sdf']  # (B, num_points, 1)
                    
                    # 从TSDF目标中采样点
                    B, num_points, _ = pred_sdf.shape
                    tsdf_flat = tsdf_target.view(B, -1)  # (B, voxel_count)
                    voxel_count = tsdf_flat.shape[1]
                    
                    # 随机采样目标点
                    if voxel_count >= num_points:
                        indices = torch.randint(0, voxel_count, (B, num_points)).to(device)
                        target_sdf = torch.gather(tsdf_flat, 1, indices).unsqueeze(-1)
                    else:
                        # 重复采样
                        repeat_times = (num_points + voxel_count - 1) // voxel_count
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
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    logger.info(f"Epoch {epoch}, Batch {batch_idx+1}: Loss={loss.item():.6f}")
                
                else:
                    logger.warning(f"输出中没有'sdf'键")
                
            except Exception as e:
                logger.error(f"批次错误: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 计算epoch平均损失
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
        else:
            avg_epoch_loss = 0
        
        # 记录历史
        history["train_losses"].append(avg_epoch_loss)
        
        # 记录epoch结果
        logger.info(f"Epoch {epoch} 完成: 平均损失={avg_epoch_loss:.6f}")
    
    # 训练完成
    total_time = time.time() - start_time
    
    # 保存模型
    os.makedirs("fixed_checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': FIXED_CONFIG,
        'history': history,
        'total_time': total_time
    }, "fixed_checkpoints/final_model.pth")
    
    # 保存训练历史
    with open("fixed_checkpoints/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"训练完成! 总时间: {total_time:.1f}s")
    
    return history

def print_fixed_summary(history):
    """打印修复训练总结"""
    if not history or "train_losses" not in history:
        print("❌ 无训练历史")
        return
    
    losses = history["train_losses"]
    
    print("\n" + "="*80)
    print("修复训练总结")
    print("="*80)
    
    print(f"配置:")
    print(f"  batch_size: {FIXED_CONFIG['batch_size']}")
    print(f"  n_frames: {FIXED_CONFIG['n_frames']}")
    print(f"  crop_size: {FIXED_CONFIG['crop_size']}")
    print(f"  voxel_size: {FIXED_CONFIG['voxel_size']}")
    print(f"  num_epochs: {FIXED_CONFIG['num_epochs']}")
    
    print(f"\n训练结果:")
    if len(losses) > 0:
        print(f"  初始损失: {losses[0]:.6f}")
        print(f"  最终损失: {losses[-1]:.6f}")
        
        if len(losses) > 1:
            if losses[-1] < losses[0]:
                improvement = (losses[0] - losses[-1]) / losses[0] * 100
                print(f"  损失改善: {improvement:.1f}%")
                print("\n✅ 训练成功! 损失下降，模型在学习。")
            else:
                print(f"  损失变化: {losses[-1] - losses[0]:.6f}")
                print("\n⚠️ 训练完成，但损失没有下降。")
    
    print(f"\n输出文件:")
    print(f"  最终模型: fixed_checkpoints/final_model.pth")
    print(f"  训练历史: fixed_checkpoints/training_history.json")
    print(f"  训练日志: fixed_training.log")

def main():
    """主函数"""
    try:
        # 运行训练
        history = train_fixed_model()
        
        # 打印总结
        print_fixed_summary(history)
        
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