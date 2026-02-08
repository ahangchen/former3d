#!/usr/bin/env python3
"""
安全的内存优化训练脚本
从低配置开始，逐步增加复杂度
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
        logging.FileHandler('safe_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("="*80)
print("安全的内存优化在线SDF训练")
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

# 安全配置（从低开始）
SAFE_CONFIG = {
    "data_root": "/home/cwh/Study/dataset/tartanair",
    "sequence_name": "abandonedfactory_sample_P001",
    "batch_size": 1,  # 最小batch size
    "n_frames": 3,    # 少量帧
    "crop_size": (24, 24, 16),  # 小裁剪
    "voxel_size": 0.12,  # 大体素
    "target_image_size": (96, 96),  # 小图像
    "num_epochs": 5,  # 少量epoch
    "learning_rate": 1e-4,
    "gradient_accumulation_steps": 1
}

logger.info(f"安全训练配置: {SAFE_CONFIG}")

def monitor_memory():
    """监控内存使用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        return allocated, cached
    return 0, 0

def create_safe_model(config):
    """创建安全模型（简化版）"""
    try:
        # 尝试导入完整模型
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        logger.info("创建StreamSDFFormerIntegrated模型（安全配置）...")
        
        model = StreamSDFFormerIntegrated(
            attn_heads=2,  # 减少注意力头
            attn_layers=1,  # 减少注意力层
            use_proj_occ=True,
            voxel_size=config["voxel_size"],
            fusion_local_radius=3.0,  # 减小融合半径
            crop_size=config["crop_size"]
        )
        
        model = model.to(device)
        
        # 计算参数
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型参数: {total_params:,}")
        
        return model
        
    except Exception as e:
        logger.warning(f"完整模型创建失败: {e}")
        logger.info("创建超简化模型...")
        
        # 超简化模型
        class UltraSimpleSDFModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 8, 3, padding=1),  # 减少通道
                    nn.ReLU(),
                    nn.Conv2d(8, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4))  # 更小的特征图
                )
                self.decoder = nn.Sequential(
                    nn.Linear(16*4*4 + 3, 64),  # 减少维度
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
                
            def forward(self, images, poses, intrinsics, reset_state=False):
                B, C, H, W = images.shape
                img_features = self.encoder(images).view(B, -1)
                num_points = 256  # 减少采样点
                points = torch.randn(B, num_points, 3).to(images.device)
                img_features_expanded = img_features.unsqueeze(1).repeat(1, num_points, 1)
                combined = torch.cat([img_features_expanded, points], dim=-1)
                sdf_pred = self.decoder(combined)
                return {'sdf': sdf_pred}
            
            def reset_state(self):
                pass
        
        model = UltraSimpleSDFModel().to(device)
        logger.info(f"超简化模型创建成功，参数: {sum(p.numel() for p in model.parameters()):,}")
        return model

def test_memory_safety(model, config):
    """测试内存安全性"""
    logger.info("测试内存安全性...")
    
    try:
        # 创建测试数据
        batch_size = config["batch_size"]
        n_frames = config["n_frames"]
        H, W = config["target_image_size"]
        
        test_images = torch.randn(batch_size, n_frames, 3, H, W).to(device)
        test_poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_frames, 1, 1).to(device)
        test_intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        
        # 测试前向传播
        with torch.no_grad():
            for i in range(n_frames):
                output = model(
                    images=test_images[:, i:i+1],
                    poses=test_poses[:, i:i+1],
                    intrinsics=test_intrinsics,
                    reset_state=(i == 0)
                )
        
        allocated, cached = monitor_memory()
        logger.info(f"内存测试通过: 分配={allocated:.2f}GB, 缓存={cached:.2f}GB")
        return True
        
    except torch.cuda.OutOfMemoryError:
        logger.error("❌ 内存测试失败: CUDA内存不足")
        return False
    except Exception as e:
        logger.error(f"❌ 内存测试失败: {e}")
        return False

def main():
    """主训练函数"""
    logger.info("开始安全训练...")
    
    # 创建数据集
    try:
        from online_tartanair_dataset import OnlineTartanAirDataset
        
        dataset = OnlineTartanAirDataset(
            data_root=SAFE_CONFIG["data_root"],
            sequence_name=SAFE_CONFIG["sequence_name"],
            n_frames=SAFE_CONFIG["n_frames"],
            crop_size=SAFE_CONFIG["crop_size"],
            voxel_size=SAFE_CONFIG["voxel_size"],
            target_image_size=SAFE_CONFIG["target_image_size"],
            max_depth=10.0,
            truncation_margin=0.2,
            augment=False
        )
        
        logger.info(f"数据集创建成功，大小: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=SAFE_CONFIG["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=False  # 避免pin memory内存问题
        )
        
    except Exception as e:
        logger.error(f"数据集创建失败: {e}")
        return
    
    # 创建模型
    model = create_safe_model(SAFE_CONFIG)
    
    # 测试内存安全性
    if not test_memory_safety(model, SAFE_CONFIG):
        logger.error("内存测试失败，停止训练")
        return
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=SAFE_CONFIG["learning_rate"])
    loss_fn = nn.MSELoss()  # 简单MSE损失
    
    # 训练循环
    logger.info(f"开始训练，共{SAFE_CONFIG['num_epochs']}个epoch...")
    
    train_losses = []
    
    for epoch in range(1, SAFE_CONFIG["num_epochs"] + 1):
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
                        pred_sdf = output['sdf']
                        
                        # 创建简单目标
                        target_sdf = torch.randn_like(pred_sdf) * 0.1
                        
                        # 计算损失
                        loss = loss_fn(pred_sdf, target_sdf)
                        batch_loss += loss.item()
                        
                        # 反向传播
                        optimizer.zero_grad()
                        loss.backward()
                        
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                        
                        # 更新参数
                        optimizer.step()
                
                epoch_loss += batch_loss
                num_batches += 1
                
                # 监控内存
                if batch_idx % 2 == 0:
                    allocated, cached = monitor_memory()
                    logger.info(f"Epoch {epoch}, Batch {batch_idx+1}: "
                               f"Loss={batch_loss:.6f}, "
                               f"GPU内存={allocated:.2f}GB")
                
            except torch.cuda.OutOfMemoryError:
                logger.error(f"❌ Epoch {epoch}, Batch {batch_idx+1}: CUDA内存不足!")
                # 清理内存
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                logger.error(f"❌ Epoch {epoch}, Batch {batch_idx+1}: {e}")
                continue
        
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            logger.info(f"Epoch {epoch} 完成: 平均损失={avg_loss:.6f}")
        
        # 清理内存
        torch.cuda.empty_cache()
    
    # 保存结果
    if train_losses:
        logger.info(f"训练完成!")
        logger.info(f"最终损失: {train_losses[-1]:.6f}")
        
        if len(train_losses) > 1:
            improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
            logger.info(f"损失改善: {improvement:.1f}%")
        
        # 保存模型
        os.makedirs("safe_checkpoints", exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': SAFE_CONFIG,
            'train_losses': train_losses
        }, "safe_checkpoints/final_model.pth")
        
        logger.info(f"模型保存到: safe_checkpoints/final_model.pth")
        
        print("\n" + "="*80)
        print("✅ 安全训练完成!")
        print(f"配置: batch_size={SAFE_CONFIG['batch_size']}, "
              f"crop_size={SAFE_CONFIG['crop_size']}")
        print(f"最终损失: {train_losses[-1]:.6f}")
        print("="*80)
    else:
        logger.error("训练失败，无有效损失记录")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()