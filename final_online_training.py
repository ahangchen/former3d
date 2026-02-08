#!/usr/bin/env python3
"""
最终版在线SDF训练脚本
使用安全配置，监控内存，优化训练效率
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
        logging.FileHandler('final_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print("="*80)
print("最终版在线SDF训练")
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

# 最终训练配置（经过测试的安全配置）
FINAL_CONFIG = {
    "data_root": "/home/cwh/Study/dataset/tartanair",
    "sequence_name": "abandonedfactory_sample_P001",
    "batch_size": 1,  # 安全batch size
    "n_frames": 4,    # 平衡帧数
    "crop_size": (32, 32, 24),  # 中等裁剪
    "voxel_size": 0.08,  # 标准体素
    "target_image_size": (128, 128),  # 标准图像
    "num_epochs": 8,  # 合理epoch数
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "gradient_clip": 1.0,
    "checkpoint_interval": 2
}

logger.info(f"最终训练配置:")
for key, value in FINAL_CONFIG.items():
    logger.info(f"  {key}: {value}")

def get_memory_info():
    """获取内存信息"""
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

def create_final_model():
    """创建最终模型"""
    try:
        # 尝试使用完整模型
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        logger.info("创建StreamSDFFormerIntegrated模型...")
        
        model = StreamSDFFormerIntegrated(
            attn_heads=4,
            attn_layers=2,
            use_proj_occ=True,
            voxel_size=FINAL_CONFIG["voxel_size"],
            fusion_local_radius=3.0,
            crop_size=FINAL_CONFIG["crop_size"]
        )
        
        model = model.to(device)
        
        # 计算参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"模型信息:")
        logger.info(f"  总参数: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
        
        return model
        
    except Exception as e:
        logger.warning(f"完整模型创建失败: {e}")
        logger.info("创建高效简化模型...")
        
        # 高效简化模型
        class EfficientSDFModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 图像编码器
                self.image_encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1, stride=2),  # 下采样
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, padding=1, stride=2),  # 再次下采样
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4))  # 固定大小
                )
                
                # 3D解码器
                self.decoder = nn.Sequential(
                    nn.Linear(64*4*4 + 3, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1)
                )
                
            def forward(self, images, poses, intrinsics, reset_state=False):
                B, C, H, W = images.shape
                
                # 编码图像
                img_features = self.image_encoder(images)  # (B, 64, 4, 4)
                img_features = img_features.view(B, -1)  # (B, 1024)
                
                # 采样3D点（更多表面点）
                num_points = 1024
                
                # 50%表面附近点，50%随机点
                surface_points = torch.randn(B, num_points//2, 3).to(device) * 0.1  # 表面附近
                random_points = torch.randn(B, num_points//2, 3).to(device)  # 随机点
                points = torch.cat([surface_points, random_points], dim=1)
                
                # 重复图像特征
                img_features_expanded = img_features.unsqueeze(1).repeat(1, num_points, 1)
                
                # 拼接特征
                combined = torch.cat([img_features_expanded, points], dim=-1)
                
                # 解码SDF
                sdf_pred = self.decoder(combined)
                
                return {'sdf': sdf_pred}
            
            def reset_state(self):
                pass
        
        model = EfficientSDFModel().to(device)
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"高效简化模型创建成功，参数: {params:,}")
        return model

def train():
    """主训练函数"""
    logger.info("开始最终训练...")
    
    start_time = time.time()
    
    # 创建数据集
    try:
        from online_tartanair_dataset import OnlineTartanAirDataset
        
        dataset = OnlineTartanAirDataset(
            data_root=FINAL_CONFIG["data_root"],
            sequence_name=FINAL_CONFIG["sequence_name"],
            n_frames=FINAL_CONFIG["n_frames"],
            crop_size=FINAL_CONFIG["crop_size"],
            voxel_size=FINAL_CONFIG["voxel_size"],
            target_image_size=FINAL_CONFIG["target_image_size"],
            max_depth=10.0,
            truncation_margin=0.2,
            augment=True  # 启用数据增强
        )
        
        logger.info(f"数据集创建成功，大小: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=FINAL_CONFIG["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
    except Exception as e:
        logger.error(f"数据集创建失败: {e}")
        return None
    
    # 创建模型
    model = create_final_model()
    
    # 创建优化器和损失函数
    optimizer = optim.Adam(
        model.parameters(),
        lr=FINAL_CONFIG["learning_rate"],
        weight_decay=FINAL_CONFIG["weight_decay"]
    )
    
    # Huber损失对SDF回归更鲁棒
    loss_fn = nn.HuberLoss(delta=0.1)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # 训练历史
    history = {
        "config": FINAL_CONFIG,
        "train_losses": [],
        "learning_rates": [],
        "memory_usage": [],
        "timestamps": []
    }
    
    # 训练循环
    logger.info(f"开始训练，共{FINAL_CONFIG['num_epochs']}个epoch...")
    
    best_loss = float('inf')
    
    for epoch in range(1, FINAL_CONFIG["num_epochs"] + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch_start = time.time()
            
            try:
                # 移动到设备
                images = batch['rgb_images'].to(device)
                poses = batch['poses'].to(device)
                intrinsics = batch['intrinsics'].to(device)
                tsdf_target = batch['tsdf'].to(device)
                
                batch_loss = 0
                frame_count = 0
                
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
                        
                        # 从TSDF目标中采样点
                        B, num_points, _ = pred_sdf.shape
                        tsdf_flat = tsdf_target.view(B, -1)
                        num_voxels = tsdf_flat.shape[1]
                        
                        # 随机采样目标点
                        if num_voxels >= num_points:
                            indices = torch.randint(0, num_voxels, (B, num_points))
                            target_sdf = torch.gather(tsdf_flat, 1, indices).unsqueeze(-1)
                        else:
                            # 重复采样
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
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            max_norm=FINAL_CONFIG["gradient_clip"]
                        )
                        
                        # 更新参数
                        optimizer.step()
                
                if frame_count > 0:
                    avg_batch_loss = batch_loss / frame_count
                    epoch_loss += avg_batch_loss
                    num_batches += 1
                    
                    # 记录批次信息
                    if (batch_idx + 1) % 2 == 0:
                        batch_time = time.time() - batch_start
                        mem_info = get_memory_info()
                        
                        logger.info(f"Epoch {epoch}, Batch {batch_idx+1}: "
                                   f"Loss={avg_batch_loss:.6f}, "
                                   f"Time={batch_time:.1f}s, "
                                   f"GPU={mem_info['allocated_gb']:.2f}GB")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"内存不足! 清理缓存并跳过批次")
                    torch.cuda.empty_cache()
                else:
                    logger.error(f"批次错误: {e}")
                continue
        
        # 计算epoch平均损失
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
        else:
            avg_epoch_loss = 0
        
        # 更新学习率
        scheduler.step(avg_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history["train_losses"].append(avg_epoch_loss)
        history["learning_rates"].append(current_lr)
        history["memory_usage"].append(get_memory_info())
        history["timestamps"].append(datetime.now().isoformat())
        
        # 记录epoch结果
        epoch_time = time.time() - epoch_start
        mem_info = get_memory_info()
        
        logger.info(f"Epoch {epoch} 完成: "
                   f"Loss={avg_epoch_loss:.6f}, "
                   f"LR={current_lr:.2e}, "
                   f"Time={epoch_time:.1f}s, "
                   f"峰值内存={mem_info['max_allocated_gb']:.2f}GB")
        
        # 保存最佳模型
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': FINAL_CONFIG
            }, "final_checkpoints/best_model.pth")
            logger.info(f"最佳模型保存到: final_checkpoints/best_model.pth (损失={best_loss:.6f})")
        
        # 定期保存检查点
        if epoch % FINAL_CONFIG["checkpoint_interval"] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'config': FINAL_CONFIG
            }, f"final_checkpoints/epoch_{epoch}.pth")
    
    # 训练完成
    total_time = time.time() - start_time
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': FINAL_CONFIG,
        'history': history,
        'total_time': total_time
    }, "final_checkpoints/final_model.pth")
    
    # 保存训练历史
    with open("final_checkpoints/training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"训练完成! 总时间: {total_time:.1f}s")
    
    return history

def print_summary(history):
    """打印训练总结"""
    if not history or "train_losses" not in history:
        return
    
    losses = history["train_losses"]
    if len(losses) < 2:
        return
    
    print("\n" + "="*80)
    print("训练总结")
    print("="*80)
    
    print(f"配置:")
    print(f"  batch_size: {FINAL_CONFIG['batch_size']}")
    print(f"  n_frames: {FINAL_CONFIG['n_frames']}")
    print(f"  crop_size: {FINAL_CONFIG['crop_size']}")
    print(f"  voxel_size: {FINAL_CONFIG['voxel_size']}")
    print(f"  num_epochs: {FINAL_CONFIG['num_epochs']}")
    
    print(f"\n训练结果:")
    print(f"  初始损失: {losses[0]:.6f}")
    print(f"  最终损失: {losses[-1]:.6f}")
    
    if losses[-1] < losses[0]:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"  损失改善: {improvement:.1f}%")
    
    # 内存使用
    if history["memory_usage"]:
        max_memory = max([m["max_allocated_gb"] for m in history["memory_usage"]])
        print(f"  峰值GPU内存: {max_memory:.2f}GB")
    
    # 学习率变化
    if history["learning_rates"]:
        print(f"  最终学习率: {history['learning_rates'][-1]:.2e}")
    
    print(f"\n输出文件:")
    print(f"  最佳模型: final_checkpoints/best_model.pth")
    print(f"  最终模型: final_checkpoints/final_model.pth")
    print(f"  训练历史: final_checkpoints/training_history.json")
    print(f"  训练日志: final_training.log")
    
    print("\n✅ 训练完成!")

def main():
    """主函数"""
    # 创建输出目录
    os.makedirs("final_checkpoints", exist_ok=True)
    
    try:
        # 运行训练
        history = train()
        
        # 打印总结
        if history:
            print_summary(history)
        
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