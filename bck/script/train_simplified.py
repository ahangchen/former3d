#!/usr/bin/env python3
"""
使用简化模型的训练脚本
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os

# 导入简化模型
from simplified_model import SimplifiedSDFFormer

def train():
    print("开始训练简化模型...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimplifiedSDFFormer(
        voxel_size=0.15,
        crop_size=(16, 16, 16)
    ).to(device)
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建模拟数据集
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=100):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            images = torch.randn(3, 64, 64)
            poses = torch.eye(4)
            intrinsics = torch.eye(3)
            tsdf_target = torch.randn(16, 16, 16)
            
            return {
                'rgb_images': images.unsqueeze(0),  # [1, 3, 64, 64]
                'poses': poses.unsqueeze(0),        # [1, 4, 4]
                'intrinsics': intrinsics,           # [3, 3]
                'tsdf': tsdf_target                 # [16, 16, 16]
            }
    
    # 数据加载器
    dataset = MockDataset(50)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.HuberLoss(delta=0.1)
    
    # 训练循环
    num_epochs = 5
    checkpoint_dir = "simplified_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"开始训练，共{num_epochs}个epoch...")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            # 移动到设备
            images = batch['rgb_images'].to(device)
            poses = batch['poses'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            tsdf_target = batch['tsdf'].to(device)
            
            # 使用第一帧
            current_images = images[:, 0]  # [B, 3, 64, 64]
            current_poses = poses[:, 0]    # [B, 4, 4]
            
            # 前向传播
            output = model(current_images, current_poses, intrinsics, reset_state=True)
            
            if 'sdf' in output:
                pred_sdf = output['sdf']  # [B, num_points, 1]
                
                # 采样目标
                B, num_points, _ = pred_sdf.shape
                tsdf_flat = tsdf_target.view(B, -1)
                
                if tsdf_flat.shape[1] >= num_points:
                    indices = torch.randint(0, tsdf_flat.shape[1], (B, num_points)).to(device)
                    target_sdf = torch.gather(tsdf_flat, 1, indices).unsqueeze(-1)
                else:
                    repeat_factor = (num_points + tsdf_flat.shape[1] - 1) // tsdf_flat.shape[1]
                    target_sdf = tsdf_flat.repeat(1, repeat_factor)[:, :num_points].unsqueeze(-1)
                
                # 计算损失
                loss = loss_fn(pred_sdf, target_sdf)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx+1}: Loss={loss.item():.6f}")
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        print(f"Epoch {epoch}/{num_epochs}: Loss={avg_loss:.6f}, Time={epoch_time:.1f}s")
        
        # 保存检查点
        if epoch % 2 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': avg_loss
            }, checkpoint_path)
            print(f"检查点保存到: {checkpoint_path}")
    
    # 保存最终模型
    final_path = os.path.join(checkpoint_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    
    print(f"
训练完成! 模型保存到: {final_path}")
    print("✅ 简化模型训练成功!")

if __name__ == "__main__":
    train()
