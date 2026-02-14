#!/usr/bin/env python3
"""
单GPU训练验证脚本
验证PoseAwareStreamSdfFormerSparse在多尺度特征融合下的训练流程
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from former3d.pose_aware_stream_sdfformer_sparse import PoseAwareStreamSdfFormerSparse


class SimpleStreamDataset(Dataset):
    """简单的流式数据集"""

    def __init__(self, num_samples=10):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        H, W = 48, 64
        n_view = 2

        return {
            'rgb_images': torch.randn(n_view, 3, H, W),
            'poses': torch.eye(4).unsqueeze(0).repeat(n_view, 1, 1),
            'intrinsics': torch.eye(3).unsqueeze(0).repeat(n_view, 1, 1),
            'tsdf': torch.randn(1, 128, 128, 128)
        }


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()

    loss_meter = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch['rgb_images']
        poses = batch['poses']
        intrinsics = batch['intrinsics']

        images = images.to(device)
        poses = poses.to(device)
        intrinsics = intrinsics.to(device)

        # 调用forward_sequence
        try:
            outputs, states = model.forward_sequence(images, poses, intrinsics, reset_state=True)

            # 计算损失
            if outputs['sdf'] is not None:
                loss = outputs['sdf'].mean()

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_meter += loss.item()
                num_batches += 1

                print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss.item():.6f} SDF: {outputs['sdf'].shape}")

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    avg_loss = loss_meter / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    """主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    print("创建模型...")
    model = PoseAwareStreamSdfFormerSparse(
        attn_heads=2,
        attn_layers=2,
        use_proj_occ=False,
        voxel_size=0.0625,
        crop_size=(16, 24, 24),
        use_checkpoint=False
    ).to(device)

    # 创建优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # 创建数据集和数据加载器
    print("创建数据集...")
    train_dataset = SimpleStreamDataset(num_samples=5)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # 训练
    print("\n开始训练...")
    print("="*60)

    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)

        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        print(f"Epoch {epoch+1} 完成:")
        print(f"  - 训练损失: {train_loss:.6f}")
        print(f"  - 样本数: {len(train_loader)}")
        print("-" * 60)

    # 完成训练
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)

    # 梯度检查
    print("\n梯度检查:")
    grad_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0:
                grad_count += 1
                if grad_count <= 5:
                    print(f"  {name}: grad_norm={grad_norm:.6f}")

    print(f"总共有 {grad_count} 个参数有梯度")


if __name__ == "__main__":
    main()
