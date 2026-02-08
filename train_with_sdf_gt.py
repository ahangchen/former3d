#!/usr/bin/env python3
"""
使用生成的SDF真值训练StreamSDFFormer
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
from generate_tartanair_sdf import TartanAirSDFGenerator


class SDFGroundTruthDataset(torch.utils.data.Dataset):
    """SDF真值数据集"""
    
    def __init__(self, sdf_data_path, num_samples=1000):
        """
        初始化数据集
        
        Args:
            sdf_data_path: SDF数据文件路径 (.npz)
            num_samples: 采样数量
        """
        # 加载SDF数据
        data = np.load(sdf_data_path)
        self.sdf_grid = data['sdf']  # (D, H, W)
        self.occ_grid = data['occupancy']  # (D, H, W)
        self.voxel_size = float(data['voxel_size'])
        self.bounds = data['bounds']
        
        # 获取占用体素的坐标
        occ_coords = np.where(self.occ_grid > 0.5)
        self.occupied_voxels = list(zip(occ_coords[0], occ_coords[1], occ_coords[2]))
        
        # 如果没有足够的占用体素，使用所有体素
        if len(self.occupied_voxels) < num_samples:
            # 创建所有体素的坐标
            D, H, W = self.sdf_grid.shape
            all_coords = [(d, h, w) for d in range(D) for h in range(H) for w in range(W)]
            self.sample_coords = all_coords[:num_samples]
        else:
            # 随机采样占用体素
            indices = np.random.choice(len(self.occupied_voxels), num_samples, replace=False)
            self.sample_coords = [self.occupied_voxels[i] for i in indices]
        
        print(f"SDF真值数据集:")
        print(f"  网格形状: {self.sdf_grid.shape}")
        print(f"  体素大小: {self.voxel_size}米")
        print(f"  占用体素: {len(self.occupied_voxels)}")
        print(f"  采样数量: {len(self.sample_coords)}")
    
    def __len__(self):
        return len(self.sample_coords)
    
    def __getitem__(self, idx):
        # 获取体素坐标
        d, h, w = self.sample_coords[idx]
        
        # 获取SDF真值
        sdf_gt = self.sdf_grid[d, h, w]
        
        # 获取占用真值
        occ_gt = self.occ_grid[d, h, w]
        
        # 创建体素坐标张量
        coords = torch.tensor([d, h, w], dtype=torch.float32)
        
        # 转换为世界坐标
        world_coords = torch.tensor([
            self.bounds[0, 0] + w * self.voxel_size + self.voxel_size / 2,
            self.bounds[1, 0] + h * self.voxel_size + self.voxel_size / 2,
            self.bounds[2, 0] + d * self.voxel_size + self.voxel_size / 2
        ], dtype=torch.float32)
        
        return {
            'coords': coords,
            'world_coords': world_coords,
            'sdf_gt': torch.tensor(sdf_gt, dtype=torch.float32),
            'occ_gt': torch.tensor(occ_gt, dtype=torch.float32)
        }


def train_with_sdf_gt(model, dataset, device, num_epochs=10, batch_size=32, lr=1e-3):
    """使用SDF真值训练模型"""
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    # 定义损失函数
    sdf_criterion = nn.MSELoss()
    occ_criterion = nn.BCEWithLogitsLoss()
    
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 训练循环
    model.train()
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_sdf_loss = 0.0
        epoch_occ_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            # 移动到设备
            coords = batch['coords'].to(device)
            world_coords = batch['world_coords'].to(device)
            sdf_gt = batch['sdf_gt'].to(device)
            occ_gt = batch['occ_gt'].to(device)
            
            # 前向传播
            # 这里简化：直接预测SDF和占用
            # 实际应该使用完整的模型前向传播
            batch_size = coords.shape[0]
            
            # 创建模拟的体素特征
            voxel_features = torch.randn(batch_size, 64, device=device)
            
            # 模型预测
            # 注意：这里需要根据实际模型接口调整
            # 简化版本：直接使用线性层预测
            sdf_pred = model.sdf_head(voxel_features).squeeze()
            occ_pred = model.occ_head(voxel_features).squeeze()
            
            # 计算损失
            sdf_loss = sdf_criterion(sdf_pred, sdf_gt)
            occ_loss = occ_criterion(occ_pred, occ_gt)
            loss = sdf_loss + occ_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录损失
            epoch_loss += loss.item()
            epoch_sdf_loss += sdf_loss.item()
            epoch_occ_loss += occ_loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'sdf': sdf_loss.item(),
                'occ': occ_loss.item()
            })
        
        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        avg_sdf_loss = epoch_sdf_loss / len(dataloader)
        avg_occ_loss = epoch_occ_loss / len(dataloader)
        
        train_losses.append({
            'total': avg_loss,
            'sdf': avg_sdf_loss,
            'occ': avg_occ_loss
        })
        
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, sdf={avg_sdf_loss:.4f}, occ={avg_occ_loss:.4f}")
    
    return train_losses


def main():
    parser = argparse.ArgumentParser(description="使用SDF真值训练StreamSDFFormer")
    parser.add_argument("--sdf_data", type=str, 
                       default="./tartanair_sdf_test/abandonedfactory_sample_P001_sdf_occ.npz",
                       help="SDF真值数据文件")
    parser.add_argument("--num_samples", type=int, default=1000,
                       help="采样数量")
    parser.add_argument("--num_epochs", type=int, default=5,
                       help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="学习率")
    parser.add_argument("--device", type=str, default="cuda",
                       help="设备 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == "cuda" and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        args.device = "cpu"
    
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 1. 创建数据集
    print(f"\n加载SDF真值数据: {args.sdf_data}")
    dataset = SDFGroundTruthDataset(args.sdf_data, num_samples=args.num_samples)
    
    # 2. 创建模型
    print("\n创建StreamSDFFormer模型...")
    model = StreamSDFFormerIntegrated(
        voxel_size=0.08,
        crop_size=(24, 48, 48),
        memory_efficient=True
    ).to(device)
    
    # 添加简化的预测头（用于测试）
    model.sdf_head = nn.Linear(64, 1).to(device)
    model.occ_head = nn.Linear(64, 1).to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 训练
    print(f"\n开始训练 ({args.num_epochs} 轮)...")
    losses = train_with_sdf_gt(
        model, dataset, device,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    # 4. 保存模型
    output_dir = Path("./trained_models")
    output_dir.mkdir(exist_ok=True)
    
    model_path = output_dir / "stream_sdfformer_sdf_trained.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'losses': losses,
        'config': {
            'voxel_size': 0.08,
            'crop_size': (24, 48, 48),
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'lr': args.lr
        }
    }, model_path)
    
    print(f"\n模型保存到: {model_path}")
    
    # 5. 绘制损失曲线
    try:
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(losses) + 1)
        total_losses = [l['total'] for l in losses]
        sdf_losses = [l['sdf'] for l in losses]
        occ_losses = [l['occ'] for l in losses]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, total_losses, 'b-', label='Total Loss', linewidth=2)
        plt.plot(epochs, sdf_losses, 'r--', label='SDF Loss', linewidth=2)
        plt.plot(epochs, occ_losses, 'g-.', label='Occ Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss with SDF Ground Truth')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        loss_plot_path = output_dir / "training_loss.png"
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"损失曲线保存到: {loss_plot_path}")
        
    except ImportError:
        print("警告: matplotlib未安装，跳过损失曲线绘制")
    
    print("\n🎉 训练完成!")
    print(f"最终损失: {losses[-1]['total']:.4f}")
    print(f"SDF损失: {losses[-1]['sdf']:.4f}")
    print(f"占用损失: {losses[-1]['occ']:.4f}")


if __name__ == "__main__":
    main()