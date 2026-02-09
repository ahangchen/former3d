#!/usr/bin/env python3
"""
使用多序列数据集的训练脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 尝试导入数据集
try:
    from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
    DATASET_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入MultiSequenceTartanAirDataset: {e}")
    print("将使用简化的数据集进行演示")
    DATASET_AVAILABLE = False

# 简化的模型定义（用于演示）
class SimpleStreamSDFFormer(nn.Module):
    """简化的Stream-SDFFormer模型"""
    
    def __init__(self, input_channels=3, hidden_dim=128, output_dim=1):
        super().__init__()
        
        # 简单的编码器-解码器结构
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, output_dim, kernel_size=3, padding=1),
        )
        
        # 状态（用于流式处理）
        self.hidden_state = None
        
    def forward(self, x, reset_state=False):
        """
        前向传播
        
        Args:
            x: 输入图像 (batch_size, 3, H, W)
            reset_state: 是否重置隐藏状态
            
        Returns:
            sdf_pred: 预测的SDF (batch_size, 1, H, W)
        """
        if reset_state or self.hidden_state is None:
            self.hidden_state = torch.zeros(x.size(0), 256, x.size(2)//4, x.size(3)//4, device=x.device)
        
        # 编码
        features = self.encoder(x)
        
        # 调整特征图尺寸以匹配隐藏状态
        if self.hidden_state is not None:
            # 确保尺寸匹配
            if features.shape != self.hidden_state.shape:
                features = torch.nn.functional.interpolate(
                    features, size=self.hidden_state.shape[-2:], mode='bilinear', align_corners=False
                )
        
        # 更新状态（简单平均）
        if self.hidden_state is None:
            self.hidden_state = features
        else:
            self.hidden_state = 0.5 * self.hidden_state + 0.5 * features
        
        # 解码
        sdf_pred = self.decoder(self.hidden_state)
        
        return sdf_pred
    
    def reset_state(self, batch_size, device, img_size=(256, 256)):
        """重置隐藏状态"""
        self.hidden_state = None
        # 预初始化隐藏状态尺寸
        h, w = img_size[0] // 4, img_size[1] // 4
        self.hidden_state = torch.zeros(batch_size, 256, h, w, device=device)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # 移动到设备
        rgb_images = batch['rgb_images'].to(device)  # (batch_size, n_view, 3, H, W)
        tsdf_gt = batch['tsdf'].to(device)           # (batch_size, 1, D, H, W)
        
        batch_size, n_view = rgb_images.shape[:2]
        
        # 重置模型状态（每个片段开始时）
        model.reset_state(batch_size, device, img_size=rgb_images.shape[-2:])
        
        # 遍历片段中的每个时刻
        batch_loss = 0
        for frame_idx in range(n_view):
            # 获取当前帧
            current_images = rgb_images[:, frame_idx]  # (batch_size, 3, H, W)
            
            # 前向传播
            sdf_pred = model(current_images, reset_state=(frame_idx == 0))
            
            # 计算损失（简化：只使用第一帧的预测）
            if frame_idx == 0:
                # 调整TSDF GT的尺寸以匹配预测
                tsdf_gt_resized = torch.nn.functional.interpolate(
                    tsdf_gt, size=sdf_pred.shape[-2:], mode='bilinear', align_corners=False
                )
                loss = criterion(sdf_pred, tsdf_gt_resized)
                batch_loss = loss
        
        # 反向传播
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += batch_loss.item() * batch_size
        total_samples += batch_size
        
        # 打印进度
        if batch_idx % 10 == 0:
            avg_loss = total_loss / max(1, total_samples)
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {batch_loss.item():.6f}, Avg Loss: {avg_loss:.6f}, "
                  f"Time: {elapsed:.2f}s")
    
    avg_loss = total_loss / max(1, total_samples)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            rgb_images = batch['rgb_images'].to(device)
            tsdf_gt = batch['tsdf'].to(device)
            
            batch_size, n_view = rgb_images.shape[:2]
            
            # 重置模型状态
            model.reset_state(batch_size, device, img_size=rgb_images.shape[-2:])
            
            # 只使用第一帧进行验证
            current_images = rgb_images[:, 0]
            sdf_pred = model(current_images, reset_state=True)
            
            # 调整TSDF GT尺寸
            tsdf_gt_resized = torch.nn.functional.interpolate(
                tsdf_gt, size=sdf_pred.shape[-2:], mode='bilinear', align_corners=False
            )
            
            loss = criterion(sdf_pred, tsdf_gt_resized)
            
            total_loss += loss.item() * batch_size
            total_samples += batch_size
    
    avg_loss = total_loss / max(1, total_samples)
    return avg_loss


def create_simple_dataloader(data_root, batch_size=2, n_view=5):
    """创建简化的数据加载器（用于演示）"""
    from torch.utils.data import Dataset
    
    class SimpleDemoDataset(Dataset):
        def __init__(self, num_samples=100, n_view=5):
            self.num_samples = num_samples
            self.n_view = n_view
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # 生成模拟数据
            batch_size = 1  # 简化：每个样本batch_size=1
            
            # RGB图像 (n_view, 3, 256, 256)
            rgb_images = torch.randn(self.n_view, 3, 256, 256)
            
            # 位姿 (n_view, 4, 4)
            poses = torch.eye(4).unsqueeze(0).repeat(self.n_view, 1, 1)
            
            # TSDF (1, 48, 48, 32) -> (1, 32, 48, 48) 调整维度顺序
            tsdf = torch.randn(1, 32, 48, 48)
            
            # 占用网格
            occupancy = (torch.rand(1, 32, 48, 48) > 0.5).float()
            
            return {
                'rgb_images': rgb_images,
                'poses': poses,
                'tsdf': tsdf,
                'occupancy': occupancy,
                'sequence_name': f"demo_seq_{idx}",
                'segment_idx': idx
            }
    
    dataset = SimpleDemoDataset(num_samples=50, n_view=n_view)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    return dataloader


def main():
    """主训练函数"""
    print("=" * 60)
    print("多序列TartanAir训练脚本")
    print("=" * 60)
    
    # 配置参数
    config = {
        'data_root': "/home/cwh/Study/dataset/tartanair",
        'batch_size': 2,
        'n_view': 5,
        'stride': 2,
        'crop_size': (48, 48, 32),
        'voxel_size': 0.04,
        'target_image_size': (256, 256),
        'max_sequences': 3,  # 限制序列数量用于测试
        'num_epochs': 3,
        'learning_rate': 1e-4,
        'device': 'cpu',  # 强制使用CPU，因为CUDA有问题
        'use_real_data': DATASET_AVAILABLE,
    }
    
    print(f"配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 设置设备
    device = torch.device(config['device'])
    print(f"\n使用设备: {device}")
    
    # 创建数据加载器
    print(f"\n创建数据加载器...")
    if config['use_real_data']:
        try:
            dataset = MultiSequenceTartanAirDataset(
                data_root=config['data_root'],
                n_view=config['n_view'],
                stride=config['stride'],
                crop_size=config['crop_size'],
                voxel_size=config['voxel_size'],
                target_image_size=config['target_image_size'],
                max_sequences=config['max_sequences'],
                shuffle=True
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=0,  # 简化：不使用多进程
                pin_memory=True if device.type == 'cuda' else False
            )
            
            print(f"✅ 使用真实数据，数据集大小: {len(dataset)}")
            
        except Exception as e:
            print(f"❌ 加载真实数据失败: {e}")
            print("将使用模拟数据")
            config['use_real_data'] = False
    
    if not config['use_real_data']:
        dataloader = create_simple_dataloader(
            config['data_root'],
            batch_size=config['batch_size'],
            n_view=config['n_view']
        )
        print(f"✅ 使用模拟数据，批次数量: {len(dataloader)}")
    
    # 创建模型
    print(f"\n创建模型...")
    model = SimpleStreamSDFFormer(
        input_channels=3,
        hidden_dim=128,
        output_dim=1
    ).to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # 训练循环
    print(f"\n开始训练...")
    print(f"总epoch数: {config['num_epochs']}")
    
    for epoch in range(1, config['num_epochs'] + 1):
        epoch_start_time = time.time()
        
        # 训练
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        
        # 验证（简化：使用训练集）
        val_loss = validate(model, dataloader, criterion, device)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f"\nEpoch {epoch}/{config['num_epochs']} 完成:")
        print(f"  训练损失: {train_loss:.6f}")
        print(f"  验证损失: {val_loss:.6f}")
        print(f"  时间: {epoch_time:.2f}秒")
        print("-" * 40)
    
    # 保存模型
    print(f"\n保存模型...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"multi_sequence_model_{timestamp}.pth"
    
    torch.save({
        'epoch': config['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'config': config
    }, model_path)
    
    print(f"✅ 模型已保存到: {model_path}")
    
    # 总结
    print(f"\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"最终训练损失: {train_loss:.6f}")
    print(f"最终验证损失: {val_loss:.6f}")
    print(f"模型文件: {model_path}")
    print(f"\n下一步:")
    print("1. 安装完整依赖以使用真实数据")
    print("2. 增加训练epoch数")
    print("3. 使用更复杂的模型架构")
    print("4. 添加更多的验证指标")


if __name__ == "__main__":
    main()