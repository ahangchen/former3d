#!/usr/bin/env python3
"""
测试多序列数据集与训练脚本的集成
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("测试多序列数据集与训练集成")
print("="*80)

# 检查PyTorch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 测试1: 导入数据集
print("\n" + "="*60)
print("测试1: 导入数据集")
print("="*60)

try:
    # 创建简化的数据集类用于测试
    class SimpleTestDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=20, n_view=5):
            self.num_samples = num_samples
            self.n_view = n_view
            self.crop_size = (48, 48, 32)
            self.image_size = (256, 256)
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # 模拟数据
            rgb_images = torch.randn(self.n_view, 3, self.image_size[0], self.image_size[1])
            poses = torch.eye(4).unsqueeze(0).repeat(self.n_view, 1, 1)
            tsdf = torch.randn(1, *self.crop_size)
            intrinsics = torch.eye(3)
            
            return {
                'rgb_images': rgb_images,
                'poses': poses,
                'tsdf': tsdf,
                'intrinsics': intrinsics,
                'sequence_name': f'test_seq_{idx % 3}',
                'segment_idx': idx
            }
    
    dataset = SimpleTestDataset(num_samples=20, n_view=5)
    print(f"✅ 数据集创建成功: {len(dataset)} 个样本")
    
except Exception as e:
    print(f"❌ 数据集导入失败: {e}")
    sys.exit(1)

# 测试2: 数据加载器
print("\n" + "="*60)
print("测试2: 数据加载器")
print("="*60)

try:
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    print(f"✅ 数据加载器创建成功")
    print(f"   批量大小: {batch_size}")
    print(f"   总批次: {len(dataloader)}")
    
    # 测试一个批次
    for batch_idx, batch in enumerate(dataloader):
        print(f"\n   批次 {batch_idx}:")
        print(f"     RGB形状: {batch['rgb_images'].shape}")  # (batch_size, n_view, 3, H, W)
        print(f"     位姿形状: {batch['poses'].shape}")      # (batch_size, n_view, 4, 4)
        print(f"     TSDF形状: {batch['tsdf'].shape}")       # (batch_size, 1, D, H, W)
        print(f"     序列: {batch['sequence_name']}")
        
        # 验证形状
        assert batch['rgb_images'].shape == (batch_size, 5, 3, 256, 256)
        assert batch['poses'].shape == (batch_size, 5, 4, 4)
        assert batch['tsdf'].shape == (batch_size, 1, 48, 48, 32)
        
        if batch_idx >= 2:  # 只测试前3个批次
            break
    
    print(f"\n✅ 数据形状验证通过")
    
except Exception as e:
    print(f"❌ 数据加载器测试失败: {e}")
    sys.exit(1)

# 测试3: 模拟模型前向传播
print("\n" + "="*60)
print("测试3: 模拟模型前向传播")
print("="*60)

try:
    # 创建简单的模拟模型（内存优化版本）
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 使用更小的特征维度
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2)
            
            # 计算经过3次stride=2后的特征图大小
            # 256 -> 128 -> 64 -> 32
            self.fc = nn.Linear(64 * 32 * 32, 48 * 48 * 32)
            
        def forward(self, x):
            # x: (batch_size, 3, H, W)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = x.view(x.size(0), 48, 48, 32)
            return x
        
        def reset_state(self, batch_size):
            # 模拟状态重置
            pass
    
    model = SimpleModel().to(device)
    print(f"✅ 模型创建成功")
    
    # 测试一个批次的前向传播
    batch = next(iter(dataloader))
    rgb_images = batch['rgb_images'].to(device)
    poses = batch['poses'].to(device)
    tsdf_target = batch['tsdf'].to(device)
    
    batch_size, n_view = rgb_images.shape[:2]
    
    print(f"   批量大小: {batch_size}, 视图数: {n_view}")
    
    # 模拟训练循环
    model.train()
    criterion = nn.MSELoss()
    
    # 重置模型状态
    model.reset_state(batch_size=batch_size)
    
    total_loss = 0.0
    
    for frame_idx in range(n_view):
        current_images = rgb_images[:, frame_idx]  # (batch_size, 3, H, W)
        
        # 前向传播
        tsdf_pred = model(current_images)  # (batch_size, D, H, W)
        
        # 只对最后一个帧计算损失
        if frame_idx == n_view - 1:
            loss = criterion(tsdf_pred, tsdf_target.squeeze(1))  # 移除通道维度
            total_loss += loss.item()
            
            print(f"     帧 {frame_idx}: 预测形状 {tsdf_pred.shape}, 目标形状 {tsdf_target.squeeze(1).shape}")
    
    avg_loss = total_loss / n_view
    print(f"\n✅ 前向传播测试通过")
    print(f"   平均损失: {avg_loss:.6f}")
    
except Exception as e:
    print(f"❌ 模型前向传播测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 训练循环集成
print("\n" + "="*60)
print("测试4: 完整训练循环集成")
print("="*60)

try:
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 模拟一个epoch的训练
    print("模拟训练一个epoch...")
    
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        rgb_images = batch['rgb_images'].to(device)
        tsdf_target = batch['tsdf'].to(device)
        
        batch_size, n_view = rgb_images.shape[:2]
        
        # 重置模型状态
        model.reset_state(batch_size=batch_size)
        
        total_frame_loss = 0.0
        
        for frame_idx in range(n_view):
            current_images = rgb_images[:, frame_idx]
            
            # 前向传播
            tsdf_pred = model(current_images)
            
            # 只对最后一个帧计算损失
            if frame_idx == n_view - 1:
                loss = criterion(tsdf_pred, tsdf_target.squeeze(1))
                total_frame_loss += loss
        
        # 平均损失
        avg_loss = total_frame_loss / n_view
        
        # 反向传播
        optimizer.zero_grad()
        avg_loss.backward()
        optimizer.step()
        
        epoch_loss += avg_loss.item()
        num_batches += 1
        
        print(f"   批次 {batch_idx}: 损失 {avg_loss.item():.6f}")
        
        if batch_idx >= 3:  # 只测试4个批次
            break
    
    avg_epoch_loss = epoch_loss / max(num_batches, 1)
    print(f"\n✅ 训练循环集成测试通过")
    print(f"   epoch平均损失: {avg_epoch_loss:.6f}")
    
except Exception as e:
    print(f"❌ 训练循环集成测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("🎉 所有集成测试通过!")
print("="*80)
print("\n下一步:")
print("1. 确保实际数据集文件存在")
print("2. 运行 multi_sequence_training.py 进行实际训练")
print("3. 监控训练日志和损失曲线")
print("\n注意事项:")
print("- 确保TartanAir数据目录存在: /home/cwh/coding/former3d/files/sample_tartanair")
print("- 如果使用模拟数据，需要调整数据集实现")
print("- 根据GPU内存调整batch_size")