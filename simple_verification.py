#!/usr/bin/env python3
"""
最简单的训练验证
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("最简单的训练验证")
print("="*80)

# 检查环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ 使用CPU")

# 1. 测试数据集加载
print("\n1. 测试数据集加载...")
try:
    from online_tartanair_dataset import OnlineTartanAirDataset
    
    dataset = OnlineTartanAirDataset(
        data_root="/home/cwh/Study/dataset/tartanair",
        sequence_name="abandonedfactory_sample_P001",
        n_frames=2,
        crop_size=(16, 16, 12),
        voxel_size=0.16,
        target_image_size=(64, 64)
    )
    
    print(f"✅ 数据集大小: {len(dataset)}")
    
    # 获取样本
    sample = dataset[0]
    print(f"✅ 样本类型: {type(sample)}")
    
    if isinstance(sample, dict):
        print(f"✅ 样本是字典，键: {list(sample.keys())}")
    else:
        print(f"⚠️ 样本不是字典: {type(sample)}")
        
except Exception as e:
    print(f"❌ 数据集加载失败: {e}")

# 2. 测试数据加载器
print("\n2. 测试数据加载器...")
try:
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )
    
    print(f"✅ 数据加载器创建成功")
    
    # 获取一个批次
    for batch in dataloader:
        print(f"✅ 批次类型: {type(batch)}")
        
        if isinstance(batch, dict):
            print(f"✅ 批次是字典")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")
        elif isinstance(batch, (list, tuple)):
            print(f"⚠️ 批次是列表/元组，长度: {len(batch)}")
            for i, item in enumerate(batch):
                print(f"  元素{i}: {type(item)}")
                if isinstance(item, torch.Tensor):
                    print(f"    形状: {item.shape}")
        break
        
except Exception as e:
    print(f"❌ 数据加载器失败: {e}")

# 3. 测试简单训练
print("\n3. 测试简单训练...")

# 创建简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.net(x)

model = SimpleModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

try:
    # 训练几步
    for step in range(3):
        # 生成随机数据
        x = torch.randn(8, 10).to(device)
        y = torch.randn(8, 1).to(device)
        
        # 训练步骤
        optimizer.zero_grad()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        
        print(f"  步骤 {step+1}: 损失={loss.item():.6f}")
    
    print("✅ 简单训练测试通过")
    
except Exception as e:
    print(f"❌ 简单训练失败: {e}")

# 4. 测试实际数据训练
print("\n4. 测试实际数据训练...")
try:
    # 创建简化SDF模型
    class SimpleSDFModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4))
            )
            self.decoder = nn.Sequential(
                nn.Linear(8*4*4 + 3, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, images, poses, intrinsics, reset_state=False):
            B, C, H, W = images.shape
            img_features = self.encoder(images).view(B, -1)
            points = torch.randn(B, 100, 3).to(images.device)
            img_features_expanded = img_features.unsqueeze(1).repeat(1, 100, 1)
            combined = torch.cat([img_features_expanded, points], dim=-1)
            return {'sdf': self.decoder(combined)}
    
    model = SimpleSDFModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    # 使用实际数据
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 1:  # 只测试一个批次
            break
            
        print(f"✅ 处理批次 {batch_idx+1}")
        
        if isinstance(batch, dict):
            images = batch['rgb_images'].to(device)
            poses = batch['poses'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            
            print(f"  图像形状: {images.shape}")
            print(f"  位姿形状: {poses.shape}")
            print(f"  内参形状: {intrinsics.shape}")
            
            # 训练步骤
            optimizer.zero_grad()
            output = model(images[:, 0:1], poses[:, 0:1], intrinsics)
            
            if 'sdf' in output:
                pred_sdf = output['sdf']
                target_sdf = torch.randn_like(pred_sdf)
                loss = loss_fn(pred_sdf, target_sdf)
                loss.backward()
                optimizer.step()
                
                print(f"  训练成功! 损失={loss.item():.6f}")
            else:
                print(f"⚠️ 输出中没有'sdf'键")
                
        else:
            print(f"❌ 批次不是字典，无法处理")
    
    print("✅ 实际数据训练测试通过")
    
except Exception as e:
    print(f"❌ 实际数据训练失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("验证完成")
print("="*80)