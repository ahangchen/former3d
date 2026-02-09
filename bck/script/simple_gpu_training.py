#!/usr/bin/env python3
"""
简化GPU训练测试
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入数据集
from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset

# 简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 1, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

def prepare_input(rgb_images, tsdf_gt, frame_idx=0):
    """准备输入数据"""
    current_images = rgb_images[:, frame_idx]
    
    # 调整图像尺寸以匹配TSDF
    tsdf_height = tsdf_gt.shape[3]
    tsdf_width = tsdf_gt.shape[4]
    
    current_images_resized = torch.nn.functional.interpolate(
        current_images,
        size=(tsdf_height, tsdf_width),
        mode='bilinear',
        align_corners=False
    )
    
    # 扩展为3D
    tsdf_depth = tsdf_gt.shape[2]
    current_images_3d = current_images_resized.unsqueeze(2).repeat(1, 1, tsdf_depth, 1, 1)
    
    return current_images_3d

def main():
    print("=" * 60)
    print("简化GPU训练测试")
    print("=" * 60)
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 配置
    data_root = "/home/cwh/Study/dataset/tartanair"
    batch_size = 2
    n_view = 5
    stride = 2
    crop_size = (48, 48, 32)
    voxel_size = 0.04
    target_image_size = (256, 256)
    max_sequences = 2
    
    print("1. 创建数据集...")
    dataset = MultiSequenceTartanAirDataset(
        data_root=data_root,
        n_view=n_view,
        stride=stride,
        crop_size=crop_size,
        voxel_size=voxel_size,
        target_image_size=target_image_size,
        max_sequences=max_sequences,
        shuffle=True
    )
    
    # 只取前50个样本用于快速测试
    from torch.utils.data import Subset
    dataset = Subset(dataset, indices=range(min(50, len(dataset))))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"   数据集大小: {len(dataset)}")
    print()
    
    # 创建模型
    print("2. 创建模型...")
    model = SimpleModel().to(device)
    
    # 计算参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   模型参数: {total_params:,}")
    print()
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 训练
    print("3. 开始训练...")
    print()
    
    model.train()
    total_loss = 0.0
    total_samples = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # 获取数据并移动到设备
        rgb_images = batch['rgb_images'].to(device)
        tsdf_gt = batch['tsdf'].to(device)
        
        batch_size = rgb_images.shape[0]
        
        # 准备输入（使用第一帧）
        input_3d = prepare_input(rgb_images, tsdf_gt, frame_idx=0)
        
        # 前向传播
        output = model(input_3d)
        
        # 计算损失
        loss = criterion(output, tsdf_gt)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # 打印进度
        if (batch_idx + 1) % 5 == 0:
            avg_loss = total_loss / total_samples
            print(f"   批次 {batch_idx+1:03d}, 损失: {avg_loss:.6f}")
    
    training_time = time.time() - start_time
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    print()
    print(f"   训练完成!")
    print(f"   平均损失: {avg_loss:.6f}")
    print(f"   训练时间: {training_time:.2f}秒")
    print(f"   总样本数: {total_samples}")
    print()
    
    # 验证
    print("4. 验证...")
    model.eval()
    
    val_loss = 0.0
    val_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:  # 只验证3个批次
                break
            
            rgb_images = batch['rgb_images'].to(device)
            tsdf_gt = batch['tsdf'].to(device)
            
            input_3d = prepare_input(rgb_images, tsdf_gt, frame_idx=0)
            output = model(input_3d)
            
            loss = criterion(output, tsdf_gt)
            
            val_loss += loss.item() * rgb_images.shape[0]
            val_samples += rgb_images.shape[0]
    
    avg_val_loss = val_loss / val_samples if val_samples > 0 else 0.0
    print(f"   验证损失: {avg_val_loss:.6f}")
    print()
    
    # 保存模型
    print("5. 保存模型...")
    os.makedirs("checkpoints", exist_ok=True)
    model_path = "checkpoints/simple_gpu_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'val_loss': avg_val_loss,
    }, model_path)
    print(f"   模型已保存到: {model_path}")
    print()
    
    # GPU内存信息
    if device.type == 'cuda':
        print("GPU内存使用:")
        print(f"   已分配: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
        print(f"   缓存: {torch.cuda.memory_reserved(0) / 1e6:.2f} MB")
        print(f"   最大已分配: {torch.cuda.max_memory_allocated(0) / 1e6:.2f} MB")
        print()
    
    print("=" * 60)
    print("简化GPU训练测试完成!")
    print(f"训练损失: {avg_loss:.6f}")
    print(f"验证损失: {avg_val_loss:.6f}")
    print(f"训练时间: {training_time:.2f}秒")
    print("=" * 60)

if __name__ == "__main__":
    main()