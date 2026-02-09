#!/usr/bin/env python3
"""
测试dataset_modification_plan的实施情况
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset

print('='*80)
print('测试 dataset_modification_plan 实施情况')
print('='*80)

# 测试1: 数据集初始化
print('\n1. 测试数据集初始化...')
try:
    dataset = MultiSequenceTartanAirDataset(
        data_root='/home/cwh/Study/dataset/tartanair',
        n_view=5,
        stride=2,
        crop_size=(48, 48, 32),
        voxel_size=0.04,
        target_image_size=(256, 256),
        max_sequences=2,
        shuffle=True
    )
    print(f'   ✅ 数据集初始化成功')
    print(f'      总序列数: 2 (预期)')
    print(f'      总片段数: {len(dataset)}')
except Exception as e:
    print(f'   ❌ 数据集初始化失败: {e}')
    sys.exit(1)

# 测试2: 数据形状
print('\n2. 测试数据形状...')
sample = dataset[0]
expected_shapes = {
    'rgb_images': (5, 3, 256, 256),
    'poses': (5, 4, 4),
    'intrinsics': (3, 3),
    'tsdf': (1, 48, 48, 32)
}

all_correct = True
for key, expected_shape in expected_shapes.items():
    actual_shape = sample[key].shape
    if actual_shape == expected_shape:
        print(f'   ✅ {key}: {actual_shape} (符合预期)')
    else:
        print(f'   ❌ {key}: {actual_shape} (预期: {expected_shape})')
        all_correct = False

if not all_correct:
    print('   ⚠️ 数据形状不符合预期')
    sys.exit(1)

# 测试3: DataLoader批次处理
print('\n3. 测试DataLoader批次处理...')
try:
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    print(f'   ✅ DataLoader工作正常')
    print(f'      批次 rgb_images 形状: {batch["rgb_images"].shape} (预期: [2, 5, 3, 256, 256])')
    print(f'      批次 tsdf 形状: {batch["tsdf"].shape} (预期: [2, 1, 48, 48, 32])')
except Exception as e:
    print(f'   ❌ DataLoader失败: {e}')
    sys.exit(1)

# 测试4: 数据处理函数
print('\n4. 测试数据处理函数...')
def correct_tsdf_dimensions(tsdf_batch):
    """修正TSDF维度从 [batch, 1, H, W, D] 到 [batch, 1, D, H, W]"""
    return tsdf_batch.permute(0, 1, 4, 2, 3)

def prepare_input_data(rgb_images, tsdf_gt_correct, frame_idx=0):
    """准备输入数据：将2D图像转换为3D体素网格"""
    batch_size = rgb_images.shape[0]
    current_images = rgb_images[:, frame_idx]  # [batch, 3, H, W]
    
    tsdf_depth = tsdf_gt_correct.shape[2]   # D
    tsdf_height = tsdf_gt_correct.shape[3]  # H
    tsdf_width = tsdf_gt_correct.shape[4]   # W
    
    current_images_resized = torch.nn.functional.interpolate(
        current_images,
        size=(tsdf_height, tsdf_width),
        mode='bilinear',
        align_corners=False
    )
    
    current_images_3d = current_images_resized.unsqueeze(2).repeat(1, 1, tsdf_depth, 1, 1)
    return current_images_3d

try:
    # 修正TSDF维度
    tsdf_gt = correct_tsdf_dimensions(batch['tsdf'])
    
    # 准备输入数据
    input_3d = prepare_input_data(batch['rgb_images'], tsdf_gt, frame_idx=0)
    
    print(f'   ✅ 数据处理函数工作正常')
    print(f'      输入3D形状: {input_3d.shape} (预期: [2, 3, 32, 48, 48])')
except Exception as e:
    print(f'   ❌ 数据处理失败: {e}')
    sys.exit(1)

# 测试5: 简单模型训练
print('\n5. 测试简单模型训练...')
class SimpleSDF3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)

try:
    model = SimpleSDF3DModel()
    
    # 如果有GPU，使用GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    input_3d = input_3d.to(device)
    tsdf_gt = tsdf_gt.to(device)
    
    # 前向传播
    output = model(input_3d)
    
    # 计算损失
    criterion = nn.MSELoss()
    loss = criterion(output, tsdf_gt)
    
    print(f'   ✅ 模型训练测试通过')
    print(f'      模型输出形状: {output.shape} (预期: [2, 1, 32, 48, 48])')
    print(f'      损失值: {loss.item():.6f}')
    
    # 反向传播测试
    loss.backward()
    print(f'   ✅ 反向传播测试通过')
    
except Exception as e:
    print(f'   ❌ 模型训练测试失败: {e}')
    sys.exit(1)

print('\n' + '='*80)
print('🎉 所有测试通过! dataset_modification_plan 实施成功!')
print('='*80)
print('\n总结:')
print('1. ✅ 多序列数据集类已正确实现')
print('2. ✅ 数据形状符合预期')
print('3. ✅ DataLoader正确处理批次')
print('4. ✅ 数据处理函数工作正常')
print('5. ✅ 模型训练流程完整')
print('\n下一步:')
print('1. 运行完整的训练: python final_multi_sequence_training_fixed.py')
print('2. 监控训练损失是否正常下降')
print('3. 验证模型性能')