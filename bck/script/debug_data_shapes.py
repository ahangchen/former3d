#!/usr/bin/env python3
"""
调试数据形状
"""

import os
import sys
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入数据集
try:
    from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
    print("✅ 成功导入MultiSequenceTartanAirDataset")
except ImportError as e:
    print(f"❌ 无法导入MultiSequenceTartanAirDataset: {e}")
    sys.exit(1)

def main():
    # 配置
    data_root = "/home/cwh/Study/dataset/tartanair"
    batch_size = 2
    n_view = 5
    stride = 2
    crop_size = (48, 48, 32)
    voxel_size = 0.04
    target_image_size = (256, 256)
    max_sequences = 2
    
    print("============================================================")
    print("调试数据形状")
    print("============================================================")
    print(f"数据根目录: {data_root}")
    print(f"批次大小: {batch_size}")
    print(f"视图数: {n_view}")
    print(f"步长: {stride}")
    print(f"裁剪尺寸: {crop_size}")
    print(f"体素大小: {voxel_size}")
    print(f"图像大小: {target_image_size}")
    print(f"最大序列数: {max_sequences}")
    print()
    
    # 创建数据集
    print("创建数据集...")
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
    
    print(f"数据集大小: {len(dataset)}")
    print()
    
    # 获取一个样本
    print("获取第一个样本...")
    sample = dataset[0]
    
    # 打印形状
    print("样本形状:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} (dtype: {value.dtype})")
        elif isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape} (dtype: {value.dtype})")
        else:
            print(f"  {key}: {type(value)}")
    
    print()
    
    # 创建数据加载器
    print("创建数据加载器...")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    
    # 获取一个批次
    print("获取第一个批次...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"批次 {batch_idx}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} (dtype: {value.dtype})")
            elif isinstance(value, list):
                print(f"  {key}: 列表, 长度 {len(value)}")
                for i, item in enumerate(value[:2]):  # 只显示前两个
                    if isinstance(item, str):
                        print(f"    元素 {i}: '{item}'")
                    else:
                        print(f"    元素 {i}: {type(item)}")
            else:
                print(f"  {key}: {type(value)}")
        
        # 检查TSDF GT的形状
        if 'tsdf' in batch:
            tsdf = batch['tsdf']
            print(f"\nTSDF GT详细形状分析:")
            print(f"  维度数: {tsdf.dim()}")
            print(f"  形状: {tsdf.shape}")
            print(f"  数据类型: {tsdf.dtype}")
            
            # 检查是否需要调整维度
            if tsdf.dim() == 5:
                print(f"  TSDF是5D张量: (batch, channel, depth, height, width)")
                print(f"  需要转换为4D图像格式: (batch, channel, height, width)")
                print(f"  对于体素网格，可能需要选择深度切片或使用3D插值")
        
        break  # 只处理第一个批次

if __name__ == "__main__":
    main()