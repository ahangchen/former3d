#!/usr/bin/env python3
"""
测试数据格式和模型输入输出
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from online_tartanair_dataset import OnlineTartanAirDataset

# 创建数据集
data_root = "/home/cwh/Study/dataset/tartanair"
sequence_name = "abandonedfactory_sample_P001"

print("创建数据集...")
dataset = OnlineTartanAirDataset(
    data_root=data_root,
    sequence_name=sequence_name,
    n_frames=4,
    crop_size=(32, 32, 24),
    voxel_size=0.08,
    target_image_size=(128, 128),
    max_depth=10.0,
    truncation_margin=0.2,
    augment=False
)

# 获取一个样本
print("\n获取样本...")
sample = dataset[0]

print("\n样本内容:")
for key, value in sample.items():
    if isinstance(value, torch.Tensor):
        print(f"  {key}: {value.shape} {value.dtype}")
    else:
        print(f"  {key}: {value}")

# 检查关键维度
print("\n关键维度检查:")
print(f"rgb_images: {sample['rgb_images'].shape}")  # 期望: (F, 3, H, W)
print(f"poses: {sample['poses'].shape}")  # 期望: (F, 4, 4)
print(f"intrinsics: {sample['intrinsics'].shape}")  # 期望: (3, 3)
print(f"tsdf: {sample['tsdf'].shape}")  # 期望: (D, H, W)

# 检查模型期望的输入格式
print("\n检查模型期望的输入格式...")
print("模型期望:")
print("  images: (batch, 3, height, width)")
print("  poses: (batch, 4, 4)")
print("  intrinsics: (batch, 3, 3)")

# 调整数据格式以匹配模型期望
print("\n调整数据格式...")
# 添加批次维度
rgb_images = sample['rgb_images'].unsqueeze(0)  # (1, F, 3, H, W)
poses = sample['poses'].unsqueeze(0)  # (1, F, 4, 4)
intrinsics = sample['intrinsics'].unsqueeze(0)  # (1, 3, 3)

print(f"调整后:")
print(f"  rgb_images: {rgb_images.shape}")
print(f"  poses: {poses.shape}")
print(f"  intrinsics: {intrinsics.shape}")

# 检查是否能够导入模型
print("\n尝试导入模型...")
try:
    from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
    
    print("创建模型...")
    model = StreamSDFFormerIntegrated(
        attn_heads=8,
        attn_layers=4,
        use_proj_occ=True,
        voxel_size=0.08,
        crop_size=(32, 32, 24)
    )
    
    print(f"模型创建成功!")
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试单帧推理
    print("\n测试单帧推理...")
    # 使用第一帧
    image0 = rgb_images[0, 0:1]  # (1, 3, H, W)
    pose0 = poses[0, 0:1]  # (1, 4, 4)
    intrinsics0 = intrinsics[0:1]  # (1, 3, 3)
    
    print(f"输入:")
    print(f"  image: {image0.shape}")
    print(f"  pose: {pose0.shape}")
    print(f"  intrinsics: {intrinsics0.shape}")
    
    with torch.no_grad():
        output, new_state = model.forward_single_frame(
            images=image0,
            poses=pose0,
            intrinsics=intrinsics0,
            reset_state=True
        )
    
    print(f"输出类型: {type(output)}")
    if isinstance(output, dict):
        print(f"输出键: {list(output.keys())}")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成!")