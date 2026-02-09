#!/usr/bin/env python3
"""
调试形状问题
"""

import os
import sys
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 导入模型
from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated

# 创建模型
model = StreamSDFFormerIntegrated(
    attn_heads=2,
    attn_layers=1,
    use_proj_occ=True,
    voxel_size=0.10,
    fusion_local_radius=0.0,
    crop_size=(24, 24, 16)
)
model = model.to(device)

# 创建测试数据
batch_size = 1
images = torch.randn(batch_size, 3, 96, 96).to(device)
poses = torch.randn(batch_size, 4, 4).to(device)
intrinsics = torch.randn(batch_size, 3, 3).to(device)

print(f"输入形状:")
print(f"  images: {images.shape}")  # 应该是 [1, 3, 96, 96]
print(f"  poses: {poses.shape}")    # 应该是 [1, 4, 4]
print(f"  intrinsics: {intrinsics.shape}")  # 应该是 [1, 3, 3]

# 测试convert_to_sdfformer_batch
print("\n测试convert_to_sdfformer_batch:")
batch = model.convert_to_sdfformer_batch(images, poses, intrinsics)
print(f"batch keys: {list(batch.keys())}")

if 'rgb_imgs' in batch:
    rgb_imgs = batch['rgb_imgs']
    print(f"rgb_imgs形状: {rgb_imgs.shape}")
    print(f"rgb_imgs维度数: {len(rgb_imgs.shape)}")
    
    # 手动解包
    if len(rgb_imgs.shape) == 5:
        batchsize, n_imgs, channels, imheight, imwidth = rgb_imgs.shape
        print(f"解包结果: batchsize={batchsize}, n_imgs={n_imgs}, channels={channels}, imheight={imheight}, imwidth={imwidth}")
    else:
        print(f"错误: 期望5维，得到{len(rgb_imgs.shape)}维")
        print(f"形状: {rgb_imgs.shape}")

# 测试forward_single_frame
print("\n测试forward_single_frame:")
try:
    with torch.no_grad():
        output, state = model.forward_single_frame(images, poses, intrinsics, reset_state=True)
    print("✅ forward_single_frame成功")
except Exception as e:
    print(f"❌ forward_single_frame失败: {e}")
    import traceback
    traceback.print_exc()