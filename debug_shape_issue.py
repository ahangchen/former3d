#!/usr/bin/env python3
"""
调试形状问题
"""

import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("调试形状问题")
print("="*80)

# 导入模型
from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated

# 创建模型
model = StreamSDFFormerIntegrated(
    attn_heads=2,
    attn_layers=1,
    use_proj_occ=True,
    voxel_size=0.04,
    fusion_local_radius=2.0,
    crop_size=(32, 32, 24)
)

# 测试convert_to_sdfformer_batch方法
print("\n1. 测试convert_to_sdfformer_batch方法...")

# 创建测试数据
batch_size = 1
n_frames = 1
height, width = 256, 256

images = torch.randn(batch_size, n_frames, 3, height, width)
poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_frames, 1, 1)
intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_frames, 1, 1)
intrinsics[:, :, 0, 0] = 500
intrinsics[:, :, 1, 1] = 500
intrinsics[:, :, 0, 2] = width / 2
intrinsics[:, :, 1, 2] = height / 2

print(f"输入形状:")
print(f"  images: {images.shape}")
print(f"  poses: {poses.shape}")
print(f"  intrinsics: {intrinsics.shape}")

# 调用convert_to_sdfformer_batch
try:
    batch = model.convert_to_sdfformer_batch(images, poses, intrinsics)
    print(f"\n✅ convert_to_sdfformer_batch成功")
    print(f"返回的batch键: {list(batch.keys())}")
    
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, dict):
            print(f"  {key} (字典):")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, torch.Tensor):
                    print(f"    {subkey}: {subvalue.shape}")
        else:
            print(f"  {key}: {type(value)}")
            
except Exception as e:
    print(f"\n❌ convert_to_sdfformer_batch失败: {e}")
    import traceback
    traceback.print_exc()

# 测试forward_single_frame方法
print("\n2. 测试forward_single_frame方法...")

# 设置模型为评估模式
model.eval()

try:
    with torch.no_grad():
        output, state = model.forward_single_frame(
            images=images,
            poses=poses,
            intrinsics=intrinsics,
            reset_state=True
        )
    
    print(f"✅ forward_single_frame成功")
    print(f"输出类型: {type(output)}")
    print(f"状态类型: {type(state)}")
    
except Exception as e:
    print(f"\n❌ forward_single_frame失败: {e}")
    import traceback
    traceback.print_exc()
    
    # 尝试更详细的调试
    print("\n尝试逐步调试...")
    try:
        # 手动调用内部方法
        print("调用convert_to_sdfformer_batch...")
        batch = model.convert_to_sdfformer_batch(images, poses, intrinsics)
        
        print(f"batch['rgb_imgs']形状: {batch['rgb_imgs'].shape}")
        
        # 检查get_img_feats期望的形状
        print("\n检查get_img_feats期望的形状...")
        rgb_imgs = batch['rgb_imgs']
        print(f"rgb_imgs形状: {rgb_imgs.shape}")
        print(f"rgb_imgs维度数: {len(rgb_imgs.shape)}")
        
        # 尝试解包
        try:
            batchsize, n_imgs, _, imheight, imwidth = rgb_imgs.shape
            print(f"解包成功: batchsize={batchsize}, n_imgs={n_imgs}, imheight={imheight}, imwidth={imwidth}")
        except ValueError as e:
            print(f"解包失败: {e}")
            print(f"实际形状: {rgb_imgs.shape}")
            
    except Exception as e2:
        print(f"逐步调试失败: {e2}")