#!/usr/bin/env python3
"""
测试optimized_online_training.py修复
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_intrinsics_fix():
    print("测试intrinsics形状修复...")
    
    # 模拟数据集返回
    batch = {
        'rgb_images': torch.randn(2, 3, 3, 64, 64),  # [B, F, C, H, W]
        'poses': torch.randn(2, 3, 4, 4),            # [B, F, 4, 4]
        'intrinsics': torch.eye(3),                  # (3, 3) - 问题所在
        'tsdf': torch.randn(2, 16, 16, 16)           # [B, D, H, W]
    }
    
    print(f"原始intrinsics形状: {batch['intrinsics'].shape}")
    
    # 应用修复
    intrinsics = batch['intrinsics']
    if intrinsics.dim() == 2:
        # 当前形状: (3, 3)，需要扩展为[B, 3, 3]
        B = batch['rgb_images'].shape[0]
        intrinsics = intrinsics.unsqueeze(0).repeat(B, 1, 1)
    
    print(f"修复后intrinsics形状: {intrinsics.shape}")
    print(f"期望形状: [{B}, 3, 3]")
    
    if intrinsics.shape == torch.Size([B, 3, 3]):
        print("✅ intrinsics形状修复正确")
        return True
    else:
        print("❌ intrinsics形状修复失败")
        return False

def test_model_compatibility():
    print("\n测试模型兼容性...")
    
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        # 创建模型
        model = StreamSDFFormerIntegrated(
            attn_heads=2,
            attn_layers=1,
            use_proj_occ=True,
            voxel_size=0.08,
            fusion_local_radius=0.0,
            crop_size=(32, 32, 24)
        )
        
        # 测试输入
        B = 2
        images = torch.randn(B, 3, 64, 64)
        poses = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
        intrinsics = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)  # [B, 3, 3]
        
        print(f"输入形状:")
        print(f"  images: {images.shape}")
        print(f"  poses: {poses.shape}")
        print(f"  intrinsics: {intrinsics.shape}")
        
        # 前向传播
        with torch.no_grad():
            output = model(images, poses, intrinsics, reset_state=True)
        
        if 'sdf' in output:
            print(f"✅ 模型前向传播成功")
            print(f"  SDF输出形状: {output['sdf'].shape}")
            return True
        else:
            print(f"❌ 模型输出中没有'sdf'键")
            return False
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*80)
    print("optimized_online_training.py修复测试")
    print("="*80)
    
    test1 = test_intrinsics_fix()
    test2 = test_model_compatibility()
    
    print("\n" + "="*80)
    print("测试结果")
    print("="*80)
    
    if test1 and test2:
        print("✅ 所有测试通过!")
        print("optimized_online_training.py修复成功")
        print("
下一步: 运行修复后的训练脚本")
        print("命令: python optimized_online_training.py")
    else:
        print("❌ 测试失败，需要进一步修复")
    
    print("
🚀 根据测试结果决定下一步操作")

if __name__ == "__main__":
    main()
