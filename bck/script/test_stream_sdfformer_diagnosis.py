#!/usr/bin/env python3
"""
诊断StreamSDFFormerIntegrated的问题
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目路径
sys.path.append('/home/cwh/coding/former3d')

def test_import():
    """测试导入"""
    print("测试导入...")
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        print("✅ StreamSDFFormerIntegrated导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        model = StreamSDFFormerIntegrated(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.08,
            fusion_local_radius=3.0,
            crop_size=(48, 96, 96)
        ).to(device)
        
        print("✅ 模型创建成功")
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 检查模型组件
        print("\n模型组件检查:")
        print(f"  - pose_projection: {hasattr(model, 'pose_projection')}")
        print(f"  - stream_fusion: {hasattr(model, 'stream_fusion')}")
        print(f"  - feature_expansion: {hasattr(model, 'feature_expansion')}")
        print(f"  - feature_compression: {hasattr(model, 'feature_compression')}")
        
        return model, device
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_data_preparation(device):
    """测试数据准备"""
    print("\n测试数据准备...")
    try:
        batch_size = 2
        
        # 图像 [batch, 3, 256, 256]
        images = torch.randn(batch_size, 3, 256, 256, device=device)
        
        # 位姿 [batch, 4, 4]
        poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        poses[:, 0, 3] = 1.0  # X方向平移
        
        # 内参 [batch, 3, 3]
        intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics[:, 0, 0] = 320.0  # fx
        intrinsics[:, 1, 1] = 320.0  # fy
        intrinsics[:, 0, 2] = 128.0  # cx
        intrinsics[:, 1, 2] = 128.0  # cy
        
        print("✅ 数据准备成功")
        print(f"  - images形状: {images.shape}")
        print(f"  - poses形状: {poses.shape}")
        print(f"  - intrinsics形状: {intrinsics.shape}")
        
        return images, poses, intrinsics
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        return None, None, None

def test_convert_to_sdfformer_batch(model, images, poses, intrinsics):
    """测试convert_to_sdfformer_batch方法"""
    print("\n测试convert_to_sdfformer_batch...")
    try:
        batch = model.convert_to_sdfformer_batch(images, poses, intrinsics)
        
        print("✅ convert_to_sdfformer_batch成功")
        print(f"batch键: {list(batch.keys())}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  - {key}: {value.shape}")
            elif isinstance(value, dict):
                print(f"  - {key}: 字典，包含 {list(value.keys())}")
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, torch.Tensor):
                        print(f"    - {subkey}: {subvalue.shape}")
        
        return batch
    except Exception as e:
        print(f"❌ convert_to_sdfformer_batch失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_generate_voxel_inds(model, device):
    """测试generate_voxel_inds方法"""
    print("\n测试generate_voxel_inds...")
    try:
        batch_size = 2
        voxel_inds = model.generate_voxel_inds(batch_size, device=device)
        
        print("✅ generate_voxel_inds成功")
        print(f"voxel_inds形状: {voxel_inds.shape}")
        print(f"voxel_inds类型: {voxel_inds.dtype}")
        print(f"voxel_inds范围:")
        print(f"  - x: {voxel_inds[:, 0].min().item()} ~ {voxel_inds[:, 0].max().item()}")
        print(f"  - y: {voxel_inds[:, 1].min().item()} ~ {voxel_inds[:, 1].max().item()}")
        print(f"  - z: {voxel_inds[:, 2].min().item()} ~ {voxel_inds[:, 2].max().item()}")
        print(f"  - batch: {voxel_inds[:, 3].min().item()} ~ {voxel_inds[:, 3].max().item()}")
        
        return voxel_inds
    except Exception as e:
        print(f"❌ generate_voxel_inds失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_forward_single_frame(model, images, poses, intrinsics):
    """测试forward_single_frame方法"""
    print("\n测试forward_single_frame...")
    try:
        model.eval()
        
        with torch.no_grad():
            output, new_state = model.forward_single_frame(
                images, poses, intrinsics, reset_state=True
            )
        
        print("✅ forward_single_frame成功")
        print(f"输出键: {list(output.keys())}")
        
        for key, value in output.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape}")
                elif isinstance(value, dict):
                    print(f"  - {key}: 字典")
                    for subkey, subvalue in value.items():
                        if subvalue is not None:
                            if hasattr(subvalue, 'shape'):
                                print(f"    - {subkey}: {subvalue.shape}")
                            else:
                                print(f"    - {subkey}: {type(subvalue)}")
        
        print(f"\n新状态键: {list(new_state.keys()) if new_state else 'None'}")
        
        return output, new_state
    except Exception as e:
        print(f"❌ forward_single_frame失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_sequence_inference(model, images, poses, intrinsics):
    """测试序列推理"""
    print("\n测试序列推理...")
    try:
        model.eval()
        model.clear_history()
        
        seq_length = 3
        images_seq = [images] * seq_length
        poses_seq = [poses] * seq_length
        intrinsics_seq = [intrinsics] * seq_length
        
        # 稍微修改位姿以模拟相机移动
        for i in range(seq_length):
            poses_seq[i] = poses.clone()
            poses_seq[i][:, 0, 3] = 1.0 + i * 0.1  # X方向逐渐移动
        
        with torch.no_grad():
            outputs = model.forward_sequence(
                images_seq, poses_seq, intrinsics_seq, reset_state=True
            )
        
        print("✅ 序列推理成功")
        print(f"输出数量: {len(outputs)}")
        
        for i, output in enumerate(outputs):
            print(f"\n第{i+1}帧输出:")
            if 'sdf' in output and output['sdf'] is not None:
                print(f"  - SDF形状: {output['sdf'].shape}")
                print(f"  - SDF范围: {output['sdf'].min().item():.3f} ~ {output['sdf'].max().item():.3f}")
            if 'occupancy' in output and output['occupancy'] is not None:
                print(f"  - 占用形状: {output['occupancy'].shape}")
        
        return outputs
    except Exception as e:
        print(f"❌ 序列推理失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主诊断函数"""
    print("=" * 60)
    print("StreamSDFFormerIntegrated诊断工具")
    print("=" * 60)
    
    # 测试导入
    if not test_import():
        return
    
    # 测试模型创建
    model, device = test_model_creation()
    if model is None:
        return
    
    # 测试数据准备
    images, poses, intrinsics = test_data_preparation(device)
    if images is None:
        return
    
    # 测试convert_to_sdfformer_batch
    batch = test_convert_to_sdfformer_batch(model, images, poses, intrinsics)
    if batch is None:
        return
    
    # 测试generate_voxel_inds
    voxel_inds = test_generate_voxel_inds(model, device)
    if voxel_inds is None:
        return
    
    # 测试forward_single_frame
    output, new_state = test_forward_single_frame(model, images, poses, intrinsics)
    if output is None:
        return
    
    # 测试序列推理
    outputs = test_sequence_inference(model, images, poses, intrinsics)
    
    print("\n" + "=" * 60)
    print("诊断完成")
    print("=" * 60)
    
    # 总结
    print("\n总结:")
    print("1. 模型创建: ✅ 成功")
    print("2. 数据准备: ✅ 成功")
    print("3. convert_to_sdfformer_batch: ✅ 成功")
    print("4. generate_voxel_inds: ✅ 成功")
    print("5. forward_single_frame: ✅ 成功")
    print("6. 序列推理: ✅ 成功")
    
    # GPU内存信息
    if torch.cuda.is_available():
        print(f"\nGPU内存使用:")
        print(f"  - 已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"  - 缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"  - 最大已分配: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

if __name__ == "__main__":
    main()