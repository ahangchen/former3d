#!/usr/bin/env python3
"""
测试设备一致性修复
"""

import os
import sys
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 测试张量设备一致性
def test_tensor_device():
    print("\n=== 测试张量设备一致性 ===")
    
    # 创建张量
    tensor_cpu = torch.randn(2, 3, 4)
    tensor_gpu = torch.randn(2, 3, 4).to(device)
    
    print(f"CPU张量设备: {tensor_cpu.device}")
    print(f"GPU张量设备: {tensor_gpu.device}")
    
    # 测试操作
    try:
        result = tensor_cpu + tensor_gpu
        print("❌ 错误：不同设备的张量可以相加！")
    except RuntimeError as e:
        print(f"✅ 正确：捕获到设备不一致错误: {e}")
    
    # 正确的方式
    tensor_cpu_to_gpu = tensor_cpu.to(device)
    result = tensor_gpu + tensor_cpu_to_gpu
    print(f"✅ 正确：相同设备的张量可以相加，结果设备: {result.device}")

# 测试模型导入
def test_model_import():
    print("\n=== 测试模型导入 ===")
    
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        print("✅ StreamSDFFormerIntegrated导入成功")
        
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
        print(f"✅ 模型创建成功，设备: {next(model.parameters()).device}")
        
        # 测试前向传播 - 使用正确的形状
        batch_size = 1
        # 根据stream_sdfformer_integrated.py，forward_single_frame期望:
        # images: [batch, 3, height, width]
        # poses: [batch, 4, 4]
        # intrinsics: [batch, 3, 3]
        images = torch.randn(batch_size, 3, 96, 96).to(device)
        poses = torch.randn(batch_size, 4, 4).to(device)
        intrinsics = torch.randn(batch_size, 3, 3).to(device)
        
        with torch.no_grad():
            output, state = model.forward_single_frame(images, poses, intrinsics, reset_state=True)
        
        print(f"✅ 前向传播成功，输出形状: {output['sdf'].shape if 'sdf' in output else 'N/A'}")
        
    except Exception as e:
        print(f"❌ 模型导入/创建失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tensor_device()
    test_model_import()
    print("\n=== 测试完成 ===")