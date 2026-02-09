#!/usr/bin/env python3
"""
基础流式测试脚本 - 只测试基本功能
"""

import os
import sys
import torch
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("基础流式测试脚本")
print("="*80)

# 使用CPU以避免GPU内存问题
device = torch.device("cpu")
print(f"使用设备: {device}")
print(f"PyTorch版本: {torch.__version__}")
print()

# 测试1: 导入模型
print("1. 导入StreamSDFFormerIntegrated...")
try:
    from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
    print("✅ StreamSDFFormerIntegrated导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试2: 创建模型（使用最小配置）
print("\n2. 创建模型（最小配置）...")
try:
    model = StreamSDFFormerIntegrated(
        attn_heads=1,           # 最小注意力头
        attn_layers=1,          # 最小注意力层
        use_proj_occ=False,     # 禁用投影占用以节省内存
        voxel_size=0.08,        # 更大的体素大小
        fusion_local_radius=2.0,
        crop_size=(16, 16, 12)  # 更小的裁剪尺寸
    )
    
    model = model.to(device)
    model.eval()
    
    print("✅ 模型创建成功")
    print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   注意力头数: 1")
    print(f"   注意力层数: 1")
    print(f"   体素大小: 0.08")
    print(f"   裁剪尺寸: (16, 16, 12)")
    
except Exception as e:
    print(f"❌ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 创建最小模拟数据
print("\n3. 创建最小模拟数据...")
try:
    batch_size = 1
    height, width = 64, 64  # 最小图像尺寸
    
    # 图像: [batch, 3, H, W]
    images = torch.randn(batch_size, 3, height, width).to(device)
    
    # 位姿: [batch, 4, 4]
    poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    
    # 内参: [batch, 3, 3]
    intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    intrinsics[:, 0, 0] = 500  # fx
    intrinsics[:, 1, 1] = 500  # fy
    intrinsics[:, 0, 2] = width / 2  # cx
    intrinsics[:, 1, 2] = height / 2  # cy
    
    print(f"✅ 模拟数据创建成功")
    print(f"   图像形状: {images.shape}")
    print(f"   位姿形状: {poses.shape}")
    print(f"   内参形状: {intrinsics.shape}")
    
except Exception as e:
    print(f"❌ 模拟数据创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 测试单帧处理
print("\n4. 测试单帧处理...")
try:
    # 前向传播（重置状态）
    print("   执行前向传播...")
    output, state = model.forward_single_frame(
        images=images,
        poses=poses,
        intrinsics=intrinsics,
        reset_state=True
    )
    
    print("✅ 单帧处理成功")
    if isinstance(output, dict):
        print(f"   输出字典键: {list(output.keys())}")
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                print(f"     {k}: {v.shape}")
    else:
        print(f"   输出形状: {output.shape}")
    
    if state is not None:
        print(f"   状态类型: {type(state)}")
        if isinstance(state, dict):
            print(f"   状态键: {list(state.keys())}")
    
except Exception as e:
    print(f"❌ 单帧处理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: 测试训练脚本导入
print("\n5. 测试训练脚本导入...")
try:
    # 模拟命令行参数
    class Args:
        def __init__(self):
            self.attn_heads = 1
            self.attn_layers = 1
            self.voxel_size = 0.08
            self.crop_size = '16,16,12'
            self.no_cuda = True
    
    args = Args()
    
    # 测试模型创建函数
    def test_create_model():
        from train_stream_integrated import create_model
        model = create_model(args, device)
        return model
    
    model2 = test_create_model()
    print("✅ 训练脚本导入成功")
    
except Exception as e:
    print(f"❌ 训练脚本导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("所有基础测试通过! ✅")
print("="*80)
print("\n总结:")
print("1. ✅ StreamSDFFormerIntegrated可以正确导入")
print("2. ✅ 模型可以使用最小配置创建")
print("3. ✅ 单帧处理功能正常")
print("4. ✅ 训练脚本框架可以导入")
print("\n下一步:")
print("1. 运行训练脚本干运行: python train_stream_integrated.py --dry-run --no-cuda")
print("2. 测试真实数据: python train_stream_integrated.py --test-only --no-cuda")
print("3. 在GPU上测试（需要更多内存）: python train_stream_integrated.py --dry-run")