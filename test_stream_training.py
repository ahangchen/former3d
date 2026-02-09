#!/usr/bin/env python3
"""
流式训练测试脚本
"""

import os
import sys
import torch
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("流式训练测试脚本")
print("="*80)

# 检查环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ 使用CPU")

print()

# 测试1: 导入模型
print("1. 导入StreamSDFFormerIntegrated...")
try:
    from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
    print("✅ StreamSDFFormerIntegrated导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试2: 创建模型
print("\n2. 创建模型...")
try:
    model = StreamSDFFormerIntegrated(
        attn_heads=2,
        attn_layers=1,
        use_proj_occ=True,
        voxel_size=0.04,
        fusion_local_radius=2.0,
        crop_size=(32, 32, 24)
    )
    
    model = model.to(device)
    model.eval()
    
    print("✅ 模型创建成功")
    print(f"   参数数量: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"❌ 模型创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 创建模拟数据
print("\n3. 创建模拟数据...")
try:
    batch_size = 1  # 减小批次大小
    n_frames = 3    # 减少序列长度
    height, width = 128, 128  # 减小图像尺寸
    
    # 图像: [batch, n_frames, 3, H, W]
    images = torch.randn(batch_size, n_frames, 3, height, width).to(device)
    
    # 位姿: [batch, n_frames, 4, 4]
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_frames, 1, 1).to(device)
    
    # 内参: [batch, n_frames, 3, 3]
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_frames, 1, 1).to(device)
    intrinsics[:, :, 0, 0] = 500  # fx
    intrinsics[:, :, 1, 1] = 500  # fy
    intrinsics[:, :, 0, 2] = width / 2  # cx
    intrinsics[:, :, 1, 2] = height / 2  # cy
    
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
    # 提取第一帧并调整形状
    frame_images = images[:, 0]  # [batch, 3, H, W] - 移除n_frames维度
    frame_poses = poses[:, 0]    # [batch, 4, 4] - 移除n_frames维度
    frame_intrinsics = intrinsics[:, 0]  # [batch, 3, 3] - 移除n_frames维度
    
    print(f"   单帧图像形状: {frame_images.shape}")
    print(f"   单帧位姿形状: {frame_poses.shape}")
    print(f"   单帧内参形状: {frame_intrinsics.shape}")
    
    # 前向传播（重置状态）
    output, state = model.forward_single_frame(
        images=frame_images,
        poses=frame_poses,
        intrinsics=frame_intrinsics,
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
    
except Exception as e:
    print(f"❌ 单帧处理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: 测试流式处理
print("\n5. 测试流式处理...")
try:
    state = None
    total_outputs = []
    
    for frame_idx in range(n_frames):
        # 提取当前帧并调整形状
        frame_images = images[:, frame_idx]  # [batch, 3, H, W]
        frame_poses = poses[:, frame_idx]    # [batch, 4, 4]
        frame_intrinsics = intrinsics[:, frame_idx]  # [batch, 3, 3]
        
        # 前向传播
        output, state = model.forward_single_frame(
            images=frame_images,
            poses=frame_poses,
            intrinsics=frame_intrinsics,
            reset_state=(frame_idx == 0)
        )
        
        # 收集输出
        if isinstance(output, dict):
            if 'sdf' in output:
                total_outputs.append(output['sdf'])
            else:
                total_outputs.append(list(output.values())[0])
        else:
            total_outputs.append(output)
        
        print(f"   帧 {frame_idx+1}/{n_frames}: 处理完成")
    
    print(f"✅ 流式处理成功，处理了 {len(total_outputs)} 帧")
    
except Exception as e:
    print(f"❌ 流式处理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试6: 测试训练脚本导入
print("\n6. 测试训练脚本导入...")
try:
    # 模拟命令行参数
    class Args:
        def __init__(self):
            self.attn_heads = 2
            self.attn_layers = 1
            self.voxel_size = 0.04
            self.crop_size = '32,32,24'
            self.no_cuda = False
    
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
print("所有测试通过! ✅")
print("="*80)
print("\n下一步:")
print("1. 运行训练脚本: python train_stream_integrated.py --dry-run")
print("2. 测试真实数据: python train_stream_integrated.py --test-only")
print("3. 开始训练: python train_stream_integrated.py --epochs 1 --batch-size 2")