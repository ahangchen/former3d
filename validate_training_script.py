#!/usr/bin/env python3
"""
验证训练脚本基本功能
"""

import os
import sys
import torch
import argparse

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("训练脚本验证")
print("="*80)

# 使用CPU
device = torch.device("cpu")
print(f"使用设备: {device}")
print()

# 测试1: 导入必要的模块
print("1. 导入必要的模块...")
try:
    from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
    from stream_state_manager import StreamStateManager
    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 测试2: 创建最小模型
print("\n2. 创建最小模型...")
try:
    model = StreamSDFFormerIntegrated(
        attn_heads=1,
        attn_layers=1,
        use_proj_occ=False,
        voxel_size=0.08,
        fusion_local_radius=2.0,
        crop_size=(16, 16, 12)
    )
    print("✅ 模型创建成功")
except Exception as e:
    print(f"❌ 模型创建失败: {e}")
    sys.exit(1)

# 测试3: 创建状态管理器
print("\n3. 创建状态管理器...")
try:
    state_manager = StreamStateManager()
    print("✅ 状态管理器创建成功")
except Exception as e:
    print(f"❌ 状态管理器创建失败: {e}")
    sys.exit(1)

# 测试4: 测试数据提取函数
print("\n4. 测试数据提取函数...")
try:
    # 创建模拟批次数据
    batch = {
        'rgb_images': torch.randn(2, 5, 3, 64, 64),  # [batch, n_frames, 3, H, W]
        'poses': torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(2, 5, 1, 1),
        'intrinsics': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(2, 5, 1, 1),
        'tsdf': torch.randn(2, 1, 16, 16, 12),
        'sequence_id': torch.tensor([0, 1])
    }
    
    # 测试extract_frame_data函数
    from train_stream_integrated import extract_frame_data
    
    frame_data = extract_frame_data(batch, 0, device)
    
    print("✅ 数据提取函数测试成功")
    print(f"   图像形状: {frame_data['images'].shape}")
    print(f"   位姿形状: {frame_data['poses'].shape}")
    print(f"   内参形状: {frame_data['intrinsics'].shape}")
    
except Exception as e:
    print(f"❌ 数据提取函数测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: 测试损失计算函数
print("\n5. 测试损失计算函数...")
try:
    from train_stream_integrated import compute_loss
    
    # 创建模拟输出和真值
    output = {
        'sdf': torch.randn(2, 1, 12, 16, 16)  # [batch, 1, D, H, W]
    }
    ground_truth = torch.randn(2, 1, 16, 16, 12)  # [batch, 1, H, W, D]
    frame_data = {'images': torch.randn(2, 3, 64, 64)}
    
    loss = compute_loss(output, ground_truth, frame_data)
    
    print(f"✅ 损失计算函数测试成功，损失值: {loss.item():.6f}")
    
except Exception as e:
    print(f"❌ 损失计算函数测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试6: 测试参数解析
print("\n6. 测试参数解析...")
try:
    from train_stream_integrated import parse_args
    
    # 模拟命令行参数
    test_args = ['--dry-run', '--no-cuda', '--batch-size', '1', '--sequence-length', '3']
    
    # 保存原始sys.argv
    original_argv = sys.argv
    sys.argv = ['train_stream_integrated.py'] + test_args
    
    args = parse_args()
    
    # 恢复原始sys.argv
    sys.argv = original_argv
    
    print("✅ 参数解析测试成功")
    print(f"   干运行模式: {args.dry_run}")
    print(f"   禁用CUDA: {args.no_cuda}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   序列长度: {args.sequence_length}")
    
except Exception as e:
    print(f"❌ 参数解析测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("所有验证测试通过! ✅")
print("="*80)
print("\n总结:")
print("1. ✅ 所有必要的模块可以正确导入")
print("2. ✅ StreamSDFFormerIntegrated模型可以创建")
print("3. ✅ StreamStateManager可以创建")
print("4. ✅ 数据提取函数正常工作")
print("5. ✅ 损失计算函数正常工作")
print("6. ✅ 命令行参数解析正常工作")
print("\n训练脚本框架已准备就绪!")
print("\n下一步:")
print("1. 收集更多训练数据到 tartanair_sdf_output 目录")
print("2. 运行完整训练: python train_stream_integrated.py --epochs 1 --batch-size 1")
print("3. 监控训练进度: tail -f stream_training.log")