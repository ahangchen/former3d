#!/usr/bin/env python3
"""
简化版模型导入测试 - 专注于基本功能
"""

import os
import sys
import torch

# 禁用CUDA警告
import warnings
warnings.filterwarnings("ignore")

print("="*80)
print("简化版StreamSDFFormerIntegrated模型导入测试")
print("="*80)

# 强制使用CPU
device = torch.device("cpu")
print(f"强制使用设备: {device}")

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, '..'))

print(f"项目根目录: {project_root}")
print(f"Python路径: {sys.path[:3]}")

# 测试1: 导入模型
print("\n1. 导入StreamSDFFormerIntegrated...")
try:
    # 尝试直接导入
    from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
    print("✅ StreamSDFFormerIntegrated导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("尝试从源码导入...")
    
    # 尝试查找模块
    module_path = os.path.join(project_root, '..', 'former3d', 'stream_sdfformer_integrated.py')
    if os.path.exists(module_path):
        print(f"找到模块文件: {module_path}")
        
        # 手动添加路径
        sys.path.insert(0, os.path.join(project_root, '..', 'former3d'))
        try:
            from stream_sdfformer_integrated import StreamSDFFormerIntegrated
            print("✅ StreamSDFFormerIntegrated导入成功（从源码）")
        except ImportError as e2:
            print(f"❌ 源码导入失败: {e2}")
            sys.exit(1)
    else:
        print(f"❌ 找不到模块文件: {module_path}")
        sys.exit(1)

# 测试2: 创建模型实例
print("\n2. 创建模型实例...")
try:
    # 使用最小配置
    model = StreamSDFFormerIntegrated(
        attn_heads=1,           # 最小注意力头
        attn_layers=1,          # 最小注意力层
        use_proj_occ=False,     # 禁用占用预测
        voxel_size=0.04,
        fusion_local_radius=1.0,
        crop_size=(16, 16, 16)  # 最小裁剪尺寸
    )
    
    print("✅ 模型实例创建成功")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
except Exception as e:
    print(f"❌ 模型实例创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 移动到设备
print("\n3. 移动模型到设备...")
try:
    model = model.to(device)
    print(f"✅ 模型已移动到设备: {device}")
    
except Exception as e:
    print(f"❌ 模型移动失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 创建模拟输入数据
print("\n4. 创建模拟输入数据...")
try:
    batch_size = 1  # 最小批量
    height, width = 128, 128  # 较小尺寸
    
    # 图像: [batch, 1, 3, H, W] - 注意是单帧
    images = torch.randn(batch_size, 1, 3, height, width)
    
    # 位姿: [batch, 1, 4, 4]
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
    # 内参: [batch, 1, 3, 3]
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    intrinsics[:, :, 0, 0] = 250  # fx
    intrinsics[:, :, 1, 1] = 250  # fy
    intrinsics[:, :, 0, 2] = width / 2  # cx
    intrinsics[:, :, 1, 2] = height / 2  # cy
    
    print("✅ 模拟输入数据创建成功")
    print(f"   图像形状: {images.shape}")
    print(f"   位姿形状: {poses.shape}")
    print(f"   内参形状: {intrinsics.shape}")
    
except Exception as e:
    print(f"❌ 模拟输入数据创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试5: 测试前向传播
print("\n5. 测试前向传播...")
try:
    model.eval()  # 设置为评估模式
    
    with torch.no_grad():
        print("   调用forward_single_frame...")
        output, state = model.forward_single_frame(
            images=images,
            poses=poses,
            intrinsics=intrinsics,
            reset_state=True
        )
    
    print("✅ 前向传播成功")
    print(f"   输出类型: {type(output)}")
    print(f"   状态类型: {type(state)}")
    
    # 检查输出格式
    if isinstance(output, dict):
        print(f"   输出字典键: {list(output.keys())}")
        for key in list(output.keys())[:3]:  # 只显示前3个键
            value = output[key]
            if isinstance(value, torch.Tensor):
                print(f"     {key}: {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"     {key}: {type(value)}, 长度: {len(value)}")
    else:
        print(f"   输出: {type(output)}")
    
    # 检查状态格式
    if state is not None:
        if isinstance(state, dict):
            print(f"   状态字典键: {list(state.keys())[:3]}")  # 只显示前3个键
        else:
            print(f"   状态类型: {type(state)}")
    
except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("简化版模型导入测试完成!")
print("="*80)

# 总结
print("\n📊 测试总结:")
print(f"1. 模型导入: ✅ 成功")
print(f"2. 模型创建: ✅ 成功")
print(f"3. 设备移动: ✅ 成功")
print(f"4. 输入数据: ✅ 成功")
print(f"5. 前向传播: ✅ 成功")

print("\n🎯 关键发现:")
print("1. StreamSDFFormerIntegrated可以在CPU上运行")
print("2. 模型支持forward_single_frame方法")
print("3. 输出是字典格式，包含多个预测结果")

print("\n🚀 下一步:")
print("1. 基于此模型创建训练脚本")
print("2. 实现数据适配器")
print("3. 创建完整的训练循环")