#!/usr/bin/env python3
"""
简单设备一致性测试
"""

import os
import sys
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 测试1: 张量设备一致性
print("\n=== 测试1: 张量设备一致性 ===")
tensor_cpu = torch.randn(2, 3)
tensor_gpu = torch.randn(2, 3).to(device)

print(f"CPU张量: {tensor_cpu.device}")
print(f"GPU张量: {tensor_gpu.device}")

# 测试错误情况
try:
    result = tensor_cpu + tensor_gpu
    print("❌ 错误：不同设备的张量可以相加！")
except RuntimeError as e:
    print(f"✅ 正确：捕获到设备不一致错误")

# 测试正确情况
tensor_cpu_to_gpu = tensor_cpu.to(device)
result = tensor_gpu + tensor_cpu_to_gpu
print(f"✅ 正确：相同设备的张量可以相加")

# 测试2: 模型导入和设备移动
print("\n=== 测试2: 模型导入和设备移动 ===")
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
    
    # 移动到设备
    model = model.to(device)
    print(f"✅ 模型移动到设备: {next(model.parameters()).device}")
    
    # 测试前向传播
    images = torch.randn(1, 3, 96, 96).to(device)
    poses = torch.randn(1, 4, 4).to(device)
    intrinsics = torch.randn(1, 3, 3).to(device)
    
    with torch.no_grad():
        output, state = model.forward_single_frame(images, poses, intrinsics, reset_state=True)
    
    print(f"✅ 前向传播成功")
    print(f"   输出包含: {list(output.keys())}")
    
except Exception as e:
    print(f"❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 修复non_distributed_training.py中的设备问题
print("\n=== 测试3: 修复原始训练脚本 ===")
try:
    # 读取原始脚本
    with open("bck/script/non_distributed_training.py", "r") as f:
        content = f.read()
    
    # 检查是否有设备不一致的问题
    issues = []
    
    # 检查.to(images.device)的使用
    if ".to(images.device)" in content:
        issues.append("找到 .to(images.device) - 可能有问题")
    
    # 检查设备变量使用
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if "device = torch.device" in line:
            print(f"   第{i+1}行: {line.strip()}")
    
    if issues:
        print(f"⚠️  发现潜在问题: {issues}")
    else:
        print("✅ 未发现明显的设备不一致问题")
        
except Exception as e:
    print(f"❌ 脚本检查失败: {e}")

print("\n=== 测试完成 ===")