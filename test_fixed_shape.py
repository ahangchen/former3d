#!/usr/bin/env python3
"""
修复形状问题后的测试
"""

import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("修复形状问题后的测试")
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

# 设置模型为评估模式
model.eval()

print("\n1. 测试正确的输入形状...")

# 创建正确的测试数据
# convert_to_sdfformer_batch期望: [batch, 3, height, width]
batch_size = 1
height, width = 256, 256

# 使用CPU进行测试
device = torch.device('cpu')
model = model.to(device)

images = torch.randn(batch_size, 3, height, width, device=device)  # 注意：没有n_frames维度！
poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, 4, 4]
intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # [batch, 3, 3]
intrinsics[:, 0, 0] = 500  # fx
intrinsics[:, 1, 1] = 500  # fy
intrinsics[:, 0, 2] = width / 2  # cx
intrinsics[:, 1, 2] = height / 2  # cy

print(f"输入形状:")
print(f"  images: {images.shape} (应该是 [batch, 3, H, W])")
print(f"  poses: {poses.shape} (应该是 [batch, 4, 4])")
print(f"  intrinsics: {intrinsics.shape} (应该是 [batch, 3, 3])")

# 测试forward_single_frame
print("\n2. 测试forward_single_frame...")
try:
    with torch.no_grad():
        output, state = model.forward_single_frame(
            images=images,
            poses=poses,
            intrinsics=intrinsics,
            reset_state=True
        )
    
    print(f"✅ forward_single_frame成功!")
    print(f"输出类型: {type(output)}")
    print(f"状态类型: {type(state)}")
    
    if isinstance(output, dict):
        print(f"输出字典键: {list(output.keys())}")
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: {type(value)}, 长度: {len(value)}")
    
    if isinstance(state, dict):
        print(f"状态字典键: {list(state.keys())}")
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
                
except Exception as e:
    print(f"❌ forward_single_frame失败: {e}")
    import traceback
    traceback.print_exc()

# 测试训练模式
print("\n3. 测试训练模式...")
try:
    model.train()
    
    # 前向传播（启用梯度）
    output, state = model.forward_single_frame(
        images=images,
        poses=poses,
        intrinsics=intrinsics,
        reset_state=True
    )
    
    # 创建模拟ground truth
    if isinstance(output, dict) and 'sdf' in output:
        sdf_pred = output['sdf']
        tsdf_gt = torch.randn_like(sdf_pred)
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(sdf_pred, tsdf_gt)
        loss.backward()
        
        # 检查梯度
        has_gradients = False
        grad_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_gradients = True
                grad_norm += param.grad.norm().item()
        
        print(f"✅ 训练模式测试通过!")
        print(f"  损失值: {loss.item():.6f}")
        print(f"  梯度存在: {has_gradients}")
        if has_gradients:
            print(f"  梯度范数: {grad_norm:.6f}")
    else:
        print("⚠️ 输出中没有'sdf'键")
        if isinstance(output, dict):
            print(f"  可用键: {list(output.keys())}")
            
except Exception as e:
    print(f"❌ 训练模式测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("测试完成!")
print("="*80)

print("\n🎯 关键发现:")
print("1. StreamSDFFormerIntegrated期望单帧输入，形状为 [batch, 3, H, W]")
print("2. 不是 [batch, n_frames, 3, H, W]")
print("3. 需要修改数据集或训练脚本来提供正确的输入形状")

print("\n🚀 下一步:")
print("1. 修改MultiSequenceTartanAirDataset以提供单帧数据")
print("2. 或者修改训练脚本从多帧中选择一帧")
print("3. 创建流式训练循环")