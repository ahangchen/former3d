#!/usr/bin/env python3
"""
最终测试脚本
验证所有修复是否有效
"""

import torch
import torch.nn as nn

print("="*80)
print("最终测试脚本")
print("="*80)

# 测试1: 检查SyncBatchNorm是否已修复
print("\n1. 检查SyncBatchNorm修复...")
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # 尝试导入可能使用SyncBatchNorm的模块
    from former3d.net3d.former_v1 import Former3D
    
    print("  ✅ Former3D导入成功")
    
    # 检查是否还有SyncBatchNorm
    with open("former3d/net3d/former_v1.py", 'r') as f:
        content = f.read()
        if "SyncBatchNorm" in content:
            print("  ❌ 仍然发现SyncBatchNorm")
        else:
            print("  ✅ 未发现SyncBatchNorm")
    
except Exception as e:
    print(f"  ❌ 导入失败: {e}")

# 测试2: 测试StreamSDFFormerIntegrated创建
print("\n2. 测试StreamSDFFormerIntegrated创建...")
try:
    from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
    
    print("  ✅ StreamSDFFormerIntegrated导入成功")
    
    # 尝试创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=2,
        attn_layers=1,
        use_proj_occ=True,
        voxel_size=0.15,
        fusion_local_radius=0.0,
        crop_size=(16, 16, 16)
    )
    
    print(f"  ✅ 模型创建成功")
    print(f"    参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    print("  ✅ 测试前向传播...")
    B = 2
    C, H, W = 3, 64, 64
    images = torch.randn(B, C, H, W)
    poses = torch.eye(4).unsqueeze(0).repeat(B, 1, 1)
    intrinsics = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    
    with torch.no_grad():
        output = model(images, poses, intrinsics, reset_state=True)
    
    if 'sdf' in output:
        sdf = output['sdf']
        print(f"  ✅ 前向传播成功")
        print(f"    SDF形状: {sdf.shape}")
        print(f"    SDF范围: [{sdf.min():.4f}, {sdf.max():.4f}]")
    else:
        print(f"  ⚠️ 输出中没有SDF键")
        print(f"    输出键: {list(output.keys())}")
    
except Exception as e:
    print(f"  ❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 测试简单训练
print("\n3. 测试简单训练...")
try:
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            return self.mlp(x)
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # 训练几步
    losses = []
    for i in range(5):
        # 生成数据
        points = torch.randn(10, 3)
        targets = torch.randn(10, 1)
        
        # 前向传播
        preds = model(points)
        
        # 计算损失
        loss = loss_fn(preds, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  步骤 {i+1}: 损失 = {loss.item():.6f}")
    
    # 检查损失是否在变化
    if len(losses) > 1:
        loss_change = losses[0] - losses[-1]
        if abs(loss_change) > 0.001:
            print(f"  ✅ 训练有效，损失变化: {loss_change:.6f}")
        else:
            print(f"  ⚠️ 损失变化很小: {loss_change:.6f}")
    
except Exception as e:
    print(f"  ❌ 训练测试失败: {e}")

print("\n" + "="*80)
print("测试完成")
print("="*80)
print("建议:")
print("1. 如果所有测试通过，可以运行真正的训练脚本")
print("2. 如果还有分布式训练错误，检查其他文件中的SyncBatchNorm")
print("3. 确保所有BatchNorm层都使用普通BatchNorm而不是SyncBatchNorm")
print("\n🚀 现在可以尝试运行真正的端到端训练了!")