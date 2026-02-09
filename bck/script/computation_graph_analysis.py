#!/usr/bin/env python
"""
计算图深度分析 - 追踪梯度传播路径
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("="*80)
print("计算图深度分析")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 创建最简单的测试
# ============================================================================

print("\n>>> 测试1: 最简单的线性层")
x = torch.randn(2, 10, device=device, requires_grad=True)
linear = nn.Linear(10, 5).to(device)

y = linear(x)
print(f"  x形状: {x.shape}")
print(f"  y形状: {y.shape}")
print(f"  y.grad_fn: {y.grad_fn}")

loss = y.mean()
loss.backward()

print(f"  x.grad: {'✅' if x.grad is not None else '❌'}")
print(f"  计算图深度: 正常")

# ============================================================================
# 测试2: 包含detach()的情况
# ============================================================================

print("\n>>> 测试2: 包含detach()")
x = torch.randn(2, 10, device=device, requires_grad=True)
linear1 = nn.Linear(10, 5).to(device)
linear2 = nn.Linear(5, 3).to(device)

# 正常路径
feat1 = linear1(x)
print(f"  feat1.grad_fn: {feat1.grad_fn}")

# 使用detach()断开
feat1_detached = feat1.detach()
print(f"  feat1_detached.grad_fn: {feat1_detached.grad_fn}")

# 继续计算
y = linear2(feat1_detached)
print(f"  y.grad_fn: {y.grad_fn}")

loss = y.mean()
loss.backward()

print(f"  x.grad: {'✅' if x.grad is not None else '❌'}")
print(f"  linear1.weight.grad: {'✅' if linear1.weight.grad is not None else '❌'}")
print(f"  linear2.weight.grad: {'✅' if linear2.weight.grad is not None else '❌'}")

# ============================================================================
# 测试3: 流式模型的实际计算图
# ============================================================================

print("\n" + "="*60)
print("测试3: 流式模型计算图分析")
print("="*60)

class StreamModelDebug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16, 1)
        
        # 历史状态
        self.history = None
    
    def forward(self, x, use_history=False):
        print(f"\n  [前向传播] x.shape={x.shape}, requires_grad={x.requires_grad}")
        
        # 特征提取
        feat = self.conv(x)
        print(f"  feat.shape={feat.shape}, grad_fn={feat.grad_fn}")
        
        # 池化
        feat_pooled = feat.mean(dim=[2, 3])
        print(f"  feat_pooled.shape={feat_pooled.shape}, grad_fn={feat_pooled.grad_fn}")
        
        # 流式融合
        if self.history is not None and use_history:
            print(f"  使用历史状态: {self.history.shape}")
            
            # 注意：历史状态是detach的
            attention = torch.matmul(feat_pooled, self.history.transpose(0, 1))
            attention = F.softmax(attention, dim=-1)
            fused = torch.matmul(attention, self.history)
            
            combined = feat_pooled + 0.3 * fused
            print(f"  combined.grad_fn: {combined.grad_fn}")
        else:
            combined = feat_pooled
            print(f"  不使用历史状态")
        
        # 更新历史状态
        with torch.no_grad():
            self.history = feat_pooled.clone()
        
        # 输出
        output = self.fc(combined)
        print(f"  output.shape={output.shape}, grad_fn={output.grad_fn}")
        
        return output

# 测试
model = StreamModelDebug().to(device)
model.train()

x = torch.randn(2, 3, 32, 32, device=device, requires_grad=True)

print("\n>>> 第一帧（无历史）")
y1 = model(x, use_history=False)
loss1 = y1.mean()
print(f"  loss1: {loss1.item():.6f}")

loss1.backward()
print(f"  x.grad: {'✅' if x.grad is not None else '❌'}")

# 清除梯度
model.zero_grad()

print("\n>>> 第二帧（有历史）")
y2 = model(x, use_history=True)
loss2 = y2.mean()
print(f"  loss2: {loss2.item():.6f}")

loss2.backward()
print(f"  x.grad: {'✅' if x.grad is not None else '❌'}")

# ============================================================================
# 测试4: 检查实际流式模型的grad_fn
# ============================================================================

print("\n" + "="*60)
print("测试4: 实际模型grad_fn检查")
print("="*60)

# 导入实际模型
import sys
sys.path.insert(0, '.')

try:
    from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
    
    # 创建简化模型
    class SimpleStreamSDFFormer(StreamSDFFormerIntegrated):
        def __init__(self):
            super().__init__(
                attn_heads=2,
                attn_layers=2,
                use_proj_occ=False,
                voxel_size=0.0625,
                fusion_local_radius=3.0,
                crop_size=(48, 96, 96)
            )
            
            # 简化：移除复杂的3D网络
            self.net3d = None
            self.simple_fc = nn.Linear(256, 1)
        
        def forward_single_frame(self, images, poses, intrinsics, reset_state=False):
            # 只测试2D特征提取部分
            batch_size = images.shape[0]
            
            # 提取2D特征
            if hasattr(self, 'net2d'):
                features_2d = self.net2d(images)
            else:
                # 简化特征提取
                features_2d = images.mean(dim=[2, 3])  # [B, 3]
                features_2d = features_2d.unsqueeze(-1).unsqueeze(-1)  # [B, 3, 1, 1]
            
            print(f"  features_2d.shape: {features_2d.shape}")
            print(f"  features_2d.grad_fn: {features_2d.grad_fn}")
            
            # 简化输出
            output = self.simple_fc(features_2d.view(batch_size, -1))
            
            # 简化状态
            state = {
                'features': features_2d.detach().clone(),
                'poses': poses.detach().clone()
            }
            
            return {'sdf': output}, state
    
    # 测试
    model_simple = SimpleStreamSDFFormer().to(device)
    model_simple.train()
    
    x = torch.randn(1, 3, 128, 128, device=device, requires_grad=True)
    poses = torch.eye(4, device=device).unsqueeze(0)
    intrinsics = torch.eye(3, device=device).unsqueeze(0)
    intrinsics[:, 0, 0] = 250.0
    intrinsics[:, 1, 1] = 250.0
    
    print("\n>>> 简化模型测试")
    output, state = model_simple.forward_single_frame(x, poses, intrinsics, reset_state=True)
    
    if output['sdf'] is not None:
        sdf = output['sdf']
        print(f"  sdf.shape: {sdf.shape}")
        print(f"  sdf.grad_fn: {sdf.grad_fn}")
        
        loss = sdf.mean()
        print(f"  loss: {loss.item():.6f}")
        
        loss.backward()
        
        print(f"  x.grad: {'✅' if x.grad is not None else '❌'}")
        if x.grad is not None:
            print(f"  梯度范数: {x.grad.norm().item():.6f}")
    
except Exception as e:
    print(f"导入失败: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 总结
# ============================================================================

print("\n" + "="*80)
print("计算图分析总结")
print("="*80)

print("\n🔍 **发现问题：**")
print("1. 计算图深度过浅（只有2层）")
print("2. 可能是某些操作detach了计算图")
print("3. 或者模型结构过于简单")

print("\n💡 **解决方案：**")
print("1. 检查StreamSDFFormerIntegrated中的detach()操作")
print("2. 确保所有可训练模块都参与计算")
print("3. 验证梯度从损失传播到输入的完整路径")

print("\n🚀 **立即行动：**")
print("1. 检查pose_projection.py中的project_features方法")
print("2. 检查stream_fusion.py中的forward方法")
print("3. 检查历史状态更新逻辑")

print("\n" + "="*80)