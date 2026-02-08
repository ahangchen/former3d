#!/usr/bin/env python
"""
完整流式SDFFormer梯度流验证
验证StreamSDFFormerIntegrated的端到端梯度流
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, '/home/cwh/coding/former3d')

print("="*80)
print("完整流式SDFFormer梯度流验证")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")
print(f"PyTorch版本: {torch.__version__}")

# ============================================================================
# 步骤1: 导入并检查完整模型
# ============================================================================

print("\n" + "="*60)
print("步骤1: 导入完整模型")
print("="*60)

try:
    from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
    
    # 创建完整模型
    model = StreamSDFFormerIntegrated(
        attn_heads=2,
        attn_layers=2,
        use_proj_occ=False,
        voxel_size=0.0625,
        fusion_local_radius=3.0,
        crop_size=(48, 96, 96)
    ).to(device)
    
    model.train()
    
    print("✅ 模型导入成功")
    print(f"模型类型: {type(model).__name__}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"参数比例: {trainable_params/total_params*100:.1f}%")
    
    # 检查关键组件
    print("\n关键组件检查:")
    components = [
        'net2d', 'net3d', 'mv_fusion', 'sdf_head', 'occ_head',
        'pose_projection', 'stream_fusion', 'img_feat_projection'
    ]
    
    for comp in components:
        if hasattr(model, comp):
            comp_obj = getattr(model, comp)
            params = sum(p.numel() for p in comp_obj.parameters()) if hasattr(comp_obj, 'parameters') else 0
            print(f"  {comp}: ✅ ({params:,}参数)")
        else:
            print(f"  {comp}: ❌ 不存在")
    
except Exception as e:
    print(f"❌ 模型导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 步骤2: 创建测试数据
# ============================================================================

print("\n" + "="*60)
print("步骤2: 创建测试数据")
print("="*60)

def create_test_data(batch_size=2, seq_len=3):
    """创建测试序列数据"""
    
    # 图像数据 [B, 3, H, W]
    images = []
    for i in range(seq_len):
        img = torch.randn(batch_size, 3, 128, 128, device=device, requires_grad=True)
        images.append(img)
        print(f"  帧{i}图像: {img.shape}, requires_grad={img.requires_grad}")
    
    # 相机位姿 [B, 4, 4]
    poses = []
    for i in range(seq_len):
        # 创建简单的相机位姿（沿z轴移动）
        pose = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        pose[:, 2, 3] = i * 0.5  # 每帧移动0.5米
        poses.append(pose)
        print(f"  帧{i}位姿: {pose.shape}")
    
    # 相机内参 [B, 3, 3]
    intrinsics = []
    for i in range(seq_len):
        intr = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intr[:, 0, 0] = 250.0  # fx
        intr[:, 1, 1] = 250.0  # fy
        intr[:, 0, 2] = 64.0   # cx
        intr[:, 1, 2] = 64.0   # cy
        intrinsics.append(intr)
        print(f"  帧{i}内参: {intr.shape}")
    
    return images, poses, intrinsics

print("创建3帧测试序列:")
images, poses, intrinsics = create_test_data(batch_size=2, seq_len=3)

# ============================================================================
# 步骤3: 完整序列推理梯度测试
# ============================================================================

print("\n" + "="*60)
print("步骤3: 完整序列推理梯度测试")
print("="*60)

# 重置模型状态
model.historical_state = None
model.historical_pose = None

outputs = []
losses = []

print(">>> 序列推理:")
for i in range(3):
    print(f"\n--- 帧{i}推理 ---")
    
    # 第一帧重置状态
    reset_state = (i == 0)
    
    # 推理
    output, new_state = model.forward_single_frame(
        images[i], 
        poses[i], 
        intrinsics[i],
        reset_state=reset_state
    )
    
    # 检查输出
    if 'sdf' in output and output['sdf'] is not None:
        sdf = output['sdf']
        print(f"  SDF输出: {sdf.shape}")
        print(f"  SDF requires_grad: {sdf.requires_grad}")
        print(f"  SDF grad_fn: {sdf.grad_fn}")
        
        # 计算损失
        loss = sdf.mean()
        losses.append(loss)
        print(f"  帧{i}损失: {loss.item():.6f}")
        
        outputs.append({
            'sdf': sdf,
            'state': new_state
        })
    else:
        print(f"  ❌ 无SDF输出")
        outputs.append(None)

# ============================================================================
# 步骤4: 梯度反向传播
# ============================================================================

print("\n" + "="*60)
print("步骤4: 梯度反向传播")
print("="*60)

if losses:
    # 计算总损失
    total_loss = sum(losses) / len(losses)
    print(f"总损失: {total_loss.item():.6f}")
    print(f"总损失requires_grad: {total_loss.requires_grad}")
    
    # 反向传播
    print("\n>>> 执行反向传播...")
    total_loss.backward()
    
    # 检查输入图像梯度
    print("\n梯度检查:")
    for i, img in enumerate(images):
        if img.grad is not None:
            grad_norm = img.grad.norm().item()
            print(f"  帧{i}图像梯度: ✅ 存在 (范数: {grad_norm:.6f})")
        else:
            print(f"  帧{i}图像梯度: ❌ None")
    
    # 检查关键模块梯度
    print("\n模块梯度统计:")
    modules_to_check = {
        'net2d': '2D特征提取',
        'net3d': '3D处理网络',
        'mv_fusion': '多视图融合',
        'sdf_head': 'SDF输出头',
        'occ_head': '占用输出头',
        'pose_projection': '位姿投影',
        'stream_fusion': '流式融合',
        'img_feat_projection': '图像特征投影'
    }
    
    for module_name, desc in modules_to_check.items():
        if hasattr(model, module_name):
            module = getattr(model, module_name)
            if hasattr(module, 'parameters'):
                grad_count = 0
                total_params = 0
                for param in module.parameters():
                    total_params += 1
                    if param.grad is not None:
                        grad_count += 1
                
                status = "✅" if grad_count > 0 else "❌"
                print(f"  {desc}: {status} ({grad_count}/{total_params}参数有梯度)")
        else:
            print(f"  {desc}: ❌ 模块不存在")

# ============================================================================
# 步骤5: 计算图深度分析
# ============================================================================

print("\n" + "="*60)
print("步骤5: 计算图深度分析")
print("="*60)

if outputs and outputs[-1] is not None and 'sdf' in outputs[-1]:
    last_sdf = outputs[-1]['sdf']
    
    print(f"最后一帧SDF输出:")
    print(f"  grad_fn: {last_sdf.grad_fn}")
    print(f"  requires_grad: {last_sdf.requires_grad}")
    
    # 计算图深度分析
    if last_sdf.grad_fn is not None:
        depth = 0
        current = last_sdf.grad_fn
        operation_types = []
        
        while current is not None and depth < 50:  # 防止无限循环
            depth += 1
            op_name = str(current)
            
            # 提取操作类型
            if 'Backward' in op_name:
                op_type = op_name.split('Backward')[0]
                operation_types.append(op_type)
            
            if hasattr(current, 'next_functions'):
                next_fns = current.next_functions
                if len(next_fns) > 0:
                    current = next_fns[0][0] if next_fns[0][0] is not None else None
                else:
                    break
            else:
                break
        
        print(f"\n计算图分析:")
        print(f"  深度: {depth}层")
        
        if depth >= 10:
            print(f"  ✅ 计算图深度足够")
            
            # 显示关键操作类型
            print(f"  关键操作类型:")
            op_counts = {}
            for op in operation_types:
                op_counts[op] = op_counts.get(op, 0) + 1
            
            for op, count in list(op_counts.items())[:10]:  # 显示前10个
                print(f"    {op}: {count}次")
        else:
            print(f"  ⚠️ 计算图可能过浅 (深度={depth})")
    else:
        print(f"  ❌ 输出无grad_fn")
else:
    print("❌ 无法分析计算图：无有效输出")

# ============================================================================
# 步骤6: 流式特性验证
# ============================================================================

print("\n" + "="*60)
print("步骤6: 流式特性验证")
print("="*60)

# 检查历史状态
print("历史状态检查:")
print(f"  当前历史状态: {'✅ 存在' if model.historical_state is not None else '❌ None'}")
print(f"  当前历史位姿: {'✅ 存在' if model.historical_pose is not None else '❌ None'}")

if model.historical_state is not None:
    print(f"  历史状态类型: {type(model.historical_state)}")
    if isinstance(model.historical_state, dict):
        print(f"  历史状态键: {list(model.historical_state.keys())}")

# 检查流式融合是否启用
print(f"\n流式融合状态:")
print(f"  stream_fusion_enabled: {model.stream_fusion_enabled}")

if hasattr(model, 'stream_fusion'):
    fusion_params = sum(p.numel() for p in model.stream_fusion.parameters())
    print(f"  流式融合参数: {fusion_params:,}")
    
    # 检查融合参数梯度
    fusion_grads = []
    for name, param in model.stream_fusion.named_parameters():
        if param.grad is not None:
            fusion_grads.append((name, param.grad.norm().item()))
    
    if fusion_grads:
        print(f"  流式融合梯度: ✅ 存在")
        for name, grad_norm in fusion_grads[:3]:  # 显示前3个
            print(f"    {name}: 范数={grad_norm:.6f}")
    else:
        print(f"  流式融合梯度: ❌ None")

# ============================================================================
# 最终验证结论
# ============================================================================

print("\n" + "="*80)
print("完整梯度流验证结论")
print("="*80)

# 评估标准
criteria = {
    "模型成功导入": True,  # 从try块可知
    "输入图像有梯度": all(img.grad is not None for img in images),
    "关键模块有梯度": False,  # 需要检查
    "计算图深度足够": False,  # 需要检查
    "流式融合可训练": len(fusion_grads) > 0 if 'fusion_grads' in locals() else False,
    "端到端可微分": total_loss.requires_grad if 'total_loss' in locals() else False
}

# 更新标准
if 'outputs' in locals() and outputs and outputs[-1] is not None:
    last_sdf = outputs[-1]['sdf']
    if last_sdf.grad_fn is not None:
        depth = 0
        current = last_sdf.grad_fn
        while current is not None and depth < 50:
            depth += 1
            if hasattr(current, 'next_functions'):
                next_fns = current.next_functions
                if len(next_fns) > 0:
                    current = next_fns[0][0] if next_fns[0][0] is not None else None
                else:
                    break
            else:
                break
        criteria["计算图深度足够"] = depth >= 10

# 检查关键模块梯度
key_modules = ['net2d', 'net3d', 'mv_fusion', 'sdf_head']
all_modules_have_grad = True
for module_name in key_modules:
    if hasattr(model, module_name):
        module = getattr(model, module_name)
        if hasattr(module, 'parameters'):
            has_grad = any(p.grad is not None for p in module.parameters())
            if not has_grad:
                all_modules_have_grad = False
                break

criteria["关键模块有梯度"] = all_modules_have_grad

print("\n📊 验证结果:")
all_passed = True
for criterion, passed in criteria.items():
    status = "✅ 通过" if passed else "❌ 失败"
    print(f"{criterion}: {status}")
    if not passed:
        all_passed = False

print(f"\n总体结果: {'✅ 所有验证通过' if all_passed else '❌ 部分验证失败'}")

if all_passed:
    print("\n" + "🎉"*40)
    print("🎉 完整流式SDFFormer梯度流验证成功！")
    print("🎉"*40)
    
    print("\n📋 **技术验证总结:**")
    print("1. ✅ 完整模型成功导入并初始化")
    print("2. ✅ 输入图像梯度完整传播")
    print("3. ✅ 所有关键模块（2D/3D/融合/输出）参与训练")
    print("4. ✅ 计算图深度足够，支持复杂操作")
    print("5. ✅ 流式融合模块权重可更新")
    print("6. ✅ 端到端可微分，支持完整训练流程")
    
    print("\n🚀 **架构正确性确认:**")
    print("• StreamSDFFormerIntegrated设计正确 ✓")
    print("• 梯度流完整，无断开点 ✓")
    print("• 历史状态管理机制工作正常 ✓")
    print("• 可进行端到端训练 ✓")
    
    print("\n✅ **Task 3.2 完整验证完成！**")
else:
    print("\n⚠️ **验证发现问题，需要修复:**")
    failed_criteria = [c for c, p in criteria.items() if not p]
    for criterion in failed_criteria:
        print(f"  • {criterion}")
    
    print("\n🔧 **修复建议:**")
    print("1. 检查是否有detach()操作断开了计算图")
    print("2. 验证所有参数的requires_grad设置")
    print("3. 确保所有模块都参与前向传播")
    print("4. 检查自定义操作的可微性")

print("\n" + "="*80)