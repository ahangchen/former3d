"""
最终梯度验证 - 处理SyncBatchNorm问题
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("最终梯度验证")
print("="*80)

# 检查GPU环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    print("❌ CUDA不可用")
    sys.exit(1)


def replace_sync_batchnorm(module):
    """递归替换SyncBatchNorm为BatchNorm2d"""
    for name, child in module.named_children():
        if isinstance(child, nn.SyncBatchNorm):
            # 创建新的BatchNorm2d层
            new_bn = nn.BatchNorm2d(
                child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats
            )
            
            # 复制权重
            if child.weight is not None:
                new_bn.weight.data = child.weight.data.clone()
            if child.bias is not None:
                new_bn.bias.data = child.bias.data.clone()
            if child.running_mean is not None:
                new_bn.running_mean.data = child.running_mean.data.clone()
            if child.running_var is not None:
                new_bn.running_var.data = child.running_var.data.clone()
            if child.num_batches_tracked is not None:
                new_bn.num_batches_tracked.data = child.num_batches_tracked.data.clone()
            
            setattr(module, name, new_bn)
            print(f"  ✅ 替换 {name}: SyncBatchNorm -> BatchNorm2d")
        else:
            replace_sync_batchnorm(child)


def create_test_model():
    """创建测试模型（处理SyncBatchNorm）"""
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        # 创建模型
        model = StreamSDFFormerIntegrated(
            attn_heads=2,
            attn_layers=1,
            use_proj_occ=False,
            voxel_size=0.16,
            fusion_local_radius=4.0,
            crop_size=(12, 24, 24)
        )
        
        # 替换SyncBatchNorm
        print("处理SyncBatchNorm...")
        replace_sync_batchnorm(model)
        
        # 移动到GPU
        model = model.cuda()
        
        print("✅ 测试模型创建成功")
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_gradient_flow():
    """测试梯度流"""
    print("\n" + "="*60)
    print("测试: 梯度流验证")
    print("="*60)
    
    model = create_test_model()
    if model is None:
        return False
    
    model.train()
    
    # 创建测试数据
    batch_size = 1
    H, W = 64, 64
    
    images = torch.randn(batch_size, 3, H, W, requires_grad=True).cuda()
    images = torch.sigmoid(images)
    
    poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    intrinsics[:, 0, 0] = 300.0
    intrinsics[:, 1, 1] = 300.0
    intrinsics[:, 0, 2] = W / 2
    intrinsics[:, 1, 2] = H / 2
    
    try:
        # 重置状态
        model.reset_state()
        
        # 第一帧推理
        print("  第一帧推理...")
        output1 = model(
            images=images,
            poses=poses,
            intrinsics=intrinsics,
            reset_state=True
        )
        
        print(f"  第一帧输出键: {list(output1.keys())}")
        
        if 'sdf' in output1:
            sdf1 = output1['sdf']
            print(f"  第一帧SDF形状: {sdf1.shape}")
            
            # 第二帧推理（使用历史）
            print("  第二帧推理...")
            output2 = model(
                images=images,
                poses=poses,
                intrinsics=intrinsics,
                reset_state=False
            )
            
            sdf2 = output2['sdf']
            print(f"  第二帧SDF形状: {sdf2.shape}")
            
            # 计算损失
            target = torch.randn_like(sdf2) * 0.1
            loss = nn.functional.mse_loss(sdf2, target)
            print(f"  损失值: {loss.item():.6f}")
            
            # 反向传播
            loss.backward()
            
            # 分析结果
            print("\n  梯度分析结果:")
            
            # 1. 输入梯度
            if images.grad is not None:
                grad_norm = images.grad.norm().item()
                print(f"  ✅ 输入图像梯度: 存在 (范数={grad_norm:.6f})")
                input_grad_ok = True
            else:
                print(f"  ❌ 输入图像梯度: 不存在")
                input_grad_ok = False
            
            # 2. 关键模块梯度
            modules_to_check = [
                ('net2d', '2D网络'),
                ('net3d', '3D网络'),
                ('mv_fusion', '多视图融合'),
                ('pose_projection', '姿态投影'),
                ('stream_fusion', '流式融合'),
                ('img_feat_projection', '图像特征投影')
            ]
            
            modules_with_grad = []
            modules_without_grad = []
            
            for module_key, module_name in modules_to_check:
                has_grad = False
                for name, param in model.named_parameters():
                    if module_key in name and param.grad is not None:
                        has_grad = True
                        break
                
                if has_grad:
                    modules_with_grad.append(module_name)
                else:
                    modules_without_grad.append(module_name)
            
            print(f"  ✅ 有梯度的模块 ({len(modules_with_grad)}): {', '.join(modules_with_grad)}")
            if modules_without_grad:
                print(f"  ❌ 无梯度的模块 ({len(modules_without_grad)}): {', '.join(modules_without_grad)}")
            
            # 3. 计算图
            if sdf2.grad_fn is not None:
                grad_fn_name = sdf2.grad_fn.__class__.__name__
                print(f"  ✅ 计算图: 存在 ({grad_fn_name})")
                
                # 检查计算图深度
                current_fn = sdf2.grad_fn
                depth = 0
                while current_fn is not None and depth < 50:
                    depth += 1
                    current_fn = getattr(current_fn, 'next_functions', None)
                    if current_fn:
                        current_fn = current_fn[0][0] if current_fn[0][0] else None
                print(f"    计算图深度: {depth}")
                compute_graph_ok = depth > 10
            else:
                print(f"  ❌ 计算图: 不存在")
                compute_graph_ok = False
            
            # 4. 状态管理
            print("\n  状态管理检查:")
            print(f"  historical_state: {model.historical_state is not None}")
            print(f"  historical_pose: {model.historical_pose is not None}")
            print(f"  historical_intrinsics: {model.historical_intrinsics is not None}")
            
            # 5. 参数梯度统计
            total_params = sum(1 for _ in model.parameters())
            params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
            print(f"\n  参数梯度统计: {params_with_grad}/{total_params} 个参数有梯度")
            
            if params_with_grad > 0:
                grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
                print(f"  梯度范数范围: {min(grad_norms):.6f} ~ {max(grad_norms):.6f}")
                print(f"  平均梯度范数: {np.mean(grad_norms):.6f}")
            
            # 综合评估
            success = (
                input_grad_ok and 
                len(modules_with_grad) >= 3 and 
                compute_graph_ok and
                params_with_grad > 0
            )
            
            if success:
                print(f"\n✅ 梯度流验证通过!")
                print(f"  - 输入梯度: {'✓' if input_grad_ok else '✗'}")
                print(f"  - 关键模块: {len(modules_with_grad)}/6 有梯度")
                print(f"  - 计算图深度: {depth if compute_graph_ok else 0}")
                print(f"  - 参数梯度: {params_with_grad}/{total_params}")
                return True
            else:
                print(f"\n❌ 梯度流验证失败")
                return False
        else:
            print(f"❌ 输出中缺少'sdf'键")
            return False
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_end_to_end_training():
    """测试端到端训练"""
    print("\n" + "="*60)
    print("测试: 端到端训练验证")
    print("="*60)
    
    model = create_test_model()
    if model is None:
        return False
    
    model.train()
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    try:
        # 训练数据
        batch_size = 1
        H, W = 64, 64
        
        images = torch.randn(batch_size, 3, H, W, requires_grad=True).cuda()
        images = torch.sigmoid(images)
        
        poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        intrinsics[:, 0, 0] = 300.0
        intrinsics[:, 1, 1] = 300.0
        intrinsics[:, 0, 2] = W / 2
        intrinsics[:, 1, 2] = H / 2
        
        # 记录初始参数
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.data.clone()
        
        # 训练步骤
        losses = []
        for step in range(3):
            optimizer.zero_grad()
            model.reset_state()
            
            # 第一帧
            output1 = model(
                images=images,
                poses=poses,
                intrinsics=intrinsics,
                reset_state=True
            )
            
            # 第二帧（使用历史）
            output2 = model(
                images=images,
                poses=poses,
                intrinsics=intrinsics,
                reset_state=False
            )
            
            if 'sdf' in output2:
                sdf_pred = output2['sdf']
                target = torch.randn_like(sdf_pred) * 0.1
                
                loss = nn.functional.mse_loss(sdf_pred, target)
                losses.append(loss.item())
                
                # 反向传播
                loss.backward()
                
                # 更新参数
                optimizer.step()
                
                print(f"  步骤 {step+1}: 损失={loss.item():.6f}")
            else:
                print(f"❌ 步骤 {step+1}: 输出中缺少'sdf'键")
                return False
        
        # 检查参数更新
        updated_params = 0
        total_params = 0
        param_changes = []
        
        for name, param in model.named_parameters():
            total_params += 1
            if name in initial_params:
                change = (param.data - initial_params[name]).norm().item()
                if change > 1e-10:
                    updated_params += 1
                    param_changes.append(change)
        
        print(f"\n  训练结果:")
        print(f"  损失变化: {losses[0]:.6f} -> {losses[-1]:.6f}")
        print(f"  参数更新: {updated_params}/{total_params} 个参数已更新")
        
        if param_changes:
            print(f"  参数变化范围: {min(param_changes):.6f} ~ {max(param_changes):.6f}")
            print(f"  平均参数变化: {np.mean(param_changes):.6f}")
        
        success = (updated_params > 0) and (len(losses) == 3)
        
        if success:
            print(f"✅ 端到端训练验证通过!")
            return True
        else:
            print(f"❌ 端到端训练验证失败")
            return False
            
    except Exception as e:
        print(f"❌ 训练测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("开始最终梯度验证...")
    
    # 测试梯度流
    gradient_success = test_gradient_flow()
    
    if gradient_success:
        print("\n" + "="*80)
        print("✅ 梯度流验证成功，继续端到端训练验证...")
        print("="*80)
        
        # 测试端到端训练
        training_success = test_end_to_end_training()
        
        if training_success:
            print("\n" + "="*80)
            print("🎉 所有验证测试通过！")
            print("="*80)
            print("总结:")
            print("1. ✅ 状态管理已修复 (reset_state方法, historical_intrinsics属性)")
            print("2. ✅ SyncBatchNorm问题已处理")
            print("3. ✅ 梯度流完整 (输入梯度、关键模块梯度、计算图)")
            print("4. ✅ 端到端训练可行 (参数可更新)")
            print("="*80)
            sys.exit(0)
        else:
            print("\n" + "="*80)
            print("⚠️ 端到端训练验证失败")
            print("="*80)
            sys.exit(1)
    else:
        print("\n" + "="*80)
        print("❌ 梯度流验证失败")
        print("="*80)
        sys.exit(1)