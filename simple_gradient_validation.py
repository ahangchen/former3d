"""
简化版梯度验证 - 专注于核心梯度流
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("简化版梯度验证")
print("="*80)

# 检查GPU环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

if not torch.cuda.is_available():
    print("❌ CUDA不可用")
    sys.exit(1)


def create_simple_model():
    """创建简化模型"""
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        # 创建最小化模型
        model = StreamSDFFormerIntegrated(
            attn_heads=2,
            attn_layers=1,
            use_proj_occ=False,
            voxel_size=0.16,  # 增大体素减少计算
            fusion_local_radius=4.0,
            crop_size=(12, 24, 24)  # 大幅减小裁剪空间
        )
        
        # 移动到GPU
        model = model.cuda()
        
        print("✅ 简化模型创建成功")
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_basic_gradient():
    """测试基础梯度流"""
    print("\n" + "="*60)
    print("测试: 基础梯度流验证")
    print("="*60)
    
    model = create_simple_model()
    if model is None:
        return False
    
    model.train()
    
    # 创建最小化测试数据
    batch_size = 1
    H, W = 32, 32  # 更小的图像
    
    images = torch.randn(batch_size, 3, H, W, requires_grad=True).cuda()
    images = torch.sigmoid(images)
    
    poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    intrinsics[:, 0, 0] = 150.0  # 更小的焦距
    intrinsics[:, 1, 1] = 150.0
    intrinsics[:, 0, 2] = W / 2
    intrinsics[:, 1, 2] = H / 2
    
    # 重置状态
    model.reset_state()
    
    try:
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
            
            # 分析梯度
            print("\n  梯度分析:")
            
            # 检查输入梯度
            if images.grad is not None:
                grad_norm = images.grad.norm().item()
                print(f"  ✅ 输入图像梯度: 存在 (范数={grad_norm:.6f})")
            else:
                print(f"  ❌ 输入图像梯度: 不存在")
            
            # 检查关键模块梯度
            modules_to_check = [
                ('net2d', '2D网络'),
                ('net3d', '3D网络'),
                ('mv_fusion', '多视图融合'),
                ('pose_projection', '姿态投影'),
                ('stream_fusion', '流式融合')
            ]
            
            modules_with_grad = 0
            for module_key, module_name in modules_to_check:
                has_grad = False
                for name, param in model.named_parameters():
                    if module_key in name and param.grad is not None:
                        has_grad = True
                        break
                
                if has_grad:
                    modules_with_grad += 1
                    print(f"  ✅ {module_name}: 有梯度")
                else:
                    print(f"  ❌ {module_name}: 无梯度")
            
            # 检查计算图
            if sdf2.grad_fn is not None:
                grad_fn_name = sdf2.grad_fn.__class__.__name__
                print(f"  ✅ 计算图: 存在 ({grad_fn_name})")
                
                # 检查计算图深度
                current_fn = sdf2.grad_fn
                depth = 0
                while current_fn is not None and depth < 30:
                    depth += 1
                    current_fn = getattr(current_fn, 'next_functions', None)
                    if current_fn:
                        current_fn = current_fn[0][0] if current_fn[0][0] else None
                print(f"    计算图深度: {depth}")
            else:
                print(f"  ❌ 计算图: 不存在")
            
            # 检查状态管理
            print("\n  状态管理检查:")
            print(f"  historical_state: {model.historical_state is not None}")
            print(f"  historical_pose: {model.historical_pose is not None}")
            print(f"  historical_intrinsics: {model.historical_intrinsics is not None}")
            
            success = (images.grad is not None) and (modules_with_grad >= 3)
            if success:
                print(f"\n✅ 基础梯度验证通过: {modules_with_grad}/5 个模块有梯度")
                return True
            else:
                print(f"\n❌ 基础梯度验证失败")
                return False
        else:
            print(f"❌ 输出中缺少'sdf'键")
            return False
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """测试单步训练"""
    print("\n" + "="*60)
    print("测试: 单步训练验证")
    print("="*60)
    
    model = create_simple_model()
    if model is None:
        return False
    
    model.train()
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    try:
        # 训练数据
        batch_size = 1
        H, W = 32, 32
        
        images = torch.randn(batch_size, 3, H, W, requires_grad=True).cuda()
        images = torch.sigmoid(images)
        
        poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        intrinsics[:, 0, 0] = 150.0
        intrinsics[:, 1, 1] = 150.0
        intrinsics[:, 0, 2] = W / 2
        intrinsics[:, 1, 2] = H / 2
        
        # 训练步骤
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
            print(f"  损失值: {loss.item():.6f}")
            
            # 反向传播
            loss.backward()
            
            # 检查梯度
            total_grad_norm = 0
            param_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item()
                    param_count += 1
            
            if param_count > 0:
                avg_grad_norm = total_grad_norm / param_count
                print(f"  平均梯度范数: {avg_grad_norm:.6f}")
                print(f"  有梯度的参数: {param_count}/{sum(1 for _ in model.parameters())}")
            else:
                print(f"  ❌ 没有参数有梯度")
                return False
            
            # 更新参数
            optimizer.step()
            
            # 检查参数是否更新
            param_changes = []
            for name, param in model.named_parameters():
                if hasattr(param, 'old_data'):
                    change = (param.data - param.old_data).norm().item()
                    param_changes.append(change)
            
            if param_changes:
                avg_change = np.mean(param_changes)
                print(f"  参数平均变化: {avg_change:.6f}")
                
                if avg_change > 0:
                    print(f"✅ 单步训练验证通过: 参数已更新")
                    return True
                else:
                    print(f"❌ 单步训练验证失败: 参数未更新")
                    return False
            else:
                print(f"❌ 无法检查参数更新")
                return False
        else:
            print(f"❌ 输出中缺少'sdf'键")
            return False
            
    except Exception as e:
        print(f"❌ 训练测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("开始简化梯度验证...")
    
    test1_success = test_basic_gradient()
    
    if test1_success:
        print("\n" + "="*80)
        print("✅ 基础梯度验证成功，继续单步训练验证...")
        print("="*80)
        
        test2_success = test_training_step()
        
        if test2_success:
            print("\n" + "="*80)
            print("🎉 所有简化验证测试通过！")
            print("="*80)
            sys.exit(0)
        else:
            print("\n" + "="*80)
            print("⚠️ 单步训练验证失败")
            print("="*80)
            sys.exit(1)
    else:
        print("\n" + "="*80)
        print("❌ 基础梯度验证失败")
        print("="*80)
        sys.exit(1)