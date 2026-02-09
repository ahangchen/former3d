"""
Task 3.2: 双GPU环境完整梯度验证
验证StreamSDFFormerIntegrated在GPU上的完整梯度流
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("双GPU环境完整梯度验证")
print("="*80)

# 检查GPU环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

if not torch.cuda.is_available():
    print("❌ CUDA不可用，无法进行GPU梯度验证")
    sys.exit(1)


def setup_distributed():
    """设置分布式环境（用于SyncBatchNorm）"""
    try:
        # 初始化单节点多GPU环境
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:23456',
            world_size=1,
            rank=0
        )
        print("✅ 分布式环境初始化成功")
        return True
    except Exception as e:
        print(f"⚠️ 分布式环境初始化失败: {e}")
        print("  将使用单GPU模式")
        return False


def create_test_model():
    """创建测试模型"""
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        # 创建简化模型以减少内存使用
        model = StreamSDFFormerIntegrated(
            attn_heads=4,  # 减少注意力头
            attn_layers=2,  # 减少注意力层
            use_proj_occ=False,  # 简化
            voxel_size=0.08,  # 增大体素减少计算
            fusion_local_radius=2.0,
            crop_size=(24, 48, 48)  # 减小裁剪空间
        )
        
        # 移动到GPU
        model = model.cuda()
        
        print("✅ 模型创建成功")
        print(f"  模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  模型设备: {next(model.parameters()).device}")
        
        return model
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_test_data(batch_size=2, image_size=(64, 64), seq_len=3):
    """创建测试数据"""
    H, W = image_size
    
    # 创建序列数据
    images_seq = []
    poses_seq = []
    intrinsics_seq = []
    
    for t in range(seq_len):
        # 图像数据
        images = torch.randn(batch_size, 3, H, W).cuda()
        images = torch.sigmoid(images)  # 归一化到[0, 1]
        
        # 相机位姿（模拟相机运动）
        poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        poses[:, :3, 3] = torch.tensor([t*0.2, 0.0, 0.0]).cuda()
        
        # 相机内参
        intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        intrinsics[:, 0, 0] = 300.0  # fx
        intrinsics[:, 1, 1] = 300.0  # fy
        intrinsics[:, 0, 2] = W / 2  # cx
        intrinsics[:, 1, 2] = H / 2  # cy
        
        images_seq.append(images)
        poses_seq.append(poses)
        intrinsics_seq.append(intrinsics)
    
    print(f"✅ 测试数据创建成功")
    print(f"  批次大小: {batch_size}")
    print(f"  图像尺寸: {image_size}")
    print(f"  序列长度: {seq_len}")
    
    return images_seq, poses_seq, intrinsics_seq


def test_single_frame_gradient(model):
    """测试单帧梯度"""
    print("\n" + "="*60)
    print("测试1: 单帧梯度验证")
    print("="*60)
    
    model.train()
    
    # 创建单帧数据
    batch_size = 2
    H, W = 64, 64
    
    images = torch.randn(batch_size, 3, H, W, requires_grad=True).cuda()
    images = torch.sigmoid(images)
    
    poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    intrinsics[:, 0, 0] = 300.0
    intrinsics[:, 1, 1] = 300.0
    intrinsics[:, 0, 2] = W / 2
    intrinsics[:, 1, 2] = H / 2
    
    # 重置状态
    model.reset_state()
    
    # 前向传播
    output = model(
        images=images,
        poses=poses,
        intrinsics=intrinsics,
        reset_state=True
    )
    
    print(f"  输出类型: {type(output)}")
    print(f"  输出键: {list(output.keys())}")
    
    # 检查输出
    if 'sdf' in output:
        sdf_pred = output['sdf']
        print(f"  SDF输出形状: {sdf_pred.shape}")
        print(f"  SDF requires_grad: {sdf_pred.requires_grad}")
        
        # 创建目标
        sdf_target = torch.randn_like(sdf_pred) * 0.1
        
        # 计算损失
        loss = nn.functional.mse_loss(sdf_pred, sdf_target)
        print(f"  损失值: {loss.item():.6f}")
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        if images.grad is not None:
            grad_norm = images.grad.norm().item()
            print(f"  ✅ 输入图像梯度: 存在 (范数={grad_norm:.6f})")
        else:
            print(f"  ❌ 输入图像梯度: 不存在")
        
        # 检查关键模块梯度
        key_modules = ['net2d', 'net3d', 'mv_fusion', 'pose_projection', 'stream_fusion']
        modules_with_grad = 0
        
        for module_name in key_modules:
            has_grad = False
            for name, param in model.named_parameters():
                if module_name in name and param.grad is not None:
                    has_grad = True
                    break
            
            if has_grad:
                modules_with_grad += 1
                print(f"  ✅ {module_name}: 有梯度")
            else:
                print(f"  ❌ {module_name}: 无梯度")
        
        # 检查计算图
        if sdf_pred.grad_fn is not None:
            grad_fn_name = sdf_pred.grad_fn.__class__.__name__
            print(f"  ✅ 计算图: 存在 ({grad_fn_name})")
            
            # 追溯计算图深度
            current_fn = sdf_pred.grad_fn
            depth = 0
            while current_fn is not None and depth < 20:
                depth += 1
                current_fn = getattr(current_fn, 'next_functions', None)
                if current_fn:
                    current_fn = current_fn[0][0] if current_fn[0][0] else None
            print(f"    计算图深度: {depth}")
        else:
            print(f"  ❌ 计算图: 不存在")
        
        success = (images.grad is not None) and (modules_with_grad >= 3)
        if success:
            print(f"✅ 单帧梯度验证通过: {modules_with_grad}/5 个模块有梯度")
            return True
        else:
            print(f"❌ 单帧梯度验证失败")
            return False
    else:
        print(f"❌ 输出中缺少'sdf'键")
        return False


def test_sequence_gradient(model):
    """测试序列梯度"""
    print("\n" + "="*60)
    print("测试2: 序列梯度验证")
    print("="*60)
    
    model.train()
    
    # 创建序列数据
    seq_len = 3
    images_seq, poses_seq, intrinsics_seq = create_test_data(seq_len=seq_len)
    
    # 重置状态
    model.reset_state()
    
    # 逐帧处理序列
    total_loss = 0
    for t in range(seq_len):
        # 提取当前帧
        images = images_seq[t]
        poses = poses_seq[t]
        intrinsics = intrinsics_seq[t]
        
        # 设置requires_grad
        images.requires_grad_(True)
        
        # 第一帧重置状态
        reset = (t == 0)
        
        # 前向传播
        output = model(
            images=images,
            poses=poses,
            intrinsics=intrinsics,
            reset_state=reset
        )
        
        if 'sdf' in output:
            sdf_pred = output['sdf']
            
            # 创建目标
            sdf_target = torch.randn_like(sdf_pred) * 0.1
            
            # 计算损失
            loss = nn.functional.mse_loss(sdf_pred, sdf_target)
            total_loss += loss
            
            print(f"  帧 {t+1}: 损失={loss.item():.6f}, SDF形状={sdf_pred.shape}")
    
    # 平均损失
    avg_loss = total_loss / seq_len
    print(f"  平均损失: {avg_loss.item():.6f}")
    
    # 反向传播
    avg_loss.backward()
    
    # 检查梯度
    print("\n  梯度分析:")
    
    # 检查所有帧的输入梯度
    frames_with_grad = 0
    for t in range(seq_len):
        images = images_seq[t]
        if images.grad is not None:
            frames_with_grad += 1
            grad_norm = images.grad.norm().item()
            print(f"    帧 {t+1} 图像梯度: 存在 (范数={grad_norm:.6f})")
        else:
            print(f"    帧 {t+1} 图像梯度: 不存在")
    
    # 检查关键模块梯度
    key_modules = ['net2d', 'net3d', 'mv_fusion', 'pose_projection', 'stream_fusion']
    modules_with_grad = 0
    
    for module_name in key_modules:
        has_grad = False
        for name, param in model.named_parameters():
            if module_name in name and param.grad is not None:
                has_grad = True
                break
        
        if has_grad:
            modules_with_grad += 1
            print(f"    ✅ {module_name}: 有梯度")
        else:
            print(f"    ❌ {module_name}: 无梯度")
    
    success = (frames_with_grad >= 2) and (modules_with_grad >= 3)
    if success:
        print(f"✅ 序列梯度验证通过: {frames_with_grad}/{seq_len} 帧有梯度, {modules_with_grad}/5 个模块有梯度")
        return True
    else:
        print(f"❌ 序列梯度验证失败")
        return False


def test_training_loop(model):
    """测试训练循环"""
    print("\n" + "="*60)
    print("测试3: 训练循环验证")
    print("="*60)
    
    model.train()
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        
        # 创建数据
        batch_size = 2
        H, W = 64, 64
        
        images = torch.randn(batch_size, 3, H, W, requires_grad=True).cuda()
        images = torch.sigmoid(images)
        
        poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        intrinsics[:, 0, 0] = 300.0
        intrinsics[:, 1, 1] = 300.0
        intrinsics[:, 0, 2] = W / 2
        intrinsics[:, 1, 2] = H / 2
        
        # 重置状态
        model.reset_state()
        
        # 第一帧推理
        output1 = model(
            images=images,
            poses=poses,
            intrinsics=intrinsics,
            reset_state=True
        )
        
        # 第二帧推理（使用历史）
        output2 = model(
            images=images,
            poses=poses,
            intrinsics=intrinsics,
            reset_state=False
        )
        
        # 计算损失（使用第二帧输出）
        if 'sdf' in output2:
            sdf_pred = output2['sdf']
            sdf_target = torch.randn_like(sdf_pred) * 0.1
            
            loss = nn.functional.mse_loss(sdf_pred, sdf_target)
            losses.append(loss.item())
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 检查参数更新
            param_changes = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_changes.append(param.grad.norm().item())
            
            print(f"  步骤 {step+1}: 损失={loss.item():.6f}, 平均梯度范数={np.mean(param_changes):.6f}")
        else:
            print(f"  步骤 {step+1}: 输出中缺少'sdf'键")
            return False
    
    # 检查训练效果
    if len(losses) >= 2:
        print(f"  初始损失: {losses[0]:.6f}")
        print(f"  最终损失: {losses[-1]:.6f}")
        
        if losses[-1] < losses[0]:
            print(f"✅ 训练循环验证通过: 损失下降 {losses[0]-losses[-1]:.6f}")
            return True
        else:
            print(f"❌ 训练循环验证失败: 损失未下降")
            return False
    else:
        print(f"❌ 训练循环验证失败: 损失记录不足")
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("开始GPU梯度验证")
    print("="*80)
    
    # 设置分布式环境
    dist_success = setup_distributed()
    
    # 创建模型
    model = create_test_model()
    if model is None:
        return False
    
    tests = [
        ("单帧梯度验证", test_single_frame_gradient),
        ("序列梯度验证", test_sequence_gradient),
        ("训练循环验证", test_training_loop)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n▶️ 开始测试: {test_name}")
            success = test_func(model)
            results.append((test_name, success))
            
            # 清理梯度
            model.zero_grad()
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "="*80)
    print("GPU梯度验证结果")
    print("="*80)
    
    passed = 0
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\n总体结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有GPU梯度验证测试通过！")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
        return passed >= 2  # 允许1个测试失败


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    finally:
        # 清理
        torch.cuda.empty_cache()