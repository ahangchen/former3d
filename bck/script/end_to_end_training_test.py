"""
Task 3.2: 端到端训练测试
测试StreamSDFFormerIntegrated模型的完整训练流程
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 尝试导入模型
try:
    from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
    print("✅ 成功导入StreamSDFFormerIntegrated")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def create_test_batch(batch_size=2, seq_len=3, image_size=(128, 128)):
    """
    创建测试批次数据
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        image_size: 图像尺寸 (H, W)
    
    Returns:
        images: 图像序列 [batch_size, seq_len, 3, H, W]
        poses: 相机位姿 [batch_size, seq_len, 4, 4]
        intrinsics: 相机内参 [batch_size, seq_len, 3, 3]
    """
    H, W = image_size
    
    # 创建图像 (归一化到[0, 1])
    images = torch.randn(batch_size, seq_len, 3, H, W)
    images = torch.sigmoid(images)  # 确保在[0, 1]范围内
    
    # 创建相机位姿 (单位矩阵 + 小扰动)
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
    for b in range(batch_size):
        for t in range(seq_len):
            # 添加小扰动模拟相机运动
            poses[b, t, :3, 3] = torch.tensor([t*0.1, 0.0, 0.0])
    
    # 创建相机内参 (假设固定内参)
    intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
    intrinsics[:, :, 0, 0] = 500.0  # fx
    intrinsics[:, :, 1, 1] = 500.0  # fy
    intrinsics[:, :, 0, 2] = W / 2  # cx
    intrinsics[:, :, 1, 2] = H / 2  # cy
    
    return images, poses, intrinsics


def test_single_frame_training():
    """测试单帧训练"""
    print("\n" + "="*60)
    print("测试1: 单帧训练")
    print("="*60)
    
    # 创建模型（使用正确的构造函数参数）
    model = StreamSDFFormerIntegrated(
        attn_heads=8,
        attn_layers=4,
        use_proj_occ=True,
        voxel_size=0.04,
        fusion_local_radius=3.0,
        crop_size=(48, 96, 96)
    )
    model.train()
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建单帧数据
    batch_size = 2
    images = torch.randn(batch_size, 3, 128, 128)
    poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics[:, 0, 0] = 500.0
    intrinsics[:, 1, 1] = 500.0
    intrinsics[:, 0, 2] = 64.0
    intrinsics[:, 1, 2] = 64.0
    
    # 训练循环
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(
            images=images,
            poses=poses,
            intrinsics=intrinsics,
            reset_state=True
        )
        
        # 计算损失（模拟SDF损失）
        sdf_pred = outputs['sdf']
        sdf_target = torch.randn_like(sdf_pred) * 0.1
        
        loss = nn.functional.mse_loss(sdf_pred, sdf_target)
        losses.append(loss.item())
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        # 更新参数
        optimizer.step()
        
        print(f"  步骤 {step+1}: 损失={loss.item():.6f}, 梯度范数={np.mean(grad_norms):.6f}")
    
    # 检查训练效果
    if losses[-1] < losses[0]:
        print(f"✅ 单帧训练成功: 损失从{losses[0]:.6f}下降到{losses[-1]:.6f}")
        return True
    else:
        print(f"❌ 单帧训练失败: 损失未下降")
        return False


def test_sequence_training():
    """测试序列训练"""
    print("\n" + "="*60)
    print("测试2: 序列训练")
    print("="*60)
    
    # 创建模型（使用正确的构造函数参数）
    model = StreamSDFFormerIntegrated(
        attn_heads=8,
        attn_layers=4,
        use_proj_occ=True,
        voxel_size=0.04,
        fusion_local_radius=3.0,
        crop_size=(48, 96, 96)
    )
    model.train()
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建序列数据
    batch_size = 2
    seq_len = 3
    images, poses, intrinsics = create_test_batch(batch_size, seq_len)
    
    # 训练循环
    losses = []
    for step in range(5):
        optimizer.zero_grad()
        
        total_loss = 0
        grad_info = {}
        
        # 逐帧处理序列
        for t in range(seq_len):
            # 提取当前帧
            frame_images = images[:, t]
            frame_poses = poses[:, t]
            frame_intrinsics = intrinsics[:, t]
            
            # 第一帧重置状态
            reset = (t == 0)
            
            # 前向传播
            outputs = model(
                images=frame_images,
                poses=frame_poses,
                intrinsics=frame_intrinsics,
                reset_state=reset
            )
            
            # 计算损失
            sdf_pred = outputs['sdf']
            sdf_target = torch.randn_like(sdf_pred) * 0.1
            loss = nn.functional.mse_loss(sdf_pred, sdf_target)
            total_loss += loss
        
        # 平均损失
        avg_loss = total_loss / seq_len
        losses.append(avg_loss.item())
        
        # 反向传播
        avg_loss.backward()
        
        # 收集梯度信息
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if name not in grad_info:
                    grad_info[name] = []
                grad_info[name].append(grad_norm)
        
        # 更新参数
        optimizer.step()
        
        # 打印关键模块梯度
        key_modules = ['net2d', 'net3d', 'mv_fusion', 'pose_projection', 'stream_fusion']
        grad_summary = []
        for module in key_modules:
            for name, grad_norm in grad_info.items():
                if module in name:
                    grad_summary.append(f"{module}:{np.mean(grad_info[name]):.6f}")
                    break
        
        print(f"  步骤 {step+1}: 平均损失={avg_loss.item():.6f}, 梯度[{', '.join(grad_summary)}]")
    
    # 检查训练效果
    if losses[-1] < losses[0]:
        print(f"✅ 序列训练成功: 损失从{losses[0]:.6f}下降到{losses[-1]:.6f}")
        return True
    else:
        print(f"❌ 序列训练失败: 损失未下降")
        return False


def test_gradient_flow_analysis():
    """测试梯度流分析"""
    print("\n" + "="*60)
    print("测试3: 梯度流分析")
    print("="*60)
    
    # 创建模型（使用正确的构造函数参数）
    model = StreamSDFFormerIntegrated(
        attn_heads=8,
        attn_layers=4,
        use_proj_occ=True,
        voxel_size=0.04,
        fusion_local_radius=3.0,
        crop_size=(48, 96, 96)
    )
    model.train()
    
    # 创建测试数据
    images = torch.randn(2, 3, 128, 128, requires_grad=True)
    poses = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)
    intrinsics = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)
    intrinsics[:, 0, 0] = 500.0
    intrinsics[:, 1, 1] = 500.0
    intrinsics[:, 0, 2] = 64.0
    intrinsics[:, 1, 2] = 64.0
    
    # 第一帧推理
    outputs1 = model(
        images=images,
        poses=poses,
        intrinsics=intrinsics,
        reset_state=True
    )
    
    # 第二帧推理（使用历史）
    outputs2 = model(
        images=images,
        poses=poses,
        intrinsics=intrinsics,
        reset_state=False
    )
    
    # 计算损失
    sdf_target = torch.randn_like(outputs2['sdf']) * 0.1
    loss = nn.functional.mse_loss(outputs2['sdf'], sdf_target)
    
    # 反向传播
    loss.backward()
    
    # 分析梯度流
    print("梯度流分析:")
    
    # 1. 检查输入图像梯度
    if images.grad is not None:
        grad_norm = images.grad.norm().item()
        print(f"  ✅ 输入图像梯度: 存在 (范数={grad_norm:.6f})")
    else:
        print(f"  ❌ 输入图像梯度: 不存在")
    
    # 2. 检查关键模块梯度
    key_modules = {
        'net2d': '2D特征提取网络',
        'net3d': '3D处理网络',
        'mv_fusion': '多视角融合',
        'pose_projection': '位姿投影',
        'stream_fusion': '流式融合'
    }
    
    modules_with_grad = 0
    for module_name, description in key_modules.items():
        has_grad = False
        for name, param in model.named_parameters():
            if module_name in name and param.grad is not None:
                has_grad = True
                break
        
        if has_grad:
            modules_with_grad += 1
            print(f"  ✅ {description}: 有梯度")
        else:
            print(f"  ❌ {description}: 无梯度")
    
    # 3. 检查计算图
    if outputs2['sdf'].grad_fn is not None:
        grad_fn_name = outputs2['sdf'].grad_fn.__class__.__name__
        print(f"  ✅ 计算图: 存在 ({grad_fn_name})")
        
        # 尝试追溯计算图
        current_fn = outputs2['sdf'].grad_fn
        depth = 0
        while current_fn is not None and depth < 10:
            depth += 1
            current_fn = getattr(current_fn, 'next_functions', None)
            if current_fn:
                current_fn = current_fn[0][0] if current_fn[0][0] else None
        print(f"    计算图深度: {depth}")
    else:
        print(f"  ❌ 计算图: 不存在")
    
    success = (images.grad is not None) and (modules_with_grad >= 3)
    if success:
        print(f"✅ 梯度流分析通过: {modules_with_grad}/5 个模块有梯度")
        return True
    else:
        print(f"❌ 梯度流分析失败")
        return False


def test_memory_efficiency():
    """测试内存效率"""
    print("\n" + "="*60)
    print("测试4: 内存效率")
    print("="*60)
    
    import gc
    import os
    
    # 简化内存测量，不使用psutil
    def get_memory_usage():
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            # 对于CPU，返回一个估计值
            return 0.0
    
    def get_memory_usage():
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    # 创建模型（使用正确的构造函数参数）
    model = StreamSDFFormerIntegrated(
        attn_heads=8,
        attn_layers=4,
        use_proj_occ=True,
        voxel_size=0.04,
        fusion_local_radius=3.0,
        crop_size=(48, 96, 96)
    )
    model.eval()  # 评估模式节省内存
    
    # 初始内存
    initial_memory = get_memory_usage()
    print(f"初始内存: {initial_memory:.2f} MB")
    
    # 测试不同序列长度
    seq_lengths = [1, 3, 5, 10]
    memory_increase = []
    
    for seq_len in seq_lengths:
        # 创建序列数据
        images, poses, intrinsics = create_test_batch(2, seq_len)
        
        # 处理序列
        with torch.no_grad():
            for t in range(seq_len):
                reset = (t == 0)
                outputs = model(
                    images=images[:, t],
                    poses=poses[:, t],
                    intrinsics=intrinsics[:, t],
                    reset_state=reset
                )
        
        # 测量内存
        current_memory = get_memory_usage()
        increase = current_memory - initial_memory
        memory_increase.append(increase)
        
        print(f"  序列长度 {seq_len}: 内存增加 {increase:.2f} MB")
        
        # 清理
        del images, poses, intrinsics, outputs
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 检查内存增长是否线性
    if len(memory_increase) >= 2:
        # 计算内存增长比率
        ratios = []
        for i in range(1, len(memory_increase)):
            if memory_increase[i-1] > 0:
                ratio = memory_increase[i] / memory_increase[i-1]
                ratios.append(ratio)
        
        avg_ratio = np.mean(ratios) if ratios else 0
        print(f"平均内存增长比率: {avg_ratio:.2f}")
        
        # 理想情况：内存增长应小于序列长度增长
        if avg_ratio < 1.5:  # 允许一些额外开销
            print(f"✅ 内存效率良好")
            return True
        else:
            print(f"⚠️ 内存增长较快，可能需要优化")
            return False
    else:
        print(f"✅ 内存测试完成")
        return True


def run_all_tests():
    """运行所有测试"""
    print("="*80)
    print("Task 3.2: 端到端训练测试")
    print("="*80)
    
    tests = [
        ("单帧训练", test_single_frame_training),
        ("序列训练", test_sequence_training),
        ("梯度流分析", test_gradient_flow_analysis),
        ("内存效率", test_memory_efficiency)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n▶️ 开始测试: {test_name}")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "="*80)
    print("测试结果总结")
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
        print("🎉 所有测试通过！")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)