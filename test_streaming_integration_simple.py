"""
流式SDFFormer简单集成测试
"""

import torch
import numpy as np
from former3d.stream_sdfformer_simple import SimpleStreamSDFFormer


def create_test_sequence():
    """创建测试序列"""
    sequence_length = 3
    batch_size = 1
    
    # 图像序列
    image_sequence = torch.randn(sequence_length, batch_size, 3, 128, 128).float()
    
    # 位姿序列（模拟相机运动）
    pose_sequence = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(
        sequence_length, batch_size, 1, 1
    )
    
    # 添加相机运动
    for t in range(sequence_length):
        angle = torch.tensor(t * 0.1)
        pose_sequence[t, :, 0, 0] = torch.cos(angle)
        pose_sequence[t, :, 0, 2] = torch.sin(angle)
        pose_sequence[t, :, 2, 0] = -torch.sin(angle)
        pose_sequence[t, :, 2, 2] = torch.cos(angle)
        pose_sequence[t, :, 0, 3] = t * 0.05
    
    return image_sequence, pose_sequence


def test_end_to_end():
    """测试端到端流式推理"""
    print("=" * 60)
    print("测试端到端流式推理")
    print("=" * 60)
    
    # 创建模型
    model = SimpleStreamSDFFormer(
        feature_dim=32,
        voxel_size=0.0625,
        crop_size=(16, 16, 8),
        fusion_local_radius=2
    )
    
    # 创建测试序列
    image_sequence, pose_sequence = create_test_sequence()
    
    print(f"模型: {model.__class__.__name__}")
    print(f"体素数量: {model.num_voxels}")
    print(f"序列长度: {image_sequence.shape[0]}")
    print(f"批量大小: {image_sequence.shape[1]}")
    
    # 处理序列
    model.reset_state()
    sequence_output = model.forward_sequence(image_sequence, pose_sequence)
    
    # 验证输出
    print(f"\\n✅ 端到端测试通过")
    print(f"  SDF形状: {sequence_output['sdf'].shape}")
    print(f"  Occupancy形状: {sequence_output['occupancy'].shape}")
    print(f"  Features形状: {sequence_output['features'].shape}")
    
    return True


def test_state_management():
    """测试状态管理"""
    print("\\n" + "=" * 60)
    print("测试状态管理")
    print("=" * 60)
    
    model = SimpleStreamSDFFormer(
        feature_dim=32,
        voxel_size=0.0625,
        crop_size=(16, 16, 8),
        fusion_local_radius=2
    )
    
    # 创建测试数据
    image = torch.randn(1, 3, 128, 128).float()
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    
    # 测试1：初始状态
    model.reset_state()
    assert model.historical_state is None
    print("✅ 初始状态正确")
    
    # 测试2：第一帧推理
    output1 = model(image, pose, reset_state=True)
    assert model.historical_state is not None
    print("✅ 第一帧后状态更新")
    
    # 测试3：第二帧推理（使用历史状态）
    output2 = model(image, pose, reset_state=False)
    assert model.historical_state is not None
    print("✅ 第二帧后状态更新")
    
    # 检查输出不同（因为使用了历史状态）
    assert not torch.allclose(output1['sdf'], output2['sdf'], rtol=1e-4)
    print("✅ 使用历史状态产生不同输出")
    
    return True


def test_gradient_flow():
    """测试梯度流"""
    print("\\n" + "=" * 60)
    print("测试梯度流")
    print("=" * 60)
    
    model = SimpleStreamSDFFormer(
        feature_dim=32,
        voxel_size=0.0625,
        crop_size=(16, 16, 8),
        fusion_local_radius=2
    )
    
    # 创建需要梯度的输入
    image = torch.randn(1, 3, 128, 128).float().requires_grad_(True)
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    
    # 推理
    model.reset_state()
    output = model(image, pose, reset_state=True)
    
    # 计算损失
    loss = output['sdf'].sum() + output['occupancy'].sum()
    
    # 反向传播
    loss.backward()
    
    # 验证梯度
    assert image.grad is not None
    assert not torch.all(image.grad == 0)
    
    grad_norm = torch.norm(image.grad)
    print(f"✅ 梯度流测试通过")
    print(f"  梯度范数: {grad_norm:.6f}")
    
    return True


def test_fusion_module():
    """测试融合模块"""
    print("\\n" + "=" * 60)
    print("测试融合模块")
    print("=" * 60)
    
    model = SimpleStreamSDFFormer(
        feature_dim=32,
        voxel_size=0.0625,
        crop_size=(16, 16, 8),
        fusion_local_radius=2
    )
    
    # 检查融合模块
    assert hasattr(model, 'stream_fusion')
    assert hasattr(model, 'pose_projection')
    
    print("✅ 融合模块存在")
    print(f"  融合模块: {model.stream_fusion.__class__.__name__}")
    print(f"  投影模块: {model.pose_projection.__class__.__name__}")
    
    # 测试融合模块的梯度
    image1 = torch.randn(1, 3, 128, 128).float().requires_grad_(True)
    image2 = torch.randn(1, 3, 128, 128).float().requires_grad_(True)
    
    pose1 = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    pose2 = pose1.clone()
    pose2[:, 0, 3] = 0.1
    
    # 第一帧
    model.reset_state()
    output1 = model(image1, pose1, reset_state=True)
    
    # 第二帧（使用历史状态）
    output2 = model(image2, pose2, reset_state=False)
    
    # 计算损失和梯度
    loss = output2['sdf'].sum()
    loss.backward()
    
    assert image1.grad is not None or image2.grad is not None
    print("✅ 融合模块梯度测试通过")
    
    return True


def test_performance():
    """测试性能"""
    print("\\n" + "=" * 60)
    print("测试性能")
    print("=" * 60)
    
    import time
    
    model = SimpleStreamSDFFormer(
        feature_dim=32,
        voxel_size=0.0625,
        crop_size=(16, 16, 8),
        fusion_local_radius=2
    )
    
    # 创建测试数据
    image_sequence, pose_sequence = create_test_sequence()
    
    # 预热
    model.reset_state()
    _ = model.forward_sequence(image_sequence[:1], pose_sequence[:1])
    
    # 性能测试
    num_runs = 5
    times = []
    
    for i in range(num_runs):
        model.reset_state()
        
        start_time = time.time()
        output = model.forward_sequence(image_sequence, pose_sequence)
        end_time = time.time()
        
        times.append(end_time - start_time)
        
        if i == 0:
            print(f"  第{i+1}次运行: {times[-1]:.3f}秒 (包含初始化)")
        else:
            print(f"  第{i+1}次运行: {times[-1]:.3f}秒")
    
    avg_time = np.mean(times[1:])  # 跳过第一次（包含初始化）
    fps = image_sequence.shape[0] / avg_time
    
    print(f"\\n✅ 性能测试通过")
    print(f"  平均处理时间: {avg_time:.3f}秒")
    print(f"  帧率: {fps:.1f} FPS")
    print(f"  每帧时间: {1000*avg_time/image_sequence.shape[0]:.1f}毫秒")
    
    return True


def main():
    """主测试函数"""
    print("流式SDFFormer集成测试")
    print("=" * 60)
    
    tests = [
        ("端到端流式推理", test_end_to_end),
        ("状态管理", test_state_management),
        ("梯度流", test_gradient_flow),
        ("融合模块", test_fusion_module),
        ("性能", test_performance),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\\n运行测试: {test_name}")
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 打印总结
    print("\\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\\n🎉 所有测试通过！")
    else:
        print("\\n⚠️  部分测试失败，需要进一步调试")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)