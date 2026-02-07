"""
流式SDFFormer集成测试
测试整个流式架构的完整功能
"""

import torch
import pytest
import numpy as np
from former3d.stream_sdfformer_simple import SimpleStreamSDFFormer


@pytest.fixture
def streaming_model():
    """创建流式模型fixture"""
    return SimpleStreamSDFFormer(
        feature_dim=64,
        voxel_size=0.0625,
        crop_size=(24, 24, 12),
        fusion_local_radius=3
    )


@pytest.fixture
def test_sequence():
    """创建测试序列fixture"""
    sequence_length = 4
    batch_size = 2
    
    # 图像序列
    image_sequence = torch.randn(sequence_length, batch_size, 3, 128, 128).float()
    
    # 位姿序列（模拟相机运动）
    pose_sequence = torch.eye(4, dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(
        sequence_length, batch_size, 1, 1
    )
    
    # 添加相机运动：绕圈
    for t in range(sequence_length):
        angle = t * 0.1  # 每帧旋转0.1弧度
        # 绕Y轴旋转
        pose_sequence[t, :, 0, 0] = torch.cos(angle)
        pose_sequence[t, :, 0, 2] = torch.sin(angle)
        pose_sequence[t, :, 2, 0] = -torch.sin(angle)
        pose_sequence[t, :, 2, 2] = torch.cos(angle)
        # 添加平移
        pose_sequence[t, :, 0, 3] = t * 0.05
        pose_sequence[t, :, 1, 3] = 0.0
        pose_sequence[t, :, 2, 3] = t * 0.05
    
    return {
        'image_sequence': image_sequence,
        'pose_sequence': pose_sequence,
        'sequence_length': sequence_length,
        'batch_size': batch_size
    }


def test_end_to_end_streaming(streaming_model, test_sequence):
    """测试端到端的流式推理"""
    print("测试端到端流式推理...")
    
    # 重置模型状态
    streaming_model.reset_state()
    
    # 处理整个序列
    sequence_output = streaming_model.forward_sequence(
        test_sequence['image_sequence'],
        test_sequence['pose_sequence']
    )
    
    # 验证输出
    assert 'sdf' in sequence_output
    assert 'occupancy' in sequence_output
    assert 'features' in sequence_output
    
    # 验证序列维度
    assert sequence_output['sdf'].shape[0] == test_sequence['sequence_length']
    assert sequence_output['occupancy'].shape[0] == test_sequence['sequence_length']
    assert sequence_output['features'].shape[0] == test_sequence['sequence_length']
    
    print(f"✅ 端到端测试通过")
    print(f"  SDF形状: {sequence_output['sdf'].shape}")
    print(f"  Occupancy形状: {sequence_output['occupancy'].shape}")
    print(f"  Features形状: {sequence_output['features'].shape}")


def test_state_consistency(streaming_model, test_sequence):
    """测试状态一致性"""
    print("测试状态一致性...")
    
    # 方法1：使用forward_sequence处理整个序列
    streaming_model.reset_state()
    sequence_output = streaming_model.forward_sequence(
        test_sequence['image_sequence'],
        test_sequence['pose_sequence']
    )
    
    # 方法2：手动逐帧处理
    streaming_model.reset_state()
    manual_outputs = []
    
    for t in range(test_sequence['sequence_length']):
        output = streaming_model(
            test_sequence['image_sequence'][t],
            test_sequence['pose_sequence'][t],
            reset_state=(t == 0)
        )
        manual_outputs.append(output)
    
    # 比较两种方法的输出
    for t in range(test_sequence['sequence_length']):
        sdf_seq = sequence_output['sdf'][t]
        sdf_manual = manual_outputs[t]['sdf']
        
        # 允许微小差异（由于数值精度）
        assert torch.allclose(sdf_seq, sdf_manual, rtol=1e-4, atol=1e-5)
    
    print("✅ 状态一致性测试通过")


def test_gradient_flow_through_sequence(streaming_model, test_sequence):
    """测试序列中的梯度流"""
    print("测试序列梯度流...")
    
    # 创建需要梯度的输入
    image_sequence = test_sequence['image_sequence'].clone().requires_grad_(True)
    
    # 重置模型状态
    streaming_model.reset_state()
    
    # 处理序列
    sequence_output = streaming_model.forward_sequence(
        image_sequence,
        test_sequence['pose_sequence']
    )
    
    # 计算损失（使用所有帧的SDF和占用）
    loss = sequence_output['sdf'].sum() + sequence_output['occupancy'].sum()
    
    # 反向传播
    loss.backward()
    
    # 验证梯度存在
    assert image_sequence.grad is not None
    assert not torch.all(image_sequence.grad == 0)
    
    grad_norm = torch.norm(image_sequence.grad)
    print(f"✅ 梯度流测试通过，梯度范数: {grad_norm:.6f}")


def test_different_batch_sizes(streaming_model):
    """测试不同批量大小"""
    print("测试不同批量大小...")
    
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        print(f"  测试 batch_size={batch_size}...")
        
        # 创建测试数据
        image = torch.randn(batch_size, 3, 128, 128).float()
        pose = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 重置状态
        streaming_model.reset_state()
        
        # 推理
        output = streaming_model(image, pose, reset_state=True)
        
        # 验证输出
        assert output['sdf'].shape[0] == batch_size * streaming_model.num_voxels
        assert output['features'].shape[0] == batch_size * streaming_model.num_voxels
        
        print(f"    ✅ batch_size={batch_size} 处理成功")
    
    print("✅ 不同批量大小测试通过")


def test_memory_usage(streaming_model, test_sequence):
    """测试内存使用"""
    print("测试内存使用...")
    
    import gc
    
    # 记录初始内存
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        memory_before = torch.cuda.memory_allocated()
    else:
        memory_before = 0
    
    # 处理序列
    streaming_model.reset_state()
    sequence_output = streaming_model.forward_sequence(
        test_sequence['image_sequence'],
        test_sequence['pose_sequence']
    )
    
    # 清理
    del sequence_output
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        memory_after = torch.cuda.memory_allocated()
        memory_used = memory_after - memory_before
        
        print(f"✅ 内存使用测试通过，内存使用量: {memory_used / 1024**2:.2f} MB")
        assert memory_used < 500 * 1024**2  # 应小于500MB
    else:
        print("✅ CPU模式下内存测试跳过")


def test_temporal_consistency(streaming_model, test_sequence):
    """测试时间一致性"""
    print("测试时间一致性...")
    
    # 处理序列
    streaming_model.reset_state()
    sequence_output = streaming_model.forward_sequence(
        test_sequence['image_sequence'],
        test_sequence['pose_sequence']
    )
    
    # 检查相邻帧之间的变化
    sdf_sequence = sequence_output['sdf']
    
    for t in range(1, test_sequence['sequence_length']):
        # 计算相邻帧之间的差异
        diff = torch.mean(torch.abs(sdf_sequence[t] - sdf_sequence[t-1]))
        
        # 差异应该在一定范围内（既不能太大也不能太小）
        assert diff > 1e-6  # 应该有变化
        assert diff < 1.0   # 变化不应太大
        
        print(f"  帧{t-1}到帧{t}的SDF平均差异: {diff:.6f}")
    
    print("✅ 时间一致性测试通过")


def test_fusion_effectiveness(streaming_model):
    """测试融合效果"""
    print("测试融合效果...")
    
    # 创建两帧数据，第二帧有微小变化
    batch_size = 1
    image1 = torch.randn(batch_size, 3, 128, 128).float()
    image2 = image1.clone() + torch.randn_like(image1) * 0.01  # 添加微小噪声
    
    pose1 = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    pose2 = pose1.clone()
    pose2[:, 0, 3] = 0.1  # 微小平移
    
    # 方法1：无历史状态（第一帧）
    streaming_model.reset_state()
    output1 = streaming_model(image1, pose1, reset_state=True)
    
    # 方法2：有历史状态（第二帧）
    output2 = streaming_model(image2, pose2, reset_state=False)
    
    # 检查输出不同（因为输入不同）
    assert not torch.allclose(output1['sdf'], output2['sdf'], rtol=1e-4)
    
    # 检查特征不同
    assert not torch.allclose(output1['features'], output2['features'], rtol=1e-4)
    
    print("✅ 融合效果测试通过")
    print(f"  第一帧SDF范围: [{output1['sdf'].min():.3f}, {output1['sdf'].max():.3f}]")
    print(f"  第二帧SDF范围: [{output2['sdf'].min():.3f}, {output2['sdf'].max():.3f}]")


def test_edge_cases(streaming_model):
    """测试边界情况"""
    print("测试边界情况...")
    
    # 测试1：极小图像
    image_tiny = torch.randn(1, 3, 32, 32).float()
    pose = torch.eye(4, dtype=torch.float32).unsqueeze(0)
    
    streaming_model.reset_state()
    output_tiny = streaming_model(image_tiny, pose, reset_state=True)
    assert output_tiny['sdf'].shape[0] == streaming_model.num_voxels
    print("  ✅ 极小图像处理成功")
    
    # 测试2：重复帧
    image = torch.randn(1, 3, 128, 128).float()
    
    streaming_model.reset_state()
    output1 = streaming_model(image, pose, reset_state=True)
    output2 = streaming_model(image, pose, reset_state=False)
    
    # 由于历史状态，输出应该不同
    assert not torch.allclose(output1['sdf'], output2['sdf'], rtol=1e-4)
    print("  ✅ 重复帧处理成功（输出不同）")
    
    # 测试3：大平移
    pose_large = pose.clone()
    pose_large[:, :3, 3] = torch.tensor([1.0, 0.5, 0.5])  # 大平移
    
    streaming_model.reset_state()
    output_large = streaming_model(image, pose_large, reset_state=True)
    assert output_large['sdf'].shape[0] == streaming_model.num_voxels
    print("  ✅ 大平移处理成功")
    
    print("✅ 所有边界情况测试通过")


if __name__ == "__main__":
    """运行所有集成测试"""
    print("=" * 60)
    print("运行流式SDFFormer集成测试")
    print("=" * 60)
    
    # 创建fixtures
    model = streaming_model()
    sequence = test_sequence()
    
    # 运行测试
    test_end_to_end_streaming(model, sequence)
    test_state_consistency(model, sequence)
    test_gradient_flow_through_sequence(model, sequence)
    test_different_batch_sizes(model)
    test_memory_usage(model, sequence)
    test_temporal_consistency(model, sequence)
    test_fusion_effectiveness(model)
    test_edge_cases(model)
    
    print("=" * 60)
    print("所有集成测试通过！ ✅")
    print("=" * 60)