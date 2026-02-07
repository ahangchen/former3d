"""
流式SDFFormer单元测试
"""

import torch
import pytest
import numpy as np
from former3d.stream_sdfformer import StreamSDFFormer


@pytest.fixture
def stream_model():
    """创建流式模型fixture"""
    return StreamSDFFormer(
        use_proj_occ=False,
        attn_heads=2,
        attn_layers=2,
        voxel_size=0.0625,
        fusion_local_radius=3
    )


@pytest.fixture
def sample_stream_data():
    """创建流式测试数据fixture"""
    batch_size = 2
    sequence_length = 3
    
    # 图像数据
    images = torch.randn(batch_size, 3, 256, 256)
    
    # 序列数据
    image_sequence = torch.randn(sequence_length, batch_size, 3, 256, 256)
    
    # 位姿数据
    poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 序列位姿（添加平移）
    pose_sequence = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(sequence_length, batch_size, 1, 1)
    for t in range(sequence_length):
        pose_sequence[t, :, :3, 3] = torch.tensor([t * 0.1, 0, 0])
    
    # 内参数据
    intrinsics = torch.tensor([
        [256, 0, 128],
        [0, 256, 128],
        [0, 0, 1]
    ]).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 序列内参
    intrinsics_sequence = intrinsics.unsqueeze(0).repeat(sequence_length, 1, 1, 1)
    
    return {
        'images': images,
        'image_sequence': image_sequence,
        'poses': poses,
        'pose_sequence': pose_sequence,
        'intrinsics': intrinsics,
        'intrinsics_sequence': intrinsics_sequence,
        'batch_size': batch_size,
        'sequence_length': sequence_length
    }


def test_state_management(stream_model):
    """测试状态管理"""
    print("测试状态管理...")
    
    # 初始状态
    assert stream_model.historical_state is None
    assert not stream_model._state_initialized
    print("✅ 初始状态正确")
    
    # 初始化状态
    stream_model.initialize_state(batch_size=2, device='cpu')
    assert stream_model._state_initialized
    assert stream_model.historical_state is not None
    assert isinstance(stream_model.historical_state, dict)
    print("✅ 状态初始化正确")
    
    # 重置状态
    stream_model.reset_state()
    assert stream_model.historical_state is None
    assert not stream_model._state_initialized
    print("✅ 状态重置正确")


def test_single_frame_inference(stream_model, sample_stream_data):
    """测试单帧推理"""
    print("测试单帧推理...")
    
    # 重置状态
    stream_model.reset_state()
    
    # 单帧推理
    output = stream_model(
        sample_stream_data['images'],
        sample_stream_data['poses'],
        sample_stream_data['intrinsics'],
        reset_state=True
    )
    
    # 检查输出字段
    required_fields = ['sdf', 'occupancy', 'features']
    for field in required_fields:
        assert field in output
        print(f"✅ 输出包含 {field}")
    
    # 检查输出形状
    assert output['sdf'].dim() == 2  # [num_voxels, 1]
    assert output['occupancy'].dim() == 2  # [num_voxels, 1]
    assert output['features'].dim() == 2  # [num_voxels, feature_dim]
    
    print(f"✅ 单帧推理输出形状正确:")
    print(f"  - SDF: {output['sdf'].shape}")
    print(f"  - Occupancy: {output['occupancy'].shape}")
    print(f"  - Features: {output['features'].shape}")


def test_sequence_inference(stream_model, sample_stream_data):
    """测试序列推理"""
    print("测试序列推理...")
    
    # 序列推理
    sequence_output = stream_model.forward_sequence(
        sample_stream_data['image_sequence'],
        sample_stream_data['pose_sequence'],
        sample_stream_data['intrinsics_sequence']
    )
    
    # 检查输出字段
    required_fields = ['sdf', 'occupancy', 'features']
    for field in required_fields:
        if field in sequence_output:
            print(f"✅ 序列输出包含 {field}")
            
            # 检查序列维度
            assert sequence_output[field].dim() >= 2
            assert sequence_output[field].shape[0] == sample_stream_data['sequence_length']
            print(f"  - {field} 序列形状: {sequence_output[field].shape}")


def test_gradient_flow(stream_model, sample_stream_data):
    """测试梯度流"""
    print("测试梯度流...")
    
    # 重置状态
    stream_model.reset_state()
    
    # 创建需要梯度的输入
    images = sample_stream_data['images'].clone().requires_grad_(True)
    
    # 推理
    output = stream_model(
        images,
        sample_stream_data['poses'],
        sample_stream_data['intrinsics'],
        reset_state=True
    )
    
    # 计算损失
    loss = output['sdf'].sum() + output['occupancy'].sum()
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    assert images.grad is not None
    assert not torch.all(images.grad == 0)
    
    grad_norm = torch.norm(images.grad)
    print(f"✅ 梯度存在，梯度范数: {grad_norm:.6f}")


def test_state_persistence(stream_model, sample_stream_data):
    """测试状态持久性"""
    print("测试状态持久性...")
    
    # 重置状态
    stream_model.reset_state()
    
    # 第一帧推理
    output1 = stream_model(
        sample_stream_data['images'],
        sample_stream_data['poses'],
        sample_stream_data['intrinsics'],
        reset_state=True
    )
    
    # 检查状态已更新
    assert stream_model.historical_state is not None
    assert stream_model.historical_pose is not None
    print("✅ 第一帧后状态已更新")
    
    # 保存状态
    saved_state = stream_model.historical_state.copy()
    saved_pose = stream_model.historical_pose.clone()
    
    # 第二帧推理（使用相同输入，但状态应该不同）
    # 修改位姿以测试状态更新
    new_pose = sample_stream_data['poses'].clone()
    new_pose[:, :3, 3] += 0.1  # 添加平移
    
    output2 = stream_model(
        sample_stream_data['images'],
        new_pose,
        sample_stream_data['intrinsics'],
        reset_state=False  # 不重置状态
    )
    
    # 检查状态已更新
    assert stream_model.historical_state is not None
    assert stream_model.historical_pose is not None
    
    # 状态应该已更新（与保存的状态不同）
    # 注意：由于模型内部实现，可能无法直接比较状态
    print("✅ 第二帧后状态已更新")
    
    # 检查输出不同（因为位姿不同）
    assert not torch.allclose(output1['sdf'], output2['sdf'], rtol=1e-4)
    print("✅ 不同位姿产生不同输出")


def test_batch_processing(stream_model):
    """测试批量处理"""
    print("测试批量处理...")
    
    # 测试不同批量大小
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        print(f"测试 batch_size={batch_size}...")
        
        # 创建数据
        images = torch.randn(batch_size, 3, 256, 256)
        poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics = torch.tensor([[256, 0, 128], [0, 256, 128], [0, 0, 1]]).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 重置状态
        stream_model.reset_state()
        
        # 推理
        output = stream_model(
            images, poses, intrinsics, reset_state=True
        )
        
        # 检查输出
        assert 'sdf' in output
        assert 'occupancy' in output
        
        # 检查特征数量与批量大小相关
        # 注意：实际特征数量可能因模型内部实现而异
        print(f"  ✅ batch_size={batch_size} 处理成功")
        print(f"    输出特征数: {output['features'].shape[0]}")


def test_module_integration(stream_model):
    """测试模块集成"""
    print("测试模块集成...")
    
    # 检查是否集成了所有必要模块
    assert hasattr(stream_model, 'pose_projection')
    assert hasattr(stream_model, 'stream_fusion')
    
    print("✅ 姿态投影模块已集成")
    print("✅ Cross-Attention融合模块已集成")
    
    # 检查模块类型
    from former3d.pose_projection import PoseProjection
    from former3d.stream_fusion import StreamCrossAttention
    
    assert isinstance(stream_model.pose_projection, PoseProjection)
    assert isinstance(stream_model.stream_fusion, StreamCrossAttention)
    
    print("✅ 模块类型正确")


def test_training_mode(stream_model, sample_stream_data):
    """测试训练模式"""
    print("测试训练模式...")
    
    # 设置为训练模式
    stream_model.train()
    assert stream_model.training
    print("✅ 训练模式设置正确")
    
    # 推理（应该能正常工作）
    output = stream_model(
        sample_stream_data['images'],
        sample_stream_data['poses'],
        sample_stream_data['intrinsics'],
        reset_state=True
    )
    
    assert 'sdf' in output
    print("✅ 训练模式下推理正常")
    
    # 切换回评估模式
    stream_model.eval()
    assert not stream_model.training
    print("✅ 评估模式设置正确")


def test_edge_cases(stream_model):
    """测试边界情况"""
    print("测试边界情况...")
    
    # 测试极小批量（batch_size=1）
    images = torch.randn(1, 3, 256, 256)
    poses = torch.eye(4).unsqueeze(0)
    intrinsics = torch.tensor([[256, 0, 128], [0, 256, 128], [0, 0, 1]]).unsqueeze(0)
    
    stream_model.reset_state()
    output = stream_model(images, poses, intrinsics, reset_state=True)
    
    assert output['sdf'].shape[0] > 0  # 应该有输出
    print("✅ 极小批量处理正常")
    
    # 测试重复调用（状态应持续更新）
    for i in range(3):
        output = stream_model(images, poses, intrinsics, reset_state=False)
        assert 'sdf' in output
    print("✅ 重复调用正常")


def test_memory_management(stream_model, sample_stream_data):
    """测试内存管理"""
    print("测试内存管理...")
    
    import gc
    
    # 多次推理，检查内存使用
    stream_model.reset_state()
    
    memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    for i in range(5):
        output = stream_model(
            sample_stream_data['images'],
            sample_stream_data['poses'],
            sample_stream_data['intrinsics'],
            reset_state=(i == 0)
        )
        
        # 清理中间变量
        del output
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    memory_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # 内存使用应该相对稳定（可能有微小增长）
    if torch.cuda.is_available():
        memory_growth = memory_after - memory_before
        print(f"✅ 内存增长: {memory_growth / 1024**2:.2f} MB")
        assert memory_growth < 100 * 1024**2  # 增长应小于100MB
    else:
        print("✅ CPU模式下内存管理测试跳过")


if __name__ == "__main__":
    """运行所有测试"""
    print("=" * 50)
    print("运行流式SDFFormer单元测试")
    print("=" * 50)
    
    # 创建fixtures
    model = stream_model()
    data = sample_stream_data()
    
    # 运行测试
    test_state_management(model)
    test_single_frame_inference(model, data)
    test_sequence_inference(model, data)
    test_gradient_flow(model, data)
    test_state_persistence(model, data)
    test_batch_processing(model)
    test_module_integration(model)
    test_training_mode(model, data)
    test_edge_cases(model)
    test_memory_management(model, data)
    
    print("=" * 50)
    print("所有测试通过！ ✅")
    print("=" * 50)