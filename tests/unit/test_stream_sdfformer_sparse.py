"""
流式SDFFormer单元测试（稀疏表示版本）
"""

import torch
import pytest
import numpy as np
from former3d.stream_sdfformer_sparse import StreamSDFFormerSparse


@pytest.fixture
def stream_model():
    """创建流式SDFFormer模型fixture"""
    return StreamSDFFormerSparse(
        use_proj_occ=False,
        attn_heads=2,
        attn_layers=2,
        voxel_size=0.0625,
        fusion_local_radius=3,
        crop_size=(48, 96, 96)
    )


@pytest.fixture
def sample_sequence_data():
    """创建测试序列数据fixture"""
    batch_size = 2
    seq_length = 3
    
    images_seq = []
    poses_seq = []
    intrinsics_seq = []
    
    for i in range(seq_length):
        # 图像 [batch, 3, 256, 256]
        images = torch.randn(batch_size, 3, 256, 256)
        
        # 位姿 [batch, 4, 4]
        pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        # 添加微小平移模拟相机运动
        pose[:, 0, 3] = i * 0.1  # X方向平移
        
        # 内参 [batch, 3, 3]
        intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics[:, 0, 0] = 500.0  # fx
        intrinsics[:, 1, 1] = 500.0  # fy
        intrinsics[:, 0, 2] = 128.0  # cx
        intrinsics[:, 1, 2] = 128.0  # cy
        
        images_seq.append(images)
        poses_seq.append(pose)
        intrinsics_seq.append(intrinsics)
    
    return {
        'images_seq': images_seq,
        'poses_seq': poses_seq,
        'intrinsics_seq': intrinsics_seq,
        'batch_size': batch_size,
        'seq_length': seq_length
    }


def test_sparse_single_frame_no_history(stream_model, sample_sequence_data):
    """测试稀疏单帧推理（无历史状态）"""
    print("测试稀疏单帧推理（无历史状态）...")
    
    # 获取第一帧数据
    images = sample_sequence_data['images_seq'][0]
    poses = sample_sequence_data['poses_seq'][0]
    intrinsics = sample_sequence_data['intrinsics_seq'][0]
    
    # 重置状态
    stream_model.historical_state = None
    stream_model.historical_pose = None
    
    # 单帧推理
    output = stream_model(images, poses, intrinsics, reset_state=True)
    
    # 验证输出结构
    required_keys = ['sdf', 'occupancy', 'features', 'coords', 'batch_inds', 'num_voxels']
    for key in required_keys:
        assert key in output, f"输出缺少键: {key}"
    
    # 验证形状
    num_voxels = output['num_voxels']
    assert output['sdf'].shape == (num_voxels, 1)
    assert output['occupancy'].shape == (num_voxels, 1)
    assert output['features'].shape[0] == num_voxels
    assert output['coords'].shape == (num_voxels, 3)
    assert output['batch_inds'].shape == (num_voxels,)
    
    print(f"✅ 输出形状正确:")
    print(f"   SDF: {output['sdf'].shape}")
    print(f"   占用: {output['occupancy'].shape}")
    print(f"   特征: {output['features'].shape}")
    print(f"   坐标: {output['coords'].shape}")
    print(f"   批次索引: {output['batch_inds'].shape}")
    print(f"   体素数量: {output['num_voxels']}")


def test_sparse_single_frame_with_history(stream_model, sample_sequence_data):
    """测试稀疏单帧推理（有历史状态）"""
    print("测试稀疏单帧推理（有历史状态）...")
    
    # 获取两帧数据
    images1 = sample_sequence_data['images_seq'][0]
    poses1 = sample_sequence_data['poses_seq'][0]
    intrinsics1 = sample_sequence_data['intrinsics_seq'][0]
    
    images2 = sample_sequence_data['images_seq'][1]
    poses2 = sample_sequence_data['poses_seq'][1]
    intrinsics2 = sample_sequence_data['intrinsics_seq'][1]
    
    # 重置状态并处理第一帧
    stream_model.historical_state = None
    stream_model.historical_pose = None
    output1 = stream_model(images1, poses1, intrinsics1, reset_state=True)
    
    # 处理第二帧（应该有历史状态）
    output2 = stream_model(images2, poses2, intrinsics2, reset_state=False)
    
    # 验证输出结构
    required_keys = ['sdf', 'occupancy', 'features', 'coords', 'batch_inds', 'num_voxels']
    for key in required_keys:
        assert key in output2, f"输出2缺少键: {key}"
    
    # 验证形状一致
    assert output2['sdf'].shape == output1['sdf'].shape
    assert output2['occupancy'].shape == output1['occupancy'].shape
    assert output2['features'].shape == output1['features'].shape
    assert output2['coords'].shape == output1['coords'].shape
    assert output2['batch_inds'].shape == output1['batch_inds'].shape
    
    print(f"✅ 有历史状态推理完成:")
    print(f"   第一帧体素: {output1['num_voxels']}")
    print(f"   第二帧体素: {output2['num_voxels']}")
    print(f"   形状一致性检查通过")


def test_sparse_sequence_inference(stream_model, sample_sequence_data):
    """测试稀疏序列推理"""
    print("测试稀疏序列推理...")
    
    # 序列推理
    outputs = stream_model.forward_sequence(
        sample_sequence_data['images_seq'],
        sample_sequence_data['poses_seq'],
        sample_sequence_data['intrinsics_seq']
    )
    
    # 验证输出数量
    assert len(outputs) == sample_sequence_data['seq_length']
    print(f"✅ 生成 {len(outputs)} 帧输出")
    
    # 验证每帧输出
    for i, output in enumerate(outputs):
        assert 'sdf' in output
        assert 'occupancy' in output
        assert 'features' in output
        assert 'coords' in output
        assert 'batch_inds' in output
        assert 'num_voxels' in output
        
        num_voxels = output['num_voxels']
        print(f"  第 {i+1} 帧: {num_voxels} 个体素")
        
        # 验证形状一致性
        assert output['sdf'].shape == (num_voxels, 1)
        assert output['occupancy'].shape == (num_voxels, 1)
        assert output['features'].shape[0] == num_voxels
        assert output['coords'].shape == (num_voxels, 3)
        assert output['batch_inds'].shape == (num_voxels,)
    
    print("✅ 序列推理测试通过")


def test_sparse_state_reset(stream_model, sample_sequence_data):
    """测试稀疏状态重置"""
    print("测试稀疏状态重置...")
    
    # 获取数据
    images = sample_sequence_data['images_seq'][0]
    poses = sample_sequence_data['poses_seq'][0]
    intrinsics = sample_sequence_data['intrinsics_seq'][0]
    
    # 第一次推理（应该创建历史状态）
    output1 = stream_model(images, poses, intrinsics, reset_state=True)
    
    # 验证历史状态存在
    assert stream_model.historical_state is not None
    assert stream_model.historical_pose is not None
    
    # 保存历史状态
    saved_state = stream_model.historical_state
    saved_pose = stream_model.historical_pose
    
    # 重置状态
    stream_model.historical_state = None
    stream_model.historical_pose = None
    
    # 验证状态已重置
    assert stream_model.historical_state is None
    assert stream_model.historical_pose is None
    
    # 再次推理（应该重新创建历史状态）
    output2 = stream_model(images, poses, intrinsics, reset_state=False)
    
    # 验证历史状态重新创建
    assert stream_model.historical_state is not None
    assert stream_model.historical_pose is not None
    
    print("✅ 状态重置测试通过")


def test_sparse_gradient_flow(stream_model, sample_sequence_data):
    """测试稀疏梯度流（简化版本：只测试第一帧）"""
    print("测试稀疏梯度流（简化版本）...")
    
    # 获取数据
    images = sample_sequence_data['images_seq'][0]
    poses = sample_sequence_data['poses_seq'][0]
    intrinsics = sample_sequence_data['intrinsics_seq'][0]
    
    # 创建需要梯度的输入
    images.requires_grad_(True)
    
    # 推理（重置状态）
    output = stream_model(images, poses, intrinsics, reset_state=True)
    
    # 计算损失和梯度
    loss = output['sdf'].sum() + output['occupancy'].sum()
    loss.backward()
    
    # 验证梯度存在
    assert images.grad is not None
    assert not torch.all(images.grad == 0)
    
    grad_norm = torch.norm(images.grad)
    print(f"✅ 梯度存在且非零，梯度范数: {grad_norm:.6f}")


def test_sparse_batch_consistency(stream_model):
    """测试稀疏批次一致性"""
    print("测试稀疏批次一致性...")
    
    # 测试不同batch size
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        # 创建数据
        images = torch.randn(batch_size, 3, 256, 256)
        poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 重置状态
        stream_model.historical_state = None
        stream_model.historical_pose = None
        
        # 推理
        output = stream_model(images, poses, intrinsics, reset_state=True)
        
        # 验证批次索引正确
        batch_inds = output['batch_inds']
        unique_batch_inds = torch.unique(batch_inds)
        
        assert len(unique_batch_inds) == batch_size
        assert torch.all(unique_batch_inds == torch.arange(batch_size))
        
        # 验证每个批次的体素数量大致相等
        num_voxels = output['num_voxels']
        expected_voxels_per_batch = num_voxels // batch_size
        
        for i in range(batch_size):
            batch_mask = batch_inds == i
            batch_voxels = torch.sum(batch_mask).item()
            # 允许一定误差
            assert abs(batch_voxels - expected_voxels_per_batch) <= 10
        
        print(f"✅ Batch size {batch_size} 处理正确")


def test_sparse_coordinate_range(stream_model, sample_sequence_data):
    """测试稀疏坐标范围"""
    print("测试稀疏坐标范围...")
    
    # 获取数据
    images = sample_sequence_data['images_seq'][0]
    poses = sample_sequence_data['poses_seq'][0]
    intrinsics = sample_sequence_data['intrinsics_seq'][0]
    
    # 推理
    output = stream_model(images, poses, intrinsics, reset_state=True)
    
    # 获取坐标
    coords = output['coords']
    
    # 验证坐标在物理范围内
    voxel_size = stream_model.voxel_size
    crop_size = torch.tensor(stream_model.crop_size).float()
    max_coord = crop_size * voxel_size
    
    # 坐标应该在 [0, max_coord) 范围内
    assert torch.all(coords >= 0)
    assert torch.all(coords[:, 0] < max_coord[0])
    assert torch.all(coords[:, 1] < max_coord[1])
    assert torch.all(coords[:, 2] < max_coord[2])
    
    print(f"✅ 坐标范围正确:")
    print(f"   最小坐标: {coords.min(dim=0)[0]}")
    print(f"   最大坐标: {coords.max(dim=0)[0]}")
    print(f"   理论最大: {max_coord}")


if __name__ == "__main__":
    """运行所有测试"""
    print("=" * 50)
    print("运行稀疏流式SDFFormer单元测试")
    print("=" * 50)
    
    # 创建fixture
    model = stream_model()
    data = sample_sequence_data()
    
    # 运行测试
    test_sparse_single_frame_no_history(model, data)
    test_sparse_single_frame_with_history(model, data)
    test_sparse_sequence_inference(model, data)
    test_sparse_state_reset(model, data)
    test_sparse_gradient_flow(model, data)
    test_sparse_batch_consistency(model)
    test_sparse_coordinate_range(model, data)
    
    print("=" * 50)
    print("所有稀疏流式SDFFormer测试通过！ ✅")
    print("=" * 50)