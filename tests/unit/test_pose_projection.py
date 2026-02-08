"""
姿态投影模块单元测试（稀疏表示版本）
"""

import torch
import pytest
import numpy as np
from former3d.pose_projection import PoseProjection


@pytest.fixture
def projector():
    """创建投影模块fixture"""
    return PoseProjection(voxel_size=0.0625, crop_size=(48, 96, 96))


@pytest.fixture
def sample_sparse_data():
    """创建稀疏测试数据fixture"""
    batch_size = 2
    num_voxels = 1000
    channels = 64
    
    # 创建体素坐标（物理坐标，单位：米）
    # 在裁剪范围内生成坐标
    crop_size = (48, 96, 96)
    voxel_size = 0.0625
    
    # 生成在裁剪范围内的坐标
    max_coord = torch.tensor(crop_size).float() * voxel_size
    voxel_coords = torch.rand(num_voxels, 3) * max_coord
    
    # 批次索引
    voxel_batch_inds = torch.randint(0, batch_size, (num_voxels,))
    
    # 创建特征
    features = torch.randn(num_voxels, channels)
    
    # 创建恒等变换pose
    identity_pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 创建平移变换pose
    translation_pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    translation_pose[:, :3, 3] = torch.tensor([0.1, 0.0, 0.0])  # X方向平移0.1米
    
    return {
        'voxel_coords': voxel_coords,
        'voxel_batch_inds': voxel_batch_inds,
        'features': features,
        'identity_pose': identity_pose,
        'translation_pose': translation_pose,
        'batch_size': batch_size,
        'channels': channels,
        'num_voxels': num_voxels
    }


def test_sparse_identity_transform(projector, sample_sparse_data):
    """测试稀疏表示的恒等变换"""
    print("测试稀疏恒等变换...")
    
    # 创建历史状态
    historical_state = {
        'features': sample_sparse_data['features'],
        'coords': sample_sparse_data['voxel_coords'],
        'batch_inds': sample_sparse_data['voxel_batch_inds']
    }
    
    # 投影（应该不变）
    projected_state = projector(
        historical_state,
        sample_sparse_data['identity_pose'],
        sample_sparse_data['identity_pose']
    )
    
    # 验证结果形状
    assert projected_state['features'].shape == sample_sparse_data['features'].shape
    print(f"✅ 特征形状正确: {projected_state['features'].shape}")
    
    # 验证坐标不变
    assert torch.allclose(
        projected_state['coords'], 
        sample_sparse_data['voxel_coords'], 
        rtol=1e-4
    )
    print("✅ 坐标恒等变换测试通过")
    
    # 验证批次索引不变
    assert torch.all(projected_state['batch_inds'] == sample_sparse_data['voxel_batch_inds'])
    print("✅ 批次索引恒等变换测试通过")


def test_sparse_translation(projector, sample_sparse_data):
    """测试稀疏表示的平移变换"""
    print("测试稀疏平移变换...")
    
    # 创建历史状态
    historical_state = {
        'features': sample_sparse_data['features'],
        'coords': sample_sparse_data['voxel_coords'],
        'batch_inds': sample_sparse_data['voxel_batch_inds']
    }
    
    # 投影
    projected_state = projector(
        historical_state,
        sample_sparse_data['identity_pose'],
        sample_sparse_data['translation_pose']
    )
    
    # 验证结果形状
    assert projected_state['features'].shape == sample_sparse_data['features'].shape
    print(f"✅ 平移后特征形状正确: {projected_state['features'].shape}")
    
    # 验证坐标不变（因为当前坐标系平移了，但体素坐标是物理坐标）
    # 在恒等变换到平移变换的情况下，坐标应该保持不变
    assert torch.allclose(
        projected_state['coords'], 
        sample_sparse_data['voxel_coords'], 
        rtol=1e-4
    )
    print("✅ 坐标平移变换测试通过")
    
    # 验证掩码存在
    assert 'mask' in projected_state
    assert projected_state['mask'].shape == (sample_sparse_data['num_voxels'],)
    print(f"✅ 掩码形状正确: {projected_state['mask'].shape}")


def test_sparse_gradient_flow(projector, sample_sparse_data):
    """测试稀疏表示的梯度存在性"""
    print("测试稀疏梯度流...")
    
    # 创建需要梯度的特征
    features = sample_sparse_data['features'].clone().requires_grad_(True)
    
    # 创建历史状态
    historical_state = {
        'features': features,
        'coords': sample_sparse_data['voxel_coords'],
        'batch_inds': sample_sparse_data['voxel_batch_inds']
    }
    
    # 投影
    projected_state = projector(
        historical_state,
        sample_sparse_data['identity_pose'],
        sample_sparse_data['translation_pose']
    )
    
    # 计算损失和梯度
    loss = projected_state['features'].sum()
    loss.backward()
    
    # 验证梯度存在
    assert features.grad is not None
    assert not torch.all(features.grad == 0)
    print(f"✅ 梯度存在且非零，梯度范数: {torch.norm(features.grad):.6f}")


def test_sparse_batch_processing(projector):
    """测试稀疏表示的批量处理"""
    print("测试稀疏批量处理...")
    
    # 创建不同batch size的数据
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        num_voxels = 500
        channels = 64
        
        # 生成数据
        voxel_coords = torch.randn(num_voxels, 3) * 0.5
        voxel_batch_inds = torch.randint(0, batch_size, (num_voxels,))
        features = torch.randn(num_voxels, channels)
        pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        
        historical_state = {
            'features': features,
            'coords': voxel_coords,
            'batch_inds': voxel_batch_inds
        }
        
        projected_state = projector(
            historical_state,
            pose,
            pose
        )
        
        assert projected_state['features'].shape[0] == num_voxels
        assert projected_state['batch_inds'].max().item() < batch_size
        print(f"✅ Batch size {batch_size} 处理正确")
    
    print("✅ 稀疏批量处理测试通过")


def test_sparse_coordinate_mapping(projector, sample_sparse_data):
    """测试稀疏坐标映射"""
    print("测试稀疏坐标映射...")
    
    # 计算坐标映射
    coordinate_mapping = projector.compute_coordinate_mapping(
        sample_sparse_data['identity_pose'],
        sample_sparse_data['identity_pose'],
        sample_sparse_data['voxel_coords'],
        sample_sparse_data['voxel_batch_inds']
    )
    
    # 验证输出结构
    assert 'historical_coords' in coordinate_mapping
    assert 'batch_inds' in coordinate_mapping
    assert 'mask' in coordinate_mapping
    
    # 验证形状
    assert coordinate_mapping['historical_coords'].shape == sample_sparse_data['voxel_coords'].shape
    assert coordinate_mapping['batch_inds'].shape == sample_sparse_data['voxel_batch_inds'].shape
    assert coordinate_mapping['mask'].shape == (sample_sparse_data['num_voxels'],)
    
    print(f"✅ 历史坐标形状: {coordinate_mapping['historical_coords'].shape}")
    print(f"✅ 批次索引形状: {coordinate_mapping['batch_inds'].shape}")
    print(f"✅ 掩码形状: {coordinate_mapping['mask'].shape}")
    
    # 在恒等变换下，历史坐标应该等于当前坐标
    assert torch.allclose(
        coordinate_mapping['historical_coords'], 
        sample_sparse_data['voxel_coords'], 
        rtol=1e-4
    )
    print("✅ 坐标映射恒等变换测试通过")


def test_sparse_rotation(projector, sample_sparse_data):
    """测试稀疏表示的旋转变换"""
    print("测试稀疏旋转变换...")
    
    # 创建旋转变换（绕Z轴旋转45度）
    angle = np.pi / 4  # 45度
    rotation_pose = torch.eye(4).unsqueeze(0).repeat(sample_sparse_data['batch_size'], 1, 1)
    
    # 绕Z轴旋转
    angle_tensor = torch.tensor(angle)
    rotation_pose[:, 0, 0] = torch.cos(angle_tensor)
    rotation_pose[:, 0, 1] = -torch.sin(angle_tensor)
    rotation_pose[:, 1, 0] = torch.sin(angle_tensor)
    rotation_pose[:, 1, 1] = torch.cos(angle_tensor)
    
    # 创建历史状态
    historical_state = {
        'features': sample_sparse_data['features'],
        'coords': sample_sparse_data['voxel_coords'],
        'batch_inds': sample_sparse_data['voxel_batch_inds']
    }
    
    # 投影
    projected_state = projector(
        historical_state,
        sample_sparse_data['identity_pose'],
        rotation_pose
    )
    
    # 验证结果形状
    assert projected_state['features'].shape == sample_sparse_data['features'].shape
    print(f"✅ 旋转后特征形状正确: {projected_state['features'].shape}")
    
    # 验证坐标不变（物理坐标在旋转变换下应该变化）
    # 注意：这里坐标应该变化，因为坐标系旋转了
    assert not torch.allclose(
        projected_state['coords'], 
        sample_sparse_data['voxel_coords'], 
        rtol=1e-4
    )
    print("✅ 坐标旋转变换测试通过")


def test_sparse_missing_coords(projector, sample_sparse_data):
    """测试缺少坐标信息的情况"""
    print("测试缺少坐标信息...")
    
    # 创建缺少坐标的历史状态
    historical_state = {
        'features': sample_sparse_data['features']
        # 缺少 'coords' 和 'batch_inds'
    }
    
    # 应该抛出错误
    with pytest.raises(ValueError, match="历史状态必须包含 'coords'"):
        projector(
            historical_state,
            sample_sparse_data['identity_pose'],
            sample_sparse_data['identity_pose']
        )
    
    print("✅ 缺少坐标信息测试通过")


def test_sparse_valid_mask(projector, sample_sparse_data):
    """测试有效掩码"""
    print("测试有效掩码...")
    
    # 创建历史状态
    historical_state = {
        'features': sample_sparse_data['features'],
        'coords': sample_sparse_data['voxel_coords'],
        'batch_inds': sample_sparse_data['voxel_batch_inds']
    }
    
    # 投影
    projected_state = projector(
        historical_state,
        sample_sparse_data['identity_pose'],
        sample_sparse_data['identity_pose']
    )
    
    # 验证掩码
    mask = projected_state['mask']
    assert mask.dtype == torch.bool
    assert mask.shape == (sample_sparse_data['num_voxels'],)
    
    # 所有在裁剪范围内的坐标都应该有效
    # 计算应该在裁剪范围内的坐标
    crop_size = torch.tensor(projector.crop_size).float()
    voxel_size = projector.voxel_size
    max_coord = crop_size * voxel_size
    
    in_range_x = (sample_sparse_data['voxel_coords'][:, 0] >= 0) & (sample_sparse_data['voxel_coords'][:, 0] < max_coord[0])
    in_range_y = (sample_sparse_data['voxel_coords'][:, 1] >= 0) & (sample_sparse_data['voxel_coords'][:, 1] < max_coord[1])
    in_range_z = (sample_sparse_data['voxel_coords'][:, 2] >= 0) & (sample_sparse_data['voxel_coords'][:, 2] < max_coord[2])
    expected_mask = in_range_x & in_range_y & in_range_z
    
    # 验证掩码正确
    assert torch.all(mask == expected_mask)
    print(f"✅ 有效掩码测试通过，有效体素: {mask.sum().item()}/{sample_sparse_data['num_voxels']}")


if __name__ == "__main__":
    """运行所有测试"""
    print("=" * 50)
    print("运行稀疏姿态投影模块单元测试")
    print("=" * 50)
    
    # 创建fixture
    proj = projector()
    data = sample_sparse_data()
    
    # 运行测试
    test_sparse_identity_transform(proj, data)
    test_sparse_translation(proj, data)
    test_sparse_gradient_flow(proj, data)
    test_sparse_batch_processing(proj)
    test_sparse_coordinate_mapping(proj, data)
    test_sparse_rotation(proj, data)
    test_sparse_missing_coords(proj, data)
    test_sparse_valid_mask(proj, data)
    
    print("=" * 50)
    print("所有稀疏表示测试通过！ ✅")
    print("=" * 50)