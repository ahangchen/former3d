"""
姿态投影模块单元测试
"""

import torch
import pytest
import numpy as np
from former3d.pose_projection import PoseProjection


@pytest.fixture
def projector():
    """创建投影模块fixture"""
    return PoseProjection(voxel_size=0.0625, crop_size=(48, 48, 24))


@pytest.fixture
def sample_data():
    """创建测试数据fixture"""
    batch_size = 2
    channels = 64
    
    # 创建特征
    features = torch.randn(batch_size, channels, 24, 48, 48)
    
    # 创建恒等变换pose
    identity_pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 创建平移变换pose
    translation_pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    translation_pose[:, :3, 3] = torch.tensor([0.1, 0.0, 0.0])  # X方向平移0.1米
    
    return {
        'features': features,
        'identity_pose': identity_pose,
        'translation_pose': translation_pose,
        'batch_size': batch_size,
        'channels': channels
    }


def test_identity_transform(projector, sample_data):
    """测试恒等变换（pose相同）"""
    print("测试恒等变换...")
    
    # 投影（应该不变）
    projected_state = projector(
        {'features': sample_data['features']},
        sample_data['identity_pose'],
        sample_data['identity_pose']
    )
    
    # 验证结果形状
    assert projected_state['features'].shape == sample_data['features'].shape
    print(f"✅ 特征形状正确: {projected_state['features'].shape}")
    
    # 验证结果接近原始（允许微小插值误差）
    max_diff = torch.max(torch.abs(projected_state['features'] - sample_data['features']))
    print(f"最大差异: {max_diff.item():.6f}")
    
    # 使用更宽松的容差，因为grid_sample可能有微小插值误差
    assert torch.allclose(
        projected_state['features'], 
        sample_data['features'], 
        rtol=1e-3, atol=1e-4
    )
    print("✅ 恒等变换测试通过")


def test_simple_translation(projector, sample_data):
    """测试简单平移"""
    print("测试平移变换...")
    
    # 投影
    projected_state = projector(
        {'features': sample_data['features']},
        sample_data['identity_pose'],
        sample_data['translation_pose']
    )
    
    # 验证结果形状
    assert projected_state['features'].shape == sample_data['features'].shape
    print(f"✅ 平移后特征形状正确: {projected_state['features'].shape}")
    
    # 验证结果与原始不同（因为平移了）
    assert not torch.allclose(
        projected_state['features'], 
        sample_data['features'], 
        rtol=1e-3, atol=1e-4
    )
    print("✅ 平移变换测试通过")


def test_gradient_flow(projector, sample_data):
    """测试梯度存在性"""
    print("测试梯度流...")
    
    # 创建需要梯度的特征
    features = sample_data['features'].clone().requires_grad_(True)
    
    # 投影
    projected_state = projector(
        {'features': features},
        sample_data['identity_pose'],
        sample_data['translation_pose']
    )
    
    # 计算损失和梯度
    loss = projected_state['features'].sum()
    loss.backward()
    
    # 验证梯度存在
    assert features.grad is not None
    assert not torch.all(features.grad == 0)
    print(f"✅ 梯度存在且非零，梯度范数: {torch.norm(features.grad):.6f}")


def test_batch_processing(projector):
    """测试批量处理"""
    print("测试批量处理...")
    
    # 创建不同batch size的数据
    batch_sizes = [1, 2, 4]
    
    for batch_size in batch_sizes:
        features = torch.randn(batch_size, 64, 24, 48, 48)
        pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        
        projected_state = projector(
            {'features': features},
            pose,
            pose
        )
        
        assert projected_state['features'].shape[0] == batch_size
        print(f"✅ Batch size {batch_size} 处理正确")
    
    print("✅ 批量处理测试通过")


def test_coordinate_mapping_shape(projector, sample_data):
    """测试坐标映射形状"""
    print("测试坐标映射形状...")
    
    # 计算坐标映射
    coordinate_mapping = projector.compute_coordinate_mapping(
        sample_data['identity_pose'],
        sample_data['identity_pose']
    )
    
    # 验证形状 [batch, depth, height, width, 3]
    expected_shape = (
        sample_data['batch_size'], 
        24,  # depth (z)
        48,  # height (y) 
        48,  # width (x)
        3    # coordinates
    )
    
    assert coordinate_mapping.shape == expected_shape
    print(f"✅ 坐标映射形状正确: {coordinate_mapping.shape}")
    
    # 验证坐标范围在[-1, 1]内
    assert torch.all(coordinate_mapping >= -1.0) and torch.all(coordinate_mapping <= 1.0)
    print("✅ 坐标映射范围正确")


def test_rotation_transform(projector, sample_data):
    """测试旋转变换"""
    print("测试旋转变换...")
    
    # 创建旋转变换（绕Z轴旋转45度）
    angle = np.pi / 4  # 45度
    rotation_pose = torch.eye(4).unsqueeze(0).repeat(sample_data['batch_size'], 1, 1)
    
    # 绕Z轴旋转
    rotation_pose[:, 0, 0] = torch.cos(angle)
    rotation_pose[:, 0, 1] = -torch.sin(angle)
    rotation_pose[:, 1, 0] = torch.sin(angle)
    rotation_pose[:, 1, 1] = torch.cos(angle)
    
    # 投影
    projected_state = projector(
        {'features': sample_data['features']},
        sample_data['identity_pose'],
        rotation_pose
    )
    
    # 验证结果形状
    assert projected_state['features'].shape == sample_data['features'].shape
    print(f"✅ 旋转后特征形状正确: {projected_state['features'].shape}")
    
    # 验证结果与原始不同
    assert not torch.allclose(
        projected_state['features'], 
        sample_data['features'], 
        rtol=1e-3, atol=1e-4
    )
    print("✅ 旋转变换测试通过")


def test_empty_state(projector, sample_data):
    """测试空状态处理"""
    print("测试空状态处理...")
    
    # 空状态字典
    empty_state = {}
    
    # 应该能正常处理空状态
    projected_state = projector(
        empty_state,
        sample_data['identity_pose'],
        sample_data['identity_pose']
    )
    
    assert isinstance(projected_state, dict)
    assert len(projected_state) == 1  # 只有coordinate_mapping
    assert 'coordinate_mapping' in projected_state
    print("✅ 空状态处理测试通过")


if __name__ == "__main__":
    """运行所有测试"""
    print("=" * 50)
    print("运行姿态投影模块单元测试")
    print("=" * 50)
    
    # 创建fixture
    proj = projector()
    data = sample_data()
    
    # 运行测试
    test_identity_transform(proj, data)
    test_simple_translation(proj, data)
    test_gradient_flow(proj, data)
    test_batch_processing(proj)
    test_coordinate_mapping_shape(proj, data)
    test_rotation_transform(proj, data)
    test_empty_state(proj, data)
    
    print("=" * 50)
    print("所有测试通过！ ✅")
    print("=" * 50)