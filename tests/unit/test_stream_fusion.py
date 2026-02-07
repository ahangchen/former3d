"""
Cross-Attention融合模块单元测试
"""

import torch
import pytest
import numpy as np
from former3d.stream_fusion import (
    LocalCrossAttention, 
    HierarchicalAttention, 
    StreamCrossAttention
)


@pytest.fixture
def sample_attention_data():
    """创建注意力测试数据fixture"""
    N_current = 50
    N_historical = 100
    feature_dim = 64
    
    # 创建特征
    current_feats = torch.randn(N_current, feature_dim)
    historical_feats = torch.randn(N_historical, feature_dim)
    
    # 创建坐标（确保有些在局部半径内，有些在半径外）
    current_coords = torch.randint(0, 20, (N_current, 3))
    historical_coords = torch.randint(0, 20, (N_historical, 3))
    
    # 图像特征（可选）
    img_feats = torch.randn(N_current, feature_dim)
    
    return {
        'current_feats': current_feats,
        'historical_feats': historical_feats,
        'current_coords': current_coords,
        'historical_coords': historical_coords,
        'img_feats': img_feats,
        'N_current': N_current,
        'N_historical': N_historical,
        'feature_dim': feature_dim
    }


def test_local_attention_shape(sample_attention_data):
    """测试局部注意力输出形状"""
    print("测试局部注意力形状...")
    
    attention = LocalCrossAttention(
        feature_dim=sample_attention_data['feature_dim'],
        num_heads=8,
        local_radius=5
    )
    
    output = attention(
        sample_attention_data['current_feats'],
        sample_attention_data['historical_feats'],
        sample_attention_data['current_coords'],
        sample_attention_data['historical_coords']
    )
    
    expected_shape = (sample_attention_data['N_current'], sample_attention_data['feature_dim'])
    assert output.shape == expected_shape
    print(f"✅ 局部注意力输出形状正确: {output.shape}")


def test_local_attention_gradients(sample_attention_data):
    """测试局部注意力梯度"""
    print("测试局部注意力梯度...")
    
    attention = LocalCrossAttention(feature_dim=64)
    
    # 创建需要梯度的特征
    current_feats = sample_attention_data['current_feats'].clone().requires_grad_(True)
    historical_feats = sample_attention_data['historical_feats'].clone().requires_grad_(True)
    
    output = attention(
        current_feats,
        historical_feats,
        sample_attention_data['current_coords'],
        sample_attention_data['historical_coords']
    )
    
    # 计算损失和梯度
    loss = output.sum()
    loss.backward()
    
    # 验证梯度存在
    assert current_feats.grad is not None
    assert historical_feats.grad is not None
    assert not torch.all(current_feats.grad == 0)
    assert not torch.all(historical_feats.grad == 0)
    
    print(f"✅ 梯度存在，当前特征梯度范数: {torch.norm(current_feats.grad):.6f}")
    print(f"✅ 历史特征梯度范数: {torch.norm(historical_feats.grad):.6f}")


def test_local_mask_construction():
    """测试局部掩码构建"""
    print("测试局部掩码构建...")
    
    attention = LocalCrossAttention(feature_dim=64, local_radius=3)
    
    # 创建测试坐标
    current_coords = torch.tensor([
        [0, 0, 0],
        [5, 5, 5]
    ])
    
    historical_coords = torch.tensor([
        [0, 0, 0],  # 距离0，在半径内
        [1, 0, 0],  # 距离1，在半径内
        [4, 0, 0],  # 距离4，在半径外
        [5, 5, 5],  # 距离0，在半径内
        [8, 5, 5]   # 距离3，在半径边界
    ])
    
    local_mask = attention.build_local_mask(current_coords, historical_coords)
    
    # 检查掩码形状
    assert local_mask.shape == (2, 5)
    
    # 检查具体值
    # 第一个当前体素（0,0,0）
    assert local_mask[0, 0] == True  # (0,0,0) 距离0
    assert local_mask[0, 1] == True  # (1,0,0) 距离1
    assert local_mask[0, 2] == False  # (4,0,0) 距离4 > 3
    assert local_mask[0, 3] == False  # (5,5,5) 距离√75 > 3
    assert local_mask[0, 4] == False  # (8,5,5) 距离√73 > 3
    
    # 第二个当前体素（5,5,5）
    assert local_mask[1, 0] == False  # (0,0,0) 距离√75 > 3
    assert local_mask[1, 1] == False  # (1,0,0) 距离√66 > 3
    assert local_mask[1, 2] == False  # (4,0,0) 距离√66 > 3
    assert local_mask[1, 3] == True   # (5,5,5) 距离0
    assert local_mask[1, 4] == True   # (8,5,5) 距离3
    
    print("✅ 局部掩码构建测试通过")


def test_hierarchical_attention_shape(sample_attention_data):
    """测试分层注意力输出形状"""
    print("测试分层注意力形状...")
    
    attention = HierarchicalAttention(
        feature_dim=sample_attention_data['feature_dim'],
        num_levels=3
    )
    
    output = attention(
        sample_attention_data['current_feats'],
        sample_attention_data['historical_feats']
    )
    
    expected_shape = (sample_attention_data['N_current'], sample_attention_data['feature_dim'])
    assert output.shape == expected_shape
    print(f"✅ 分层注意力输出形状正确: {output.shape}")


def test_stream_cross_attention_with_hierarchical(sample_attention_data):
    """测试带分层注意力的流式交叉注意力"""
    print("测试带分层注意力的流式交叉注意力...")
    
    fusion = StreamCrossAttention(
        feature_dim=sample_attention_data['feature_dim'],
        num_heads=8,
        local_radius=5,
        hierarchical=True
    )
    
    output = fusion(
        sample_attention_data['current_feats'],
        sample_attention_data['historical_feats'],
        sample_attention_data['current_coords'],
        sample_attention_data['historical_coords'],
        sample_attention_data['img_feats']
    )
    
    expected_shape = (sample_attention_data['N_current'], sample_attention_data['feature_dim'])
    assert output.shape == expected_shape
    print(f"✅ 带分层注意力输出形状正确: {output.shape}")


def test_stream_cross_attention_without_hierarchical(sample_attention_data):
    """测试不带分层注意力的流式交叉注意力"""
    print("测试不带分层注意力的流式交叉注意力...")
    
    fusion = StreamCrossAttention(
        feature_dim=sample_attention_data['feature_dim'],
        num_heads=8,
        local_radius=5,
        hierarchical=False
    )
    
    output = fusion(
        sample_attention_data['current_feats'],
        sample_attention_data['historical_feats'],
        sample_attention_data['current_coords'],
        sample_attention_data['historical_coords']
    )
    
    expected_shape = (sample_attention_data['N_current'], sample_attention_data['feature_dim'])
    assert output.shape == expected_shape
    print(f"✅ 不带分层注意力输出形状正确: {output.shape}")


def test_attention_with_different_radii():
    """测试不同局部半径的影响"""
    print("测试不同局部半径...")
    
    N_current = 10
    N_historical = 20
    feature_dim = 32
    
    current_feats = torch.randn(N_current, feature_dim)
    historical_feats = torch.randn(N_historical, feature_dim)
    current_coords = torch.randint(0, 10, (N_current, 3))
    historical_coords = torch.randint(0, 10, (N_historical, 3))
    
    # 测试不同半径
    radii = [1, 3, 5, 10]
    outputs = []
    
    for radius in radii:
        attention = LocalCrossAttention(
            feature_dim=feature_dim,
            num_heads=4,
            local_radius=radius
        )
        
        output = attention(
            current_feats, historical_feats,
            current_coords, historical_coords
        )
        outputs.append(output)
    
    # 验证不同半径的输出不同
    for i in range(len(radii) - 1):
        assert not torch.allclose(outputs[i], outputs[i + 1], rtol=1e-4)
    
    print(f"✅ 不同局部半径({radii})产生不同输出")


def test_multihead_attention_heads():
    """测试多头注意力"""
    print("测试多头注意力...")
    
    feature_dim = 64
    num_heads = 8
    
    attention = LocalCrossAttention(
        feature_dim=feature_dim,
        num_heads=num_heads,
        local_radius=3
    )
    
    # 验证头维度计算
    assert attention.head_dim == feature_dim // num_heads
    print(f"✅ 头维度计算正确: {attention.head_dim}")
    
    # 验证特征维度能被头数整除
    assert feature_dim % num_heads == 0
    print(f"✅ 特征维度{feature_dim}能被头数{num_heads}整除")


def test_dropout_effect(sample_attention_data):
    """测试dropout效果"""
    print("测试dropout效果...")
    
    # 训练模式下的注意力（启用dropout）
    attention_train = LocalCrossAttention(
        feature_dim=sample_attention_data['feature_dim'],
        num_heads=8,
        local_radius=5,
        dropout=0.5
    )
    attention_train.train()
    
    # 评估模式下的注意力（禁用dropout）
    attention_eval = LocalCrossAttention(
        feature_dim=sample_attention_data['feature_dim'],
        num_heads=8,
        local_radius=5,
        dropout=0.5
    )
    attention_eval.eval()
    
    # 相同输入
    current_feats = sample_attention_data['current_feats'].clone()
    historical_feats = sample_attention_data['historical_feats'].clone()
    
    # 多次运行训练模式（由于dropout随机性，输出应该不同）
    train_outputs = []
    for _ in range(5):
        output = attention_train(
            current_feats, historical_feats,
            sample_attention_data['current_coords'],
            sample_attention_data['historical_coords']
        )
        train_outputs.append(output)
    
    # 评估模式输出（应该一致）
    eval_output = attention_eval(
        current_feats, historical_feats,
        sample_attention_data['current_coords'],
        sample_attention_data['historical_coords']
    )
    
    # 检查训练模式输出由于dropout而有差异
    has_variation = False
    for i in range(len(train_outputs) - 1):
        if not torch.allclose(train_outputs[i], train_outputs[i + 1], rtol=1e-4):
            has_variation = True
            break
    
    assert has_variation, "训练模式下dropout应产生输出变化"
    print("✅ Dropout在训练模式下产生随机性")
    
    # 评估模式多次运行应该一致
    eval_output2 = attention_eval(
        current_feats, historical_feats,
        sample_attention_data['current_coords'],
        sample_attention_data['historical_coords']
    )
    assert torch.allclose(eval_output, eval_output2, rtol=1e-6)
    print("✅ 评估模式输出一致")


def test_residual_connection():
    """测试残差连接"""
    print("测试残差连接...")
    
    fusion = StreamCrossAttention(
        feature_dim=64,
        num_heads=8,
        local_radius=3,
        hierarchical=True
    )
    
    # 创建数据
    N_current = 10
    current_feats = torch.randn(N_current, 64)
    historical_feats = torch.randn(20, 64)
    current_coords = torch.randint(0, 10, (N_current, 3))
    historical_coords = torch.randint(0, 10, (20, 3))
    
    output = fusion(
        current_feats, historical_feats,
        current_coords, historical_coords
    )
    
    # 输出应该与输入形状相同
    assert output.shape == current_feats.shape
    print("✅ 残差连接保持形状")


if __name__ == "__main__":
    """运行所有测试"""
    print("=" * 50)
    print("运行Cross-Attention融合模块单元测试")
    print("=" * 50)
    
    # 创建测试数据
    data = sample_attention_data()
    
    # 运行测试
    test_local_attention_shape(data)
    test_local_attention_gradients(data)
    test_local_mask_construction()
    test_hierarchical_attention_shape(data)
    test_stream_cross_attention_with_hierarchical(data)
    test_stream_cross_attention_without_hierarchical(data)
    test_attention_with_different_radii()
    test_multihead_attention_heads()
    test_dropout_effect(data)
    test_residual_connection()
    
    print("=" * 50)
    print("所有测试通过！ ✅")
    print("=" * 50)