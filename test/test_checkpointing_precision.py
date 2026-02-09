#!/usr/bin/env python3
"""
测试checkpointing的数值精度
验证使用gradient checkpointing不会影响模型的数值输出
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from former3d.stream_fusion import StreamCrossAttention

def test_checkpointing_precision():
    """测试checkpointing不影响数值精度"""

    print("="*70)
    print("测试checkpointing数值精度")
    print("="*70)

    # 设置随机种子以确保可重复性
    torch.manual_seed(42)

    # 创建模型配置
    feature_dim = 128
    num_heads = 4
    local_radius = 100.0  # 增大半径以确保有邻域

    # 创建输入数据
    N_current = 100
    N_historical = 500

    current_feats = torch.randn(N_current, feature_dim)
    historical_feats = torch.randn(N_historical, feature_dim)
    current_coords = torch.randn(N_current, 3) * 10
    historical_coords = torch.randn(N_historical, 3) * 10

    print(f"\n输入配置：")
    print(f"  - 当前体素数: {N_current}")
    print(f"  - 历史体素数: {N_historical}")
    print(f"  - 特征维度: {feature_dim}")
    print(f"  - 注意力头数: {num_heads}")
    print(f"  - 局部半径: {local_radius}")

    # 创建不使用checkpointing的模型
    print(f"\n创建不使用checkpointing的模型...")
    model_original = StreamCrossAttention(
        feature_dim=feature_dim,
        num_heads=num_heads,
        local_radius=local_radius,
        hierarchical=False,  # 简化测试，只测试局部注意力
        dropout=0.0,  # 禁用dropout以确保数值一致
        use_checkpoint=False
    )

    # 创建使用checkpointing的模型（加载相同的权重）
    print(f"创建使用checkpointing的模型...")
    model_checkpointed = StreamCrossAttention(
        feature_dim=feature_dim,
        num_heads=num_heads,
        local_radius=local_radius,
        hierarchical=False,
        dropout=0.0,
        use_checkpoint=True
    )
    model_checkpointed.load_state_dict(model_original.state_dict())

    # 设置为eval模式（禁用dropout和batch norm）
    model_original.eval()
    model_checkpointed.eval()

    # 手动设置requires_grad以启用梯度计算
    current_feats.requires_grad = True
    historical_feats.requires_grad = True

    # 前向传播 - 不使用checkpointing
    print(f"\n前向传播 - 不使用checkpointing...")
    output_original = model_original(
        current_feats,
        historical_feats,
        current_coords,
        historical_coords
    )

    # 前向传播 - 使用checkpointing
    print(f"前向传播 - 使用checkpointing...")
    output_checkpointed = model_checkpointed(
        current_feats,
        historical_feats,
        current_coords,
        historical_coords
    )

    # 检查输出形状
    print(f"\n输出形状：")
    print(f"  - 原始方法: {output_original.shape}")
    print(f"  - Checkpointing: {output_checkpointed.shape}")
    assert output_original.shape == output_checkpointed.shape, "输出形状不匹配"

    # 计算数值差异
    diff = torch.abs(output_original - output_checkpointed)

    print(f"\n数值差异分析：")
    print(f"  - 最大差异: {diff.max().item():.2e}")
    print(f"  - 平均差异: {diff.mean().item():.2e}")
    print(f"  - 均方根误差: {torch.sqrt((diff ** 2).mean()).item():.2e}")

    # 验证数值精度
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    if max_diff < 1e-5:  # 降低阈值到1e-5
        print(f"\n✅ 测试通过：数值差异在可接受范围内（< 1e-5）")
        print(f"   最大差异: {max_diff:.2e}")
        print(f"   平均差异: {mean_diff:.2e}")
    else:
        print(f"\n❌ 测试失败：数值差异过大")
        print(f"   最大差异: {max_diff:.2e} (应该 < 1e-5)")
        print(f"   平均差异: {mean_diff:.2e} (应该 < 1e-5)")
        raise AssertionError(f"数值差异过大: {max_diff}")

    # 额外检查：相对误差
    relative_error = diff / (torch.abs(output_original) + 1e-8)
    max_relative_error = relative_error.max().item()

    print(f"\n相对误差分析：")
    print(f"  - 最大相对误差: {max_relative_error:.2%}")

    if max_relative_error < 1e-4:  # 降低阈值到1e-4
        print(f"  ✅ 相对误差在可接受范围内（< 0.01%）")
    else:
        print(f"  ⚠️  相对误差较大: {max_relative_error:.2%}")

    print(f"\n{'='*70}")
    print("所有测试通过！✅")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_checkpointing_precision()
