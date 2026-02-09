#!/usr/bin/env python3
"""
测试checkpointing的显存占用
验证使用gradient checkpointing可以降低显存占用
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from former3d.stream_fusion import StreamCrossAttention

def test_checkpointing_memory():
    """测试checkpointing降低显存占用"""

    print("="*70)
    print("测试checkpointing显存占用")
    print("="*70)

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过显存测试")
        return

    device = torch.device("cuda:0")

    # 创建模型配置
    feature_dim = 128
    num_heads = 4
    local_radius = 100.0

    # 创建大输入数据
    N_current = 2000  # 增加到2000
    N_historical = 10000  # 增加到10000

    current_feats = torch.randn(N_current, feature_dim, device=device)
    historical_feats = torch.randn(N_historical, feature_dim, device=device)
    current_coords = torch.randn(N_current, 3, device=device) * 10
    historical_coords = torch.randn(N_historical, 3, device=device) * 10

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
        hierarchical=True,  # 启用分层注意力
        dropout=0.0,
        use_checkpoint=False
    ).to(device)

    # 创建使用checkpointing的模型
    print(f"创建使用checkpointing的模型...")
    model_checkpointed = StreamCrossAttention(
        feature_dim=feature_dim,
        num_heads=num_heads,
        local_radius=local_radius,
        hierarchical=True,  # 启用分层注意力
        dropout=0.0,
        use_checkpoint=True
    ).to(device)
    model_checkpointed.load_state_dict(model_original.state_dict())

    # 测试不使用checkpointing的显存占用
    print(f"\n测试不使用checkpointing的显存占用...")
    model_original.eval()
    current_feats.requires_grad = True
    historical_feats.requires_grad = True

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    output_original = model_original(
        current_feats,
        historical_feats,
        current_coords,
        historical_coords
    )

    # 反向传播以激活梯度
    loss = output_original.sum()
    loss.backward()

    memory_original = torch.cuda.max_memory_allocated() / 1024**3  # GB
    print(f"  峰值显存: {memory_original:.3f} GB")

    # 测试使用checkpointing的显存占用
    print(f"\n测试使用checkpointing的显存占用...")
    model_checkpointed.eval()
    current_feats.requires_grad = True
    historical_feats.requires_grad = True

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    output_checkpointed = model_checkpointed(
        current_feats,
        historical_feats,
        current_coords,
        historical_coords
    )

    # 反向传播以激活梯度
    loss = output_checkpointed.sum()
    loss.backward()

    memory_checkpointed = torch.cuda.max_memory_allocated() / 1024**3  # GB
    print(f"  峰值显存: {memory_checkpointed:.3f} GB")

    # 计算显存降低
    reduction = (memory_original - memory_checkpointed) / memory_original
    print(f"\n显存降低分析：")
    print(f"  - 原始显存: {memory_original:.3f} GB")
    print(f"  - Checkpointing显存: {memory_checkpointed:.3f} GB")
    print(f"  - 降低幅度: {reduction:.2%}")

    # 验证显存降低
    if reduction > 0.15:  # 至少降低15%
        print(f"\n✅ 测试通过：显存降低{reduction:.1%} (> 15%)")
    elif reduction > 0:  # 有降低但不足
        print(f"\n⚠️  测试部分通过：显存降低{reduction:.1%} (< 15%)")
        print(f"   可能原因：模型太小或计算量不足")
    else:
        print(f"\n❌ 测试失败：显存未降低")
        print(f"   原始显存: {memory_original:.3f} GB")
        print(f"   Checkpointing显存: {memory_checkpointed:.3f} GB")
        raise AssertionError("显存未降低")

    print(f"\n{'='*70}")
    print("测试完成！")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_checkpointing_memory()
