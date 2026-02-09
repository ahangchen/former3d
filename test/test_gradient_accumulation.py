#!/usr/bin/env python3
"""
测试梯度累积的正确性
验证梯度累积与大批次训练产生相同的结果
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated

def test_gradient_accumulation():
    """测试梯度累积的正确性"""

    print("="*70)
    print("测试梯度累积正确性")
    print("="*70)

    # 设置随机种子
    torch.manual_seed(42)

    # 配置
    batch_size_large = 4
    batch_size_small = 2
    accumulation_steps = batch_size_large // batch_size_small
    sequence_length = 3
    image_size = (64, 64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n配置：")
    print(f"  - 大批次大小: {batch_size_large}")
    print(f"  - 小批次大小: {batch_size_small}")
    print(f"  - 累积步数: {accumulation_steps}")
    print(f"  - 序列长度: {sequence_length}")
    print(f"  - 设备: {device}")

    # 创建两个相同的模型
    model_large = StreamSDFFormerIntegrated(
        attn_heads=1,
        attn_layers=1,
        use_proj_occ=False,
        voxel_size=0.16,
        fusion_local_radius=2.0,
        crop_size=(24, 24, 16)
    ).to(device)

    model_small = StreamSDFFormerIntegrated(
        attn_heads=1,
        attn_layers=1,
        use_proj_occ=False,
        voxel_size=0.16,
        fusion_local_radius=2.0,
        crop_size=(24, 24, 16)
    ).to(device)

    # 复制权重
    model_small.load_state_dict(model_large.state_dict())

    # 创建优化器
    optimizer_large = optim.Adam(model_large.parameters(), lr=1e-4)
    optimizer_small = optim.Adam(model_small.parameters(), lr=1e-4)

    # 生成模拟数据（大批次）
    large_batch = {
        'images': torch.randn(batch_size_large, sequence_length, 3, *image_size).to(device),
        'poses': torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size_large, sequence_length, 1, 1).to(device),
        'intrinsics': torch.eye(3).unsqueeze(0).repeat(batch_size_large, sequence_length, 1, 1).to(device)
    }

    # 生成模拟数据（小批次）
    small_batches = []
    for i in range(accumulation_steps):
        small_batch = {
            'images': large_batch['images'][i*batch_size_small:(i+1)*batch_size_small],
            'poses': large_batch['poses'][i*batch_size_small:(i+1)*batch_size_small],
            'intrinsics': large_batch['intrinsics'][i*batch_size_small:(i+1)*batch_size_small]
        }
        small_batches.append(small_batch)

    # 训练大批次
    print(f"\n训练大批次模型...")
    optimizer_large.zero_grad()

    state_large = None
    for frame_idx in range(sequence_length):
        images = large_batch['images'][:, frame_idx]
        poses = large_batch['poses'][:, frame_idx]
        intrinsics = large_batch['intrinsics'][:, frame_idx]

        output_large, state_large = model_large.forward_single_frame(
            images, poses, intrinsics, reset_state=(frame_idx == 0)
        )

        # 简单损失
        loss_large = sum(v.sum() if hasattr(v, 'sum') else 0 for v in output_large.values() if v is not None)
        loss_large = loss_large / accumulation_steps
        loss_large.backward()

    optimizer_large.step()

    # 训练小批次（使用梯度累积）
    print(f"训练小批次模型（使用梯度累积）...")
    optimizer_small.zero_grad()
    accumulation_counter = 0

    for batch_idx, batch in enumerate(small_batches):
        state_small = None
        batch_loss = 0.0

        for frame_idx in range(sequence_length):
            images = batch['images'][:, frame_idx]
            poses = batch['poses'][:, frame_idx]
            intrinsics = batch['intrinsics'][:, frame_idx]

            output_small, state_small = model_small.forward_single_frame(
                images, poses, intrinsics, reset_state=(frame_idx == 0)
            )

            # 简单损失
            loss_small = sum(v.sum() if hasattr(v, 'sum') else 0 for v in output_small.values() if v is not None)
            loss_small = loss_small / accumulation_steps
            batch_loss += loss_small

            # 反向传播（累积梯度）
            loss_small.backward()

        # 检查是否需要更新参数
        accumulation_counter += 1
        if accumulation_counter % accumulation_steps == 0:
            optimizer_small.step()
            optimizer_small.zero_grad()

    # 比较参数
    print(f"\n比较模型参数...")
    params_large = list(model_large.parameters())
    params_small = list(model_small.parameters())

    max_diff = 0.0
    total_diff = 0.0
    param_count = 0

    for p_large, p_small in zip(params_large, params_small):
        diff = torch.abs(p_large - p_small).max().item()
        max_diff = max(max_diff, diff)
        total_diff += diff
        param_count += 1

    avg_diff = total_diff / param_count

    print(f"  - 最大差异: {max_diff:.6e}")
    print(f"  - 平均差异: {avg_diff:.6e}")
    print(f"  - 参数数量: {param_count}")

    # 验证结果
    if max_diff < 1e-4:  # 允许一定的数值误差
        print(f"\n✅ 测试通过：梯度累积与大批次训练结果一致")
        print(f"   最大差异: {max_diff:.6e} < 1e-4")
    else:
        print(f"\n❌ 测试失败：梯度累积与大批次训练结果不一致")
        print(f"   最大差异: {max_diff:.6e} > 1e-4")
        raise AssertionError("梯度累积不正确")

    print(f"\n{'='*70}")
    print("梯度累积测试完成！")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_gradient_accumulation()
