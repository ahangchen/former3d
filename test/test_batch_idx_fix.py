#!/usr/bin/env python3
"""
快速测试batch_idx修复
验证在batch_size=1的情况下，所有batch_inds都被归一化到0
"""

import os
import sys
import torch

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated

def test_batch_idx_normalization():
    """测试batch_idx归一化"""

    print("="*70)
    print("测试batch_idx归一化")
    print("="*70)

    # 创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=1,
        attn_layers=1,
        use_proj_occ=False,
        voxel_size=0.16,
        fusion_local_radius=2.0,
        crop_size=(24, 24, 16)
    )

    # 确保轻量级模式启用
    model.enable_lightweight_state(True)

    # 创建模拟输入
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 模拟的图像和位姿
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).to(device)

    # 前向传播（第一次，reset=True）
    print(f"\n第1次前向传播 (batch_size={batch_size}, reset=True)")
    output1, state1 = model.forward_single_frame(images, poses, intrinsics, reset_state=True)

    if state1 is not None and 'batch_inds' in state1:
        batch_inds_1 = state1['batch_inds']
        print(f"  batch_inds形状: {batch_inds_1.shape}")
        print(f"  batch_inds唯一值: {torch.unique(batch_inds_1).tolist()}")
        print(f"  batch_inds范围: [{batch_inds_1.min().item()}, {batch_inds_1.max().item()}]")

        # 验证所有batch_inds都是0
        assert torch.all(batch_inds_1 == 0), "所有batch_inds应该都是0"
        print("  ✅ 所有batch_inds都是0（batch_size=1）")
    else:
        print("  ⚠️  状态中没有batch_inds")

    # 前向传播（第二次，reset=False，使用历史状态）
    print(f"\n第2次前向传播 (reset=False，使用历史状态)")
    output2, state2 = model.forward_single_frame(images, poses, intrinsics, reset_state=False)

    if state2 is not None and 'batch_inds' in state2:
        batch_inds_2 = state2['batch_inds']
        print(f"  batch_inds形状: {batch_inds_2.shape}")
        print(f"  batch_inds唯一值: {torch.unique(batch_inds_2).tolist()}")
        print(f"  batch_inds范围: [{batch_inds_2.min().item()}, {batch_inds_2.max().item()}]")

        # 验证所有batch_inds都是0
        assert torch.all(batch_inds_2 == 0), "所有batch_inds应该都是0"
        print("  ✅ 所有batch_inds都是0（batch_size=1）")
    else:
        print("  ⚠️  状态中没有batch_inds")

    print(f"\n{'='*70}")
    print("✅ 测试通过：batch_idx归一化正确")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_batch_idx_normalization()
