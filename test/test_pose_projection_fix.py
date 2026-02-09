#!/usr/bin/env python3
"""
测试pose_projection的batch_idx归一化修复
验证在不同batch_size下的历史状态处理
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated

def test_pose_projection_batch_idx():
    """测试pose_projection的batch_idx归一化"""

    print("="*70)
    print("测试pose_projection的batch_idx归一化修复")
    print("="*70)

    # 测试batch_size=1和batch_size=2的情况
    for test_batch_size in [1, 2]:
        print(f"\n{'='*70}")
        print(f"测试batch_size={test_batch_size}")
        print(f"{'='*70}")

        # 创建模型
        model = StreamSDFFormerIntegrated(
            attn_heads=1,
            attn_layers=1,
            use_proj_occ=False,
            voxel_size=0.16,
            fusion_local_radius=2.0,
            crop_size=(24, 24, 16)
        )

        model.enable_lightweight_state(True)

        # 创建模拟状态（在batch_size=2时创建）
        num_voxels = 1000
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        if test_batch_size == 2:
            # 模拟batch_size=2时创建的状态，batch_inds包含[0, 1]
            batch_inds = torch.cat([
                torch.zeros(num_voxels // 2, dtype=torch.long, device=device),
                torch.ones(num_voxels // 2, dtype=torch.long, device=device)
            ])
        else:
            # batch_size=1时，batch_inds包含[0]
            batch_inds = torch.zeros(num_voxels, dtype=torch.long, device=device)

        features = torch.randn(num_voxels, 128, device=device)
        coords = torch.randn(num_voxels, 3, device=device) * 10

        historical_state = {
            'features': features,
            'coords': coords,
            'batch_inds': batch_inds,
        }

        print(f"  历史batch_inds唯一值: {torch.unique(batch_inds).tolist()}")
        print(f"  历史batch_inds范围: [{batch_inds.min().item()}, {batch_inds.max().item()}]")

        # 创建当前位姿（batch_size=1）
        current_batch_size = 1
        current_pose = torch.eye(4).unsqueeze(0).repeat(current_batch_size, 1, 1).to(device)

        # 创建历史位姿（模拟从历史状态中获取的位姿）
        historical_pose = torch.eye(4).unsqueeze(0).repeat(test_batch_size, 1, 1).to(device)

        # 投影历史状态到当前坐标系
        print(f"  当前batch_size={current_batch_size}")
        projected_state = model.pose_projection(
            historical_state,
            historical_pose,
            current_pose
        )

        print(f"  投影后batch_inds唯一值: {torch.unique(projected_state['batch_inds']).tolist()}")
        print(f"  投影后batch_inds范围: [{projected_state['batch_inds'].min().item()}, {projected_state['batch_inds'].max().item()}]")

        # 验证所有batch_inds都在有效范围内
        assert torch.all(projected_state['batch_inds'] < current_batch_size), \
            f"所有batch_inds应该< {current_batch_size}"

        print(f"  ✅ 所有batch_inds都在有效范围内")

    print(f"\n{'='*70}")
    print("✅ 所有测试通过！")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_pose_projection_batch_idx()
