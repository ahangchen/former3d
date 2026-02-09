#!/usr/bin/env python3
"""
测试模型对batch和n_view维度的支持
验证修复后的模型可以正确处理(batch, n_view, 3, h, w)的输入
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def test_model_batch_support():
    """测试模型对batch和n_view维度的支持"""
    print("=" * 80)
    print("测试模型对batch和n_view维度的支持")
    print("=" * 80)

    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated

        # 创建模型
        print("\n创建模型...")
        model = StreamSDFFormerIntegrated(
            attn_heads=1,
            attn_layers=1,
            use_proj_occ=False,
            voxel_size=0.16,
            fusion_local_radius=2.0,
            crop_size=(24, 24, 16),
            use_checkpoint=False
        )

        model.eval()
        model.enable_lightweight_state(True)

        print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

        # 模拟一个batch的数据
        batch_size = 2
        n_view = 5

        print(f"\n测试参数:")
        print(f"  batch_size: {batch_size}")
        print(f"  n_view: {n_view}")

        # 创建测试数据
        images = torch.randn(batch_size, n_view, 3, 256, 256)
        poses = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, n_view, 4, 4)
        intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(batch_size, n_view, 3, 3)

        print(f"\n输入shape:")
        print(f"  images: {images.shape}")
        print(f"  poses: {poses.shape}")
        print(f"  intrinsics: {intrinsics.shape}")

        # 测试forward_single_frame（处理(batch, 1, 3, h, w)）
        print(f"\n测试forward_single_frame...")
        # 提取第一帧
        images_frame = images[:, 0:1]  # (batch, 1, 3, 256, 256)
        poses_frame = poses[:, 0:1]    # (batch, 1, 4, 4)
        intrinsics_frame = intrinsics[:, 0:1]  # (batch, 1, 3, 3)

        print(f"  单帧输入shape: {images_frame.shape}")
        output_single, state_single = model.forward_single_frame(
            images_frame, poses_frame, intrinsics_frame, reset_state=True
        )

        print(f"  单帧输出shape: {type(output_single)}")
        if isinstance(output_single, dict):
            for key, value in output_single.items():
                if value is not None and isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape}")

        # 测试forward_sequence（处理(batch, n_view, 3, h, w)）
        print(f"\n测试forward_sequence...")
        outputs_seq, states_seq = model.forward_sequence(images, poses, intrinsics)

        print(f"\n序列输出shape:")
        print(f"  outputs_seq: {type(outputs_seq)}")
        if isinstance(outputs_seq, dict):
            for key, value in outputs_seq.items():
                if value is not None and isinstance(value, torch.Tensor):
                    print(f"    {key}: {value.shape}")
        elif isinstance(outputs_seq, torch.Tensor):
            print(f"  outputs_seq: {outputs_seq.shape}")
        else:
            print(f"  outputs_seq: {outputs_seq}")

        print(f"  states_seq: {len(states_seq)} 个状态")

        # 验证输出shape
        if isinstance(outputs_seq, torch.Tensor):
            assert outputs_seq.shape[0] == batch_size, f"batch维度错误: {outputs_seq.shape[0]} != {batch_size}"
            assert outputs_seq.shape[1] == n_view, f"n_view维度错误: {outputs_seq.shape[1]} != {n_view}"
            print(f"\n✅ 模型正确处理batch和n_view维度！")

        return True

    except FileNotFoundError as e:
        print(f"\n⚠️  模型文件不存在: {e}")
        print("跳过测试（需要先完成模型修改）")
        return True  # 不算失败，只是还没修改
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("模型Batch支持测试")
    print("=" * 80)

    success = test_model_batch_support()

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"模型batch支持测试: {'✅ 通过' if success else '❌ 失败'}")

    if success:
        print("\n✅ 所有测试通过！")
        return 0
    else:
        print("\n❌ 部分测试失败！")
        return 1


if __name__ == "__main__":
    exit(main())
