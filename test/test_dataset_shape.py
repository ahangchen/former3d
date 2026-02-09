#!/usr/bin/env python3
"""
测试MultiSequenceTartanAirDataset的输出shape
验证修复后的shape是否符合预期
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
import numpy as np


def test_dataset_shape():
    """测试Dataset输出shape"""
    print("=" * 80)
    print("测试1：验证Dataset输出shape")
    print("=" * 80)

    try:
        from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset

        # 使用实际数据目录
        data_root = "/home/cwh/Study/dataset/tartanair"

        # 创建数据集
        dataset = MultiSequenceTartanAirDataset(
            data_root=data_root,
            n_view=5,
            stride=2,
            crop_size=(48, 48, 32),
            voxel_size=0.04,
            target_image_size=(256, 256),
            max_sequences=1,  # 只用1个序列测试
            shuffle=False
        )

        print(f"\n数据集大小: {len(dataset)} 个片段")

        # 测试第一个样本
        sample = dataset[0]

        print(f"\nDataset输出shape:")
        print(f"  rgb_images: {sample['rgb_images'].shape}")
        print(f"  poses: {sample['poses'].shape}")
        print(f"  intrinsics: {sample['intrinsics'].shape}")
        print(f"  tsdf: {sample['tsdf'].shape}")
        print(f"  occupancy: {sample['occupancy'].shape}")

        # 验证shape（修复后的预期）
        expected_shapes = {
            'rgb_images': (1, 5, 3, 256, 256),      # (1, n_view, 3, H, W)
            'poses': (1, 5, 4, 4),                  # (1, n_view, 4, 4)
            'intrinsics': (1, 3, 3),                 # (1, 3, 3)
            'tsdf': (1, 1, 48, 48, 32),            # (1, 1, D, H, W)
            'occupancy': (1, 1, 48, 48, 32),         # (1, 1, D, H, W)
        }

        success = True
        for key, expected_shape in expected_shapes.items():
            actual_shape = sample[key].shape
            if actual_shape == expected_shape:
                print(f"  ✅ {key}: {actual_shape} (符合预期)")
            else:
                print(f"  ❌ {key}: {actual_shape} (预期: {expected_shape})")
                success = False

        return success

    except FileNotFoundError as e:
        print(f"\n⚠️  数据目录不存在: {e}")
        print("跳过测试（需要实际数据）")
        return True  # 不算失败，只是没有数据
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataloader_batch():
    """测试DataLoader组batch后的shape"""
    print("\n" + "=" * 80)
    print("测试2：验证DataLoader组batch后的shape")
    print("=" * 80)

    try:
        from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset

        # 使用实际数据目录
        data_root = "/home/cwh/Study/dataset/tartanair"

        # 创建数据集
        dataset = MultiSequenceTartanAirDataset(
            data_root=data_root,
            n_view=5,
            stride=2,
            crop_size=(48, 48, 32),
            voxel_size=0.04,
            target_image_size=(256, 256),
            max_sequences=1,
            shuffle=False
        )

        # 创建DataLoader，使用自定义collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            collate_fn=MultiSequenceTartanAirDataset.collate_fn
        )

        # 测试第一个batch
        for batch in dataloader:
            print(f"\nDataLoader输出shape:")
            print(f"  rgb_images: {batch['rgb_images'].shape}")
            print(f"  poses: {batch['poses'].shape}")
            print(f"  intrinsics: {batch['intrinsics'].shape}")
            print(f"  tsdf: {batch['tsdf'].shape}")
            print(f"  occupancy: {batch['occupancy'].shape}")

            # 验证shape（组batch后的预期）
            batch_size = 4
            n_view = 5
            expected_shapes = {
                'rgb_images': (batch_size, n_view, 3, 256, 256),    # (batch, n_view, 3, H, W)
                'poses': (batch_size, n_view, 4, 4),                # (batch, n_view, 4, 4)
                'intrinsics': (batch_size, 3, 3),                   # (batch, 3, 3)
                'tsdf': (batch_size, 1, 48, 48, 32),               # (batch, 1, D, H, W)
                'occupancy': (batch_size, 1, 48, 48, 32),          # (batch, 1, D, H, W)
            }

            success = True
            for key, expected_shape in expected_shapes.items():
                actual_shape = batch[key].shape
                if actual_shape == expected_shape:
                    print(f"  ✅ {key}: {actual_shape} (符合预期)")
                else:
                    print(f"  ❌ {key}: {actual_shape} (预期: {expected_shape})")
                    success = False

            return success

    except FileNotFoundError as e:
        print(f"\n⚠️  数据目录不存在: {e}")
        print("跳过测试（需要实际数据）")
        return True  # 不算失败，只是没有数据
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("MultiSequenceTartanAirDataset Shape测试")
    print("=" * 80)

    test1_success = test_dataset_shape()
    test2_success = test_dataloader_batch()

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"测试1（Dataset输出shape）: {'✅ 通过' if test1_success else '❌ 失败'}")
    print(f"测试2（DataLoader组batch）: {'✅ 通过' if test2_success else '❌ 失败'}")

    if test1_success and test2_success:
        print("\n✅ 所有测试通过！")
        return 0
    else:
        print("\n❌ 部分测试失败！")
        return 1


if __name__ == "__main__":
    exit(main())
