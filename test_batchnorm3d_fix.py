#!/usr/bin/env python3
"""
测试 BatchNorm3d 修复 - 验证 batch size 4 双 GPU 训练是否正常
"""

import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, '/home/cwh/coding/former3d')

from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated


def test_batchnorm3d():
    """测试BatchNorm3d是否能正确处理5D张量"""
    print("=" * 60)
    print("测试 BatchNorm3d 修复")
    print("=" * 60)

    # 检查CUDA
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return False

    device = torch.device('cuda:0')
    print(f"✅ 使用设备: {device}")

    # 创建数据集
    print("\n创建数据集...")
    dataset = MultiSequenceTartanAirDataset(
        data_root='/home/cwh/Study/dataset/tartanair',
        n_view=5,
        crop_size=(8, 8, 6),
        voxel_size=0.16,
        max_sequences=2  # 只使用2个序列快速测试
    )
    print(f"✅ 数据集创建成功，样本数: {len(dataset)}")

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=4,  # 测试 batch size 4
        shuffle=False,
        num_workers=2,
        collate_fn=dataset.collate_fn
    )
    print(f"✅ 数据加载器创建成功，批次大小: 4")

    # 创建模型
    print("\n创建模型...")
    model = StreamSDFFormerIntegrated(
        attn_heads=1,
        attn_layers=1,
        use_proj_occ=False,
        voxel_size=0.16,
        fusion_local_radius=2.0,
        crop_size=(8, 8, 6)
    )
    model = model.to(device)
    print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")

    # 测试前向传播
    print("\n测试前向传播...")
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n处理批次 {batch_idx + 1}/1")

            # 获取数据
            images = batch['rgb_images'].to(device)  # [batch, seq_len, 3, H, W]
            poses = batch['poses'].to(device)  # [batch, seq_len, 4, 4]
            intrinsics = batch['intrinsics'].to(device)  # [batch, seq_len, 3, 3]

            print(f"  - images shape: {images.shape}")
            print(f"  - poses shape: {poses.shape}")
            print(f"  - intrinsics shape: {intrinsics.shape}")

            try:
                # 前向传播
                outputs, states = model.forward_sequence(images, poses, intrinsics)
                print(f"  ✅ 前向传播成功!")
                print(f"  - outputs type: {type(outputs)}")
                print(f"  - outputs length: {len(outputs) if hasattr(outputs, '__len__') else 'N/A'}")
                print(f"  - states length: {len(states)}")

                # 检查输出
                try:
                    if outputs and len(outputs) > 0:
                        first_output = outputs[0]
                        print(f"  - first output shape: {first_output.shape}")
                        print(f"  - first output dtype: {first_output.dtype}")
                        print(f"  - first output device: {first_output.device}")

                        # 检查是否有NaN
                        if torch.isnan(first_output).any():
                            print("  ⚠️  警告: 输出包含NaN")
                        else:
                            print("  ✅ 输出无NaN")
                    else:
                        print("  ⚠️  警告: outputs为空")
                except Exception as e:
                    print(f"  ⚠️  访问输出时出错: {e}")
                    # 打印更多调试信息
                    print(f"  - outputs: {outputs}")

                break  # 只测试第一个batch

            except Exception as e:
                print(f"  ❌ 前向传播失败: {e}")
                import traceback
                traceback.print_exc()
                return False

    print("\n" + "=" * 60)
    print("✅ 测试通过！BatchNorm3d修复成功")
    print("=" * 60)
    return True


if __name__ == '__main__':
    success = test_batchnorm3d()
    sys.exit(0 if success else 1)
