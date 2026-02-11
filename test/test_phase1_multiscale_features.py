#!/usr/bin/env python3
"""
Phase 1 测试：SDFFormer多尺度特征提取

测试SDFFormer能否正确返回多尺度特征
"""

import sys
import os
import torch
import numpy as np

# 添加路径
sys.path.insert(0, '/home/cwh/coding/former3d')

from former3d.sdfformer import SDFFormer
import spconv.pytorch as spconv


def test_multiscale_feature_extraction():
    """测试SDFFormer的多尺度特征提取"""
    print("\n" + "="*60)
    print("Phase 1 测试：SDFFormer多尺度特征提取")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建SDFFormer模型
    model = SDFFormer(
        attn_heads=2,
        attn_layers=2,
        use_proj_occ=False,
        voxel_size=0.0625
    ).to(device)
    model.eval()

    print(f"✅ SDFFormer模型创建成功")

    # 创建测试数据
    batch_size = 2
    n_views = 1
    height, width = 128, 128

    # RGB图像
    rgb_imgs = torch.randn(batch_size, n_views, 3, height, width, device=device)

    # 投影矩阵
    proj_mats = {
        'coarse': torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, n_views, 4, 4),
        'medium': torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, n_views, 4, 4),
        'fine': torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, n_views, 4, 4),
    }

    # 相机位置
    cam_positions = torch.zeros(batch_size, n_views, 3, device=device)

    # 原点
    origin = torch.zeros(batch_size, 3, device=device)

    # 体素索引（增加数量以避免提前返回）
    num_voxels = 500
    max_coord = 16
    voxel_inds_16 = torch.randint(0, max_coord, (num_voxels, 4), device=device)
    voxel_inds_16[:, 3] = torch.randint(0, batch_size, (num_voxels,), device=device)
    voxel_inds_16 = voxel_inds_16.to(torch.int32)  # spconv要求int32

    print(f"测试数据创建:")
    print(f"  RGB图像: {rgb_imgs.shape}")
    print(f"  体素索引: {voxel_inds_16.shape}")

    # 测试是否支持return_multiscale_features参数
    import inspect
    forward_signature = inspect.signature(model.forward)
    params = forward_signature.parameters

    print(f"\nforward方法参数: {list(params.keys())}")

    has_multiscale_param = 'return_multiscale_features' in params

    if has_multiscale_param:
        print(f"✅ forward方法支持return_multiscale_features参数")
    else:
        print(f"❌ forward方法不支持return_multiscale_features参数")

    # 构建batch字典
    batch = {
        "rgb_imgs": rgb_imgs,
        "proj_mats": proj_mats,
        "cam_positions": cam_positions,
        "origin": origin
    }

    # 测试前向传播
    try:
        with torch.no_grad():
            if has_multiscale_param:
                voxel_outputs, proj_occ_logits, bp_data, multiscale_features = model.forward(
                    batch,
                    voxel_inds_16,
                    return_multiscale_features=True
                )

                print(f"\n✅ 前向传播成功（带多尺度特征）")

                # 验证多尺度特征
                if multiscale_features is not None:
                    print(f"\n多尺度特征:")
                    for resname, features_data in multiscale_features.items():
                        print(f"  {resname}:")
                        print(f"    features: {type(features_data['features'])}")
                        print(f"    indices: {features_data['indices'].shape}")
                        print(f"    batch_size: {features_data['batch_size']}")
                        print(f"    spatial_shape: {features_data['spatial_shape']}")
                        print(f"    resolution: {features_data['resolution']}")

                    # 验证必需的分辨率级别
                    required_resolutions = ['coarse', 'medium', 'fine']
                    all_present = all(res in multiscale_features for res in required_resolutions)

                    if all_present:
                        print(f"\n✅ 所有多尺度特征都已提取")
                        return True
                    else:
                        missing = [res for res in required_resolutions if res not in multiscale_features]
                        print(f"\n❌ 缺少分辨率级别: {missing}")
                        return False
                else:
                    print(f"\n❌ multiscale_features为None")
                    return False
            else:
                voxel_outputs, proj_occ_logits, bp_data = model.forward(
                    batch,
                    voxel_inds_16
                )

                print(f"\n✅ 前向传播成功（无多尺度特征）")
                print(f"⚠️  需要添加return_multiscale_features参数支持")
                return False

    except Exception as e:
        print(f"\n❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_multiscale_feature_extraction()
    sys.exit(0 if success else 1)
