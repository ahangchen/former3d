#!/usr/bin/env python3
"""简化版流式训练验证测试"""

import sys
import torch

sys.path.insert(0, '/home/cwh/coding/former3d')

from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated, PoseBasedFeatureProjection

def test_basic():
    """基本功能测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=2,
        attn_layers=2,
        use_proj_occ=False,
        voxel_size=0.0625,
        fusion_local_radius=3.0,
        crop_size=(48, 96, 96)
    ).to(device)
    model.eval()

    # 测试数据
    batch_size = 2
    height, width = 128, 128

    rgb_imgs = torch.randn(batch_size, 1, 3, height, width, device=device)
    proj_mats = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, 4, 4)
    cam_positions = torch.zeros(batch_size, 1, 3, device=device)
    origin = torch.zeros(batch_size, 3, device=device)

    num_voxels = 1000
    max_coord = 16
    voxel_inds_16 = torch.randint(0, max_coord, (num_voxels, 4), device=device).to(torch.int32)
    voxel_inds_16[:, 3] = torch.randint(0, batch_size, (num_voxels,), device=device)

    batch = {
        "rgb_imgs": rgb_imgs,
        "proj_mats": {
            'coarse': proj_mats,
            'medium': proj_mats,
            'fine': proj_mats
        },
        "cam_positions": cam_positions,
        "origin": origin
    }

    # 测试1: 多尺度特征提取
    print("\n" + "="*60)
    print("测试1: 多尺度特征提取")
    print("="*60)

    try:
        with torch.no_grad():
            voxel_outputs, proj_occ_logits, bp_data, multiscale_features = model.forward(
                batch, voxel_inds_16, return_multiscale_features=True
            )

        print(f"✅ 多尺度特征提取成功")
        print(f"  包含分辨率: {list(multiscale_features.keys())}")
        for resname in ['coarse', 'medium', 'fine']:
            if resname in multiscale_features:
                res_data = multiscale_features[resname]
                print(f"  {resname}: features={res_data['features'].shape}, indices={res_data['indices'].shape}")

    except Exception as e:
        print(f"❌ 多尺度特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试2: 状态创建
    print("\n" + "="*60)
    print("测试2: 状态创建（_create_new_state）")
    print("="*60)

    # 构建模拟输出
    output = {
        'multiscale_features': multiscale_features,
        'sdf': torch.randn(num_voxels, 1, device=device),
        'occupancy': torch.randn(num_voxels, 1, device=device)
    }

    test_pose = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, 4, 4)

    try:
        new_state = model._create_new_state(output, test_pose)

        if isinstance(new_state, dict) and 'dense_grids' in new_state:
            print(f"✅ 状态创建成功")
            print(f"  包含dense_grids: {list(new_state['dense_grids'].keys())}")
            print(f"  包含sparse_indices: {list(new_state['sparse_indices'].keys())}")
            print(f"  包含spatial_shapes: {list(new_state['spatial_shapes'].keys())}")
        else:
            print(f"❌ 状态创建失败，不包含dense_grids")
            print(f"  状态类型: {type(new_state)}")
            return False

    except Exception as e:
        print(f"❌ 状态创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试3: PoseBasedFeatureProjection
    print("\n" + "="*60)
    print("测试3: PoseBasedFeatureProjection")
    print("="*60)

    projector = PoseBasedFeatureProjection(voxel_size=0.0625)

    try:
        # 测试pose变换
        historical_pose = torch.eye(4, device=device).unsqueeze(0)
        current_pose = torch.eye(4, device=device).unsqueeze(0)
        current_pose[0, 0, 3] = 0.1  # 平移

        T_ch = projector.compute_transform(historical_pose, current_pose)
        print(f"✅ Pose变换计算成功")
        print(f"  T_ch: {T_ch}")

        # 测试坐标归一化
        coords = torch.randn(100, 3, device=device)
        normalized = projector.normalize_coords(coords, (32, 32, 32))
        print(f"✅ 坐标归一化成功")
        print(f"  归一化范围: [{normalized.min():.3f}, {normalized.max():.3f}]")

    except Exception as e:
        print(f"❌ PoseBasedFeatureProjection测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*60)
    print("所有基本测试通过！")
    print("="*60)
    return True

if __name__ == '__main__':
    success = test_basic()
    sys.exit(0 if success else 1)
