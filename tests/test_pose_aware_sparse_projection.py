"""
测试pose_aware_stream_sdfformer_sparse中的历史投影功能
"""

import torch
import sys
sys.path.insert(0, '/home/cwh/ubuntu18/home/ubuntu/coding/former3d')

from former3d.pose_aware_stream_sdfformer_sparse import PoseAwareStreamSdfFormerSparse


def test_historical_state_project_sparse():
    """测试_historical_state_project_sparse函数"""
    print("=" * 60)
    print("测试_historical_state_project_sparse函数")
    print("=" * 60)

    # 清理显存
    torch.cuda.empty_cache()

    # 使用GPU
    device = torch.device('cuda')

    # 创建模型实例（使用更小的配置）
    model = PoseAwareStreamSdfFormerSparse(
        attn_heads=2,
        attn_layers=1,
        use_proj_occ=True,
        voxel_size=0.04,
        fusion_local_radius=3.0,
        crop_size=(24, 48, 48),  # 更小的crop_size
        use_checkpoint=False
    )
    model = model.to(device)
    model.eval()

    # 模拟历史状态
    batch_size = 1

    # 创建模拟的历史稀疏特征（使用更小的数据量）
    n_hist = 500
    historical_features = torch.randn(n_hist, 128, device=device)
    historical_indices = torch.randint(0, 24, (n_hist, 4), device=device, dtype=torch.int32)
    historical_indices[:, 0] = 0  # batch index
    historical_spatial_shape = (24, 48, 48)  # 匹配crop_size

    # 创建SparseConvTensor作为historical_logits
    from spconv.pytorch import SparseConvTensor
    historical_logits = SparseConvTensor(
        features=torch.randn(n_hist, 1, device=device),
        indices=historical_indices,
        spatial_shape=historical_spatial_shape,
        batch_size=batch_size
    )

    # 保存历史状态
    model.historical_state = {
        'multiscale': {
            'fine': {
                'features': historical_features,
                'indices': historical_indices,
                'spatial_shape': historical_spatial_shape,
                'logits': historical_logits,
                'resolution': 0.04,
                'batch_size': batch_size
            }
        },
        'batch_size': batch_size
    }

    # 模拟历史pose
    model.historical_pose = torch.eye(4, device=device).unsqueeze(0)  # [1, 4, 4]

    # 创建当前帧数据（使用更小的数据量）
    n_cur = 300
    current_pose = torch.eye(4, device=device).unsqueeze(0)  # [1, 4, 4]
    current_features = torch.randn(n_cur, 1, device=device)
    current_indices = torch.randint(0, 24, (n_cur, 4), device=device, dtype=torch.int32)
    current_indices[:, 0] = 0  # batch index

    # 创建multiscale_features（使用真实的SparseConvTensor）
    multiscale_features = {
        'fine': {
            'features': SparseConvTensor(
                features=current_features,
                indices=current_indices,
                spatial_shape=historical_spatial_shape,
                batch_size=batch_size
            ),
            'resolution': 0.04,
            'logits': None
        }
    }

    print(f"历史稀疏点数: {n_hist}")
    print(f"当前稀疏点数: {n_cur}")
    print(f"历史pose: {model.historical_pose}")
    print(f"当前pose: {current_pose}")

    # 调用_historical_state_project_sparse
    print("\n调用_historical_state_project_sparse...")
    projected_features, projected_sdfs = model._historical_state_project_sparse(
        current_pose,
        current_features,
        current_indices,
        multiscale_features
    )

    print(f"\n投影特征形状: {projected_features.shape}")
    print(f"投影SDF形状: {projected_sdfs.shape}")
    print(f"投影特征范围: [{projected_features.min():.3f}, {projected_features.max():.3f}]")
    print(f"投影SDF范围: [{projected_sdfs.min():.3f}, {projected_sdfs.max():.3f}]")
    print(f"非零特征数: {(projected_features.abs() > 0).sum().item()}")
    print(f"非零SDF数: {(projected_sdfs.abs() > 0).sum().item()}")

    # 验证输出形状
    assert projected_features.shape == (n_cur, 128), f"投影特征形状错误: {projected_features.shape}"
    assert projected_sdfs.shape == (n_cur, 1), f"投影SDF形状错误: {projected_sdfs.shape}"

    print("\n✓ 测试通过！")
    return True


def test_forward_single_frame():
    """测试forward_single_frame函数"""
    print("\n" + "=" * 60)
    print("测试forward_single_frame函数")
    print("=" * 60)

    # 清理显存
    torch.cuda.empty_cache()

    # 使用GPU
    device = torch.device('cuda')

    # 创建模型实例（使用更小的配置）
    model = PoseAwareStreamSdfFormerSparse(
        attn_heads=2,
        attn_layers=1,
        use_proj_occ=True,
        voxel_size=0.04,
        fusion_local_radius=3.0,
        crop_size=(24, 48, 48),  # 更小的crop_size
        use_checkpoint=False
    )
    model = model.to(device)
    model.eval()

    # 创建输入数据（使用更小的图像尺寸）
    batch_size = 1
    images = torch.randn(batch_size, 3, 128, 128, device=device)
    poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    print(f"输入图像形状: {images.shape}")
    print(f"输入Pose形状: {poses.shape}")
    print(f"输入内参形状: {intrinsics.shape}")

    # 第一帧（没有历史信息）
    print("\n第一帧（没有历史信息）...")
    with torch.no_grad():
        output, state = model.forward_single_frame(images, poses, intrinsics, reset_state=True)

    print(f"输出类型: {type(output)}")
    print(f"输出keys: {output.keys() if isinstance(output, dict) else 'N/A'}")
    if 'voxel_outputs' in output:
        print(f"voxel_outputs keys: {output['voxel_outputs'].keys()}")

    # 第二帧（有历史信息）
    print("\n第二帧（有历史信息）...")
    with torch.no_grad():
        output2, state2 = model.forward_single_frame(images, poses, intrinsics, reset_state=False)

    print(f"输出2类型: {type(output2)}")
    print(f"输出2keys: {output2.keys() if isinstance(output2, dict) else 'N/A'}")

    print("\n✓ 测试通过！")
    return True


if __name__ == '__main__':
    print("\n开始测试...")
    try:
        test_historical_state_project_sparse()
        test_forward_single_frame()
        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
