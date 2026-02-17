"""
测试pose_aware_stream_sdfformer_sparse中的_historical_state_project函数（Dense版本）
严格按照任务二要求使用grid_sample
"""

import torch
import sys
sys.path.insert(0, '/home/cwh/ubuntu18/home/ubuntu/coding/former3d')

from former3d.pose_aware_stream_sdfformer_sparse import PoseAwareStreamSdfFormerSparse
from spconv.pytorch import SparseConvTensor


def test_historical_state_project():
    """测试_historical_state_project函数（Dense版本）"""
    print("=" * 60)
    print("测试_historical_state_project函数（Dense版本）")
    print("=" * 60)

    # 清理显存
    torch.cuda.empty_cache()

    # 使用GPU
    device = torch.device('cuda')

    # 创建模型实例
    model = PoseAwareStreamSdfFormerSparse(
        attn_heads=2,
        attn_layers=1,
        use_proj_occ=True,
        voxel_size=0.04,
        fusion_local_radius=3.0,
        crop_size=(48, 96, 96),
        use_checkpoint=False
    )
    model = model.to(device)
    model.eval()

    # 模拟历史状态（使用_record_state记录的格式）
    batch_size = 1

    # 创建模拟的历史稀疏特征
    n_hist = 1000
    hist_feat_dim = 16  # fine级别的特征维度
    historical_features = torch.randn(n_hist, hist_feat_dim, device=device)
    historical_indices = torch.randint(0, 48, (n_hist, 4), device=device, dtype=torch.int32)
    historical_indices[:, 0] = 0  # batch index
    # 确保索引在有效范围内
    historical_indices[:, 1] = torch.clamp(historical_indices[:, 1], 0, 95)  # x < 96
    historical_indices[:, 2] = torch.clamp(historical_indices[:, 2], 0, 95)  # y < 96
    historical_indices[:, 3] = torch.clamp(historical_indices[:, 3], 0, 47)  # z < 48
    historical_spatial_shape = (48, 96, 96)  # (D, H, W)

    # 创建historical_logits作为SparseConvTensor
    historical_logits = SparseConvTensor(
        features=torch.randn(n_hist, 1, device=device),
        indices=historical_indices,
        spatial_shape=historical_spatial_shape,
        batch_size=batch_size
    )

    # 保存历史状态（使用_record_state的格式）
    model.historical_state = {
        'multiscale': {
            'fine': {
                'features': historical_features,
                'indices': historical_indices,
                'spatial_shape': historical_spatial_shape,
                'logits': historical_logits,  # SparseConvTensor
                'resolution': 0.04,
                'batch_size': batch_size
            }
        },
        'batch_size': batch_size
    }

    # 模拟历史pose
    model.historical_pose = torch.eye(4, device=device).unsqueeze(0)  # [1, 4, 4]

    # 当前帧数据
    current_pose = torch.eye(4, device=device).unsqueeze(0)  # [1, 4, 4]
    current_spatial_shape = (48, 96, 96)  # (D, H, W)

    print(f"历史稀疏点数: {n_hist}")
    print(f"历史特征维度: {hist_feat_dim}")
    print(f"历史spatial_shape: {historical_spatial_shape}")
    print(f"当前spatial_shape: {current_spatial_shape}")
    print(f"历史pose: 单位矩阵")
    print(f"当前pose: 单位矩阵")

    # 调用_historical_state_project
    print("\n调用_historical_state_project...")
    with torch.no_grad():
        projected_features, projected_sdfs = model._historical_state_project(
            current_pose,
            current_spatial_shape
        )

    print(f"\n投影特征形状: {projected_features.shape}")
    print(f"投影SDF形状: {projected_sdfs.shape}")
    print(f"投影特征范围: [{projected_features.min():.3f}, {projected_features.max():.3f}]")
    print(f"投影SDF范围: [{projected_sdfs.min():.3f}, {projected_sdfs.max():.3f}]")
    print(f"非零特征数: {(projected_features.abs() > 1e-6).sum().item()}")
    print(f"非零SDF数: {(projected_sdfs.abs() > 1e-6).sum().item()}")

    # 验证输出形状
    D, H, W = current_spatial_shape
    assert projected_features.shape == (D, H, W, hist_feat_dim), \
        f"投影特征形状错误: {projected_features.shape}, 期望: ({D}, {H}, {W}, {hist_feat_dim})"
    assert projected_sdfs.shape == (D, H, W, 1), \
        f"投影SDF形状错误: {projected_sdfs.shape}, 期望: ({D}, {H}, {W}, 1)"

    # 验证特征维度
    assert projected_features.shape[-1] == hist_feat_dim, \
        f"投影特征维度错误: {projected_features.shape[-1]}, 期望: {hist_feat_dim}"

    print("\n✓ 测试通过！")
    return True


if __name__ == '__main__':
    print("\n开始测试...")
    try:
        test_historical_state_project()
        print("\n" + "=" * 60)
        print("_historical_state_project（Dense版本）测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
