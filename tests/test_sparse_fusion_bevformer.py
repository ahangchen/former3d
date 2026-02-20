"""
测试BEVFormer风格的稀疏特征融合
测试_historical_state_project_sparse函数
"""

import torch
import sys
sys.path.insert(0, '/home/cwh/ubuntu18/home/ubuntu/coding/former3d')

from former3d.pose_aware_stream_sdfformer_sparse import PoseAwareStreamSdfFormerSparse
from spconv.pytorch import SparseConvTensor


def test_sparse_fusion_bevformer():
    """测试BEVFormer风格的稀疏特征融合"""
    print("=" * 60)
    print("测试BEVFormer风格的稀疏特征融合")
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

    # 模拟历史状态
    batch_size = 1
    n_hist = 1000
    hist_feat_dim = 16

    # 创建历史稀疏特征
    historical_features = torch.randn(n_hist, hist_feat_dim, device=device)
    historical_indices = torch.randint(0, 48, (n_hist, 4), device=device, dtype=torch.int32)
    historical_indices[:, 0] = 0  # batch index
    historical_indices[:, 1] = torch.clamp(historical_indices[:, 1], 0, 95)  # x < 96
    historical_indices[:, 2] = torch.clamp(historical_indices[:, 2], 0, 95)  # y < 96
    historical_indices[:, 3] = torch.clamp(historical_indices[:, 3], 0, 47)  # z < 48
    historical_spatial_shape = (48, 96, 96)

    # 创建historical_logits
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
    model.historical_pose = torch.eye(4, device=device).unsqueeze(0)

    # 当前帧数据
    n_cur = 800
    current_pose = torch.eye(4, device=device).unsqueeze(0)
    current_features = torch.randn(n_cur, hist_feat_dim, device=device)
    current_indices = torch.randint(0, 48, (n_cur, 4), device=device, dtype=torch.int32)
    current_indices[:, 0] = 0  # batch index
    current_indices[:, 1] = torch.clamp(current_indices[:, 1], 0, 95)
    current_indices[:, 2] = torch.clamp(current_indices[:, 2], 0, 95)
    current_indices[:, 3] = torch.clamp(current_indices[:, 3], 0, 47)

    # 创建当前帧的multiscale_features
    current_sparse = SparseConvTensor(
        features=current_features,
        indices=current_indices,
        spatial_shape=historical_spatial_shape,
        batch_size=batch_size
    )

    multiscale_features = {
        'fine': {
            'features': current_sparse,
            'resolution': 0.04
        }
    }

    print(f"历史稀疏点数: {n_hist}")
    print(f"当前稀疏点数: {n_cur}")
    print(f"特征维度: {hist_feat_dim}")
    print(f"历史spatial_shape: {historical_spatial_shape}")

    # 调用_historical_state_project_sparse
    print("\n调用_historical_state_project_sparse...")
    with torch.no_grad():
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
    print(f"非零特征数: {(projected_features.abs() > 1e-6).sum().item()}")
    print(f"非零SDF数: {(projected_sdfs.abs() > 1e-6).sum().item()}")

    # 验证输出形状
    assert projected_features.shape == (n_cur, hist_feat_dim), \
        f"投影特征形状错误: {projected_features.shape}, 期望: ({n_cur}, {hist_feat_dim})"
    assert projected_sdfs.shape == (n_cur, 1), \
        f"投影SDF形状错误: {projected_sdfs.shape}, 期望: ({n_cur}, 1)"

    print("\n✓ 稀疏融合测试通过！")
    return True


if __name__ == '__main__':
    print("\n开始测试...")
    try:
        test_sparse_fusion_bevformer()
        print("\n" + "=" * 60)
        print("BEVFormer风格稀疏融合测试通过！")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
