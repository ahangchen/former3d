"""
测试CUDA OOM修复 - 分批距离计算
验证在大规模稀疏点情况下不会OOM
"""

import torch
import sys
sys.path.insert(0, '/home/cwh/ubuntu18/home/ubuntu/coding/former3d')

from former3d.pose_aware_stream_sdfformer_sparse import PoseAwareStreamSdfFormerSparse
from spconv.pytorch import SparseConvTensor


def test_large_scale_sparse_fusion():
    """测试大规模稀疏点融合（模拟OOM场景）"""
    print("=" * 60)
    print("测试大规模稀疏点融合 - CUDA OOM修复验证")
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

    # 模拟大规模历史状态（接近OOM触发条件）
    batch_size = 1
    n_hist = 60000  # 大量历史点
    hist_feat_dim = 16

    print(f"\n创建 {n_hist} 个历史稀疏点...")
    historical_features = torch.randn(n_hist, hist_feat_dim, device=device)
    historical_indices = torch.randint(0, 48, (n_hist, 4), device=device, dtype=torch.int32)
    historical_indices[:, 0] = 0  # batch index
    historical_indices[:, 1] = torch.clamp(historical_indices[:, 1], 0, 95)
    historical_indices[:, 2] = torch.clamp(historical_indices[:, 2], 0, 95)
    historical_indices[:, 3] = torch.clamp(historical_indices[:, 3], 0, 47)
    historical_spatial_shape = (48, 96, 96)

    historical_logits = SparseConvTensor(
        features=torch.randn(n_hist, 1, device=device),
        indices=historical_indices,
        spatial_shape=historical_spatial_shape,
        batch_size=batch_size
    )

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

    model.historical_pose = torch.eye(4, device=device).unsqueeze(0)

    # 当前帧数据（也很大）
    n_cur = 30000  # 大量当前点
    print(f"创建 {n_cur} 个当前稀疏点...")
    current_pose = torch.eye(4, device=device).unsqueeze(0)
    current_features = torch.randn(n_cur, hist_feat_dim, device=device)
    current_indices = torch.randint(0, 48, (n_cur, 4), device=device, dtype=torch.int32)
    current_indices[:, 0] = 0
    current_indices[:, 1] = torch.clamp(current_indices[:, 1], 0, 95)
    current_indices[:, 2] = torch.clamp(current_indices[:, 2], 0, 95)
    current_indices[:, 3] = torch.clamp(current_indices[:, 3], 0, 47)

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

    print(f"\n测试参数:")
    print(f"  历史稀疏点数: {n_hist}")
    print(f"  当前稀疏点数: {n_cur}")
    print(f"  特征维度: {hist_feat_dim}")
    print(f"  预估距离矩阵大小: {n_cur * n_hist * 4 / 1024**3:.2f} GB (如果不分批)")

    # 调用_historical_state_project_sparse
    print("\n调用_historical_state_project_sparse（使用分批计算）...")
    with torch.no_grad():
        projected_features, projected_sdfs = model._historical_state_project_sparse(
            current_pose,
            current_features,
            current_indices,
            multiscale_features
        )

    print(f"\n✓ 测试通过！没有OOM错误")
    print(f"投影特征形状: {projected_features.shape}")
    print(f"投影SDF形状: {projected_sdfs.shape}")
    print(f"投影特征范围: [{projected_features.min():.3f}, {projected_features.max():.3f}]")
    print(f"投影SDF范围: [{projected_sdfs.min():.3f}, {projected_sdfs.max():.3f}]")
    print(f"非零特征数: {(projected_features.abs() > 1e-6).sum().item()}")

    # 验证输出形状
    assert projected_features.shape == (n_cur, hist_feat_dim), \
        f"投影特征形状错误: {projected_features.shape}, 期望: ({n_cur}, {hist_feat_dim})"
    assert projected_sdfs.shape == (n_cur, 1), \
        f"投影SDF形状错误: {projected_sdfs.shape}, 期望: ({n_cur}, 1)"

    print("\n" + "=" * 60)
    print("✓ CUDA OOM修复验证成功！")
    print("=" * 60)
    return True


if __name__ == '__main__':
    print("\n开始测试...")
    try:
        test_large_scale_sparse_fusion()
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
