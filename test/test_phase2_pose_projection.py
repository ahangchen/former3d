#!/usr/bin/env python3
"""
Phase 2 测试：Pose-based特征投影

测试PoseBasedFeatureProjection类的功能
"""

import sys
import os
import torch
import numpy as np

# 添加路径
sys.path.insert(0, '/home/cwh/coding/former3d')


def test_pose_transform():
    """测试pose变换"""
    print("\n" + "="*60)
    print("Phase 2 测试1：Pose变换")
    print("="*60)

    from former3d.stream_sdfformer_integrated import PoseBasedFeatureProjection

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建历史和当前pose
    historical_pose = torch.eye(4, device=device).unsqueeze(0)  # [1, 4, 4]
    current_pose = torch.eye(4, device=device).unsqueeze(0)  # [1, 4, 4]

    # 在X方向平移1米
    current_pose[0, 0, 3] = 1.0

    # 在Y方向平移0.5米
    current_pose[0, 1, 3] = 0.5

    print(f"历史pose:\n{historical_pose}")
    print(f"当前pose:\n{current_pose}")

    # 创建投影器
    projector = PoseBasedFeatureProjection(voxel_size=0.0625)

    # 计算变换矩阵
    T_ch = projector.compute_transform(historical_pose, current_pose)

    print(f"\n变换矩阵 T_ch (从历史到当前):\n{T_ch}")

    # 验证变换矩阵
    expected_translation = torch.tensor([[1.0, 0.5, 0.0]], device=device)

    if torch.allclose(T_ch[0, :3, 3], expected_translation, atol=1e-6):
        print(f"✅ 平移向量正确: {T_ch[0, :3, 3]}")
        translation_correct = True
    else:
        print(f"❌ 平移向量错误: 期望 {expected_translation}, 实际 {T_ch[0, :3, 3]}")
        translation_correct = False

    # 验证旋转矩阵（应该是单位矩阵）
    rotation = T_ch[0, :3, :3]
    expected_rotation = torch.eye(3, device=device)

    if torch.allclose(rotation, expected_rotation, atol=1e-6):
        print(f"✅ 旋转矩阵正确（单位矩阵）")
        rotation_correct = True
    else:
        print(f"❌ 旋转矩阵错误: \n期望\n{expected_rotation}\n实际\n{rotation}")
        rotation_correct = False

    return translation_correct and rotation_correct


def test_voxel_coords_transform():
    """测试体素坐标变换"""
    print("\n" + "="*60)
    print("Phase 2 测试2：体素坐标变换")
    print("="*60)

    from former3d.stream_sdfformer_integrated import PoseBasedFeatureProjection

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建投影器
    projector = PoseBasedFeatureProjection(voxel_size=0.0625)

    # 创建测试体素坐标
    voxel_coords = torch.tensor([
        [1.0, 2.0, 3.0],
        [5.0, 10.0, 15.0],
        [0.0, 0.0, 0.0],
    ], device=device)  # [3, 3]

    print(f"原始体素坐标:\n{voxel_coords}")

    # 创建变换矩阵（X方向平移10个体素）
    T_ch = torch.eye(4, device=device)
    T_ch[0, 3] = 10.0 * 0.0625  # 10个体素 * 体素大小

    print(f"\n变换矩阵:\n{T_ch}")

    # 变换坐标
    transformed_coords = projector.transform_voxel_coords(voxel_coords, T_ch)

    print(f"\n变换后的坐标:\n{transformed_coords}")

    # 验证变换
    # T_ch是单位矩阵 + X方向平移10个体素(10*0.0625=0.625)
    # 第一个点：(1, 2, 3) -> (1+10*0.0625/0.0625, 2, 3) = (11, 2, 3)
    # 等等，不对。T_ch已经在世界坐标系中了，所以坐标变换应该是：
    # [1, 2, 3, 1] @ T_ch = [1*1 + 2*0 + 3*0 + 1*0.625, ...] = [1.625, 2, 3]
    # 所以期望应该是 (1, 2, 3) @ T_ch = (1.625, 2, 3)
    # 但测试用例中T_ch是单位矩阵+平移0.625，所以变换后的坐标应该是原始坐标+平移
    expected_coords = torch.tensor([
        [1.0, 2.0, 3.0],
        [5.0, 10.0, 15.0],
        [0.0, 0.0, 0.0],
    ], device=device)
    expected_coords[:, 0] += 0.625  # 添加X方向的平移

    if torch.allclose(transformed_coords, expected_coords, atol=1e-3):
        print(f"✅ 坐标变换正确")
        return True
    else:
        print(f"❌ 坐标变换错误")
        print(f"期望:\n{expected_coords}")
        print(f"实际:\n{transformed_coords}")
        return False


def test_coordinate_normalization():
    """测试坐标归一化"""
    print("\n" + "="*60)
    print("Phase 2 测试3：坐标归一化")
    print("="*60)

    from former3d.stream_sdfformer_integrated import PoseBasedFeatureProjection

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建投影器
    projector = PoseBasedFeatureProjection(voxel_size=0.0625)

    # 创建测试坐标
    grid_shape = [32, 32, 32]  # [D, H, W]

    coords = torch.tensor([
        [0, 0, 0],        # 最小值
        [31, 31, 31],     # 最大值
        [15, 15, 15],     # 中间值
    ], dtype=torch.float32, device=device)

    print(f"原始坐标:\n{coords}")
    print(f"网格形状: {grid_shape}")

    # 归一化坐标
    normalized = projector.normalize_coords(coords, grid_shape)

    print(f"\n归一化后坐标:\n{normalized}")

    # 验证归一化范围 [-1, 1]
    min_val = normalized.min().item()
    max_val = normalized.max().item()

    print(f"归一化范围: [{min_val:.3f}, {max_val:.3f}]")

    in_range = (min_val >= -1.0) and (max_val <= 1.0)

    if in_range:
        print(f"✅ 归一化范围正确")
    else:
        print(f"❌ 归一化范围错误")

    # 验证特定点的值
    # (0, 0, 0) 应该归一化为 (-1, -1, -1)
    # (31, 31, 31) 应该归一化为 (1, 1, 1)
    # (15, 15, 15) 应该归一化为 (~0, ~0, ~0)

    expected = torch.tensor([
        [-1, -1, -1],
        [1, 1, 1],
        [-0.0323, -0.0323, -0.0323],  # 15/31*2-1 ≈ -0.0323
    ], device=device)

    if torch.allclose(normalized, expected, atol=1e-3):
        print(f"✅ 归一化值正确")
        return True
    else:
        print(f"❌ 归一化值错误")
        print(f"期望:\n{expected}")
        print(f"实际:\n{normalized}")
        return False


def test_feature_projection():
    """测试特征投影（使用grid_sample）"""
    print("\n" + "="*60)
    print("Phase 2 测试4：特征投影")
    print("="*60)

    from former3d.stream_sdfformer_integrated import PoseBasedFeatureProjection

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建投影器
    projector = PoseBasedFeatureProjection(voxel_size=0.0625)

    # 创建历史特征网格
    batch_size = 2
    num_channels = 64
    grid_shape = [32, 32, 32]

    historical_features_grid = torch.randn(batch_size, num_channels, *grid_shape, device=device)

    # 创建索引：前3列是体素坐标，第4列是batch索引
    historical_indices = torch.randint(0, max(grid_shape), (100, 3), device=device).to(torch.int32)
    batch_inds = torch.randint(0, batch_size, (100,), device=device).to(torch.int32)
    historical_indices = torch.cat([historical_indices, batch_inds.unsqueeze(1)], dim=1)  # [100, 4]

    current_indices = historical_indices.clone()  # 相同的索引

    print(f"历史特征网格: {historical_features_grid.shape}")
    print(f"历史索引: {historical_indices.shape}")
    print(f"当前索引: {current_indices.shape}")

    # 创建单位变换矩阵
    T_ch = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    print(f"变换矩阵: {T_ch.shape}")

    # 投影特征
    try:
        projected = projector.project_features(
            historical_features_grid,
            historical_indices,
            current_indices,
            T_ch,
            grid_shape
        )

        print(f"\n投影后的特征: {projected.shape}")

        # 验证形状
        if projected.shape == (100, num_channels):
            print(f"✅ 投影特征形状正确")
            shape_correct = True
        else:
            print(f"❌ 投影特征形状错误: 期望 (100, {num_channels}), 实际 {projected.shape}")
            shape_correct = False

        return shape_correct

    except Exception as e:
        print(f"\n❌ 特征投影失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_phase2_tests():
    """运行Phase 2所有测试"""
    print("\n" + "="*60)
    print("Phase 2 测试套件：Pose-based特征投影")
    print("="*60)

    results = {}

    # 测试1：Pose变换
    results['Pose变换'] = test_pose_transform()

    # 测试2：体素坐标变换
    results['体素坐标变换'] = test_voxel_coords_transform()

    # 测试3：坐标归一化
    results['坐标归一化'] = test_coordinate_normalization()

    # 测试4：特征投影
    results['特征投影'] = test_feature_projection()

    # 打印测试结果汇总
    print("\n" + "="*60)
    print("Phase 2 测试结果汇总")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20s}: {status}")

    # 统计
    passed_count = sum(results.values())
    total_count = len(results)
    print(f"\n总计: {passed_count}/{total_count} 测试通过")

    return all(results.values())


if __name__ == '__main__':
    success = run_all_phase2_tests()
    sys.exit(0 if success else 1)
