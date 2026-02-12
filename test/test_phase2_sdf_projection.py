#!/usr/bin/env python3
"""
测试 Phase 2: 实现SDF投影方法

测试目标：
1. 验证PoseBasedFeatureProjection.project_sdf()方法存在
2. 验证SDF投影的坐标变换正确
3. 验证grid_sample采样正确
4. 验证输出形状正确
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from former3d.stream_sdfformer_integrated import PoseBasedFeatureProjection
    POSE_PROJECTION_AVAILABLE = True
except ImportError as e:
    print(f"警告：无法导入PoseBasedFeatureProjection: {e}")
    POSE_PROJECTION_AVAILABLE = False


class TestPhase2SDFProjection:
    """Phase 2: 测试SDF投影方法"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def test_projection_method_exists(self):
        """测试1: 验证project_sdf方法存在"""
        print("\n" + "="*60)
        print("测试1: 验证project_sdf方法存在")
        print("="*60)

        if not POSE_PROJECTION_AVAILABLE:
            print("⚠️ 跳过测试：无法导入PoseBasedFeatureProjection")
            return False

        projection = PoseBasedFeatureProjection(voxel_size=0.0625)

        if hasattr(projection, 'project_sdf'):
            print("✅ project_sdf方法存在")
            return True
        else:
            print("❌ project_sdf方法不存在")
            return False

    def test_coordinate_transform(self):
        """测试2: 验证坐标变换"""
        print("\n" + "="*60)
        print("测试2: 验证坐标变换")
        print("="*60)

        if not POSE_PROJECTION_AVAILABLE:
            print("⚠️ 跳过测试：无法导入PoseBasedFeatureProjection")
            return False

        projection = PoseBasedFeatureProjection(voxel_size=0.0625)

        # 创建测试坐标
        num_points = 1000
        coords = torch.randn(num_points, 3, device=self.device)  # [N, 3]

        # 创建变换矩阵（单位矩阵）
        T = torch.eye(4, device=self.device).unsqueeze(0)  # [1, 4, 4]

        # 变换坐标
        transformed = projection.transform_voxel_coords(coords, T[0])  # [N, 3]

        # 单位矩阵变换应该保持坐标不变
        diff = torch.norm(transformed - coords, dim=1).mean().item()

        print(f"✅ 坐标变换完成")
        print(f"  - 原始坐标形状: {coords.shape}")
        print(f"  - 变换后坐标形状: {transformed.shape}")
        print(f"  - 平均误差: {diff:.6f}")

        if diff < 1e-5:
            print(f"✅ 单位矩阵变换验证通过")
            return True
        else:
            print(f"❌ 单位矩阵变换验证失败: 误差{diff}")
            return False

    def test_grid_sample_shape(self):
        """测试3: 验证grid_sample形状"""
        print("\n" + "="*60)
        print("测试3: 验证grid_sample形状")
        print("="*60)

        batch_size = 2
        spatial_shape = [16, 16, 16]  # [D, H, W]
        num_points = 1000

        # 创建密集网格
        dense_grid = torch.randn(batch_size, 1, *spatial_shape, device=self.device)

        # 创建归一化坐标 [-1, 1]
        normalized_coords = torch.rand(num_points, 3, device=self.device) * 2 - 1

        # 调整为grid_sample期望的格式 [1, 1, 1, N, 3]
        grid = normalized_coords.view(1, 1, 1, num_points, 3)
        grid = grid.expand(batch_size, -1, -1, -1, -1)  # [B, 1, 1, N, 3]

        # 使用grid_sample采样
        sampled = F.grid_sample(
            dense_grid,  # [B, 1, D, H, W]
            grid,  # [B, 1, 1, N, 3]
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B, 1, 1, 1, N]

        print(f"✅ grid_sample采样完成")
        print(f"  - 输入网格形状: {dense_grid.shape}")
        print(f"  - grid形状: {grid.shape}")
        print(f"  - 采样结果形状: {sampled.shape}")
        print(f"  - 期望形状: [{batch_size}, 1, 1, 1, {num_points}]")

        if sampled.shape == torch.Size([batch_size, 1, 1, 1, num_points]):
            print(f"✅ 形状验证通过")
            return True
        else:
            print(f"❌ 形状验证失败")
            return False

    def test_projection_pipeline(self):
        """测试4: 完整投影流程"""
        print("\n" + "="*60)
        print("测试4: 完整投影流程")
        print("="*60)

        if not POSE_PROJECTION_AVAILABLE:
            print("⚠️ 跳过测试：无法导入PoseBasedFeatureProjection")
            return False

        projection = PoseBasedFeatureProjection(voxel_size=0.0625)

        batch_size = 2
        num_points = 1000
        spatial_shape = [16, 16, 16]

        # 创建历史SDF网格
        historical_sdf_grid = torch.randn(batch_size, 1, *spatial_shape, device=self.device)

        # 创建历史索引
        historical_indices = torch.randint(0, 16, (num_points, 4), device=self.device)
        historical_indices[:, 0] = torch.randint(0, batch_size, (num_points,), device=self.device)

        # 创建当前索引
        current_indices = torch.randint(0, 16, (num_points, 4), device=self.device)
        current_indices[:, 0] = torch.randint(0, batch_size, (num_points,), device=self.device)

        # 创建pose变换（单位矩阵）
        T_ch = torch.eye(4, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)

        resolution = 0.0625

        # 检查方法是否存在
        if not hasattr(projection, 'project_sdf'):
            print("⚠️ project_sdf方法尚未实现")
            return False

        try:
            # 执行投影
            projected_sdf = projection.project_sdf(
                historical_sdf_grid,
                historical_indices,
                current_indices,
                T_ch,
                spatial_shape,
                resolution
            )

            print(f"✅ SDF投影完成")
            print(f"  - 投影结果形状: {projected_sdf.shape}")
            print(f"  - 期望形状: [{num_points}, 1]")

            if projected_sdf.shape == torch.Size([num_points, 1]):
                print(f"✅ 形状验证通过")

                # 检查是否有NaN
                if torch.isnan(projected_sdf).any():
                    print(f"❌ 投影结果包含NaN")
                    return False
                else:
                    print(f"✅ 投影结果无NaN")
                    return True
            else:
                print(f"❌ 形状验证失败")
                return False

        except Exception as e:
            print(f"❌ 投影失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("开始 Phase 2 测试套件")
        print("="*60)

        tests = [
            ("测试1: 方法存在性", self.test_projection_method_exists),
            ("测试2: 坐标变换", self.test_coordinate_transform),
            ("测试3: grid_sample形状", self.test_grid_sample_shape),
            ("测试4: 完整投影流程", self.test_projection_pipeline),
        ]

        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"❌ {test_name} 失败: {e}")
                import traceback
                traceback.print_exc()
                results.append((test_name, False))

        # 打印总结
        print("\n" + "="*60)
        print("测试总结")
        print("="*60)
        for test_name, result in results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name}: {status}")

        passed = sum(1 for _, result in results if result)
        total = len(results)
        print(f"\n总计: {passed}/{total} 通过")

        return passed == total


if __name__ == '__main__':
    tester = TestPhase2SDFProjection(device='cuda')
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
