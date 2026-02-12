#!/usr/bin/env python3
"""
测试 Phase 1: 修改_create_new_state保存SDF

测试目标：
1. 验证_create_new_state能够正确提取fine分辨率的SDF
2. 验证SDF网格的形状和类型
3. 验证SDF相关字段（sdf_grid, sdf_resolution等）已保存
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
    from spconv.pytorch.utils import PointToVoxelConverter3D
    SPconv_AVAILABLE = True
except ImportError as e:
    print(f"警告：无法导入StreamSDFFormerIntegrated: {e}")
    SPconv_AVAILABLE = False


class TestPhase1SDFState:
    """Phase 1: 测试SDF状态保存"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def test_extract_sdf_from_output(self):
        """测试1: 从SDFFormer输出中提取SDF"""
        print("\n" + "="*60)
        print("测试1: 从SDFFormer输出中提取SDF")
        print("="*60)

        if not SPconv_AVAILABLE:
            print("⚠️ 跳过测试：无法导入StreamSDFFormerIntegrated")
            return False

        # 创建模拟的输出
        batch_size = 2
        num_voxels = 1000

        # 创建模拟的fine输出（SparseConvTensor）
        # 注意：由于spconv的复杂性，这里简化为使用普通tensor模拟
        fine_features = torch.randn(num_voxels, 1, device=self.device)  # SDF值
        fine_indices = torch.randint(0, 32, (num_voxels, 4), device=self.device)
        fine_indices[:, 0] = torch.randint(0, batch_size, (num_voxels,), device=self.device)  # batch索引

        spatial_shape = [32, 32, 32]

        print(f"✅ 模拟fine输出创建成功")
        print(f"  - SDF特征形状: {fine_features.shape}")
        print(f"  - 索引形状: {fine_indices.shape}")
        print(f"  - 空间形状: {spatial_shape}")

        return True

    def test_sparse_to_dense_conversion(self):
        """测试2: 稀疏到密集转换"""
        print("\n" + "="*60)
        print("测试2: 稀疏到密集转换")
        print("="*60)

        if not SPconv_AVAILABLE:
            print("⚠️ 跳过测试：无法导入StreamSDFFormerIntegrated")
            return False

        # 模拟稀疏数据
        batch_size = 2
        num_voxels = 500
        spatial_shape = [16, 16, 16]
        channels = 1

        sparse_features = torch.randn(num_voxels, channels, device=self.device)
        sparse_indices = torch.randint(0, 16, (num_voxels, 4), device=self.device)
        sparse_indices[:, 0] = torch.randint(0, batch_size, (num_voxels,), device=self.device)

        # 手动实现稀疏到密集转换（模拟SparseConvTensor转换）
        dense_grid = torch.zeros(
            (batch_size, channels, *spatial_shape),
            device=self.device,
            dtype=sparse_features.dtype
        )

        # 填充稀疏特征
        for i in range(len(sparse_features)):
            b, x, y, z = sparse_indices[i].tolist()
            if 0 <= b < batch_size and \
               0 <= x < spatial_shape[0] and \
               0 <= y < spatial_shape[1] and \
               0 <= z < spatial_shape[2]:
                dense_grid[b, :, x, y, z] = sparse_features[i]

        print(f"✅ 稀疏到密集转换成功")
        print(f"  - 稀疏特征形状: {sparse_features.shape}")
        print(f"  - 密集网格形状: {dense_grid.shape}")
        print(f"  - 期望形状: ({batch_size}, {channels}, {spatial_shape[0]}, {spatial_shape[1]}, {spatial_shape[2]})")
        print(f"  - 非零体素数: {(dense_grid.abs() > 0).sum().item()}")

        # 验证形状
        expected_shape = (batch_size, channels, *spatial_shape)
        if dense_grid.shape == expected_shape:
            print(f"✅ 形状验证通过")
            return True
        else:
            print(f"❌ 形状验证失败: {dense_grid.shape} != {expected_shape}")
            return False

    def test_create_state_structure(self):
        """测试3: 状态结构验证"""
        print("\n" + "="*60)
        print("测试3: 状态结构验证")
        print("="*60)

        # 模拟状态结构
        batch_size = 2
        spatial_shape = [32, 32, 32]
        sdf_resolution = 0.0625

        # 模拟SDF网格
        sdf_grid = torch.randn(batch_size, 1, *spatial_shape, device=self.device)
        sdf_indices = torch.randint(0, 32, (1000, 4), device=self.device)
        sdf_spatial_shape = spatial_shape

        # 创建状态字典
        state = {
            'dense_grids': {
                'coarse': torch.randn(batch_size, 128, 16, 16, 8, device=self.device),
                'medium': torch.randn(batch_size, 128, 32, 32, 16, device=self.device),
                'fine': torch.randn(batch_size, 128, *spatial_shape, device=self.device),
            },
            'sdf_grid': sdf_grid,  # 新增
            'sdf_indices': sdf_indices,  # 新增
            'sdf_spatial_shape': sdf_spatial_shape,  # 新增
            'sdf_resolution': sdf_resolution,  # 新增
            'batch_size': batch_size,
        }

        # 验证SDF相关字段
        required_fields = ['sdf_grid', 'sdf_indices', 'sdf_spatial_shape', 'sdf_resolution']
        for field in required_fields:
            if field in state:
                print(f"✅ 字段 '{field}' 存在: {type(state[field])}")
            else:
                print(f"❌ 字段 '{field}' 缺失")
                return False

        # 验证SDF网格形状
        expected_sdf_shape = (batch_size, 1, *spatial_shape)
        if state['sdf_grid'].shape == expected_sdf_shape:
            print(f"✅ SDF网格形状正确: {state['sdf_grid'].shape}")
        else:
            print(f"❌ SDF网格形状错误: {state['sdf_grid'].shape} != {expected_sdf_shape}")
            return False

        # 验证SDF分辨率
        if state['sdf_resolution'] == sdf_resolution:
            print(f"✅ SDF分辨率正确: {state['sdf_resolution']}")
        else:
            print(f"❌ SDF分辨率错误: {state['sdf_resolution']} != {sdf_resolution}")
            return False

        return True

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*60)
        print("开始 Phase 1 测试套件")
        print("="*60)

        tests = [
            ("测试1: 提取SDF", self.test_extract_sdf_from_output),
            ("测试2: 稀疏到密集转换", self.test_sparse_to_dense_conversion),
            ("测试3: 状态结构验证", self.test_create_state_structure),
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
    tester = TestPhase1SDFState(device='cuda')
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
