#!/usr/bin/env python3
"""
测试轻量级状态模式

验证enable_lightweight_state()方法能正确清理冗余数据
"""

import os
import sys
import torch
import gc

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
except ImportError as e:
    print(f"❌ 无法导入StreamSDFFormerIntegrated: {e}")
    sys.exit(1)


# 模拟sparse tensor（全局）
class MockSparseTensor:
    """模拟sparse tensor用于测试"""
    def __init__(self, num_voxels):
        self.features = torch.randn(num_voxels, 1)
        self.indices = torch.cat([
            torch.randint(0, 32, (num_voxels, 1)),
            torch.randint(0, 32, (num_voxels, 1)),
            torch.randint(0, 32, (num_voxels, 1)),
            torch.zeros(num_voxels, 1)
        ], dim=1)


def get_gpu_memory_mb():
    """获取GPU显存使用量（MB）"""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated() / (1024 ** 2)


def create_mock_output():
    """创建模拟的模型输出"""
    # 创建包含大量数据的输出
    num_voxels = 1000

    return {
        'voxel_outputs': {
            'fine': MockSparseTensor(num_voxels)
        },
        'sdf': torch.randn(num_voxels, 1),
        'occupancy': torch.randn(num_voxels, 1),
        'extra_data': [torch.randn(1000, 100) for _ in range(10)]  # 额外的中间变量
    }


def test_lightweight_state_mode():
    """
    测试1: 轻量级状态模式
    验证在轻量级模式下，不会保存完整的output字典
    """
    print("=" * 70)
    print("测试1: 轻量级状态模式")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=1,
        attn_layers=1,
        use_proj_occ=False,
        voxel_size=0.16,
        fusion_local_radius=2.0,
        crop_size=(24, 24, 16)
    )
    model = model.to(device)

    # 确保轻量级模式启用（默认）
    model.enable_lightweight_state(True)

    # 创建模拟输出和位姿
    mock_output = create_mock_output()
    mock_pose = torch.eye(4).unsqueeze(0).to(device)

    # 移动模拟输出中的张量到设备
    for key, value in mock_output.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if hasattr(v, 'features') and hasattr(v, 'indices'):  # 检查是否为MockSparseTensor
                    v.features = v.features.to(device)
                    v.indices = v.indices.to(device)
        elif isinstance(value, torch.Tensor):
            mock_output[key] = value.to(device)
        elif isinstance(value, list):
            mock_output[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]

    # 创建状态
    state = model._create_new_state(mock_output, mock_pose)

    print(f"\n创建的状态键: {list(state.keys())}")

    # 验证：不应该包含完整的output或original_features
    if 'output' in state:
        print("❌ 失败：状态中包含完整的output（非轻量级模式）")
        return False
    elif 'original_features' in state:
        print("❌ 失败：状态中包含original_features（非轻量级模式）")
        return False
    else:
        print("✅ 通过：轻量级模式正常工作，没有保存冗余数据")
        state_size = sum(v.numel() * 4 / 1024**2 for v in state.values() if isinstance(v, torch.Tensor))
        print(f"   状态大小估计: {state_size:.2f} MB")
        return True


def test_non_lightweight_mode():
    """
    测试2: 非轻量级模式
    验证在非轻量级模式下，会保存完整的output字典（用于调试）
    """
    print("\n" + "=" * 70)
    print("测试2: 非轻量级模式（调试模式）")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=1,
        attn_layers=1,
        use_proj_occ=False,
        voxel_size=0.16,
        fusion_local_radius=2.0,
        crop_size=(24, 24, 16)
    )
    model = model.to(device)

    # 禁用轻量级模式（非轻量级模式）
    model.enable_lightweight_state(False)

    # 创建模拟输出和位姿
    mock_output = create_mock_output()
    mock_pose = torch.eye(4).unsqueeze(0).to(device)

    # 移动模拟输出中的张量到设备
    for key, value in mock_output.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if hasattr(v, 'features') and hasattr(v, 'indices'):  # 检查是否为MockSparseTensor
                    v.features = v.features.to(device)
                    v.indices = v.indices.to(device)
        elif isinstance(value, torch.Tensor):
            mock_output[key] = value.to(device)
        elif isinstance(value, list):
            mock_output[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]

    # 创建状态
    state = model._create_new_state(mock_output, mock_pose)

    print(f"\n创建的状态键: {list(state.keys())}")

    # 验证：应该包含完整的output和original_features
    if 'output' in state and 'original_features' in state:
        print("✅ 通过：非轻量级模式正常工作，保存了完整输出")
        # 估算大小
        basic_size = sum(v.numel() * 4 / 1024**2 for k, v in state.items() if isinstance(v, torch.Tensor) and k not in ['output', 'original_features'])
        total_size = sum(v.numel() * 4 / 1024**2 for v in state.values() if isinstance(v, torch.Tensor))
        print(f"   基础状态大小: {basic_size:.2f} MB")
        print(f"   总状态大小: {total_size:.2f} MB")
        print(f"   额外数据: {total_size - basic_size:.2f} MB")
        return True
    else:
        print("❌ 失败：非轻量级模式下应该保存完整输出")
        return False


def test_switching_modes():
    """
    测试3: 切换模式
    验证可以从轻量级模式切换到非轻量级模式，反之亦然
    """
    print("\n" + "=" * 70)
    print("测试3: 切换模式")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=1,
        attn_layers=1,
        use_proj_occ=False,
        voxel_size=0.16,
        fusion_local_radius=2.0,
        crop_size=(24, 24, 16)
    )
    model = model.to(device)

    # 模拟输出
    mock_output = create_mock_output()
    mock_pose = torch.eye(4).unsqueeze(0).to(device)

    # 移动张量到设备
    for key, value in mock_output.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if hasattr(v, 'features') and hasattr(v, 'indices'):  # 检查是否为MockSparseTensor
                    v.features = v.features.to(device)
                    v.indices = v.indices.to(device)
        elif isinstance(value, torch.Tensor):
            mock_output[key] = value.to(device)
        elif isinstance(value, list):
            mock_output[key] = [v.to(device) if isinstance(v, torch.Tensor) else v for v in value]

    # 测试1: 轻量级模式
    model.enable_lightweight_state(True)
    state1 = model._create_new_state(mock_output, mock_pose)
    assert 'output' not in state1, "轻量级模式不应该包含output"
    print("  ✅ 轻量级模式: 不包含冗余数据")

    # 测试2: 切换到非轻量级模式
    model.enable_lightweight_state(False)
    state2 = model._create_new_state(mock_output, mock_pose)
    assert 'output' in state2, "非轻量级模式应该包含output"
    print("  ✅ 非轻量级模式: 包含完整输出")

    # 测试3: 切换回轻量级模式
    model.enable_lightweight_state(True)
    state3 = model._create_new_state(mock_output, mock_pose)
    assert 'output' not in state3, "轻量级模式不应该包含output"
    print("  ✅ 切换回轻量级模式: 不包含冗余数据")

    print("\n✅ 通过：模式切换功能正常")
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("轻量级状态模式测试套件")
    print("=" * 70)

    results = []

    try:
        results.append(test_lightweight_state_mode())
    except Exception as e:
        print(f"\n❌ 测试1异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    try:
        results.append(test_non_lightweight_mode())
    except Exception as e:
        print(f"\n❌ 测试2异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    try:
        results.append(test_switching_modes())
    except Exception as e:
        print(f"\n❌ 测试3异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(False)

    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)

    passed = sum(results)
    total = len(results)

    print(f"\n通过: {passed}/{total}")

    if passed == total:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    exit(exit_code)
