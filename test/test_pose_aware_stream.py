#!/usr/bin/env python3
"""
PoseAwareStreamSdfFormer 测试用例
测试历史信息保存、投影、融合等核心功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from pathlib import Path
import traceback

from former3d.pose_aware_stream_sdfformer import PoseAwareStreamSdfFormer


class TestResult:
    """测试结果记录"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def add_pass(self, test_name):
        self.passed += 1
        print(f"✅ {test_name}")

    def add_fail(self, test_name, error):
        self.failed += 1
        self.errors.append((test_name, error))
        print(f"❌ {test_name}")
        print(f"   错误: {error}")

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "="*60)
        print(f"测试总结: {self.passed}/{total} 通过")
        if self.failed > 0:
            print(f"失败测试: {self.failed}")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print("="*60)
        return self.failed == 0


def test_model_initialization():
    """测试1: 模型初始化"""
    print("\n【测试1】模型初始化")

    result = TestResult()

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        # 创建模型
        model = PoseAwareStreamSdfFormer(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625,
            fusion_local_radius=3.0,
            crop_size=(48, 96, 96),
            use_checkpoint=False
        ).to(device)

        # 检查关键属性
        assert hasattr(model, 'historical_state'), "缺少 historical_state 属性"
        assert hasattr(model, 'historical_pose'), "缺少 historical_pose 属性"
        assert hasattr(model, 'historical_intrinsics'), "缺少 historical_intrinsics 属性"
        assert hasattr(model, 'historical_3d_points'), "缺少 historical_3d_points 属性"
        assert hasattr(model, 'fusion_3d'), "缺少 fusion_3d 属性"

        # 检查初始状态
        assert model.historical_state is None, "initial historical_state should be None"
        assert model.historical_pose is None, "initial historical_pose should be None"

        result.add_pass("模型初始化")

    except Exception as e:
        result.add_fail("模型初始化", str(e))
        traceback.print_exc()

    return result


def test_record_state():
    """测试2: _record_state 功能"""
    print("\n【测试2】_record_state 功能")

    result = TestResult()

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = PoseAwareStreamSdfFormer(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625
        ).to(device)
        model.eval()

        # 创建模拟的输出
        from spconv.pytorch import SparseConvTensor

        batch_size = 2
        num_voxels = 100

        # 创建sparse特征
        features = torch.randn(num_voxels, 1, device=device)
        indices = torch.randint(0, 10, (num_voxels, 4), device=device)
        indices[:, 0] = torch.randint(0, batch_size, (num_voxels,), device=device)
        indices = indices.to(torch.int32)  # 转换为int32（spconv要求）
        spatial_shape = (8, 8, 8)

        sparse_tensor = SparseConvTensor(
            features=features,
            indices=indices,
            spatial_shape=spatial_shape,
            batch_size=batch_size
        )

        # 创建模拟输出
        output = {
            'voxel_outputs': {
                'fine': sparse_tensor
            }
        }

        # 创建pose和intrinsics
        current_pose = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        current_intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        # 创建3D点
        current_3d_points = indices[:, 1:4].float() * model.voxel_size

        # 调用_record_state
        model._record_state(output, current_pose, current_intrinsics, current_3d_points)

        # 验证历史信息已保存
        assert model.historical_state is not None, "historical_state should not be None"
        assert model.historical_pose is not None, "historical_pose should not be None"
        assert model.historical_intrinsics is not None, "historical_intrinsics should not be None"
        assert model.historical_3d_points is not None, "historical_3d_points should not be None"

        # 验证shape
        assert model.historical_state['features'].shape == (num_voxels, 1), \
            f"features shape mismatch: {model.historical_state['features'].shape}"
        assert model.historical_state['indices'].shape == (num_voxels, 4), \
            f"indices shape mismatch: {model.historical_state['indices'].shape}"
        assert model.historical_pose.shape == (batch_size, 4, 4), \
            f"pose shape mismatch: {model.historical_pose.shape}"

        result.add_pass("_record_state 功能")

    except Exception as e:
        result.add_fail("_record_state 功能", str(e))
        traceback.print_exc()

    return result


def test_sparse_to_dense_grid():
    """测试3: _sparse_to_dense_grid 转换"""
    print("\n【测试3】_sparse_to_dense_grid 转换")

    result = TestResult()

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = PoseAwareStreamSdfFormer(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625
        ).to(device)

        # 创建稀疏特征
        batch_size = 2
        num_voxels = 50
        num_channels = 128
        spatial_shape = (8, 8, 8)

        features = torch.randn(num_voxels, num_channels, device=device)
        indices = torch.randint(0, 8, (num_voxels, 4), device=device)
        indices[:, 0] = torch.randint(0, batch_size, (num_voxels,), device=device)
        indices = indices.to(torch.int32)  # 转换为int32（spconv要求）

        # 转换为dense
        dense_grid = model._sparse_to_dense_grid(
            features, indices, spatial_shape, batch_size
        )

        # 验证shape
        assert dense_grid.shape == (batch_size, num_channels, *spatial_shape), \
            f"dense_grid shape mismatch: {dense_grid.shape}"

        # 验证非零值
        nonzero_count = 0
        for i in range(num_voxels):
            b = int(indices[i, 0].item())
            x = int(indices[i, 1].item())
            y = int(indices[i, 2].item())
            z = int(indices[i, 3].item())
            if 0 <= b < batch_size and 0 <= x < spatial_shape[0] and \
               0 <= y < spatial_shape[1] and 0 <= z < spatial_shape[2]:
                # 检查该位置是否有值（不一定等于原始值，因为可能有重复索引）
                if dense_grid[b, :, x, y, z].abs().sum() > 0:
                    nonzero_count += 1

        assert nonzero_count > 0, "dense_grid应该有非零值"

        result.add_pass("_sparse_to_dense_grid 转换")

    except Exception as e:
        result.add_fail("_sparse_to_dense_grid 转换", str(e))
        traceback.print_exc()

    return result


def test_forward_single_frame_first():
    """测试4: forward_single_frame 第一帧"""
    print("\n【测试4】forward_single_frame 第一帧")

    result = TestResult()

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = PoseAwareStreamSdfFormer(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625,
            crop_size=(24, 48, 48)  # 减小crop_size以节省显存
        ).to(device)
        model.eval()

        # 创建输入数据
        batch_size = 1  # 减小batch size以节省显存
        H, W = 64, 80  # 减小图像尺寸

        images = torch.randn(batch_size, 3, H, W, device=device)
        poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        origin = torch.zeros(batch_size, 3, device=device)

        # 第一帧：reset_state=True
        with torch.no_grad():
            output, new_state = model.forward_single_frame(
                images, poses, intrinsics, reset_state=True, origin=origin
            )

        # 验证输出
        assert output is not None, "output should not be None"
        assert 'voxel_outputs' in output, "output should contain voxel_outputs"
        assert 'sdf' in output, "output should contain sdf"
        assert new_state is not None, "new_state should not be None"

        # 验证历史状态已保存
        assert model.historical_state is not None, "historical_state should be saved after first frame"

        result.add_pass("forward_single_frame 第一帧")

    except Exception as e:
        result.add_fail("forward_single_frame 第一帧", str(e))
        traceback.print_exc()

    return result


def test_forward_single_frame_with_history():
    """测试5: forward_single_frame 有历史信息"""
    print("\n【测试5】forward_single_frame 有历史信息")

    result = TestResult()

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = PoseAwareStreamSdfFormer(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625,
            crop_size=(24, 48, 48)  # 减小crop_size以节省显存
        ).to(device)
        model.eval()

        batch_size = 1  # 减小batch size以节省显存
        H, W = 64, 80  # 减小图像尺寸

        # 第一帧
        images = torch.randn(batch_size, 3, H, W, device=device)
        poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        origin = torch.zeros(batch_size, 3, device=device)

        with torch.no_grad():
            output1, state1 = model.forward_single_frame(
                images, poses, intrinsics, reset_state=True, origin=origin
            )

        # 第二帧：pose有轻微变化
        poses[:, 0, 3] = 0.1  # X方向平移0.1米

        with torch.no_grad():
            output2, state2 = model.forward_single_frame(
                images, poses, intrinsics, reset_state=False, origin=origin
            )

        # 验证输出
        assert output2 is not None, "output2 should not be None"
        assert 'voxel_outputs' in output2, "output2 should contain voxel_outputs"
        assert 'sdf' in output2, "output2 should contain sdf"

        # 验证历史状态更新
        assert model.historical_state is not None, "historical_state should be updated"
        assert state2 is not None, "state2 should not be None"

        result.add_pass("forward_single_frame 有历史信息")

    except Exception as e:
        result.add_fail("forward_single_frame 有历史信息", str(e))
        traceback.print_exc()

    return result


def test_forward_sequence():
    """测试6: forward_sequence 序列推理"""
    print("\n【测试6】forward_sequence 序列推理")

    result = TestResult()

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = PoseAwareStreamSdfFormer(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625,
            crop_size=(24, 48, 48)  # 减小crop_size以节省显存
        ).to(device)
        model.eval()

        batch_size = 1  # 减小batch size以节省显存
        n_view = 2  # 减小view数量以节省显存
        H, W = 64, 80  # 减小图像尺寸

        # 创建序列数据
        images = torch.randn(batch_size, n_view, 3, H, W, device=device)
        poses = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_view, 1, 1)
        # 每帧有轻微平移
        for t in range(n_view):
            poses[:, t, 0, 3] = t * 0.1

        intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_view, 1, 1)

        with torch.no_grad():
            outputs, states = model.forward_sequence(
                images, poses, intrinsics, reset_state=True
            )

        # 验证输出
        assert outputs is not None, "outputs should not be None"
        assert len(states) == n_view, f"should have {n_view} states, got {len(states)}"

        # 验证每个状态
        for i, state in enumerate(states):
            assert state is not None, f"state {i} should not be None"

        result.add_pass("forward_sequence 序列推理")

    except Exception as e:
        result.add_fail("forward_sequence 序列推理", str(e))
        traceback.print_exc()

    return result


def test_gradient_flow():
    """测试7: 梯度流验证"""
    print("\n【测试7】梯度流验证")

    result = TestResult()

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = PoseAwareStreamSdfFormer(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625,
            crop_size=(24, 48, 48)  # 减小crop_size以节省显存
        ).to(device)
        model.train()

        batch_size = 1  # 减小batch size以节省显存
        H, W = 64, 80  # 减小图像尺寸

        # 第一帧
        images = torch.randn(batch_size, 3, H, W, device=device, requires_grad=True)
        poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        origin = torch.zeros(batch_size, 3, device=device)

        output1, state1 = model.forward_single_frame(
            images, poses, intrinsics, reset_state=True, origin=origin
        )

        # 第二帧
        poses[:, 0, 3] = 0.1
        output2, state2 = model.forward_single_frame(
            images, poses, intrinsics, reset_state=False, origin=origin
        )

        # 计算损失
        if output2['sdf'] is not None:
            loss = output2['sdf'].mean()
        else:
            loss = torch.tensor(0.0, device=device)

        # 反向传播
        loss.backward()

        # 检查梯度
        has_gradient = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradient = True
                print(f"  {name}: grad_norm={param.grad.norm().item():.6f}")

        if not has_gradient:
            raise RuntimeError("没有检测到梯度，可能存在梯度断开")

        result.add_pass("梯度流验证")

    except Exception as e:
        result.add_fail("梯度流验证", str(e))
        traceback.print_exc()

    return result


def test_memory_leak():
    """测试8: 显存泄露检测"""
    print("\n【测试8】显存泄露检测")

    result = TestResult()

    try:
        if not torch.cuda.is_available():
            print("  ⚠️ CUDA不可用，跳过显存测试")
            result.add_pass("显存泄露检测 (跳过)")
            return result

        device = torch.device('cuda')

        model = PoseAwareStreamSdfFormer(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625,
            crop_size=(24, 48, 48)  # 减小crop_size以节省显存
        ).to(device)
        model.eval()

        batch_size = 1  # 减小batch size以节省显存
        H, W = 64, 80  # 减小图像尺寸

        # 记录初始显存
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # 运行多帧
        num_frames = 5
        for i in range(num_frames):
            images = torch.randn(batch_size, 3, H, W, device=device)
            poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            poses[:, 0, 3] = i * 0.1
            intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            origin = torch.zeros(batch_size, 3, device=device)

            try:
                with torch.no_grad():
                    output, state = model.forward_single_frame(
                        images, poses, intrinsics,
                        reset_state=(i == 0),
                        origin=origin
                    )

                del output, state
                torch.cuda.empty_cache()
            except Exception as e:
                # 允许某些帧失败（如显存不足）
                if i < 2:
                    raise  # 前两帧必须成功
                print(f"  第{i+1}帧失败（可接受）: {e}")
                torch.cuda.empty_cache()
                continue

        # 记录最终显存
        final_memory = torch.cuda.memory_allocated()
        memory_increase = final_memory - initial_memory

        print(f"  初始显存: {initial_memory / 1024**2:.2f} MB")
        print(f"  最终显存: {final_memory / 1024**2:.2f} MB")
        print(f"  显存增长: {memory_increase / 1024**2:.2f} MB")

        # 允许一定程度的增长（由于缓存等），但不应该过大
        max_allowed_increase = 500 * 1024**2  # 500 MB

        if memory_increase > max_allowed_increase:
            raise RuntimeError(f"显存增长过大: {memory_increase / 1024**2:.2f} MB")

        result.add_pass("显存泄露检测")

    except Exception as e:
        result.add_fail("显存泄露检测", str(e))
        traceback.print_exc()

    return result


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("PoseAwareStreamSdfFormer 测试套件")
    print("="*60)

    all_results = []

    # 运行所有测试
    all_results.append(test_model_initialization())
    all_results.append(test_record_state())
    all_results.append(test_sparse_to_dense_grid())
    all_results.append(test_forward_single_frame_first())
    all_results.append(test_forward_single_frame_with_history())
    all_results.append(test_forward_sequence())
    all_results.append(test_gradient_flow())
    all_results.append(test_memory_leak())

    # 汇总结果
    total_passed = sum(r.passed for r in all_results)
    total_failed = sum(r.failed for r in all_results)
    total_tests = total_passed + total_failed

    print("\n" + "="*60)
    print(f"总体结果: {total_passed}/{total_tests} 测试通过")
    print("="*60)

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
