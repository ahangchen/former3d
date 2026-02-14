#!/usr/bin/env python3
"""
简化的梯度流测试
逐步定位训练模式问题
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import traceback

from former3d.pose_aware_stream_sdfformer import PoseAwareStreamSdfFormer


def test_first_frame_gradient():
    """测试第一帧的梯度流（最简单的情况）"""
    print("\n【测试1】第一帧梯度流")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    try:
        model = PoseAwareStreamSdfFormer(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625,
            crop_size=(16, 24, 24),  # 非常小的crop size
            use_checkpoint=False
        ).to(device)
        model.train()  # 训练模式

        batch_size = 1
        H, W = 48, 64  # 小图像

        images = torch.randn(batch_size, 3, H, W, device=device, requires_grad=True)
        poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        origin = torch.zeros(batch_size, 3, device=device)

        print(f"输入形状: images={images.shape}, poses={poses.shape}")

        # 前向传播
        print("执行前向传播...")
        output, state = model.forward_single_frame(
            images, poses, intrinsics, reset_state=True, origin=origin
        )

        print(f"输出键: {list(output.keys())}")
        if 'sdf' in output and output['sdf'] is not None:
            print(f"SDF形状: {output['sdf'].shape}")

        # 计算损失
        if output['sdf'] is not None:
            loss = output['sdf'].mean()
            print(f"损失值: {loss.item()}")

            # 反向传播
            print("执行反向传播...")
            loss.backward()

            # 检查梯度
            grad_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 0:
                        grad_count += 1
                        if grad_count <= 5:  # 只打印前5个
                            print(f"  {name}: grad_norm={grad_norm:.6f}")

            print(f"总共有 {grad_count} 个参数有梯度")

        print("✅ 第一帧梯度流测试通过")
        return True

    except Exception as e:
        print(f"❌ 第一帧梯度流测试失败: {e}")
        traceback.print_exc()
        return False


def test_second_frame_gradient():
    """测试第二帧的梯度流（有历史信息）"""
    print("\n【测试2】第二帧梯度流（有历史信息）")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    try:
        model = PoseAwareStreamSdfFormer(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625,
            crop_size=(16, 24, 24),  # 非常小的crop size
            use_checkpoint=False
        ).to(device)
        model.train()  # 训练模式

        batch_size = 1
        H, W = 48, 64

        # 第一帧
        images1 = torch.randn(batch_size, 3, H, W, device=device, requires_grad=True)
        poses1 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics1 = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        origin = torch.zeros(batch_size, 3, device=device)

        print("执行第一帧...")
        with torch.no_grad():  # 第一帧不需要梯度
            output1, state1 = model.forward_single_frame(
                images1, poses1, intrinsics1, reset_state=True, origin=origin
            )
        print(f"第一帧完成，SDF形状: {output1['sdf'].shape if output1['sdf'] is not None else 'None'}")

        # 第二帧：有历史信息
        images2 = torch.randn(batch_size, 3, H, W, device=device, requires_grad=True)
        poses2 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        poses2[:, 0, 3] = 0.1  # 轻微平移

        print("\n执行第二帧（有历史信息，需要梯度）...")
        output2, state2 = model.forward_single_frame(
            images2, poses2, intrinsics1, reset_state=False, origin=origin
        )

        print(f"第二帧输出键: {list(output2.keys())}")
        if 'sdf' in output2 and output2['sdf'] is not None:
            print(f"第二帧SDF形状: {output2['sdf'].shape}")

            # 计算损失
            loss = output2['sdf'].mean()
            print(f"损失值: {loss.item()}")

            # 反向传播
            print("执行反向传播...")
            loss.backward()

            # 检查梯度
            grad_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 0:
                        grad_count += 1
                        if grad_count <= 5:
                            print(f"  {name}: grad_norm={grad_norm:.6f}")

            print(f"总共有 {grad_count} 个参数有梯度")

        print("✅ 第二帧梯度流测试通过")
        return True

    except Exception as e:
        print(f"❌ 第二帧梯度流测试失败: {e}")
        traceback.print_exc()
        return False


def test_sequence_gradient():
    """测试序列的梯度流"""
    print("\n【测试3】序列梯度流")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    try:
        model = PoseAwareStreamSdfFormer(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625,
            crop_size=(16, 24, 24),
            use_checkpoint=False
        ).to(device)
        model.train()

        batch_size = 1
        n_view = 2
        H, W = 48, 64

        images = torch.randn(batch_size, n_view, 3, H, W, device=device, requires_grad=True)
        poses = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_view, 1, 1)
        for t in range(n_view):
            poses[:, t, 0, 3] = t * 0.05

        intrinsics = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_view, 1, 1)

        print(f"输入形状: images={images.shape}, poses={poses.shape}")

        # 执行序列前向传播
        print("执行序列前向传播...")
        outputs, states = model.forward_sequence(
            images, poses, intrinsics, reset_state=True
        )

        print(f"输出序列长度: {len(states)}")

        # 计算最后一个输出的损失
        if outputs and 'sdf' in outputs and outputs['sdf'] is not None:
            loss = outputs['sdf'].mean()
            print(f"损失值: {loss.item()}")

            # 反向传播
            print("执行反向传播...")
            loss.backward()

            # 检查梯度
            grad_count = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 0:
                        grad_count += 1
                        if grad_count <= 5:
                            print(f"  {name}: grad_norm={grad_norm:.6f}")

            print(f"总共有 {grad_count} 个参数有梯度")

        print("✅ 序列梯度流测试通过")
        return True

    except Exception as e:
        print(f"❌ 序列梯度流测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """运行所有梯度流测试"""
    print("="*60)
    print("PoseAwareStreamSdfFormer 梯度流测试")
    print("="*60)

    results = []

    # 运行测试
    results.append(test_first_frame_gradient())

    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results.append(test_second_frame_gradient())

    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results.append(test_sequence_gradient())

    # 总结
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"结果: {passed}/{total} 测试通过")
    print("="*60)

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
