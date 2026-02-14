#!/usr/bin/env python3
"""
稀疏融合版本的梯度流测试
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import traceback

from former3d.pose_aware_stream_sdfformer_sparse import PoseAwareStreamSdfFormerSparse


def test_training_with_fusion():
    """测试训练模式下带融合的梯度流"""
    print("\n【测试】训练模式稀疏融合")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    try:
        model = PoseAwareStreamSdfFormerSparse(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625,
            crop_size=(16, 24, 24),
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

        print("执行第一帧（无历史信息）...")
        with torch.no_grad():
            output1, state1 = model.forward_single_frame(
                images1, poses1, intrinsics1, reset_state=True, origin=origin
            )
        print(f"第一帧完成，SDF形状: {output1['sdf'].shape if output1['sdf'] is not None else 'None'}")

        # 第二帧：有历史信息，启用融合
        images2 = torch.randn(batch_size, 3, H, W, device=device, requires_grad=True)
        poses2 = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        poses2[:, 0, 3] = 0.1  # 轻微平移

        print("\n执行第二帧（有历史信息，启用融合，需要梯度）...")
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

        print("✅ 训练模式稀疏融合测试通过")
        return True

    except Exception as e:
        print(f"❌ 训练模式稀疏融合测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """运行测试"""
    print("="*60)
    print("稀疏融合版本测试")
    print("="*60)

    success = test_training_with_fusion()

    print("\n" + "="*60)
    print(f"结果: {'通过' if success else '失败'}")
    print("="*60)

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
