#!/usr/bin/env python3
"""
流式融合综合测试
显式监控显存使用情况
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated

def test_stream_fusion_with_memory_monitoring():
    """测试流式融合并监控显存使用"""

    print("=" * 70)
    print("流式融合显存监控测试")
    print("=" * 70)

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过显存测试")
        return

    print(f"\n设备: {device}")

    # 创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=1,
        attn_layers=1,
        use_proj_occ=False,
        voxel_size=0.16,
        fusion_local_radius=2.0,
        crop_size=(24, 24, 16),
        use_checkpoint=False
    ).to(device)

    model.eval()
    model.enable_lightweight_state(True)

    # 测试参数
    num_steps = 10
    image_size = (256, 256)
    sequence_length = 5

    print(f"\n测试参数：")
    print(f"  - 测试步数: {num_steps}")
    print(f"  - 图像尺寸: {image_size}")
    print(f"  - 序列长度: {sequence_length}")

    # 记录显存使用
    memory_usage = []

    for step in range(num_steps):
        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()

        # 记录初始显存
        initial_allocated = torch.cuda.memory_allocated() / 1024**3
        initial_reserved = torch.cuda.memory_reserved() / 1024**3

        # 创建模拟数据
        images = torch.randn(1, 3, *image_size).to(device)
        poses = torch.eye(4).unsqueeze(0).to(device)
        intrinsics = torch.eye(3).unsqueeze(0).to(device)

        # 前向传播
        with torch.no_grad():
            output, new_state = model.forward_single_frame(
                images, poses, intrinsics, reset_state=(step % sequence_length == 0)
            )

        # 记录峰值显存
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

        memory_usage.append({
            'step': step,
            'initial_allocated': initial_allocated,
            'initial_reserved': initial_reserved,
            'peak_allocated': peak_allocated,
            'peak_reserved': peak_reserved,
            'delta_allocated': peak_allocated - initial_allocated
        })

        print(f"  步骤 {step}: "
              f"初始显存={initial_allocated:.3f}GB, "
              f"峰值显存={peak_allocated:.3f}GB, "
              f"增量={peak_allocated - initial_allocated:.3f}GB")

        # 重置峰值统计
        torch.cuda.reset_peak_memory_stats()

    # 分析结果
    print(f"\n{'='*70}")
    print("显存使用分析")
    print(f"{'='*70}")

    avg_initial = sum(m['initial_allocated'] for m in memory_usage) / len(memory_usage)
    avg_peak = sum(m['peak_allocated'] for m in memory_usage) / len(memory_usage)
    max_peak = max(m['peak_allocated'] for m in memory_usage)
    min_peak = min(m['peak_allocated'] for m in memory_usage)

    print(f"  平均初始显存: {avg_initial:.3f}GB")
    print(f"  平均峰值显存: {avg_peak:.3f}GB")
    print(f"  最大峰值显存: {max_peak:.3f}GB")
    print(f"  最小峰值显存: {min_peak:.3f}GB")
    print(f"  显存增长: {max_peak - avg_initial:.3f}GB")

    # 检查显存是否累积
    last_5 = memory_usage[-5:]
    growth_trend = sum(m['delta_allocated'] for m in last_5) / len(last_5)

    print(f"\n  显存增长趋势（最后5步）: {growth_trend:.3f}GB/步")

    if growth_trend > 0.01:
        print(f"  ⚠️  警告：显存正在累积（> 10MB/步）")
    elif growth_trend > 0.001:
        print(f"  ⚠️  警告：显存有轻微累积（> 1MB/步）")
    else:
        print(f"  ✅ 显存使用稳定（< 1MB/步）")

    print(f"\n{'='*70}")
    print("测试完成！")
    print(f"{'='*70}")

if __name__ == "__main__":
    import gc
    test_stream_fusion_with_memory_monitoring()
