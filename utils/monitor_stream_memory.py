#!/usr/bin/env python3
"""
StreamSDFFormer 模型逐层显存监控
帮助识别显存瓶颈
"""

import sys
import torch
import gc
from memory_monitor_layer import MemoryMonitor, memory_context, monitor_layer

sys.path.insert(0, '/home/cwh/coding/former3d')

from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated


def monitor_model_layers():
    """监控 StreamSDFFormer 模型的逐层显存占用"""
    print("="*80)
    print("StreamSDFFormer 逐层显存监控")
    print("="*80)

    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return False

    device = torch.device('cuda:0')
    print(f"✅ 使用设备: {device}\n")

    # 创建数据集和模型
    print("创建数据集和模型...")
    dataset = MultiSequenceTartanAirDataset(
        data_root='/home/cwh/Study/dataset/tartanair',
        n_view=3,  # 减少视图数量以加快测试
        crop_size=(6, 6, 4),  # 较小的crop size
        voxel_size=0.20,  # 较大的voxel size
        max_sequences=1  # 只用1个序列
    )

    model = StreamSDFFormerIntegrated(
        attn_heads=1,
        attn_layers=1,
        use_proj_occ=False,
        voxel_size=0.20,
        fusion_local_radius=2.0,
        crop_size=(6, 6, 4)
    ).to(device)

    print(f"✅ 数据集样本数: {len(dataset)}")
    print(f"✅ 模型参数数: {sum(p.numel() for p in model.parameters()):,}")

    # 获取一个batch
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1,
                           collate_fn=dataset.collate_fn)
    batch = next(iter(dataloader))

    images = batch['rgb_images'].to(device)
    poses = batch['poses'].to(device)
    intrinsics = batch['intrinsics'].to(device)

    print(f"✅ 输入数据:")
    print(f"  - images: {images.shape}")
    print(f"  - poses: {poses.shape}")
    print(f"  - intrinsics: {intrinsics.shape}\n")

    # 创建显存监控器
    monitor = MemoryMonitor(device)

    # 监控模型初始化
    with memory_context(monitor, "model_init"):
        # 模型已经在GPU上，这里只是记录状态
        pass

    model.eval()

    # 预热
    print("预热...")
    with torch.no_grad():
        for _ in range(2):
            _ = model.forward_sequence(images, poses, intrinsics)
        torch.cuda.empty_cache()
    print("✅ 预热完成\n")

    # 清理记录，开始正式监控
    monitor.clear()

    # 监控前向传播
    print("开始监控前向传播...")
    print("-"*80)

    with torch.no_grad():
        with memory_context(monitor, "forward_sequence_total"):
            outputs, states = model.forward_sequence(images, poses, intrinsics)

            # 监控每个时间步
            for t, (output, state) in enumerate(zip(outputs, states)):
                with memory_context(monitor, f"timestep_{t}"):
                    # 监控输出
                    if output is not None:
                        with memory_context(monitor, f"timestep_{t}_output"):
                            if hasattr(output, 'shape'):
                                _ = output.shape
                            elif isinstance(output, (list, tuple)) and len(output) > 0:
                                if hasattr(output[0], 'shape'):
                                    _ = output[0].shape

                    # 监控状态
                    if state:
                        for key, value in state.items():
                            with memory_context(monitor, f"timestep_{t}_state_{key}"):
                                if isinstance(value, torch.Tensor):
                                    _ = value.shape
                                elif isinstance(value, (list, tuple)) and len(value) > 0:
                                    if isinstance(value[0], torch.Tensor):
                                        _ = value[0].shape

    # 打印摘要
    print("\n" + "="*80)
    monitor.print_summary(top_n=20)
    monitor.print_detailed()

    # 分析瓶颈
    print("显存瓶颈分析:")
    print("-"*80)

    # 找出增量最大的层
    sorted_records = sorted(monitor.records, key=lambda x: x['delta_allocated'], reverse=True)
    top_5 = sorted_records[:5]

    print("\n显存增量最大的5个组件:")
    for i, record in enumerate(top_5, 1):
        delta = record['delta_allocated']
        pct = (delta / sorted_records[0]['delta_allocated'] * 100) if sorted_records[0]['delta_allocated'] > 0 else 0
        print(f"  {i}. {record['layer']:<50} {delta:>10.4f} GB ({pct:>5.1f}%)")

    # 按类型分类
    print("\n按操作类型分类:")
    timestep_ops = sum(r['delta_allocated'] for r in monitor.records if 'timestep_' in r['layer'])
    state_ops = sum(r['delta_allocated'] for r in monitor.records if 'state' in r['layer'])
    output_ops = sum(r['delta_allocated'] for r in monitor.records if 'output' in r['layer'])
    other_ops = sum(r['delta_allocated'] for r in monitor.records if 'timestep_' not in r['layer'] and 'state' not in r['layer'] and 'output' not in r['layer'])

    total_delta = max(sum(r['delta_allocated'] for r in monitor.records), 0.0001)

    print(f"  - 时间步操作: {timestep_ops:>10.4f} GB ({timestep_ops/total_delta*100:>5.1f}%)")
    print(f"  - 状态操作:   {state_ops:>10.4f} GB ({state_ops/total_delta*100:>5.1f}%)")
    print(f"  - 输出操作:   {output_ops:>10.4f} GB ({output_ops/total_delta*100:>5.1f}%)")
    print(f"  - 其他操作:   {other_ops:>10.4f} GB ({other_ops/total_delta*100:>5.1f}%)")
    print("="*80 + "\n")

    # 建议
    print("优化建议:")
    print("-"*80)

    final = monitor.records[-1]['after']
    usage_pct = (final['allocated'] / final['total'] * 100)

    if usage_pct > 90:
        print("⚠️  显存使用率超过90%，建议:")
        print("   1. 减小 crop_size")
        print("   2. 增大 voxel_size")
        print("   3. 减小 sequence_length")
        print("   4. 减小 batch_size")
        print("   5. 使用梯度累积代替大batch size")
    elif usage_pct > 80:
        print("⚠️  显存使用率超过80%，建议:")
        print("   1. 适度减小 crop_size 或增大 voxel_size")
        print("   2. 增加显存清理频率")
    else:
        print("✅ 显存使用率合理，可以适当增加:")
        print("   1. 增大 crop_size")
        print("   2. 减小 voxel_size")
        print("   3. 增大 batch_size")

    # 检查主要瓶颈
    if state_ops > timestep_ops * 0.5:
        print("\n📊 状态管理占用较多显存，建议:")
        print("   - 优化历史状态的存储方式")
        print("   - 减少历史状态的保留层数")
        print("   - 使用更紧凑的状态表示")

    print("="*80 + "\n")

    return True


if __name__ == '__main__':
    try:
        success = monitor_model_layers()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
