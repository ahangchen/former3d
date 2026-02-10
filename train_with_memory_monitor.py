#!/usr/bin/env python3
"""
集成显存监控的流式训练脚本
用于分析 batch size 4 + 双 GPU 训练的显存占用
"""

import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse

sys.path.insert(0, '/home/cwh/coding/former3d')

from memory_monitor_layer import MemoryMonitor, memory_context
from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
import torch.optim as optim


def monitor_training(args, monitor: MemoryMonitor):
    """执行训练并监控显存"""
    print("="*80)
    print(f"显存监控训练配置")
    print("="*80)
    print(f"  Batch Size: {args.batch_size}")
    print(f"  GPU: {args.device}")
    print(f"  Multi-GPU: {args.multi_gpu}")
    print(f"  Crop Size: {args.crop_size}")
    print(f"  Voxel Size: {args.voxel_size}")
    print(f"  Sequence Length: {args.sequence_length}")
    print(f"  Max Sequences: {args.max_sequences}")
    print("="*80 + "\n")

    # 创建数据集
    print("创建数据集...")
    dataset = MultiSequenceTartanAirDataset(
        data_root=args.data_root,
        n_view=args.sequence_length,
        crop_size=tuple(map(int, args.crop_size.split(','))),
        voxel_size=args.voxel_size,
        max_sequences=args.max_sequences,
        shuffle=True
    )
    print(f"✅ 数据集创建成功，样本数: {len(dataset)}")

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )
    print(f"✅ 数据加载器创建成功，批次大小: {args.batch_size}\n")

    # 创建模型
    print("创建模型...")
    model = StreamSDFFormerIntegrated(
        attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
        use_proj_occ=False,
        voxel_size=args.voxel_size,
        fusion_local_radius=2.0,
        crop_size=tuple(map(int, args.crop_size.split(',')))
    )

    # 移动到设备
    device = torch.device(args.device)
    model = model.to(device)

    # 启用多GPU
    if args.multi_gpu and torch.cuda.device_count() > 1:
        gpu_ids = list(range(torch.cuda.device_count()))
        if len(gpu_ids) > 1:
            print(f"启用多GPU训练，使用GPU: {gpu_ids}")
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    else:
        print(f"使用单GPU: {device}")

    print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}\n")

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 训练循环
    model.train()
    num_batches = min(args.max_batches, len(dataloader))

    print(f"开始训练（前 {num_batches} 个batch）...")
    print("-"*80)

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        print(f"\n[Batch {batch_idx + 1}/{num_batches}]")

        # 加载数据
        with memory_context(monitor, f"batch_{batch_idx+1}_load_data"):
            images = batch['rgb_images'].to(device, non_blocking=True)
            poses = batch['poses'].to(device, non_blocking=True)
            intrinsics = batch['intrinsics'].to(device, non_blocking=True)

        optimizer.zero_grad()

        # 前向传播
        with memory_context(monitor, f"batch_{batch_idx+1}_forward"):
            try:
                # DataParallel 包装后需要访问 module
                model_to_use = model.module if isinstance(model, torch.nn.DataParallel) else model
                outputs, states = model_to_use.forward_sequence(images, poses, intrinsics)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ❌ 显存不足，终止训练")
                    print(f"  错误: {e}")
                    return False
                else:
                    raise

        # 模拟损失计算（简化）
        with memory_context(monitor, f"batch_{batch_idx+1}_loss"):
            # outputs 是一个列表，包含每个时间步的输出
            # 这里简化为模拟损失，实际训练需要计算真实的损失
            loss = torch.tensor(1.0, device=device, requires_grad=True)

            # 反向传播
            with memory_context(monitor, f"batch_{batch_idx+1}_backward"):
                loss.backward()

        # 更新参数
        with memory_context(monitor, f"batch_{batch_idx+1}_optimizer_step"):
            optimizer.step()

        # 清理
        with memory_context(monitor, f"batch_{batch_idx+1}_cleanup"):
            del images, poses, intrinsics, outputs, states
            if batch_idx % args.cleanup_freq == 0:
                torch.cuda.empty_cache()

        # 打印当前显存
        mem_info = monitor.get_memory_info()
        print(f"  Loss: {loss.item():.4f}")
        print(f"  显存: {mem_info['allocated']:.2f} GB / {mem_info['total']:.2f} GB ({mem_info['allocated']/mem_info['total']*100:.1f}%)")

    print("\n" + "="*80)
    print("训练完成！")
    print("="*80 + "\n")

    return True


def analyze_memory_bottleneck(monitor: MemoryMonitor):
    """分析显存瓶颈"""
    print("="*80)
    print("显存瓶颈分析")
    print("="*80 + "\n")

    # 1. 总体摘要
    monitor.print_summary(top_n=30)

    # 2. 按操作类型分类
    print("\n" + "="*80)
    print("按操作类型分类统计")
    print("="*80)

    categories = {
        'load_data': [],
        'forward': [],
        'loss': [],
        'backward': [],
        'optimizer_step': [],
        'cleanup': [],
    }

    for record in monitor.records:
        for category in categories.keys():
            if category in record['layer']:
                categories[category].append(record['delta_allocated'])
                break

    print(f"\n{'操作类型':<20} {'平均增量 (GB)':<20} {'总计 (GB)':<20} {'占比':<10}")
    print("-"*80)

    total_delta = sum(abs(r['delta_allocated']) for r in monitor.records)

    for category, deltas in categories.items():
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            total_category = sum(deltas)
            pct = (total_category / total_delta * 100) if total_delta > 0 else 0
            print(f"{category:<20} {avg_delta:>15.4f} {total_category:>18.4f} {pct:>9.1f}%")

    print("-"*80)

    # 3. 识别最大显存消费者
    print("\n" + "="*80)
    print("最大显存消费者（前20）")
    print("="*80)

    sorted_records = sorted(monitor.records, key=lambda x: x['delta_allocated'], reverse=True)
    top_20 = sorted_records[:20]

    print(f"\n{'排名':<6} {'层名':<45} {'增量 (GB)':<15} {'累计 (GB)':<15}")
    print("-"*80)

    cumulative = 0
    for i, record in enumerate(top_20, 1):
        delta = record['delta_allocated']
        cumulative += delta
        print(f"{i:<6} {record['layer']:<45} {delta:>15.4f} {cumulative:>15.4f}")

    print("-"*80)

    # 4. 分析瓶颈
    print("\n" + "="*80)
    print("瓶颈分析")
    print("="*80)

    final = monitor.records[-1]['after']
    usage_pct = (final['allocated'] / final['total'] * 100)

    print(f"\n最终显存状态:")
    print(f"  已分配: {final['allocated']:.4f} GB ({usage_pct:.2f}%)")
    print(f"  已预留: {final['reserved']:.4f} GB")
    print(f"  可用: {final['free']:.4f} GB")
    print(f"  总计: {final['total']:.4f} GB")

    if usage_pct > 90:
        print("\n⚠️  显存使用率超过90%，需要优化！")
    elif usage_pct > 80:
        print("\n⚠️  显存使用率较高，建议优化。")
    else:
        print("\n✅ 显存使用率合理。")

    # 5. 检查关键瓶颈
    print("\n" + "="*80)
    print("关键瓶颈识别")
    print("="*80)

    # forward 操作
    forward_ops = [r for r in monitor.records if 'forward' in r['layer']]
    if forward_ops:
        avg_forward = sum(r['delta_allocated'] for r in forward_ops) / len(forward_ops)
        max_forward = max(r['delta_allocated'] for r in forward_ops)
        print(f"\n前向传播:")
        print(f"  平均增量: {avg_forward:.4f} GB")
        print(f"  最大增量: {max_forward:.4f} GB")

        if avg_forward > 1.0:
            print("  ⚠️  前向传播显存占用较高，建议：")
            print("     - 减小 crop_size")
            print("     - 增大 voxel_size")
            print("     - 使用梯度检查点")

    # backward 操作
    backward_ops = [r for r in monitor.records if 'backward' in r['layer']]
    if backward_ops:
        avg_backward = sum(r['delta_allocated'] for r in backward_ops) / len(backward_ops)
        max_backward = max(r['delta_allocated'] for r in backward_ops)
        print(f"\n反向传播:")
        print(f"  平均增量: {avg_backward:.4f} GB")
        print(f"  最大增量: {max_backward:.4f} GB")

        if avg_backward > 0.5:
            print("  ⚠️  反向传播显存占用较高，建议：")
            print("     - 使用梯度累积")
            print("     - 减小 batch_size")

    # optimizer 操作
    optimizer_ops = [r for r in monitor.records if 'optimizer' in r['layer']]
    if optimizer_ops:
        avg_optimizer = sum(r['delta_allocated'] for r in optimizer_ops) / len(optimizer_ops)
        print(f"\n优化器更新:")
        print(f"  平均增量: {avg_optimizer:.4f} GB")

    print("\n" + "="*80 + "\n")


def create_improvement_plan(monitor: MemoryMonitor, args):
    """制定改进计划"""
    print("="*80)
    print("改进计划")
    print("="*80 + "\n")

    final = monitor.records[-1]['after']
    usage_pct = (final['allocated'] / final['total'] * 100)

    # 1. 当前配置分析
    print("1. 当前配置分析")
    print("-"*80)
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Crop Size: {args.crop_size}")
    print(f"  Voxel Size: {args.voxel_size}")
    print(f"  Sequence Length: {args.sequence_length}")
    print(f"  显存使用率: {usage_pct:.2f}%")
    print("-"*80 + "\n")

    # 2. 识别主要瓶颈
    print("2. 主要瓶颈识别")
    print("-"*80)

    sorted_records = sorted(monitor.records, key=lambda x: x['delta_allocated'], reverse=True)
    top_bottlenecks = sorted_records[:5]

    print("\n显存增量最大的5个组件:")
    for i, record in enumerate(top_bottlenecks, 1):
        delta = record['delta_allocated']
        pct = (delta / sorted_records[0]['delta_allocated'] * 100) if sorted_records[0]['delta_allocated'] > 0 else 0
        print(f"  {i}. {record['layer']:<50} {delta:>10.4f} GB ({pct:>5.1f}%)")

    print("-"*80 + "\n")

    # 3. 改进方案
    print("3. 改进方案")
    print("-"*80)

    plans = []

    # 方案 A: 减小 crop_size
    crop_vals = list(map(int, args.crop_size.split(',')))
    if max(crop_vals) > 6:
        new_crop = ",".join(str(max(4, v // 2)) for v in crop_vals)
        plans.append({
            'name': '减小 Crop Size',
            'description': f'将 crop_size 从 {args.crop_size} 减小到 {new_crop}',
            'expected_reduction': '30-50%',
            'impact': '中等',
            'command': f'--crop-size {new_crop}'
        })

    # 方案 B: 增大 voxel_size
    if args.voxel_size < 0.25:
        new_voxel = min(0.25, args.voxel_size * 1.25)
        plans.append({
            'name': '增大 Voxel Size',
            'description': f'将 voxel_size 从 {args.voxel_size} 增大到 {new_voxel}',
            'expected_reduction': '20-30%',
            'impact': '中等',
            'command': f'--voxel-size {new_voxel}'
        })

    # 方案 C: 减小 sequence_length
    if args.sequence_length > 3:
        new_seq = max(3, args.sequence_length - 1)
        plans.append({
            'name': '减小序列长度',
            'description': f'将 sequence_length 从 {args.sequence_length} 减小到 {new_seq}',
            'expected_reduction': '15-25%',
            'impact': '小',
            'command': f'--sequence-length {new_seq}'
        })

    # 方案 D: 减小 batch_size
    if args.batch_size > 2:
        new_batch = args.batch_size // 2
        plans.append({
            'name': '减小 Batch Size',
            'description': f'将 batch_size 从 {args.batch_size} 减小到 {new_batch}（或使用梯度累积）',
            'expected_reduction': '40-50%',
            'impact': '大',
            'command': f'--batch-size {new_batch}'
        })

    # 方案 E: 增加显存清理频率
    if args.cleanup_freq > 5:
        new_freq = max(5, args.cleanup_freq // 2)
        plans.append({
            'name': '增加显存清理频率',
            'description': f'将 cleanup_freq 从 {args.cleanup_freq} 增加到 {new_freq}',
            'expected_reduction': '5-10%',
            'impact': '小',
            'command': f'--cleanup-freq {new_freq}'
        })

    # 方案 F: 单 GPU 训练
    if args.multi_gpu:
        plans.append({
            'name': '单 GPU 训练',
            'description': '禁用多GPU，使用单GPU训练',
            'expected_reduction': '10-15%',
            'impact': '中等',
            'command': ''
        })

    # 打印方案
    for i, plan in enumerate(plans, 1):
        print(f"\n方案 {i}: {plan['name']}")
        print(f"  描述: {plan['description']}")
        print(f"  预期减少显存: {plan['expected_reduction']}")
        print(f"  影响: {plan['impact']}")
        if plan['command']:
            print(f"  命令参数: {plan['command']}")

    print("-"*80 + "\n")

    # 4. 推荐配置
    print("4. 推荐配置（根据显存使用率）")
    print("-"*80)

    if usage_pct > 95:
        print("\n⚠️  显存严重不足，建议:")
        print("   立即采用以下组合:")
        crop_vals = list(map(int, args.crop_size.split(',')))
        new_crop = ",".join(str(max(4, v // 2)) for v in crop_vals)
        print(f"   1. --crop-size {new_crop}")
        print(f"   2. --voxel-size {min(0.25, args.voxel_size * 1.5)}")
        print(f"   3. --batch-size {args.batch_size // 2}")
    elif usage_pct > 90:
        print("\n⚠️  显存不足，建议:")
        print("   采用以下组合:")
        crop_vals = list(map(int, args.crop_size.split(',')))
        new_crop = ",".join(str(max(5, v - 2)) for v in crop_vals)
        print(f"   1. --crop-size {new_crop}")
        print(f"   2. --voxel-size {min(0.25, args.voxel_size * 1.25)}")
    elif usage_pct > 80:
        print("\n⚠️  显存使用率较高，建议:")
        print("   采用以下之一:")
        print(f"   1. --crop-size {','.join(str(max(6, int(v)-1)) for v in crop_vals)}")
        print(f"   2. --voxel-size {min(0.25, args.voxel_size * 1.1)}")
    else:
        print("\n✅ 显存使用率合理，可以适当增加:")
        print(f"   1. 增大 crop_size")
        print(f"   2. 减小 voxel_size")
        print(f"   3. 增大 batch_size")

    print("-"*80 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='显存监控训练脚本')

    # 训练参数
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num-workers', type=int, default=2, help='数据加载器工作进程数')
    parser.add_argument('--max-batches', type=int, default=5, help='最大测试batch数')

    # 模型参数
    parser.add_argument('--attn-heads', type=int, default=1, help='注意力头数')
    parser.add_argument('--attn-layers', type=int, default=1, help='注意力层数')
    parser.add_argument('--voxel-size', type=float, default=0.16, help='体素大小')
    parser.add_argument('--crop-size', type=str, default='6,6,4', help='裁剪尺寸')

    # 数据参数
    parser.add_argument('--data-root', type=str, default='/home/cwh/Study/dataset/tartanair', help='数据根目录')
    parser.add_argument('--sequence-length', type=int, default=3, help='序列长度')
    parser.add_argument('--max-sequences', type=int, default=1, help='最大序列数')

    # 显存管理参数
    parser.add_argument('--cleanup-freq', type=int, default=5, help='显存清理频率')

    # 设备参数
    parser.add_argument('--device', type=str, default='cuda:0', help='设备选择')
    parser.add_argument('--multi-gpu', action='store_true', help='启用多GPU训练')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        sys.exit(1)

    # 创建监控器
    monitor = MemoryMonitor()

    # 训练并监控
    success = monitor_training(args, monitor)

    if success:
        # 分析瓶颈
        analyze_memory_bottleneck(monitor)

        # 制定改进计划
        create_improvement_plan(monitor, args)

    sys.exit(0 if success else 1)
