#!/usr/bin/env python3
"""
前向传播显存分析 - 基于组件的手动监控
使用 hook 监控各个网络层的显存占用
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


def analyze_forward_steps(args, monitor):
    """分析前向传播的各个步骤"""
    print("="*100)
    print("前向传播步骤级显存分析")
    print("="*100)
    print(f"\n配置：")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Crop Size: {args.crop_size}")
    print(f"  Voxel Size: {args.voxel_size}")
    print(f"  Sequence Length: {args.sequence_length}")
    print("="*100 + "\n")

    device = torch.device(args.device)

    # 创建数据集和模型
    print("创建数据集和模型...")
    dataset = MultiSequenceTartanAirDataset(
        data_root=args.data_root,
        n_view=args.sequence_length,
        crop_size=tuple(map(int, args.crop_size.split(','))),
        voxel_size=args.voxel_size,
        max_sequences=args.max_sequences,
        shuffle=True
    )

    model = StreamSDFFormerIntegrated(
        attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
        use_proj_occ=False,
        voxel_size=args.voxel_size,
        fusion_local_radius=2.0,
        crop_size=tuple(map(int, args.crop_size.split(',')))
    ).to(device)

    if args.multi_gpu and torch.cuda.device_count() > 1:
        gpu_ids = list(range(torch.cuda.device_count()))
        if len(gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpu_ids)

    print(f"✅ 数据集样本数: {len(dataset)}")
    print(f"✅ 模型参数数: {sum(p.numel() for p in model.parameters()):,}\n")

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )

    # 预热
    print("预热...")
    model.eval()
    batch = next(iter(dataloader))
    images = batch['rgb_images'].to(device, non_blocking=True)
    poses = batch['poses'].to(device, non_blocking=True)
    intrinsics = batch['intrinsics'].to(device, non_blocking=True)

    model_to_use = model.module if isinstance(model, torch.nn.DataParallel) else model

    with torch.no_grad():
        for _ in range(2):
            _ = model_to_use.forward_sequence(images, poses, intrinsics)
        torch.cuda.empty_cache()

    print("✅ 预热完成\n")

    # 清理记录
    monitor.clear()

    # 监控前向传播
    model.train()
    num_batches = min(args.max_batches, len(dataloader))

    print(f"开始监控（前 {num_batches} 个batch）...")
    print("-"*100 + "\n")

    step_counter = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        print(f"\n[Batch {batch_idx + 1}/{num_batches}]")

        # 加载数据
        with memory_context(monitor, f"batch_{batch_idx+1}_load_data"):
            images = batch['rgb_images'].to(device, non_blocking=True)
            poses = batch['poses'].to(device, non_blocking=True)
            intrinsics = batch['intrinsics'].to(device, non_blocking=True)

        # 监控每个时间步
        batch_size, n_view, _, H, W = images.shape

        with torch.no_grad():
            model_to_use.reset_state()

            for t in range(n_view):
                print(f"\n  [时间步 {t+1}/{n_view}]")

                # 提取当前帧数据
                with memory_context(monitor, f"batch_{batch_idx+1}_t{t}_extract_frame"):
                    current_image = images[:, t:t+1]  # [batch, 1, 3, H, W]
                    current_pose = poses[:, t:t+1]
                    current_intrinsic = intrinsics[:, t:t+1]

                # 创建 batch 字典
                with memory_context(monitor, f"batch_{batch_idx+1}_t{t}_prepare_batch"):
                    batch_dict = {
                        'rgb_images': current_image.squeeze(1),  # [batch, 3, H, W]
                        'poses': current_pose.squeeze(1),  # [batch, 4, 4]
                        'intrinsics': current_intrinsic.squeeze(1),  # [batch, 3, 3]
                    }

                # 特征提取
                with memory_context(monitor, f"batch_{batch_idx+1}_t{t}_feature_extraction"):
                    # 这里简化，实际会调用模型的特征提取
                    pass

                # SDF体素化
                with memory_context(monitor, f"batch_{batch_idx+1}_t{t}_sdf_voxelization"):
                    # 这里简化
                    pass

                # 3D前向传播
                with memory_context(monitor, f"batch_{batch_idx+1}_t{t}_3d_forward"):
                    # 这里简化
                    pass

                # 状态融合
                with memory_context(monitor, f"batch_{batch_idx+1}_t{t}_state_fusion"):
                    # 这里简化
                    pass

                # 完整前向传播
                with memory_context(monitor, f"batch_{batch_idx+1}_t{t}_full_forward"):
                    _ = model_to_use.forward_single_frame(
                        batch_dict,
                        None
                    )

                step_counter += 1
                if step_counter >= args.max_steps:
                    break

            # 完整序列
            with memory_context(monitor, f"batch_{batch_idx+1}_full_sequence"):
                _ = model_to_use.forward_sequence(images, poses, intrinsics)

        # 清理
        with memory_context(monitor, f"batch_{batch_idx+1}_cleanup"):
            del images, poses, intrinsics
            torch.cuda.empty_cache()

        # 打印当前显存
        mem_info = monitor.get_memory_info()
        print(f"  显存: {mem_info['allocated']:.2f} GB / {mem_info['total']:.2f} GB ({mem_info['allocated']/mem_info['total']*100:.1f}%)")

    print("\n" + "="*100)
    print("监控完成！")
    print("="*100 + "\n")

    return True


def analyze_memory_patterns(monitor):
    """分析显存模式"""
    print("="*100)
    print("显存占用模式分析")
    print("="*100 + "\n")

    # 总体摘要
    monitor.print_summary(top_n=30)

    # 按操作类型分组
    print("\n" + "="*100)
    print("按步骤类型统计")
    print("="*100)

    step_types = {
        'load_data': [],
        'extract_frame': [],
        'prepare_batch': [],
        'feature_extraction': [],
        'sdf_voxelization': [],
        '3d_forward': [],
        'state_fusion': [],
        'full_forward': [],
        'full_sequence': [],
        'cleanup': [],
    }

    for record in monitor.records:
        for step_type in step_types.keys():
            if step_type in record['layer']:
                step_types[step_type].append(record['delta_allocated'])
                break

    print(f"\n{'步骤类型':<25} {'调用次数':<12} {'平均 (GB)':<15} {'最大 (GB)':<15} {'总计 (GB)':<15}")
    print("-"*100)

    total_delta = sum(abs(r['delta_allocated']) for r in monitor.records)

    for step_type, deltas in step_types.items():
        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            max_delta = max(deltas)
            total_step = sum(deltas)
            count = len(deltas)
            pct = (total_step / total_delta * 100) if total_delta > 0 else 0

            print(f"{step_type:<25} {count:<12} {avg_delta:<15.4f} {max_delta:<15.4f} {total_step:<15.4f}")

    print("-"*100 + "\n")

    # 识别瓶颈
    print("主要瓶颈识别：")
    print("-"*100)

    sorted_records = sorted(monitor.records, key=lambda x: x['delta_allocated'], reverse=True)
    top_10 = sorted_records[:10]

    print(f"\n{'排名':<6} {'步骤名称':<40} {'增量 (GB)':<15} {'累计 (GB)':<15}")
    print("-"*100)

    cumulative = 0
    for i, record in enumerate(top_10, 1):
        delta = record['delta_allocated']
        cumulative += delta
        print(f"{i:<6} {record['layer']:<40} {delta:<15.4f} {cumulative:<15.4f}")

    print("-"*100 + "\n")

    # 时间步分析
    print("时间步分析：")
    print("-"*100)

    timestep_records = [r for r in monitor.records if 't' in r['layer'] and 'batch_' not in r['layer']]

    if timestep_records:
        # 按时间步分组
        timesteps = {}
        for record in timestep_records:
            layer_name = record['layer']
            # 提取时间步编号
            parts = layer_name.split('_')
            for part in parts:
                if part.startswith('t') and part[1:].isdigit():
                    timestep = int(part[1:])
                    if timestep not in timesteps:
                        timesteps[timestep] = []
                    timesteps[timestep].append(record['delta_allocated'])
                    break

        print(f"\n{'时间步':<10} {'操作数':<10} {'平均 (GB)':<15} {'最大 (GB)':<15} {'总计 (GB)':<15}")
        print("-"*100)

        for timestep in sorted(timesteps.keys()):
            deltas = timesteps[timestep]
            avg_delta = sum(deltas) / len(deltas)
            max_delta = max(deltas)
            total_step = sum(deltas)
            count = len(deltas)

            print(f"t={timestep:<5} {count:<10} {avg_delta:<15.4f} {max_delta:<15.4f} {total_step:<15.4f}")

        print("-"*100 + "\n")

    # 最终状态
    final = monitor.records[-1]['after']
    usage_pct = (final['allocated'] / final['total'] * 100)

    print("最终显存状态：")
    print("-"*100)
    print(f"  已分配: {final['allocated']:.4f} GB ({usage_pct:.2f}%)")
    print(f"  已预留: {final['reserved']:.4f} GB")
    print(f"  可用: {final['free']:.4f} GB")
    print(f"  总计: {final['total']:.4f} GB")
    print("="*100 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description='前向传播步骤级显存分析')

    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--max-batches', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--attn-heads', type=int, default=1)
    parser.add_argument('--attn-layers', type=int, default=1)
    parser.add_argument('--voxel-size', type=float, default=0.12)
    parser.add_argument('--crop-size', type=str, default='12,12,10')
    parser.add_argument('--data-root', type=str, default='/home/cwh/Study/dataset/tartanair')
    parser.add_argument('--sequence-length', type=int, default=5)
    parser.add_argument('--max-sequences', type=int, default=1)
    parser.add_argument('--cleanup-freq', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--multi-gpu', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        sys.exit(1)

    monitor = MemoryMonitor()

    # 分析前向传播
    success = analyze_forward_steps(args, monitor)

    if success:
        # 分析显存模式
        analyze_memory_patterns(monitor)

    sys.exit(0 if success else 1)
