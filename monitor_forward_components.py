#!/usr/bin/env python3
"""
前向传播网络组件级显存监控
用于分析 StreamSDFFormer 模型中各个组件的显存占用
"""

import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn

sys.path.insert(0, '/home/cwh/coding/former3d')

from memory_monitor_layer import MemoryMonitor, memory_context, monitor_layer
from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated


class ComponentMonitor(nn.Module):
    """组件监控包装器"""

    def __init__(self, module, name, monitor):
        super().__init__()
        self.module = module
        self.name = name
        self.monitor = monitor

    def forward(self, *args, **kwargs):
        with memory_context(self.monitor, f"forward_{self.name}"):
            return self.module(*args, **kwargs)


def wrap_model_components(model, monitor):
    """包装模型的各个组件以监控显存"""

    # 包装 stem
    if hasattr(model, 'sdfformer') and hasattr(model.sdfformer, 'net3d'):
        net3d = model.sdfformer.net3d

        # 包装 stem
        if hasattr(net3d, 'stem'):
            net3d.stem = ComponentMonitor(net3d.stem, "stem", monitor)

        # 包装 sp_convs
        if hasattr(net3d, 'sp_convs'):
            for i, conv in enumerate(net3d.sp_convs):
                net3d.sp_convs[i] = ComponentMonitor(conv, f"sp_conv_{i}", monitor)

        # 包装 upconvs
        if hasattr(net3d, 'upconvs'):
            for i, conv in enumerate(net3d.upconvs):
                net3d.upconvs[i] = ComponentMonitor(conv, f"upconv_{i}", monitor)

        # 包装 lateral_attns
        if hasattr(net3d, 'lateral_attns'):
            for i, attn in enumerate(net3d.lateral_attns):
                net3d.lateral_attns[i] = ComponentMonitor(attn, f"lateral_attn_{i}", monitor)

        # 包装 post_attn
        if hasattr(net3d, 'post_attn'):
            net3d.post_attn = ComponentMonitor(net3d.post_attn, "post_attn", monitor)

    # 包装图像编码器
    if hasattr(model, 'image_encoder') and model.image_encoder is not None:
        model.image_encoder = ComponentMonitor(model.image_encoder, "image_encoder", monitor)

    # 包装特征提取器
    if hasattr(model, 'feature_extractor') and model.feature_extractor is not None:
        model.feature_extractor = ComponentMonitor(model.feature_extractor, "feature_extractor", monitor)

    # 包装投影头
    if hasattr(model, 'projection_head') and model.projection_head is not None:
        model.projection_head = ComponentMonitor(model.projection_head, "projection_head", monitor)

    return model


def monitor_forward_components(args):
    """监控前向传播中各个组件的显存占用"""
    print("="*100)
    print("前向传播网络组件级显存监控")
    print("="*100)
    print(f"\n配置：")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  GPU: {args.device}")
    print(f"  Multi-GPU: {args.multi_gpu}")
    print(f"  Crop Size: {args.crop_size}")
    print(f"  Voxel Size: {args.voxel_size}")
    print(f"  Sequence Length: {args.sequence_length}")
    print("="*100 + "\n")

    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return False

    device = torch.device(args.device)
    print(f"✅ 使用设备: {device}\n")

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
        shuffle=False,
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

    # 创建显存监控器
    monitor = MemoryMonitor(device)

    # 包装模型组件
    print("包装模型组件以监控显存...")
    if isinstance(model, torch.nn.DataParallel):
        model.module = wrap_model_components(model.module, monitor)
    else:
        model = wrap_model_components(model, monitor)
    print("✅ 组件包装完成\n")

    # 预热
    print("预热...")
    model.eval()
    batch = next(iter(dataloader))
    images = batch['rgb_images'].to(device, non_blocking=True)
    poses = batch['poses'].to(device, non_blocking=True)
    intrinsics = batch['intrinsics'].to(device, non_blocking=True)

    with torch.no_grad():
        for _ in range(2):
            model_to_use = model.module if isinstance(model, torch.nn.DataParallel) else model
            _ = model_to_use.forward_sequence(images, poses, intrinsics)
        torch.cuda.empty_cache()
    print("✅ 预热完成\n")

    # 清理记录，开始正式监控
    monitor.clear()

    # 监控多个 batch
    model.train()
    num_batches = min(args.max_batches, len(dataloader))

    print(f"开始监控前向传播（前 {num_batches} 个batch）...")
    print("-"*100 + "\n")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        print(f"\n[Batch {batch_idx + 1}/{num_batches}]")

        # 加载数据
        with memory_context(monitor, f"batch_{batch_idx+1}_load_data"):
            images = batch['rgb_images'].to(device, non_blocking=True)
            poses = batch['poses'].to(device, non_blocking=True)
            intrinsics = batch['intrinsics'].to(device, non_blocking=True)

        # 前向传播
        with memory_context(monitor, f"batch_{batch_idx+1}_forward_sequence"):
            with torch.no_grad():
                model_to_use = model.module if isinstance(model, torch.nn.DataParallel) else model
                outputs, states = model_to_use.forward_sequence(images, poses, intrinsics)

        # 清理
        with memory_context(monitor, f"batch_{batch_idx+1}_cleanup"):
            del images, poses, intrinsics, outputs, states
            if batch_idx % args.cleanup_freq == 0:
                torch.cuda.empty_cache()

        # 打印当前显存
        mem_info = monitor.get_memory_info()
        print(f"  显存: {mem_info['allocated']:.2f} GB / {mem_info['total']:.2f} GB ({mem_info['allocated']/mem_info['total']*100:.1f}%)")

    print("\n" + "="*100)
    print("监控完成！")
    print("="*100 + "\n")

    return True, monitor


def analyze_components(monitor):
    """分析各个组件的显存占用"""
    print("="*100)
    print("前向传播组件显存占用分析")
    print("="*100 + "\n")

    # 过滤出 forward 相关的记录
    forward_records = [r for r in monitor.records if 'forward' in r['layer']]

    # 按组件分组
    components = {}
    for record in forward_records:
        # 提取组件名称
        layer_name = record['layer']
        if '_' in layer_name:
            parts = layer_name.split('_')
            # 格式：forward_<component> 或 batch_N_forward_<component>
            if 'forward' in parts:
                forward_idx = parts.index('forward')
                if forward_idx + 1 < len(parts):
                    component = '_'.join(parts[forward_idx + 1:])
                    if component not in components:
                        components[component] = []
                    components[component].append(record['delta_allocated'])

    # 统计每个组件
    print("组件统计：")
    print("-"*100)
    print(f"{'组件名称':<30} {'调用次数':<12} {'平均 (GB)':<15} {'最大 (GB)':<15} {'总计 (GB)':<15} {'占比':<10}")
    print("-"*100)

    component_stats = []
    total_delta = sum(abs(r['delta_allocated']) for r in forward_records)

    for component, deltas in sorted(components.items(), key=lambda x: sum(x[1]), reverse=True):
        avg_delta = sum(deltas) / len(deltas) if deltas else 0
        max_delta = max(deltas) if deltas else 0
        total_component = sum(deltas)
        count = len(deltas)
        pct = (total_component / total_delta * 100) if total_delta > 0 else 0

        component_stats.append({
            'name': component,
            'count': count,
            'avg': avg_delta,
            'max': max_delta,
            'total': total_component,
            'pct': pct
        })

        print(f"{component:<30} {count:<12} {avg_delta:<15.4f} {max_delta:<15.4f} {total_component:<15.4f} {pct:<9.1f}%")

    print("-"*100 + "\n")

    # 按类型分类
    print("按类型分类：")
    print("-"*100)

    categories = {
        '图像编码': ['image_encoder', 'feature_extractor', 'stem'],
        '3D卷积': ['sp_conv'],
        '注意力': ['lateral_attn', 'post_attn', 'atten'],
        '上采样': ['upconv', 'upsample'],
        '融合': ['fusion', 'concat'],
        '投影': ['projection_head', 'proj'],
        '其他': []
    }

    print(f"\n{'类型':<15} {'组件数':<12} {'平均 (GB)':<15} {'总计 (GB)':<15} {'占比':<10}")
    print("-"*100)

    for category, keywords in categories.items():
        category_components = [c for c in components.keys() if any(kw in c for kw in keywords)]
        if category_components:
            total_category = sum(sum(components[c]) for c in category_components)
            avg_category = total_category / len(category_components)
            count_category = len(category_components)
            pct = (total_category / total_delta * 100) if total_delta > 0 else 0

            print(f"{category:<15} {count_category:<12} {avg_category:<15.4f} {total_category:<15.4f} {pct:<9.1f}%")

            # 打印该类别下的组件
            for comp in sorted(category_components, key=lambda c: sum(components[c]), reverse=True):
                comp_total = sum(components[comp])
                comp_pct = (comp_total / total_category * 100) if total_category > 0 else 0
                print(f"  └─ {comp:<28} {comp_total:<15.4f} ({comp_pct:>5.1f}%)")

    print("-"*100 + "\n")

    # 识别主要瓶颈
    print("显存瓶颈识别（前10）：")
    print("-"*100)

    sorted_by_total = sorted(component_stats, key=lambda x: x['total'], reverse=True)
    top_10 = sorted_by_total[:10]

    print(f"\n{'排名':<6} {'组件名称':<30} {'平均 (GB)':<15} {'最大 (GB)':<15} {'总计 (GB)':<15} {'占比':<10}")
    print("-"*100)

    cumulative = 0
    for i, stat in enumerate(top_10, 1):
        cumulative += stat['total']
        print(f"{i:<6} {stat['name']:<30} {stat['avg']:<15.4f} {stat['max']:<15.4f} {stat['total']:<15.4f} {stat['pct']:<9.1f}%")

    print("-"*100)

    # 瓶颈分析
    print("\n瓶颈分析：")
    print("-"*100)

    if top_10:
        top_component = top_10[0]
        print(f"\n最大显存消费者: {top_component['name']}")
        print(f"  - 总占用: {top_component['total']:.4f} GB")
        print(f"  - 平均占用: {top_component['avg']:.4f} GB")
        print(f"  - 峰值占用: {top_component['max']:.4f} GB")
        print(f"  - 占总显存: {top_component['pct']:.1f}%")

        # 根据组件类型提供优化建议
        print(f"\n优化建议：")

        if any(kw in top_component['name'].lower() for kw in ['image_encoder', 'feature_extractor']):
            print("  ⚠️  图像编码器占用较大，建议：")
            print("     - 使用更高效的骨干网络（如 MobileNet, EfficientNet）")
            print("     - 减小输入图像尺寸")
            print("     - 使用混合精度训练")

        elif any(kw in top_component['name'].lower() for kw in ['sp_conv', 'conv']):
            print("  ⚠️  3D卷积占用较大，建议：")
            print("     - 减小 crop_size")
            print("     - 增大 voxel_size")
            print("     - 使用分组卷积或深度可分离卷积")
            print("     - 使用稀疏卷积优化")

        elif any(kw in top_component['name'].lower() for kw in ['atten', 'attention']):
            print("  ⚠️  注意力机制占用较大，建议：")
            print("     - 减少注意力头数或层数")
            print("     - 使用局部注意力代替全局注意力")
            print("     - 使用 FlashAttention 等高效实现")
            print("     - 使用梯度检查点")

        elif any(kw in top_component['name'].lower() for kw in ['upconv', 'upsample']):
            print("  ⚠️  上采样操作占用较大，建议：")
            print("     - 使用转置卷积代替上采样+卷积")
            print("     - 减少上采样通道数")
            print("     - 使用更高效的上采样方法")

    # 最终状态
    final = monitor.records[-1]['after']
    usage_pct = (final['allocated'] / final['total'] * 100)

    print(f"\n最终显存状态：")
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

    print("="*100 + "\n")

    return component_stats


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='组件级显存监控')

    # 训练参数
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--num-workers', type=int, default=2, help='数据加载器工作进程数')
    parser.add_argument('--max-batches', type=int, default=3, help='最大测试batch数')

    # 模型参数
    parser.add_argument('--attn-heads', type=int, default=1, help='注意力头数')
    parser.add_argument('--attn-layers', type=int, default=1, help='注意力层数')
    parser.add_argument('--voxel-size', type=float, default=0.12, help='体素大小')
    parser.add_argument('--crop-size', type=str, default='12,12,10', help='裁剪尺寸')

    # 数据参数
    parser.add_argument('--data-root', type=str, default='/home/cwh/Study/dataset/tartanair', help='数据根目录')
    parser.add_argument('--sequence-length', type=int, default=5, help='序列长度')
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

    # 监控前向传播
    success, monitor = monitor_forward_components(args)

    if success:
        # 分析组件
        component_stats = analyze_components(monitor)

        # 保存报告
        report_file = "doc/component_memory_report.md"
        with open(report_file, 'w') as f:
            f.write("# 前向传播组件显存占用报告\n\n")
            f.write(f"## 配置\n\n")
            f.write(f"- Batch Size: {args.batch_size}\n")
            f.write(f"- Crop Size: {args.crop_size}\n")
            f.write(f"- Voxel Size: {args.voxel_size}\n")
            f.write(f"- Sequence Length: {args.sequence_length}\n")
            f.write(f"- Multi-GPU: {args.multi_gpu}\n\n")

            f.write("## 组件统计\n\n")
            f.write(f"| 组件名称 | 调用次数 | 平均 (GB) | 最大 (GB) | 总计 (GB) | 占比 |\n")
            f.write(f"|----------|----------|-----------|-----------|-----------|------|\n")

            for stat in sorted(component_stats, key=lambda x: x['total'], reverse=True):
                f.write(f"| {stat['name']} | {stat['count']} | {stat['avg']:.4f} | {stat['max']:.4f} | {stat['total']:.4f} | {stat['pct']:.1f}% |\n")

            f.write("\n## 详细分析\n\n")
            f.write(f"前 10 个显占用最大的组件已在上面的表格中列出。\n")

        print(f"✅ 报告已保存到: {report_file}\n")

    sys.exit(0 if success else 1)
