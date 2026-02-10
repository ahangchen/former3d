"""
显存分析工具 - 用于分析流式训练时不同层、不同frame、不同step的显存占用
"""
import torch
import time
from collections import defaultdict
from typing import Dict, List, Optional


class MemoryProfiler:
    """显存分析器"""

    def __init__(self, device: torch.device = None):
        """
        初始化显存分析器

        Args:
            device: 要监控的设备，如果为None则监控所有GPU
        """
        self.device = device
        self.current_step = 0
        self.current_frame = 0
        self.current_layer = None
        self.history = []

        # 按维度分类存储显存记录
        self.by_step = defaultdict(list)
        self.by_frame = defaultdict(list)
        self.by_layer = defaultdict(list)
        self.by_gpu = defaultdict(list)

    def reset(self):
        """重置分析器状态"""
        self.current_step = 0
        self.current_frame = 0
        self.current_layer = None
        self.history = []
        self.by_step.clear()
        self.by_frame.clear()
        self.by_layer.clear()
        self.by_gpu.clear()

    def set_step(self, step: int):
        """设置当前训练步"""
        self.current_step = step

    def set_frame(self, frame_idx: int):
        """设置当前帧索引"""
        self.current_frame = frame_idx

    def set_layer(self, layer_name: str):
        """设置当前层名称"""
        self.current_layer = layer_name

    def record(self, label: str = "", extra_info: Dict = None):
        """
        记录当前时刻的显存使用情况

        Args:
            label: 记录标签
            extra_info: 额外信息字典
        """
        if not torch.cuda.is_available():
            return

        # 获取所有GPU的显存信息
        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            device = torch.device(f'cuda:{i}')
            allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
            reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
            max_reserved = torch.cuda.max_memory_reserved(device) / 1024**2  # MB

            gpu_info[i] = {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'max_allocated_mb': max_allocated,
                'max_reserved_mb': max_reserved,
            }

            # 记录到by_gpu
            self.by_gpu[i].append({
                'step': self.current_step,
                'frame': self.current_frame,
                'layer': self.current_layer,
                'label': label,
                **gpu_info[i],
                **(extra_info or {})
            })

        # 创建记录
        record = {
            'step': self.current_step,
            'frame': self.current_frame,
            'layer': self.current_layer,
            'label': label,
            'timestamp': time.time(),
            'gpu_info': gpu_info,
            **(extra_info or {})
        }

        self.history.append(record)
        self.by_step[self.current_step].append(record)
        self.by_frame[self.current_frame].append(record)
        if self.current_layer:
            self.by_layer[self.current_layer].append(record)

        return record

    def get_gpu_memory(self, device_id: int = 0) -> Dict:
        """
        获取指定GPU的显存使用情况

        Args:
            device_id: GPU ID

        Returns:
            显存信息字典（单位：MB）
        """
        device = torch.device(f'cuda:{device_id}')
        return {
            'allocated_mb': torch.cuda.memory_allocated(device) / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved(device) / 1024**2,
            'max_allocated_mb': torch.cuda.max_memory_allocated(device) / 1024**2,
            'max_reserved_mb': torch.cuda.max_memory_reserved(device) / 1024**2,
        }

    def get_all_gpu_memory(self) -> Dict[int, Dict]:
        """获取所有GPU的显存使用情况"""
        result = {}
        for i in range(torch.cuda.device_count()):
            result[i] = self.get_gpu_memory(i)
        return result

    def summarize_by_step(self) -> Dict[int, Dict]:
        """
        按训练步统计显存使用

        Returns:
            {step_id: {min_allocated, max_allocated, avg_allocated, ...}}
        """
        summary = {}
        for step, records in self.by_step.items():
            if not records:
                continue

            # 统计每张GPU的显存
            gpu_stats = {}
            for gpu_id in range(torch.cuda.device_count()):
                allocated_values = [r['gpu_info'][gpu_id]['allocated_mb'] for r in records if gpu_id in r['gpu_info']]

                if allocated_values:
                    gpu_stats[gpu_id] = {
                        'min_allocated_mb': min(allocated_values),
                        'max_allocated_mb': max(allocated_values),
                        'avg_allocated_mb': sum(allocated_values) / len(allocated_values),
                        'min_reserved_mb': min([r['gpu_info'][gpu_id]['reserved_mb'] for r in records if gpu_id in r['gpu_info']]),
                        'max_reserved_mb': max([r['gpu_info'][gpu_id]['reserved_mb'] for r in records if gpu_id in r['gpu_info']]),
                    }

            summary[step] = {
                'record_count': len(records),
                'gpu_stats': gpu_stats,
            }

        return summary

    def summarize_by_frame(self) -> Dict[int, Dict]:
        """
        按帧索引统计显存使用

        Returns:
            {frame_idx: {min_allocated, max_allocated, avg_allocated, ...}}
        """
        summary = {}
        for frame_idx, records in self.by_frame.items():
            if not records:
                continue

            # 统计每张GPU的显存
            gpu_stats = {}
            for gpu_id in range(torch.cuda.device_count()):
                allocated_values = [r['gpu_info'][gpu_id]['allocated_mb'] for r in records if gpu_id in r['gpu_info']]

                if allocated_values:
                    gpu_stats[gpu_id] = {
                        'min_allocated_mb': min(allocated_values),
                        'max_allocated_mb': max(allocated_values),
                        'avg_allocated_mb': sum(allocated_values) / len(allocated_values),
                        'min_reserved_mb': min([r['gpu_info'][gpu_id]['reserved_mb'] for r in records if gpu_id in r['gpu_info']]),
                        'max_reserved_mb': max([r['gpu_info'][gpu_id]['reserved_mb'] for r in records if gpu_id in r['gpu_info']]),
                    }

            summary[frame_idx] = {
                'record_count': len(records),
                'gpu_stats': gpu_stats,
            }

        return summary

    def summarize_by_layer(self) -> Dict[str, Dict]:
        """
        按层统计显存使用

        Returns:
            {layer_name: {min_allocated, max_allocated, avg_allocated, ...}}
        """
        summary = {}
        for layer_name, records in self.by_layer.items():
            if not records:
                continue

            # 统计每张GPU的显存
            gpu_stats = {}
            for gpu_id in range(torch.cuda.device_count()):
                allocated_values = [r['gpu_info'][gpu_id]['allocated_mb'] for r in records if gpu_id in r['gpu_info']]

                if allocated_values:
                    gpu_stats[gpu_id] = {
                        'min_allocated_mb': min(allocated_values),
                        'max_allocated_mb': max(allocated_values),
                        'avg_allocated_mb': sum(allocated_values) / len(allocated_values),
                        'min_reserved_mb': min([r['gpu_info'][gpu_id]['reserved_mb'] for r in records if gpu_id in r['gpu_info']]),
                        'max_reserved_mb': max([r['gpu_info'][gpu_id]['reserved_mb'] for r in records if gpu_id in r['gpu_info']]),
                    }

            summary[layer_name] = {
                'record_count': len(records),
                'gpu_stats': gpu_stats,
            }

        return summary

    def print_summary(self):
        """打印显存使用摘要"""
        print("\n" + "=" * 80)
        print("显存使用摘要")
        print("=" * 80)

        # 按训练步统计
        print("\n【按训练步统计】")
        step_summary = self.summarize_by_step()
        for step in sorted(step_summary.keys()):
            print(f"\nStep {step}:")
            for gpu_id, stats in step_summary[step]['gpu_stats'].items():
                print(f"  GPU {gpu_id}:")
                print(f"    Allocated:  {stats['min_allocated_mb']:.2f} MB (min) -> {stats['max_allocated_mb']:.2f} MB (max)")
                print(f"    Reserved:   {stats['min_reserved_mb']:.2f} MB (min) -> {stats['max_reserved_mb']:.2f} MB (max)")
                print(f"    Avg Allocated: {stats['avg_allocated_mb']:.2f} MB")

        # 按帧索引统计
        print("\n【按帧索引统计】")
        frame_summary = self.summarize_by_frame()
        for frame_idx in sorted(frame_summary.keys()):
            print(f"\nFrame {frame_idx}:")
            for gpu_id, stats in frame_summary[frame_idx]['gpu_stats'].items():
                print(f"  GPU {gpu_id}:")
                print(f"    Allocated:  {stats['min_allocated_mb']:.2f} MB (min) -> {stats['max_allocated_mb']:.2f} MB (max)")
                print(f"    Reserved:   {stats['min_reserved_mb']:.2f} MB (min) -> {stats['max_reserved_mb']:.2f} MB (max)")
                print(f"    Avg Allocated: {stats['avg_allocated_mb']:.2f} MB")

        # 按层统计
        print("\n【按层统计】")
        layer_summary = self.summarize_by_layer()
        for layer_name in sorted(layer_summary.keys()):
            print(f"\n{layer_name}:")
            for gpu_id, stats in layer_summary[layer_name]['gpu_stats'].items():
                print(f"  GPU {gpu_id}:")
                print(f"    Allocated:  {stats['min_allocated_mb']:.2f} MB (min) -> {stats['max_allocated_mb']:.2f} MB (max)")
                print(f"    Reserved:   {stats['min_reserved_mb']:.2f} MB (min) -> {stats['max_reserved_mb']:.2f} MB (max)")
                print(f"    Avg Allocated: {stats['avg_allocated_mb']:.2f} MB")

        # 当前显存状态
        print("\n【当前显存状态】")
        current_memory = self.get_all_gpu_memory()
        for gpu_id, info in current_memory.items():
            print(f"GPU {gpu_id}:")
            print(f"  Allocated:  {info['allocated_mb']:.2f} MB")
            print(f"  Reserved:   {info['reserved_mb']:.2f} MB")
            print(f"  Max Allocated:  {info['max_allocated_mb']:.2f} MB")
            print(f"  Max Reserved:   {info['max_reserved_mb']:.2f} MB")

        print("\n" + "=" * 80)

    def export_to_file(self, filename: str):
        """
        将显存记录导出到文件

        Args:
            filename: 输出文件名
        """
        import json

        # 导出原始记录
        with open(f"{filename}_raw.json", 'w') as f:
            json.dump(self.history, f, indent=2, default=str)

        # 导出摘要
        summary = {
            'by_step': self.summarize_by_step(),
            'by_frame': self.summarize_by_frame(),
            'by_layer': self.summarize_by_layer(),
            'current_memory': self.get_all_gpu_memory(),
        }

        with open(f"{filename}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n显存记录已导出到:")
        print(f"  - {filename}_raw.json (原始记录)")
        print(f"  - {filename}_summary.json (摘要统计)")


def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(torch.device(f'cuda:{i}'))


def print_gpu_memory(prefix: str = ""):
    """打印当前GPU显存使用情况"""
    if not torch.cuda.is_available():
        return

    print(f"{prefix}")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(torch.device(f'cuda:{i}')) / 1024**2
        reserved = torch.cuda.memory_reserved(torch.device(f'cuda:{i}')) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(torch.device(f'cuda:{i}')) / 1024**2

        print(f"  GPU {i}: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB, Max Allocated: {max_allocated:.2f} MB")
