#!/usr/bin/env python3
"""
PyTorch 不同层显存占用监控工具
支持监控每一层的显存分配和释放情况
"""

import torch
import gc
from contextlib import contextmanager
from typing import Callable, Optional, Dict, List
import functools


class MemoryMonitor:
    """PyTorch 显存监控器"""

    def __init__(self, device: Optional[torch.device] = None):
        """
        初始化显存监控器

        Args:
            device: 要监控的设备，默认为 CUDA:0
        """
        self.device = device or (torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'))
        self.records: List[Dict] = []
        self.layer_counter = 0

    def get_memory_info(self) -> Dict[str, float]:
        """
        获取当前显存信息（单位：GB）

        Returns:
            包含各项显存指标的字典
        """
        if self.device.type == 'cpu':
            return {
                'allocated': 0.0,
                'reserved': 0.0,
                'free': 0.0,
                'total': 0.0
            }

        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        total = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
        free = total - reserved

        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free,
            'total': total
        }

    def record(self, layer_name: str, before: Dict[str, float], after: Dict[str, float]):
        """
        记录某一层的显存变化

        Args:
            layer_name: 层名称
            before: 执行前的显存信息
            after: 执行后的显存信息
        """
        delta_allocated = after['allocated'] - before['allocated']
        delta_reserved = after['reserved'] - before['reserved']

        self.records.append({
            'layer': layer_name,
            'before': before.copy(),
            'after': after.copy(),
            'delta_allocated': delta_allocated,
            'delta_reserved': delta_reserved
        })

    def print_summary(self, top_n: int = 10):
        """打印显存使用摘要"""
        print("\n" + "="*80)
        print("显存使用摘要")
        print("="*80)

        if not self.records:
            print("无记录")
            return

        # 按 delta_allocated 排序
        sorted_records = sorted(self.records, key=lambda x: x['delta_allocated'], reverse=True)

        print(f"\n显存增量最大的 {top_n} 层：")
        print("-"*80)
        print(f"{'层名':<40} {'增量 (GB)':>15} {'累计 (GB)':>15}")
        print("-"*80)

        total_allocated = 0
        for i, record in enumerate(sorted_records[:top_n]):
            delta = record['delta_allocated']
            total_allocated += delta
            print(f"{record['layer']:<40} {delta:>15.4f} {total_allocated:>15.4f}")

        print("-"*80)
        print(f"总计增量: {total_allocated:.4f} GB")

        # 打印最终状态
        final = self.records[-1]['after']
        print(f"\n最终显存状态：")
        print(f"  已分配: {final['allocated']:.4f} GB")
        print(f"  已预留: {final['reserved']:.4f} GB")
        print(f"  可用: {final['free']:.4f} GB")
        print(f"  总计: {final['total']:.4f} GB")
        print("="*80 + "\n")

    def print_detailed(self):
        """打印详细的逐层显存变化"""
        print("\n" + "="*100)
        print("详细逐层显存变化")
        print("="*100)
        print(f"{'层名':<40} {'前 (GB)':>12} {'后 (GB)':>12} {'增量 (GB)':>12} {'预留增量 (GB)':>14}")
        print("-"*100)

        for record in self.records:
            before = record['before']
            after = record['after']
            print(f"{record['layer']:<40} "
                  f"{before['allocated']:>10.4f}   "
                  f"{after['allocated']:>10.4f}   "
                  f"{record['delta_allocated']:>10.4f}   "
                  f"{record['delta_reserved']:>14.4f}")

        print("="*100 + "\n")

    def clear(self):
        """清空记录"""
        self.records.clear()
        self.layer_counter = 0


def monitor_layer(monitor: MemoryMonitor, layer_name: Optional[str] = None):
    """
    装饰器：监控函数或方法的显存占用

    Args:
        monitor: MemoryMonitor 实例
        layer_name: 层名称，默认使用函数名
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = layer_name or func.__name__
            before = monitor.get_memory_info()

            result = func(*args, **kwargs)

            after = monitor.get_memory_info()
            monitor.record(name, before, after)

            return result
        return wrapper
    return decorator


@contextmanager
def memory_context(monitor: MemoryMonitor, context_name: str):
    """
    上下文管理器：监控代码块的显存占用

    Args:
        monitor: MemoryMonitor 实例
        context_name: 上下文名称
    """
    before = monitor.get_memory_info()
    try:
        yield
    finally:
        after = monitor.get_memory_info()
        monitor.record(context_name, before, after)


def profile_model_forward(model: torch.nn.Module, input_data: torch.device,
                         device: torch.device, num_warmup: int = 3,
                         num_runs: int = 1) -> MemoryMonitor:
    """
    分析模型前向传播的显存占用

    Args:
        model: PyTorch 模型
        input_data: 输入数据
        device: 设备
        num_warmup: 预热次数
        num_runs: 运行次数

    Returns:
        MemoryMonitor 实例
    """
    monitor = MemoryMonitor(device)
    model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_data)
            torch.cuda.empty_cache()

    # 注册钩子
    def create_hook(name):
        def hook(module, input, output):
            with memory_context(monitor, f"{name}"):
                pass
        return hook

    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            hook = module.register_forward_hook(create_hook(name))
            hooks.append(hook)

    # 运行
    with torch.no_grad():
        for run in range(num_runs):
            print(f"\n运行 {run + 1}/{num_runs}")
            with memory_context(monitor, f"forward_run_{run+1}"):
                _ = model(input_data)
                torch.cuda.empty_cache()

    # 移除钩子
    for hook in hooks:
        hook.remove()

    return monitor


def monitor_tensor_operations(monitor: MemoryMonitor, func: Callable, *args, **kwargs):
    """
    监控特定张量操作的显存占用

    Args:
        monitor: MemoryMonitor 实例
        func: 要监控的函数
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        函数返回值
    """
    with memory_context(monitor, func.__name__):
        return func(*args, **kwargs)


# ========== 使用示例 ==========

if __name__ == '__main__':
    # 示例 1: 基本使用
    print("示例 1: 基本使用")
    print("-"*80)

    monitor = MemoryMonitor()

    def some_computation(x):
        """模拟一些计算"""
        y = x * 2
        z = torch.randn(1000, 1000, device=x.device)
        w = torch.matmul(y, z)
        return w

    x = torch.randn(1000, 1000, device='cuda:0')

    # 监控多次计算
    for i in range(3):
        with memory_context(monitor, f"computation_{i+1}"):
            result = some_computation(x)
            del result

    monitor.print_summary()
    monitor.print_detailed()

    # 示例 2: 监控模型
    print("\n示例 2: 监控模型")
    print("-"*80)

    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 2000),
        torch.nn.ReLU(),
        torch.nn.Linear(2000, 1000),
        torch.nn.ReLU(),
        torch.nn.Linear(1000, 100)
    ).cuda()

    input_data = torch.randn(32, 1000).cuda()

    monitor_model = MemoryMonitor()

    # 使用上下文管理器监控
    with memory_context(monitor_model, "model_forward"):
        output = model(input_data)

    monitor_model.print_summary()

    # 示例 3: 使用装饰器
    print("\n示例 3: 使用装饰器")
    print("-"*80)

    monitor_decorator = MemoryMonitor()

    @monitor_layer(monitor_decorator, "layer1_conv")
    def layer1(x):
        return torch.conv2d(x, torch.randn(64, 3, 3, 3).cuda())

    @monitor_layer(monitor_decorator, "layer2_relu")
    def layer2(x):
        return torch.relu(x)

    input_img = torch.randn(1, 3, 256, 256).cuda()

    output = layer2(layer1(input_img))

    monitor_decorator.print_summary(top_n=5)
