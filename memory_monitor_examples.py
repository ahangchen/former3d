#!/usr/bin/env python3
"""
PyTorch 显存监控工具使用示例
展示如何监控不同层的显存占用
"""

import torch
import torch.nn as nn
import gc
from memory_monitor_layer import MemoryMonitor, memory_context, monitor_layer


def example_1_basic_usage():
    """示例1: 基本使用 - 监控简单的张量操作"""
    print("="*80)
    print("示例1: 基本使用 - 监控张量操作")
    print("="*80)

    monitor = MemoryMonitor()

    # 创建一些张量
    with memory_context(monitor, "create_tensors"):
        x = torch.randn(1000, 1000, device='cuda:0')
        y = torch.randn(1000, 1000, device='cuda:0')

    # 矩阵乘法
    with memory_context(monitor, "matrix_multiplication"):
        z = torch.matmul(x, y)

    # 激活函数
    with memory_context(monitor, "activation"):
        result = torch.relu(z)

    # 清理
    with memory_context(monitor, "cleanup"):
        del x, y, z, result
        torch.cuda.empty_cache()

    monitor.print_summary(top_n=5)
    monitor.print_detailed()


def example_2_layer_monitoring():
    """示例2: 监控神经网络层"""
    print("\n" + "="*80)
    print("示例2: 监控神经网络层")
    print("="*80)

    # 创建一个简单的CNN
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc = nn.Linear(128 * 64 * 64, 10)  # 适配 256x256 输入

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.max_pool2d(x, 2)  # 256 -> 128
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)  # 128 -> 64
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = SimpleCNN().cuda()
    monitor = MemoryMonitor()

    # 监控前向传播的各个阶段
    input_data = torch.randn(4, 3, 256, 256, device='cuda:0')

    with memory_context(monitor, "input_creation"):
        pass  # input_data already created

    with memory_context(monitor, "conv1"):
        x = torch.relu(model.conv1(input_data))

    with memory_context(monitor, "maxpool1"):
        x = torch.max_pool2d(x, 2)

    with memory_context(monitor, "conv2"):
        x = torch.relu(model.conv2(x))

    with memory_context(monitor, "maxpool2"):
        x = torch.max_pool2d(x, 2)

    with memory_context(monitor, "flatten"):
        x = x.view(x.size(0), -1)

    with memory_context(monitor, "fc"):
        output = model.fc(x)

    monitor.print_summary(top_n=10)

    # 分析显存瓶颈
    print("\n显存瓶颈分析:")
    print("-"*80)
    sorted_records = sorted(monitor.records, key=lambda x: x['delta_allocated'], reverse=True)
    if sorted_records:
        max_delta = sorted_records[0]['delta_allocated']
        print(f"最大显存增量: {max_delta:.4f} GB")
        print(f"最大增量来源: {sorted_records[0]['layer']}")

    final = monitor.records[-1]['after']
    usage_pct = (final['allocated'] / final['total'] * 100)
    print(f"最终显存使用率: {usage_pct:.2f}%")


def example_3_decorator_usage():
    """示例3: 使用装饰器监控函数"""
    print("\n" + "="*80)
    print("示例3: 使用装饰器监控函数")
    print("="*80)

    monitor = MemoryMonitor()

    @monitor_layer(monitor, "layer1_conv")
    def conv_layer(x):
        return torch.conv2d(x, torch.randn(64, 3, 3, 3).cuda())

    @monitor_layer(monitor, "layer2_batchnorm")
    def bn_layer(x):
        bn = nn.BatchNorm2d(64).cuda()
        return bn(x)

    @monitor_layer(monitor, "layer3_relu")
    def relu_layer(x):
        return torch.relu(x)

    @monitor_layer(monitor, "layer4_pool")
    def pool_layer(x):
        return torch.max_pool2d(x, 2)

    # 测试
    input_img = torch.randn(2, 3, 128, 128).cuda()

    output = pool_layer(relu_layer(bn_layer(conv_layer(input_img))))

    monitor.print_summary(top_n=5)


def example_4_batch_processing():
    """示例4: 监控批量处理的显存变化"""
    print("\n" + "="*80)
    print("示例4: 监控批量处理的显存变化")
    print("="*80)

    monitor = MemoryMonitor()

    # 测试不同 batch size 的显存占用
    model = nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 100)
    ).cuda()

    batch_sizes = [1, 2, 4, 8, 16]

    print(f"\n{'Batch Size':<15} {'显存占用 (GB)':<20} {'增加 (GB)':<20}")
    print("-"*60)

    prev_allocated = 0

    for batch_size in batch_sizes:
        with memory_context(monitor, f"batch_{batch_size}"):
            input_data = torch.randn(batch_size, 1000).cuda()
            output = model(input_data)

        current = monitor.records[-1]['after']['allocated']
        delta = current - prev_allocated
        prev_allocated = current

        print(f"{batch_size:<15} {current:<20.4f} {delta:<20.4f}")

        del input_data, output
        torch.cuda.empty_cache()

    print("-"*60)

    # 找出最优 batch size
    print("\n最优 batch size 分析:")
    sorted_by_efficiency = sorted(
        [(r['layer'], r['after']['allocated']) for r in monitor.records if 'batch_' in r['layer']],
        key=lambda x: x[1]
    )

    print("显存占用最小的配置:")
    for name, mem in sorted_by_efficiency[:3]:
        print(f"  {name}: {mem:.4f} GB")


def example_5_memory_leak_detection():
    """示例5: 检测显存泄漏"""
    print("\n" + "="*80)
    print("示例5: 检测显存泄漏")
    print("="*80)

    monitor = MemoryMonitor()

    def function_without_leak(x):
        """无显存泄漏的函数"""
        y = x * 2
        z = torch.randn(1000, 1000, device=x.device)
        result = torch.matmul(y, z)
        return result

    def function_with_leak(x):
        """有显存泄漏的函数（模拟）"""
        global leaked_tensor
        y = x * 2
        leaked_tensor = torch.randn(1000, 1000, device=x.device)  # 泄漏！
        result = torch.matmul(y, leaked_tensor)
        return result

    print("\n测试无泄漏的函数:")
    for i in range(3):
        with memory_context(monitor, f"no_leak_run_{i+1}"):
            x = torch.randn(1000, 1000, device='cuda:0')
            result = function_without_leak(x)
            del result  # 正确清理
        torch.cuda.empty_cache()

    print("\n测试有泄漏的函数:")
    for i in range(3):
        with memory_context(monitor, f"with_leak_run_{i+1}"):
            x = torch.randn(1000, 1000, device='cuda:0')
            result = function_with_leak(x)
            del result  # 没有清理 leaked_tensor

    monitor.print_summary(top_n=10)

    # 分析
    print("\n显存泄漏分析:")
    print("-"*80)

    no_leak_runs = [r for r in monitor.records if 'no_leak' in r['layer']]
    with_leak_runs = [r for r in monitor.records if 'with_leak' in r['layer']]

    if no_leak_runs and with_leak_runs:
        no_leak_avg = sum(r['delta_allocated'] for r in no_leak_runs) / len(no_leak_runs)
        with_leak_avg = sum(r['delta_allocated'] for r in with_leak_runs) / len(with_leak_runs)

        print(f"无泄漏函数平均增量: {no_leak_avg:.4f} GB")
        print(f"有泄漏函数平均增量: {with_leak_avg:.4f} GB")
        print(f"差异: {with_leak_avg - no_leak_avg:.4f} GB")

        if with_leak_avg > no_leak_avg * 1.5:
            print("⚠️  检测到潜在的显存泄漏！")


if __name__ == '__main__':
    print("PyTorch 显存监控工具示例")
    print("="*80)

    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        exit(1)

    # 运行所有示例
    example_1_basic_usage()
    example_2_layer_monitoring()
    example_3_decorator_usage()
    example_4_batch_processing()
    example_5_memory_leak_detection()

    print("\n" + "="*80)
    print("所有示例运行完成！")
    print("="*80)
