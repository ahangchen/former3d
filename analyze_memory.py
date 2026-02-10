#!/usr/bin/env python3
"""
显存分析报告生成器
分析显存分析数据并生成详细报告
"""

import json
import sys
from typing import Dict, List

def load_memory_data(file_path: str) -> Dict:
    """加载显存分析数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def format_mb(value: float) -> str:
    """格式化MB数值"""
    return f"{value:.2f} MB"

def analyze_step_data(data: Dict) -> List[Dict]:
    """分析按训练步的数据"""
    results = []
    by_step = data.get('by_step', {})

    for step, step_data in sorted(by_step.items()):
        for gpu_id, gpu_stats in step_data.get('gpu_stats', {}).items():
            results.append({
                'step': step,
                'gpu': gpu_id,
                'min_allocated': gpu_stats.get('min_allocated_mb', 0),
                'max_allocated': gpu_stats.get('max_allocated_mb', 0),
                'avg_allocated': gpu_stats.get('avg_allocated_mb', 0),
                'min_reserved': gpu_stats.get('min_reserved_mb', 0),
                'max_reserved': gpu_stats.get('max_reserved_mb', 0),
            })

    return results

def analyze_layer_data(data: Dict) -> List[Dict]:
    """分析按层的数据"""
    results = []
    by_layer = data.get('by_layer', {})

    for layer_name, layer_data in by_layer.items():
        for gpu_id, gpu_stats in layer_data.get('gpu_stats', {}).items():
            results.append({
                'layer': layer_name,
                'gpu': gpu_id,
                'min_allocated': gpu_stats.get('min_allocated_mb', 0),
                'max_allocated': gpu_stats.get('max_allocated_mb', 0),
                'avg_allocated': gpu_stats.get('avg_allocated_mb', 0),
                'min_reserved': gpu_stats.get('min_reserved_mb', 0),
                'max_reserved': gpu_stats.get('max_reserved_mb', 0),
            })

    return results

def print_summary(data: Dict):
    """打印摘要"""
    print("\n" + "=" * 80)
    print("显存分析摘要报告")
    print("=" * 80)

    # 当前显存状态
    print("\n【当前显存状态】")
    current_memory = data.get('current_memory', {})
    for gpu_id, info in current_memory.items():
        print(f"\nGPU {gpu_id}:")
        print(f"  已分配: {format_mb(info.get('allocated_mb', 0))}")
        print(f"  已预留: {format_mb(info.get('reserved_mb', 0))}")
        print(f"  最大已分配: {format_mb(info.get('max_allocated_mb', 0))}")
        print(f"  最大已预留: {format_mb(info.get('max_reserved_mb', 0))}")

    # 按训练步分析
    print("\n\n【按训练步分析】")
    step_results = analyze_step_data(data)
    if step_results:
        print(f"\n{'步数':<10} {'GPU':<8} {'最小分配':<15} {'最大分配':<15} {'平均分配':<15} {'最小预留':<15} {'最大预留':<15}")
        print("-" * 105)
        for result in step_results:
            print(f"{'Step '+str(result['step']):<10} {result['gpu']:<8} "
                  f"{format_mb(result['min_allocated']):<15} "
                  f"{format_mb(result['max_allocated']):<15} "
                  f"{format_mb(result['avg_allocated']):<15} "
                  f"{format_mb(result['min_reserved']):<15} "
                  f"{format_mb(result['max_reserved']):<15}")

    # 按层分析
    print("\n\n【按网络层分析】")
    layer_results = analyze_layer_data(data)
    if layer_results:
        print(f"\n{'网络层':<20} {'GPU':<8} {'最小分配':<15} {'最大分配':<15} {'平均分配':<15}")
        print("-" * 85)
        for result in layer_results:
            print(f"{result['layer']:<20} {result['gpu']:<8} "
                  f"{format_mb(result['min_allocated']):<15} "
                  f"{format_mb(result['max_allocated']):<15} "
                  f"{format_mb(result['avg_allocated']):<15}")

    # 显存增长分析
    print("\n\n【显存增长分析】")
    if current_memory:
        for gpu_id, info in current_memory.items():
            initial = 0
            for layer_name, layer_data in data.get('by_layer', {}).items():
                if gpu_id in layer_data.get('gpu_stats', {}):
                    gpu_stats = layer_data['gpu_stats'][gpu_id]
                    if gpu_stats.get('min_allocated_mb', 0) > 0:
                        initial = gpu_stats['min_allocated_mb']
                        break

            final = info.get('max_allocated_mb', 0)
            growth = final - initial
            growth_rate = (growth / initial * 100) if initial > 0 else 0

            print(f"\nGPU {gpu_id}:")
            print(f"  初始显存: {format_mb(initial)}")
            print(f"  最终显存: {format_mb(final)}")
            print(f"  增长量: {format_mb(growth)}")
            print(f"  增长率: {growth_rate:.2f}%")

    print("\n" + "=" * 80)

def compare_batches(batch1_file: str, batch2_file: str):
    """比较两个batch的显存使用"""
    print("\n" + "=" * 80)
    print("多批次显存对比分析")
    print("=" * 80)

    data1 = load_memory_data(batch1_file)
    data2 = load_memory_data(batch2_file)

    # 提取GPU 0的最大分配显存
    gpu0_max1 = data1.get('current_memory', {}).get('0', {}).get('max_allocated_mb', 0)
    gpu0_max2 = data2.get('current_memory', {}).get('0', {}).get('max_allocated_mb', 0)

    print(f"\nGPU 0 最大显存使用:")
    print(f"  {batch1_file}: {format_mb(gpu0_max1)}")
    print(f"  {batch2_file}: {format_mb(gpu0_max2)}")
    print(f"  差异: {format_mb(gpu0_max2 - gpu0_max1)}")

    print("\n" + "=" * 80)

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法: python3 analyze_memory.py <memory_summary.json>")
        print("       python3 analyze_memory.py <file1.json> <file2.json>")
        return

    if len(sys.argv) == 2:
        # 分析单个文件
        file_path = sys.argv[1]
        print(f"正在分析: {file_path}")
        data = load_memory_data(file_path)
        print_summary(data)
    elif len(sys.argv) == 3:
        # 对比两个文件
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        print(f"正在对比: {file1} vs {file2}")
        compare_batches(file1, file2)

if __name__ == '__main__':
    main()
