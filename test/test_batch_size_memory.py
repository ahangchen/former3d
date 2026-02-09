#!/usr/bin/env python3
"""
实时显存监控训练脚本
测试不同batch size下的显存使用
"""

import torch
import subprocess
import time
import re

def parse_memory_output(output):
    """解析显存使用情况"""
    # 查找显存信息
    lines = output.split('\n')
    for line in lines:
        if '显存使用过高' in line or '内存清理完成' in line:
            # 提取显存使用量
            match = re.search(r'已分配:\s*([\d.]+)\s*GB', line)
            if match:
                return float(match.group(1))
    return None

def run_training_with_memory_monitor(batch_size, crop_size="24,24,16", voxel_size=0.16):
    """运行训练并监控显存"""

    print(f"\n{'='*70}")
    print(f"测试 batch_size={batch_size}")
    print(f"{'='*70}")

    # 启动训练进程
    cmd = [
        'python', 'train_stream_integrated.py',
        '--epochs', '1',
        '--test-only',
        '--batch-size', str(batch_size),
        '--crop-size', crop_size,
        '--voxel-size', str(voxel_size),
        '--num-workers', '0',
        '--max-sequences', '1'
    ]

    print(f"执行命令: {' '.join(cmd)}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 检查初始显存
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(device) / 1024**3
    print(f"初始显存: {initial_memory:.2f} GB")

    # 运行训练
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        elapsed_time = time.time() - start_time

        # 检查最终显存
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated(device) / 1024**3

        print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"运行时间: {elapsed_time:.1f}秒")
        print(f"最终显存: {final_memory:.2f} GB")

        # 分析输出
        if result.returncode == 0:
            print("✅ 训练成功完成")
            return True, initial_memory, final_memory
        else:
            print(f"❌ 训练失败，返回码: {result.returncode}")
            print("错误输出:")
            print(result.stderr[-500:])  # 只显示最后500个字符
            return False, initial_memory, final_memory

    except subprocess.TimeoutExpired:
        print(f"❌ 训练超时（180秒）")
        return False, initial_memory, None
    except Exception as e:
        print(f"❌ 训练异常: {e}")
        return False, initial_memory, None

def test_batch_sizes():
    """测试不同batch size"""

    # 定义要测试的batch size序列
    batch_sizes = [1, 2, 4, 8]

    results = []

    for batch_size in batch_sizes:
        success, initial_mem, final_mem = run_training_with_memory_monitor(batch_size)

        results.append({
            'batch_size': batch_size,
            'success': success,
            'initial_memory': initial_mem,
            'final_memory': final_mem,
            'peak_memory': final_mem if success else None
        })

        # 如果失败，停止测试更大的batch size
        if not success:
            print(f"\n⚠️  batch_size={batch_size} 失败，停止测试")
            break

        # 等待一下，让显存完全释放
        time.sleep(5)
        torch.cuda.empty_cache()

    # 打印总结
    print(f"\n{'='*70}")
    print("测试总结")
    print(f"{'='*70}")
    print(f"{'Batch Size':<12} {'状态':<8} {'峰值显存 (GB)':<15} {'结论'}")
    print("-"*70)

    for result in results:
        status = "✅ 成功" if result['success'] else "❌ 失败"
        peak_mem = f"{result['peak_memory']:.2f}" if result['peak_memory'] else "N/A"
        conclusion = ""

        if result['success']:
            if result['peak_memory'] < 8.0:
                conclusion = "推荐"
            elif result['peak_memory'] < 9.0:
                conclusion = "可用"
            else:
                conclusion = "接近上限"

        print(f"{result['batch_size']:<12} {status:<8} {peak_mem:<15} {conclusion}")

    # 找出最大的可用batch size
    max_batch = None
    for result in reversed(results):
        if result['success'] and result['peak_memory'] < 9.0:
            max_batch = result['batch_size']
            break

    print("-"*70)
    if max_batch:
        print(f"\n✅ 推荐配置: batch_size={max_batch}, peak_memory={results[max_batch-1]['peak_memory']:.2f} GB")
    else:
        print(f"\n❌ 未找到合适的batch size配置")

    return results

if __name__ == "__main__":
    print("="*70)
    print("流式训练显存监控测试")
    print("="*70)

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，无法进行GPU训练")
        exit(1)

    # 打印GPU信息
    device = torch.device("cuda:0")
    print(f"\nGPU: {torch.cuda.get_device_name(device)}")
    print(f"总显存: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")

    # 运行测试
    results = test_batch_sizes()

    exit(0)
