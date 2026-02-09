#!/usr/bin/env python3
"""
渐进式batch size测试和实际训练
先找到最大可行batch size，然后进行2个epoch的实际训练
"""

import torch
import subprocess
import time
import signal
import os

def get_gpu_memory_gb():
    """获取GPU显存使用量（GB）"""
    if not torch.cuda.is_available():
        return 0, 0, 0
    device = torch.device("cuda:0")
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return allocated, reserved, total

def test_batch_size(batch_size, crop_size="24,24,16", voxel_size=0.16, test_mode=True):
    """测试指定batch size是否可行"""

    print(f"\n{'='*70}")
    print(f"测试 batch_size={batch_size} (test_mode={test_mode})")
    print(f"{'='*70}")

    # 清理显存
    torch.cuda.empty_cache()
    initial_mem, _, total_mem = get_gpu_memory_gb()
    print(f"初始显存: {initial_mem:.2f} GB / {total_mem:.2f} GB")

    # 构建命令
    if test_mode:
        cmd = [
            'python', 'train_stream_integrated.py',
            '--epochs', '1',
            '--test-only',
            '--batch-size', str(batch_size),
            '--crop-size', crop_size,
            '--voxel-size', str(voxel_size),
            '--num-workers', '0',
            '--max-sequences', '1',
            '--sequence-length', '5'  # 减少序列长度以加快测试
        ]
    else:
        cmd = [
            'python', 'train_stream_integrated.py',
            '--epochs', '2',
            '--batch-size', str(batch_size),
            '--crop-size', crop_size,
            '--voxel-size', str(voxel_size),
            '--num-workers', '0',
            '--max-sequences', '1',
            '--sequence-length', '5'  # 减少序列长度以加快测试
        ]

    print(f"执行命令: {' '.join(cmd)}")

    # 运行训练
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300 if test_mode else 600  # 测试模式5分钟，训练模式10分钟
        )

        elapsed_time = time.time() - start_time

        # 检查显存
        torch.cuda.empty_cache()
        final_mem, reserved_mem, _ = get_gpu_memory_gb()
        peak_mem = reserved_mem  # 近似峰值

        # 分析结果
        if result.returncode == 0:
            print(f"✅ 成功完成")
            print(f"   运行时间: {elapsed_time:.1f}秒")
            print(f"   峰值显存: {peak_mem:.2f} GB")
            print(f"   剩余显存: {total_mem - peak_mem:.2f} GB")

            # 检查是否有显存警告
            if "CUDA out of memory" in result.stderr:
                print(f"⚠️  检测到CUDA OOM，但进程正常退出")
                return False, peak_mem
            else:
                return True, peak_mem
        else:
            print(f"❌ 失败，返回码: {result.returncode}")

            # 检查是否是OOM
            if "CUDA out of memory" in result.stderr or "out of memory" in result.stderr.lower():
                print(f"   原因: CUDA OOM (显存不足)")
                return False, peak_mem
            else:
                print(f"   错误: {result.stderr[-200:]}")
                return False, peak_mem

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"⏱️  超时 ({'300' if test_mode else '600'}秒)")
        return False, None

    except KeyboardInterrupt:
        print(f"\n⚠️  被中断")
        return False, None

    except Exception as e:
        print(f"❌ 异常: {e}")
        return False, None

def find_max_batch_size():
    """渐进式测试找到最大可行的batch size"""

    print("="*70)
    print("第1步: 渐进式测试batch size")
    print("="*70)

    # 定义测试序列
    batch_sizes = [1, 2, 4, 8]

    results = []

    for batch_size in batch_sizes:
        success, peak_mem = test_batch_size(batch_size, test_mode=True)

        results.append({
            'batch_size': batch_size,
            'success': success,
            'peak_memory': peak_mem
        })

        # 如果失败，停止测试更大的batch size
        if not success:
            print(f"\n⚠️  batch_size={batch_size} 失败，停止测试")
            break

        # 等待显存释放
        time.sleep(3)
        torch.cuda.empty_cache()

    # 找出最大的成功batch size
    max_successful = None
    for result in reversed(results):
        if result['success'] and result['peak_memory'] < 9.0:  # 留一些余量
            max_successful = result['batch_size']
            max_peak_mem = result['peak_memory']
            break

    # 打印总结
    print(f"\n{'='*70}")
    print("第1步测试总结")
    print(f"{'='*70}")
    print(f"{'Batch Size':<12} {'状态':<8} {'峰值显存 (GB)':<15} {'结论'}")
    print("-"*70)

    for result in results:
        status = "✅ 成功" if result['success'] else "❌ 失败"
        peak = f"{result['peak_memory']:.2f}" if result['peak_memory'] else "N/A"

        if result['success']:
            if result['peak_memory'] < 8.0:
                conclusion = "推荐"
            elif result['peak_memory'] < 9.0:
                conclusion = "可用"
            else:
                conclusion = "接近上限"
        else:
            conclusion = "OOM"

        print(f"{result['batch_size']:<12} {status:<8} {peak:<15} {conclusion}")

    print("-"*70)

    if max_successful:
        print(f"\n✅ 推荐配置: batch_size={max_successful}, 峰值显存={max_peak_mem:.2f} GB")
        return max_successful
    else:
        print(f"\n❌ 所有batch size都失败，使用默认值1")
        return 1

def run_actual_training(batch_size):
    """进行实际训练（2个epoch）"""

    print(f"\n{'='*70}")
    print(f"第2步: 实际训练（2个epoch，batch_size={batch_size})")
    print(f"{'='*70}")

    # 清理显存
    torch.cuda.empty_cache()
    initial_mem, _, total_mem = get_gpu_memory_gb()
    print(f"初始显存: {initial_mem:.2f} GB / {total_mem:.2f} GB")

    # 构建命令
    cmd = [
        'python', 'train_stream_integrated.py',
        '--epochs', '2',
        '--batch-size', str(batch_size),
        '--crop-size', "24,24,16",
        '--voxel-size', "0.16",
        '--num-workers', '0',
        '--max-sequences', '5',  # 使用更多序列进行实际训练
        '--sequence-length', '10'
    ]

    print(f"\n执行命令: {' '.join(cmd)}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # 运行训练
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 最多1小时
        )

        elapsed_time = time.time() - start_time

        # 检查显存
        torch.cuda.empty_cache()
        final_mem, _, _ = get_gpu_memory_gb()

        print(f"\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"运行时间: {elapsed_time/60:.1f}分钟")

        # 打印结果
        if result.returncode == 0:
            print(f"\n✅ 训练成功完成！")
            print(f"   最终显存: {final_mem:.2f} GB")

            # 提取训练日志
            lines = result.stdout.split('\n')
            print(f"\n训练日志摘要:")
            for line in lines:
                if "Epoch" in line and "Loss" in line:
                    print(f"  {line}")

            return True
        else:
            print(f"\n❌ 训练失败，返回码: {result.returncode}")

            if "CUDA out of memory" in result.stderr or "out of memory" in result.stderr.lower():
                print(f"   原因: CUDA OOM (显存不足)")
                print(f"   建议: 减小batch_size到{batch_size//2}")
            else:
                print(f"   错误: {result.stderr[-500:]}")

            return False

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  超时 (3600秒 = 1小时)")
        print(f"   已运行: {elapsed_time/60:.1f}分钟")
        print(f"   可能原因: 训练时间过长，建议减少序列数或epoch数")
        return False

    except KeyboardInterrupt:
        print(f"\n⚠️  训练被中断")
        return False

    except Exception as e:
        print(f"❌ 异常: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""

    print("="*70)
    print("流式训练验证 - 找到最大batch size并进行实际训练")
    print("="*70)

    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return 1

    device = torch.device("cuda:0")
    print(f"\nGPU: {torch.cuda.get_device_name(device)}")
    _, _, total_mem = get_gpu_memory_gb()
    print(f"总显存: {total_mem:.2f} GB")

    # 第1步: 找到最大可行batch size
    max_batch_size = find_max_batch_size()

    # 等待一下
    time.sleep(5)
    torch.cuda.empty_cache()

    # 第2步: 进行实际训练
    success = run_actual_training(max_batch_size)

    # 最终总结
    print(f"\n{'='*70}")
    print("最终总结")
    print(f"{'='*70}")
    if success:
        print(f"✅ 成功完成实际训练")
        print(f"   配置: batch_size={max_batch_size}, epochs=2")
        return 0
    else:
        print(f"❌ 实际训练失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
