#!/usr/bin/env python3
"""
智能训练监控脚本 - 每个epoch完成时发送状态报告
"""

import os
import sys
import re
import json
import time
from datetime import datetime
from pathlib import Path

# 配置
LOG_DIR = "/home/cwh/coding/former3d/logs"
CHECKPOINT_DIR = "/home/cwh/coding/former3d/checkpoints/ddp"
STATE_FILE = "/tmp/training_monitor_state.json"
REPORT_FILE = "/tmp/training_report.txt"

def load_state():
    """加载监控状态"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {
        'last_reported_epoch': -1,
        'epoch_start_times': {},
        'epoch_losses': {}
    }

def save_state(state):
    """保存监控状态"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def parse_latest_log():
    """解析最新日志文件"""
    if not os.path.exists(LOG_DIR):
        return None

    log_files = list(Path(LOG_DIR).glob('*.log'))
    if not log_files:
        return None

    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    log_file = str(log_files[0])

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析epoch信息
    epoch_info = {}
    batch_losses = []

    current_epoch = None
    epoch_line_pattern = r'Epoch\s+(\d+)/(\d+)\s+\(LR:\s+([\d.e-]+)\)'
    batch_pattern = r'Epoch\s+\[(\d+)/(\d+)\]\s+Batch\s+\[(\d+)/(\d+)\]\s+Loss:\s+([\d.]+)\s+LR:\s+([\d.e-]+)'
    epoch_complete_pattern = r'Epoch\s+\[(\d+)/(\d+)\].*completed'

    for line in lines:
        # 检查epoch开始
        epoch_match = re.search(epoch_line_pattern, line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
            lr = float(epoch_match.group(3))
            epoch_info['current_epoch'] = current_epoch
            epoch_info['total_epochs'] = total_epochs
            epoch_info['learning_rate'] = lr

        # 检查batch进度 - 只记录当前epoch的batch
        batch_match = re.search(batch_pattern, line)
        if batch_match:
            epoch = int(batch_match.group(1))
            batch = int(batch_match.group(3))
            total_batches = int(batch_match.group(4))
            loss = float(batch_match.group(5))
            lr = float(batch_match.group(6))

            # 只记录当前epoch或最后一个epoch的batch信息
            if current_epoch is None or epoch == current_epoch:
                epoch_info['epoch'] = epoch
                epoch_info['batch'] = batch
                epoch_info['total_batches'] = total_batches
                epoch_info['current_loss'] = loss
                epoch_info['current_lr'] = lr

                batch_losses.append(loss)

        # 检查epoch完成
        complete_match = re.search(epoch_complete_pattern, line, re.IGNORECASE)
        if complete_match:
            epoch = int(complete_match.group(1))
            epoch_info['epoch_completed'] = epoch

    if batch_losses:
        epoch_info['avg_loss'] = sum(batch_losses[-10:]) / min(10, len(batch_losses))

    # 提取验证loss
    val_loss_pattern = r'val_loss:\s+([\d.]+)'
    for line in reversed(lines):
        val_match = re.search(val_loss_pattern, line)
        if val_match:
            epoch_info['val_loss'] = float(val_match.group(1))
            break

    # 检查错误
    errors = []
    error_keywords = ['Error:', 'Exception:', 'Traceback', 'CUDA out of memory',
                      'RuntimeError', 'ValueError', 'AssertionError']
    for line in lines:
        for keyword in error_keywords:
            if keyword in line:
                if line.strip() not in errors:
                    errors.append(line.strip())
                break

    if errors:
        epoch_info['errors'] = errors[-5:]

    return epoch_info

def get_gpu_memory():
    """获取GPU显存使用情况"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=10
        )

        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpu_info.append({
                        'index': int(parts[0]),
                        'memory_used_mb': int(parts[1]),
                        'memory_total_mb': int(parts[2]),
                        'utilization_percent': int(parts[3]),
                        'temperature_c': int(parts[4])
                    })
        return gpu_info
    except Exception as e:
        return {'error': str(e)}

def format_report(epoch_info, gpu_info, state):
    """格式化训练报告"""
    report = []
    report.append("📊 分布式训练状态报告")
    report.append(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 50)
    report.append("")

    # Epoch进度
    if 'current_epoch' in epoch_info:
        current = epoch_info['current_epoch']
        total = epoch_info.get('total_epochs', 50)
        progress = (current + 1) / total * 100
        report.append(f"🔄 Epoch进度: {current + 1}/{total} ({progress:.1f}%)")
    else:
        report.append("🔄 Epoch: 初始化中...")

    # Batch进度
    if 'batch' in epoch_info:
        batch = epoch_info['batch']
        total_batches = epoch_info['total_batches']
        progress = (batch + 1) / total_batches * 100
        report.append(f"📦 Batch进度: {batch + 1}/{total_batches} ({progress:.1f}%)")

    report.append("")

    # Loss信息
    if 'current_loss' in epoch_info:
        report.append(f"📉 当前Loss: {epoch_info['current_loss']:.6f}")

    if 'avg_loss' in epoch_info:
        report.append(f"📊 平均Loss(最近10批): {epoch_info['avg_loss']:.6f}")

    if 'val_loss' in epoch_info:
        report.append(f"✅ 验证Loss: {epoch_info['val_loss']:.6f}")

    if 'learning_rate' in epoch_info:
        report.append(f"🎯 学习率: {epoch_info['learning_rate']:.2e}")

    report.append("")

    # GPU显存
    report.append("🎮 GPU状态:")
    if isinstance(gpu_info, list) and gpu_info:
        for gpu in gpu_info:
            used_gb = gpu['memory_used_mb'] / 1024
            total_gb = gpu['memory_total_mb'] / 1024
            util = gpu['utilization_percent']
            temp = gpu['temperature_c']
            report.append(f"  GPU {gpu['index']}: {used_gb:.2f}GB/{total_gb:.2f}GB "
                          f"({util}% 使用率, {temp}°C)")
    elif isinstance(gpu_info, dict) and 'error' in gpu_info:
        report.append(f"  ⚠️ 无法获取GPU信息")

    # 异常检查
    report.append("")
    if 'errors' in epoch_info and epoch_info['errors']:
        report.append("⚠️ 发现异常:")
        for error in epoch_info['errors']:
            report.append(f"  • {error}")
    else:
        report.append("✅ 未发现异常")

    report.append("")
    report.append("=" * 50)

    return "\n".join(report)

def main():
    """主函数"""
    # 加载状态
    state = load_state()

    # 解析日志
    epoch_info = parse_latest_log()
    if not epoch_info:
        print("⚠️ 未找到训练日志")
        return

    # 获取GPU信息
    gpu_info = get_gpu_memory()

    # 检查是否有新的epoch完成
    should_report = False
    if 'epoch_completed' in epoch_info:
        completed_epoch = epoch_info['epoch_completed']
        if completed_epoch > state['last_reported_epoch']:
            state['last_reported_epoch'] = completed_epoch
            should_report = True
            print(f"✅ Epoch {completed_epoch} 完成，生成报告...")
    elif 'current_epoch' in epoch_info:
        # 如果没有明确的完成标记，根据batch进度判断
        current_epoch = epoch_info['current_epoch']
        if current_epoch > state['last_reported_epoch']:
            # 检查是否接近epoch结束（batch > 95%）
            if 'batch' in epoch_info and 'total_batches' in epoch_info:
                progress = epoch_info['batch'] / epoch_info['total_batches']
                if progress > 0.95:
                    state['last_reported_epoch'] = current_epoch
                    should_report = True
                    print(f"✅ Epoch {current_epoch} 即将完成，生成报告...")

    # 如果需要报告或首次运行
    if should_report or state['last_reported_epoch'] == -1:
        report = format_report(epoch_info, gpu_info, state)

        # 保存报告
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(report)

        print(report)
        print(f"\n📄 报告已保存到: {REPORT_FILE}")

        # 保存状态
        save_state(state)
    else:
        # 不需要报告，只显示简要信息
        if 'current_epoch' in epoch_info:
            print(f"⏳ 训练进行中: Epoch {epoch_info['current_epoch']}, "
                  f"Batch {epoch_info.get('batch', 0)}, "
                  f"Loss {epoch_info.get('current_loss', 0):.6f}")
        else:
            print("⏳ 训练初始化中...")

if __name__ == '__main__':
    main()
