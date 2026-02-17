#!/usr/bin/env python3
"""
训练状态监控脚本
定期检查训练日志并提取关键信息
"""

import os
import sys
import re
import json
from datetime import datetime
from pathlib import Path

# 日志文件路径
LOG_DIR = "/home/cwh/coding/former3d/logs"
CHECKPOINT_DIR = "/home/cwh/coding/former3d/checkpoints/ddp"

def parse_training_log(log_file):
    """解析训练日志，提取关键信息"""
    if not os.path.exists(log_file):
        return None

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取最新的epoch信息
    epoch_pattern = r'Epoch\s+(\d+)/(\d+)'
    loss_pattern = r'train_loss:\s+([\d.]+)'
    val_loss_pattern = r'val_loss:\s+([\d.]+)'
    time_pattern = r'time:\s+([\d.]+)s'

    epochs = []
    current_epoch = None
    train_loss = None
    val_loss = None
    time_taken = None

    for line in content.split('\n'):
        if 'Epoch' in line and '/' in line:
            match = re.search(epoch_pattern, line)
            if match:
                current_epoch = int(match.group(1))
                epochs.append(current_epoch)

        if 'train_loss:' in line:
            match = re.search(loss_pattern, line)
            if match:
                train_loss = float(match.group(1))

        if 'val_loss:' in line:
            match = re.search(val_loss_pattern, line)
            if match:
                val_loss = float(match.group(1))

        if 'time:' in line:
            match = re.search(time_pattern, line)
            if match:
                time_taken = float(match.group(1))

    # 检查异常信息
    errors = []
    error_patterns = [
        r'Error:',
        r'Exception:',
        r'Traceback',
        r'CUDA out of memory',
        r'RuntimeError',
        r'ValueError',
        r'AssertionError'
    ]

    for line in content.split('\n'):
        for pattern in error_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                if line.strip() and line.strip() not in errors:
                    errors.append(line.strip())

    return {
        'current_epoch': current_epoch,
        'total_epochs': 50,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'time': time_taken,
        'errors': errors[-5:] if errors else None  # 只保留最后5个错误
    }

def get_gpu_memory():
    """获取GPU显存使用情况"""
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )

        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_info.append({
                        'index': int(parts[0]),
                        'memory_used_mb': int(parts[1]),
                        'memory_total_mb': int(parts[2]),
                        'utilization_percent': int(parts[3])
                    })
        return gpu_info
    except Exception as e:
        return {'error': str(e)}

def find_latest_log():
    """查找最新的训练日志文件"""
    if not os.path.exists(LOG_DIR):
        return None

    log_files = list(Path(LOG_DIR).glob('*.log'))
    if not log_files:
        return None

    # 按修改时间排序
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(log_files[0])

def format_status_message(log_info, gpu_info):
    """格式化状态消息"""
    if not log_info:
        return "⚠️ 未找到训练日志"

    msg = []
    msg.append(f"📊 训练状态报告")
    msg.append(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    msg.append("")

    if log_info['current_epoch'] is not None:
        msg.append(f"🔄 Epoch进度: {log_info['current_epoch']}/{log_info['total_epochs']}")
    else:
        msg.append("🔄 Epoch: 未找到")

    if log_info['train_loss'] is not None:
        msg.append(f"📉 训练Loss: {log_info['train_loss']:.6f}")
    else:
        msg.append("📉 训练Loss: 未找到")

    if log_info['val_loss'] is not None:
        msg.append(f"✅ 验证Loss: {log_info['val_loss']:.6f}")
    else:
        msg.append("✅ 验证Loss: 未找到")

    if log_info['time'] is not None:
        msg.append(f"⏱️  本轮用时: {log_info['time']:.2f}s")

    msg.append("")
    msg.append("🎮 GPU显存:")

    if isinstance(gpu_info, list) and gpu_info:
        for gpu in gpu_info:
            used_gb = gpu['memory_used_mb'] / 1024
            total_gb = gpu['memory_total_mb'] / 1024
            msg.append(f"  GPU {gpu['index']}: {used_gb:.2f}GB / {total_gb:.2f}GB ({gpu['utilization_percent']}%)")
    elif isinstance(gpu_info, dict) and 'error' in gpu_info:
        msg.append(f"  ⚠️ 无法获取GPU信息: {gpu_info['error']}")

    # 检查异常
    if log_info['errors']:
        msg.append("")
        msg.append("⚠️ 发现异常:")
        for error in log_info['errors']:
            msg.append(f"  • {error}")
    else:
        msg.append("")
        msg.append("✅ 未发现异常")

    return "\n".join(msg)

def main():
    """主函数"""
    # 查找最新日志
    log_file = find_latest_log()
    if not log_file:
        print("❌ 未找到训练日志")
        return

    print(f"📄 解析日志: {log_file}")

    # 解析日志
    log_info = parse_training_log(log_file)

    # 获取GPU信息
    gpu_info = get_gpu_memory()

    # 格式化消息
    message = format_status_message(log_info, gpu_info)

    # 保存到文件供发送脚本读取
    status_file = "/tmp/training_status.txt"
    with open(status_file, 'w', encoding='utf-8') as f:
        f.write(message)

    print(f"✅ 状态已保存到: {status_file}")
    print("\n" + message)

if __name__ == '__main__':
    main()
