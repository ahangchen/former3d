"""
实验配置管理模块

提供实验目录名生成和配置保存功能。
"""

import os
import json
from datetime import datetime
from typing import Any, Dict


def generate_experiment_name(args: Any) -> str:
    """
    根据命令行参数生成实验目录名

    Args:
        args: 命令行参数对象

    Returns:
        str: 实验目录名
    """
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 关键参数（用于生成目录名）
    # 选择最能区分实验的参数
    batch_size = getattr(args, 'batch_size', 2)
    learning_rate = getattr(args, 'learning_rate', 1e-4)
    seq_len = getattr(args, 'sequence_length', 10)
    attn_heads = getattr(args, 'attn_heads', 1)
    attn_layers = getattr(args, 'attn_layers', 1)
    voxel_size = getattr(args, 'voxel_size', 0.16)

    # 格式化学习率（转换为科学计数法）
    lr_str = f"{learning_rate:.0e}".replace('e-', 'em').replace('e+', 'e')

    # 生成实验名
    # 格式: {timestamp}_bs{batch_size}_lr{lr}_seq{seq_len}_h{attn_heads}_l{attn_layers}_v{voxel_size}
    experiment_name = (
        f"{timestamp}_"
        f"bs{batch_size}_"
        f"lr{lr_str}_"
        f"seq{seq_len}_"
        f"h{attn_heads}_"
        f"l{attn_layers}_"
        f"v{voxel_size}"
    )

    return experiment_name


def save_experiment_config(args: Any, experiment_dir: str, model_name: str = "StreamSDFFormerIntegrated") -> str:
    """
    保存实验配置到Markdown文件

    Args:
        args: 命令行参数对象
        experiment_dir: 实验目录路径
        model_name: 模型名称

    Returns:
        str: 配置文件路径
    """
    # 确保目录存在
    os.makedirs(experiment_dir, exist_ok=True)

    # 配置文件路径
    config_file = os.path.join(experiment_dir, "EXPERIMENT_CONFIG.md")

    # 收集所有参数
    config_dict = {}
    for key, value in vars(args).items():
        if key.startswith('_'):
            continue  # 跳过私有属性
        config_dict[key] = value

    # 生成Markdown内容
    md_content = f"""# 实验配置

**实验名称**: {os.path.basename(experiment_dir)}
**创建时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**模型**: {model_name}

## 训练参数

| 参数 | 值 |
|------|-----|
| 训练轮数 (epochs) | {args.epochs} |
| 批次大小 (batch_size) | {args.batch_size} |
| 学习率 (learning_rate) | {args.learning_rate} |
| 数据加载进程 (num_workers) | {args.num_workers} |

## 模型参数

| 参数 | 值 |
|------|-----|
| 注意力头数 (attn_heads) | {args.attn_heads} |
| 注意力层数 (attn_layers) | {args.attn_layers} |
| 体素大小 (voxel_size) | {args.voxel_size} |
| 裁剪尺寸 (crop_size) | {args.crop_size} |

## 数据参数

| 参数 | 值 |
|------|-----|
| 数据根目录 (data_root) | {args.data_root} |
| 序列长度 (sequence_length) | {args.sequence_length} |
| 最大序列数 (max_sequences) | {args.max_sequences} |

## 显存管理参数

| 参数 | 值 |
|------|-----|
| 显存清理频率 (cleanup_freq) | {args.cleanup_freq} |
| 显存阈值 (memory_threshold) | {args.memory_threshold} GB |
| 梯度累积步数 (accumulation_steps) | {args.accumulation_steps} |

## 运行模式

| 模式 | 状态 |
|------|------|
| 干运行 (dry_run) | {args.dry_run} |
| 仅测试 (test_only) | {args.test_only} |
| 调试模式 (debug) | {args.debug} |

## 设备参数

| 参数 | 值 |
|------|-----|
| 禁用CUDA (no_cuda) | {args.no_cuda} |
| 设备 (device) | {args.device} |
| 多GPU (multi_gpu) | {args.multi_gpu} |
| GPU IDs | {args.gpu_ids if args.gpu_ids else 'None'} |

## 显存分析

| 参数 | 值 |
|------|-----|
| 启用显存分析 (enable_memory_profile) | {args.enable_memory_profile} |
| 显存分析输出 (memory_profile_output) | {args.memory_profile_output} |

## Rerun可视化

| 参数 | 值 |
|------|-----|
| 启用Rerun可视化 (enable_rerun_viz) | {args.enable_rerun_viz} |
| 可视化目录 (rerun_viz_dir) | {args.rerun_viz_dir} |
| 可视化频率 (rerun_viz_freq) | {args.rerun_viz_freq} |

## 完整配置

```json
{json.dumps(config_dict, indent=2, ensure_ascii=False)}
```

## 实验目录结构

```
{experiment_dir}/
├── EXPERIMENT_CONFIG.md    # 本文件
├── training.rrd            # Rerun可视化数据（如果启用）
├── checkpoints/            # 模型检查点
│   ├── stream_model_epoch_5.pth
│   ├── stream_model_epoch_10.pth
│   └── ...
└── logs/                   # 训练日志
    └── stream_training.log
```

## 使用方法

### 训练命令

```bash
python train_stream_integrated.py \\
    --enable-rerun-viz \\
    --data-root {args.data_root} \\
    --batch-size {args.batch_size} \\
    --learning-rate {args.learning_rate} \\
    --epochs {args.epochs} \\
    --sequence-length {args.sequence_length}
```

### 查看可视化

```bash
rerun {experiment_dir}/training.rrd
```

---

*此文件由实验配置管理模块自动生成*
"""

    # 保存到文件
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"[ExperimentConfig] ✅ 配置已保存到: {config_file}")

    return config_file


def create_experiment_directory(base_dir: str, args: Any, model_name: str = "StreamSDFFormerIntegrated") -> Dict[str, str]:
    """
    创建实验目录并保存配置

    Args:
        base_dir: 基础目录（如'experiments'或'viz'）
        args: 命令行参数对象
        model_name: 模型名称

    Returns:
        dict: 包含实验相关路径的字典
        {
            'experiment_name': 实验名称,
            'experiment_dir': 实验目录路径,
            'config_file': 配置文件路径,
            'rrd_file': RRD文件路径,
            'checkpoint_dir': 检查点目录路径
        }
    """
    # 生成实验名
    experiment_name = generate_experiment_name(args)

    # 创建实验目录
    experiment_dir = os.path.join(base_dir, experiment_name)

    # 保存配置
    config_file = save_experiment_config(args, experiment_dir, model_name)

    # 定义其他路径
    rrd_file = os.path.join(experiment_dir, "training.rrd")
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    return {
        'experiment_name': experiment_name,
        'experiment_dir': experiment_dir,
        'config_file': config_file,
        'rrd_file': rrd_file,
        'checkpoint_dir': checkpoint_dir
    }
