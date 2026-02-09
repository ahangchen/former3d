#!/usr/bin/env python3
"""
长期训练显存监控工具
监控batch_size=2长期训练过程中的显存累积效应
"""

import os
import sys
import torch
import time
import logging
from typing import Dict, List, Tuple
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LongTermMemoryMonitor:
    """长期训练显存监控器"""

    def __init__(self, device: torch.device, num_steps: int = 100):
        """
        初始化监控器

        Args:
            device: 计算设备
            num_steps: 监控步数
        """
        self.device = device
        self.num_steps = num_steps
        self.snapshots: List[Dict] = []
        self.state_snapshots: List[Dict] = []

    def monitor_training(self, batch_size: int) -> Dict:
        """
        监控长期训练的显存使用

        Args:
            batch_size: 批次大小

        Returns:
            监控结果字典
        """
        print(f"\n{'='*80}")
        print(f"长期训练显存监控 - Batch Size = {batch_size}")
        print(f"{'='*80}")

        # 清理显存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect()

        # 使用类属性device
        device = self.device

        # 创建模型
        model = StreamSDFFormerIntegrated(
            attn_heads=1,
            attn_layers=1,
            use_proj_occ=False,
            voxel_size=0.16,
            fusion_local_radius=2.0,
            crop_size=(24, 24, 16),
            use_checkpoint=False
        ).to(device)

        model.eval()
        model.enable_lightweight_state(True)

        # 记录模型参数显存
        model_params_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        print(f"模型参数显存: {model_params_memory:.3f} GB")

        # 创建输入数据
        image_size = (256, 256)
        images = torch.randn(batch_size, 3, *image_size).to(device)
        poses = torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1).to(device)
        intrinsics = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(device)

        # 监控训练过程
        results = {
            'batch_size': batch_size,
            'model_params_memory': model_params_memory,
            'steps': [],
            'state_tracker': {}
        }

        print(f"\n开始监控 {self.num_steps} 步训练...")

        for step in range(self.num_steps):
            # 重置峰值显存统计
            torch.cuda.reset_peak_memory_stats()

            # 前向传播
            with torch.no_grad():
                output, state = model.forward_single_frame(
                    images, poses, intrinsics,
                    reset_state=(step % 10 == 0)  # 每10步重置一次状态
                )

            # 记录峰值显存
            peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
            peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

            # 记录当前显存（清理前）
            current_allocated_before = torch.cuda.memory_allocated() / 1024**3
            current_reserved_before = torch.cuda.memory_reserved() / 1024**3

            # 分析历史状态
            state_info = {}
            if state is not None:
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        size_gb = value.element_size() * value.numel() / 1024**3
                        state_info[key] = {
                            'shape': str(value.shape),
                            'size_gb': size_gb,
                            'size_mb': size_gb * 1024
                        }

            # 记录每10步的状态
            if step % 10 == 0:
                results['state_tracker'][step] = state_info

            # 打印进度（每10步）
            if (step + 1) % 10 == 0:
                print(f"  步骤 {step+1}/{self.num_steps}: "
                      f"峰值 {peak_allocated:.3f} GB, "
                      f"当前 {current_allocated_before:.3f} GB")

            # 记录快照
            snapshot = {
                'step': step,
                'peak_allocated_gb': peak_allocated,
                'peak_reserved_gb': peak_reserved,
                'current_allocated_gb_before': current_allocated_before,
                'current_reserved_gb_before': current_reserved_before,
                'state_size_gb': sum([info['size_gb'] for info in state_info.values()]),
                'state_num_voxels': sum([len([v for v in state.values() if isinstance(v, torch.Tensor) and 'features' in key])]),
            }

            results['steps'].append(snapshot)

            # 尝试清理（但不强制）
            torch.cuda.empty_cache()

            # 记录清理后的显存
            current_allocated_after = torch.cuda.memory_allocated() / 1024**3
            current_reserved_after = torch.cuda.memory_reserved() / 1024**3

            snapshot['current_allocated_gb_after'] = current_allocated_after
            snapshot['current_reserved_gb_after'] = current_reserved_after
            snapshot['freed_gb'] = current_allocated_before - current_allocated_after

            self.snapshots.append(snapshot)

        # 清理模型
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return results

    def analyze_trend(self, results: Dict) -> None:
        """
        分析显存趋势

        Args:
            results: 监控结果
        """
        print(f"\n{'='*80}")
        print("显存趋势分析")
        print(f"{'='*80}")

        steps = results['steps']

        # 计算统计数据
        peak_allocations = [s['peak_allocated_gb'] for s in steps]
        current_allocations_before = [s['current_allocated_gb_before'] for s in steps]
        state_sizes = [s['state_size_gb'] for s in steps]

        print(f"\n峰值显存统计:")
        print(f"  最小值: {min(peak_allocations):.3f} GB")
        print(f"  最大值: {max(peak_allocations):.3f} GB")
        print(f"  平均值: {sum(peak_allocations)/len(peak_allocations):.3f} GB")
        print(f"  标准差: {(sum((x - sum(peak_allocations)/len(peak_allocations))**2 for x in peak_allocations)/len(peak_allocations))**0.5:.3f} GB")

        print(f"\n当前显存统计（清理前）:")
        print(f"  最小值: {min(current_allocations_before):.3f} GB")
        print(f"  最大值: {max(current_allocations_before):.3f} GB")
        print(f"  平均值: {sum(current_allocations_before)/len(current_allocations_before):.3f} GB")

        print(f"\n历史状态大小统计:")
        print(f"  最小值: {min(state_sizes):.6f} GB ({min(state_sizes)*1024:.3f} MB)")
        print(f"  最大值: {max(state_sizes):.6f} GB ({max(state_sizes)*1024:.3f} MB)")
        print(f"  平均值: {sum(state_sizes)/len(state_sizes):.6f} GB ({sum(state_sizes)/len(state_sizes)*1024:.3f} MB)")

        # 检查显存累积
        print(f"\n显存累积检查:")

        # 检查是否有显存泄漏（当前显存持续增长）
        first_10_avg = sum(current_allocations_before[:10]) / 10
        last_10_avg = sum(current_allocations_before[-10:]) / 10

        print(f"  前10步平均: {first_10_avg:.3f} GB")
        print(f"  后10步平均: {last_10_avg:.3f} GB")
        print(f"  增长: {last_10_avg - first_10_avg:.3f} GB ({(last_10_avg/first_10_avg - 1)*100:.2f}%)")

        if last_10_avg > first_10_avg * 1.1:  # 超过10%增长
            print(f"  ⚠️  警告：检测到显存泄漏（增长{(last_10_avg/first_10_avg - 1)*100:.2f}%）")
        else:
            print(f"  ✅ 正常：显存无明显增长")

        # 检查是否有异常的峰值
        threshold = sum(peak_allocations) / len(peak_allocations) + 3 * (sum((x - sum(peak_allocations)/len(peak_allocations))**2 for x in peak_allocations)/len(peak_allocations))**0.5
        anomalies = [i for i, x in enumerate(peak_allocations) if x > threshold]

        if anomalies:
            print(f"  ⚠️  警告：检测到{len(anomalies)}个异常峰值")
            print(f"  异常步骤: {anomalies}")
            for i in anomalies[:5]:  # 只显示前5个
                print(f"    步骤{steps[i]['step']}: {steps[i]['peak_allocated_gb']:.3f} GB")
        else:
            print(f"  ✅ 正常：无异常峰值")

        # 分析历史状态
        print(f"\n历史状态分析:")
        print(f"  状态追踪步数: {len(results['state_tracker'])}")

        for step in sorted(results['state_tracker'].keys()):
            state_info = results['state_tracker'][step]
            if 'features' in state_info:
                features_size = state_info['features']['size_mb']
                print(f"  步骤{step}: features = {features_size:.2f} MB")

    def generate_report(self, results: Dict) -> None:
        """
        生成监控报告

        Args:
            results: 监控结果
        """
        steps = results['steps']

        peak_allocations = [s['peak_allocated_gb'] for s in steps]
        current_allocations = [s['current_allocated_gb_before'] for s in steps]
        state_sizes = [s['state_size_gb'] for s in steps]

        # 计算统计数据
        peak_min = min(peak_allocations)
        peak_max = max(peak_allocations)
        peak_avg = sum(peak_allocations) / len(peak_allocations)
        peak_std = (sum((x - peak_avg)**2 for x in peak_allocations) / len(peak_allocations))**0.5

        current_min = min(current_allocations)
        current_max = max(current_allocations)
        current_avg = sum(current_allocations) / len(current_allocations)

        state_min = min(state_sizes)
        state_max = max(state_sizes)
        state_avg = sum(state_sizes) / len(state_sizes)

        first_10_avg = sum(current_allocations[:10]) / 10
        last_10_avg = sum(current_allocations[-10:]) / 10
        growth_percent = (last_10_avg / first_10_avg - 1) * 100

        report = f"""# Batch Size {results['batch_size']} 长期训练显存监控报告

## 📊 监控信息

**监控时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**设备**: {self.device}
**总显存**: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB
**Batch Size**: {results['batch_size']}
**监控步数**: {self.num_steps}
**模型参数显存**: {results['model_params_memory']:.3f} GB

## 🎯 显存使用统计

### 峰值显存

- **最小值**: {peak_min:.3f} GB
- **最大值**: {peak_max:.3f} GB
- **平均值**: {peak_avg:.3f} GB
- **标准差**: {peak_std:.3f} GB
- **波动范围**: {peak_max - peak_min:.3f} GB

### 当前显存（清理前）

- **最小值**: {current_min:.3f} GB
- **最大值**: {current_max:.3f} GB
- **平均值**: {current_avg:.3f} GB
- **波动范围**: {current_max - current_min:.3f} GB

### 历史状态大小

- **最小值**: {state_min:.6f} GB ({state_min*1024:.3f} MB)
- **最大值**: {state_max:.6f} GB ({state_max*1024:.3f} MB)
- **平均值**: {state_avg:.6f} GB ({state_avg*1024:.3f} MB)
- **波动范围**: {(state_max - state_min)*1024:.3f} MB

## 📈 显存趋势分析

### 显存累积检查

- **前10步平均**: {first_10_avg:.3f} GB
- **后10步平均**: {last_10_avg:.3f} GB
- **总增长**: {last_10_avg - first_10_avg:.3f} GB
- **增长率**: {growth_percent:.2f}%

### 显存累积评估

"""

        if growth_percent > 10:
            report += f"⚠️  **警告**：检测到显存泄漏（增长{growth_percent:.2f}%）\n\n"
            report += f"**可能原因**：\n"
            report += f"- 历史状态未正确释放\n"
            report += f"- 中间变量累积\n"
            report += f"- CUDA缓存未清理\n\n"
        elif growth_percent > 5:
            report += f"⚠️  **注意**：显存有轻微增长（{growth_percent:.2f}%）\n\n"
            report += f"**可能原因**：\n"
            report += f"- 正常的显存波动\n"
            report += f"- 某些状态需要保持\n\n"
        else:
            report += f"✅ **正常**：显存无明显累积（{growth_percent:.2f}%）\n\n"

        report += f"""## 🔍 组件显存占比分析

基于最大峰值显存 {peak_max:.3f} GB：

| 组件 | 显存(GB) | 占比 |
|------|----------|------|
| 模型参数 | {results['model_params_memory']:.3f} | {results['model_params_memory']/peak_max*100:.2f}% |
| 峰值动态显存 | {peak_max - results['model_params_memory']:.3f} | {(peak_max - results['model_params_memory'])/peak_max*100:.2f}% |
| 当前动态显存（平均）| {current_avg - results['model_params_memory']:.3f} | {(current_avg - results['model_params_memory'])/peak_max*100:.2f}% |

## 💡 关键发现

### 1. 显存稳定性

"""

        if peak_std / peak_avg < 0.1:
            report += f"- **显存波动小**（标准差/平均值 = {peak_std/peak_avg:.2f}）\n"
            report += f"- 训练过程显存使用稳定\n\n"
        elif peak_std / peak_avg < 0.3:
            report += f"- **显存波动中等**（标准差/平均值 = {peak_std/peak_avg:.2f}）\n"
            report += f"- 训练过程显存使用有一定波动\n\n"
        else:
            report += f"- **显存波动大**（标准差/平均值 = {peak_std/peak_avg:.2f}）\n"
            report += f"- 训练过程显存使用波动较大\n\n"

        report += f"""### 2. 历史状态影响

- **历史状态大小**: {state_avg*1024:.3f} MB（平均）
- **占峰值显存比例**: {state_avg/peak_max*100:.2f}%
- **波动范围**: {(state_max - state_min)*1024:.3f} MB

"""

        if state_max / state_avg > 3:
            report += f"⚠️  **警告**：历史状态大小波动很大（{state_max/state_avg:.2f}倍）\n\n"
        else:
            report += f"✅ **正常**：历史状态大小波动适中\n\n"

        report += f"""### 3. 峰值vs当前显存

- **峰值显存**: {peak_max:.3f} GB
- **当前显存（平均）**: {current_avg:.3f} GB
- **峰值/当前比例**: {peak_max/current_avg:.2f}x

**说明**：峰值/当前比例表示训练过程中的显存峰值相对于平均使用水平的倍数。比例越大，说明显存需求波动越大。

## 🚀 达到9 GB显存占用的预估

基于当前监控结果：

### 线性外推

假设显存使用随batch_size线性增长：

- **当前峰值**: {peak_max:.3f} GB (batch_size={results['batch_size']})
- **batch_size=4 预估**: {peak_max * 4:.3f} GB
- **batch_size=8 预估**: {peak_max * 8:.3f} GB
- **batch_size=16 预估**: {peak_max * 16:.3f} GB

### 达到9 GB所需的batch_size

- **线性模型**: {9.0 / peak_max:.2f}
- **超线性模型（假设2.5x增长）**: {(9.0 / peak_max)**0.5:.2f}

**结论**：基于{results['batch_size']}的测量结果，达到9 GB显存大约需要batch_size为{9.0 / peak_max:.2f}（假设线性增长）。

## 🎯 结论

### 显存使用总结

- **模型参数**: {results['model_params_memory']:.3f} GB ({results['model_params_memory']/peak_max*100:.2f}%)
- **峰值动态显存**: {peak_max - results['model_params_memory']:.3f} GB ({(peak_max - results['model_params_memory'])/peak_max*100:.2f}%)
- **总峰值显存**: {peak_max:.3f} GB (100%)

### 显存累积评估

"""

        if growth_percent > 10:
            report += f"⚠️  **存在显存泄漏**（{growth_percent:.2f}%增长）\n"
            report += f"- 需要检查历史状态释放机制\n"
            report += f"- 建议增加显存清理频率\n\n"
        else:
            report += f"✅ **无显存泄漏**（{growth_percent:.2f}%增长）\n"
            report += f"- 显存使用稳定\n"
            report += f"- 无需额外优化\n\n"

        report += f"""### 训练稳定性

"""

        if peak_std / peak_avg < 0.1:
            report += f"✅ **训练稳定**（显存波动小）\n"
        elif peak_std / peak_avg < 0.3:
            report += f"⚠️  **训练较稳定**（显存波动中等）\n"
        else:
            report += f"⚠️  **训练不稳定**（显存波动大）\n"

        report += f"""

### 优化建议

1. **如果显存稳定且无泄漏**：
   - 可以考虑增大batch_size
   - 可以考虑增大模型配置

2. **如果存在显存泄漏**：
   - 检查历史状态释放机制
   - 增加显存清理频率
   - 使用更激进的LRU策略

3. **如果显存波动大**：
   - 使用梯度累积稳定显存使用
   - 优化体素采样策略
   - 实施分块注意力计算

---

**报告生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**监控器**: LongTermMemoryMonitor
**测试者**: Frank
"""

        # 保存报告
        report_path = f"/home/cwh/coding/former3d/doc/long_term_memory_monitor_batch_{results['batch_size']}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n报告已保存到: {report_path}")


def main():
    """主函数"""
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过显存测试")
        return

    print(f"设备: {device}")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 创建长期监控器
    monitor = LongTermMemoryMonitor(device, num_steps=100)

    # 监控batch_size=2的长期训练
    results = monitor.monitor_training(batch_size=2)

    # 分析趋势
    monitor.analyze_trend(results)

    # 生成报告
    monitor.generate_report(results)


if __name__ == "__main__":
    main()
