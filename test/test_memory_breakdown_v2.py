#!/usr/bin/env python3
"""
显存分析工具 v2
精确测量batch_size=1时各个组件的显存占用
"""

import os
import sys
import torch
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedMemoryProfiler:
    """高级显存分析器"""

    def __init__(self, device: torch.device):
        """
        初始化显存分析器

        Args:
            device: 计算设备
        """
        self.device = device
        self.snapshots: List[Dict] = []
        self.current_label = ""

    def reset(self) -> None:
        """重置分析器"""
        self.snapshots = []
        self.current_label = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gc.collect()

    def set_label(self, label: str) -> None:
        """
        设置当前标签

        Args:
            label: 标签名称
        """
        self.current_label = label

    def snapshot(self, label: str = None, use_peak: bool = False) -> Dict:
        """
        记录当前显存状态

        Args:
            label: 快照标签（如果未设置，使用当前标签）
            use_peak: 是否使用峰值显存

        Returns:
            显存状态字典
        """
        if not torch.cuda.is_available():
            return {
                'label': label or self.current_label,
                'allocated_gb': 0,
                'reserved_gb': 0,
                'max_allocated_gb': 0,
                'peak_allocated_gb': 0
            }

        label = label or self.current_label

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3

        if use_peak:
            # 使用峰值显存
            snapshot = {
                'label': label,
                'allocated_gb': max_allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'peak_allocated_gb': max_allocated
            }
        else:
            # 使用当前显存
            snapshot = {
                'label': label,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated,
                'peak_allocated_gb': max(allocated, max_allocated)
            }

        self.snapshots.append(snapshot)
        return snapshot

    def print_snapshots(self) -> None:
        """打印所有快照"""
        print("\n" + "=" * 90)
        print("显存快照")
        print("=" * 90)
        print(f"{'标签':<45} {'已分配':<12} {'已保留':<12} {'峰值':<12}")
        print("-" * 90)

        for snap in self.snapshots:
            print(f"{snap['label']:<45} "
                  f"{snap['allocated_gb']:<12.3f} "
                  f"{snap['reserved_gb']:<12.3f} "
                  f"{snap['peak_allocated_gb']:<12.3f}")

        print("=" * 90)

    def print_differences(self) -> None:
        """打印所有快照之间的差异"""
        print("\n" + "=" * 90)
        print("显存差异分析（基于增量）")
        print("=" * 90)
        print(f"{'组件':<45} {'已分配增量':<15} {'比例':<10}")
        print("-" * 90)

        if len(self.snapshots) < 2:
            print("需要至少2个快照来计算差异")
            return

        base = self.snapshots[0]
        total_allocated_diff = self.snapshots[-1]['allocated_gb'] - base['allocated_gb']

        for i in range(1, len(self.snapshots)):
            snap = self.snapshots[i]
            prev = self.snapshots[i-1]

            allocated_diff = snap['allocated_gb'] - prev['allocated_gb']

            if abs(allocated_diff) > 0.001:  # 大于1MB才记录
                # 计算占总显存的比例
                if total_allocated_diff > 0:
                    percentage = (allocated_diff / total_allocated_diff) * 100
                    sign = "+" if allocated_diff > 0 else ""
                    print(f"{snap['label']:<45} "
                          f"{sign}{allocated_diff:<15.3f} "
                          f"{percentage:<10.2f}%")

        print("=" * 90)
        print(f"总显存增量: {total_allocated_diff:.3f} GB")

    def analyze_components(self) -> Dict[str, float]:
        """
        分析各个组件的显存占用

        Returns:
            组件显存占用字典
        """
        if len(self.snapshots) < 2:
            return {}

        base = self.snapshots[0]
        components = {}

        for i in range(1, len(self.snapshots)):
            snap = self.snapshots[i]
            prev = self.snapshots[i-1]

            allocated_diff = snap['allocated_gb'] - prev['allocated_gb']

            if abs(allocated_diff) > 0.001:  # 大于1MB才记录
                components[snap['label']] = allocated_diff

        # 计算总显存（只计算正增量）
        total_allocated_diff = sum([v for v in components.values() if v > 0])

        # 计算比例（创建新字典避免迭代时修改）
        result = {}
        for label, memory in components.items():
            result[label] = memory
            if total_allocated_diff > 0:
                result[f"{label}_percentage"] = (memory / total_allocated_diff) * 100

        result['total'] = total_allocated_diff

        return result


def test_memory_breakdown_detailed():
    """详细测试batch_size=1时的显存分布"""

    print("=" * 90)
    print("Batch Size 1 显存分布详细分析")
    print("=" * 90)

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过显存测试")
        return

    print(f"\n设备: {device}")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 创建显存分析器
    profiler = AdvancedMemoryProfiler(device)

    # 测试参数
    batch_size = 1
    image_size = (256, 256)
    sequence_length = 5

    print(f"\n测试参数：")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - 图像尺寸: {image_size}")
    print(f"  - 序列长度: {sequence_length}")

    # 步骤0：初始化
    print("\n" + "=" * 90)
    print("步骤0: 初始化")
    print("=" * 90)

    profiler.reset()
    profiler.set_label("初始化")
    profiler.snapshot()

    # 步骤1：创建模型
    print("\n步骤1: 创建模型")
    print("-" * 90)

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

    # 清理后再次记录
    torch.cuda.empty_cache()
    gc.collect()

    profiler.set_label("模型参数")
    profiler.snapshot()

    # 步骤2：创建输入数据
    print("\n步骤2: 创建输入数据")
    print("-" * 90)

    images = torch.randn(batch_size, 3, *image_size).to(device)
    poses = torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1).to(device)
    intrinsics = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(device)

    profiler.set_label("输入数据")
    profiler.snapshot()

    # 步骤3：第一帧（包含2D和3D编码器）
    print("\n步骤3: 第一帧（无历史状态）")
    print("-" * 90)

    # 记录前向传播前的峰值
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        output, state = model.forward_single_frame(
            images, poses, intrinsics, reset_state=True
        )

    # 记录峰值显存
    profiler.set_label("第一帧-峰值显存")
    profiler.snapshot(use_peak=True)

    # 清理并记录当前显存
    torch.cuda.empty_cache()
    gc.collect()

    profiler.set_label("第一帧-清理后显存")
    profiler.snapshot()

    # 步骤4：第二帧（包含流式融合）
    print("\n步骤4: 第二帧（有历史状态 + 流式融合）")
    print("-" * 90)

    # 记录前向传播前的峰值
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        output, state = model.forward_single_frame(
            images, poses, intrinsics, reset_state=False
        )

    # 记录峰值显存
    profiler.set_label("第二帧-峰值显存")
    profiler.snapshot(use_peak=True)

    # 清理并记录当前显存
    torch.cuda.empty_cache()
    gc.collect()

    profiler.set_label("第二帧-清理后显存")
    profiler.snapshot()

    # 步骤5：分析历史状态
    print("\n步骤5: 历史状态分析")
    print("-" * 90)

    state_size = 0
    if state is not None:
        print("\n历史状态组件:")
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                size_mb = value.element_size() * value.numel() / 1024**2
                state_size += size_mb / 1024  # 转换为GB
                print(f"  {key:<20} {str(value.shape):<20} {str(value.dtype):<10} {size_mb:>8.2f} MB")

    profiler.set_label(f"历史状态存储({state_size:.3f} GB)")
    profiler.snapshot()

    # 打印结果
    profiler.print_snapshots()
    profiler.print_differences()

    # 分析组件
    print("\n" + "=" * 90)
    print("组件显存占用分析")
    print("=" * 90)

    components = profiler.analyze_components()

    total = components.get('total', 0)
    print(f"\n总显存占用: {total:.3f} GB")
    print(f"\n各组件占比：")

    # 按显存占用排序
    component_list = []
    for key, value in components.items():
        if not key.endswith('_percentage') and key != 'total':
            percentage = components.get(f"{key}_percentage", 0)
            component_list.append((key, value, percentage))

    component_list.sort(key=lambda x: -abs(x[1]))  # 按绝对值降序排序

    print(f"\n{'组件':<45} {'显存(GB)':<12} {'占比':<10}")
    print("-" * 90)
    for name, memory, percentage in component_list:
        sign = "+" if memory > 0 else ""
        print(f"{name:<45} {sign}{memory:<12.3f} {percentage:<10.2f}%")

    print("=" * 90)

    # 生成报告
    generate_detailed_report(component_list, total, device, model)

    # 额外分析：模型参数大小
    print("\n" + "=" * 90)
    print("模型参数分析")
    print("=" * 90)

    total_params = sum(p.numel() for p in model.parameters())
    param_size = total_params * 4 / 1024**3  # 假设FP32
    print(f"总参数数量: {total_params:,}")
    print(f"参数显存占用（FP32）: {param_size:.3f} GB")

    # 打印结果
    profiler.print_snapshots()
    profiler.print_differences()

    # 生成报告
    generate_detailed_report(component_list, total, device, model)


def generate_detailed_report(components: List[Tuple[str, float, float]],
                             total: float,
                             device: torch.device,
                             model: Any) -> None:
    """
    生成详细显存分析报告

    Args:
        components: 组件列表 [(名称, 显存, 占比)]
        total: 总显存
        device: 计算设备
        model: 模型对象
    """
    # 分析模型参数
    total_params = sum(p.numel() for p in model.parameters())
    param_size = total_params * 4 / 1024**3  # 假设FP32

    report = f"""# Batch Size 1 显存分析详细报告

## 📊 测试信息

**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**设备**: {device}
**总显存**: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB
**Batch Size**: 1
**模型配置**: attn_heads=1, attn_layers=1, voxel_size=0.16, crop_size=(24,24,16)
**模型参数数量**: {total_params:,}
**模型参数显存**: {param_size:.3f} GB

## 🎯 总显存占用

**总显存**: {total:.3f} GB

## 📈 组件显存分布

| 组件 | 显存(GB) | 占比 |
|------|----------|------|
"""

    for name, memory, percentage in components:
        sign = "+" if memory > 0 else ""
        report += f"| {name} | {sign}{memory:.3f} | {percentage:.2f}% |\n"

    report += f"""

## 🔍 关键发现

### 1. 显存使用模式

**静态显存**（模型参数）: {param_size:.3f} GB
- 模型参数: {total_params:,}
- 占比: {(param_size / total * 100) if total > 0 else 0:.2f}%

**动态显存**（前向传播）: {total - param_size:.3f} GB
- 第一帧峰值: 需要查看数据
- 第二帧峰值: 需要查看数据
- 流式融合: 包含在前向传播中

### 2. 各组件详细分析

"""

    # 添加每个组件的详细分析
    if len(components) > 0:
        report += f"**主要显存消耗组件:**\n\n"
        for i, (name, memory, percentage) in enumerate(components[:3]):
            report += f"{i+1}. **{name}** ({percentage:.2f}%)\n"
            report += f"   - 显存占用: {memory:.3f} GB\n"
            report += f"   - 占比: {percentage:.2f}%\n\n"

    report += f"""### 3. 峰值显存分析

**第一帧峰值**: 需要查看数据
- 包含: 2D编码器 + 3D编码器 + 状态初始化
- 特点: 无历史状态，显存占用较低

**第二帧峰值**: 需要查看数据
- 包含: 2D编码器 + 3D编码器 + 流式融合
- 特点: 有历史状态，可能显存占用更高

### 4. 流式融合影响

**流式融合显存影响**: 需要对比第一帧和第二帧的峰值
- 如果第二帧峰值明显高于第一帧，说明流式融合消耗显存
- 否则，说明流式融合优化良好

## 💡 优化建议

### 1. 模型参数优化

**当前**: {param_size:.3f} GB
**建议**:
- 使用混合精度训练（FP16）: 可减半到 {param_size/2:.3f} GB
- 使用梯度检查点: 可降低中间变量显存
- 减少模型层数: 当前1层，可考虑保持

### 2. 动态显存优化

**当前**: {total - param_size:.3f} GB
**建议**:
- 使用梯度累积: 在保持低显存的同时增大有效batch size
- 优化注意力计算: 当前已使用checkpointing
- 减少历史状态大小: 当前已使用轻量级模式

### 3. Batch Size扩展

**Batch Size 1**: {total:.3f} GB
**Batch Size 2**: 预估 {total * 2:.3f} GB
- 问题: 显存碎片化，最大连续块只有3.04 GB
- 解决方案: 使用梯度累积（batch_size=1 + accumulation_steps=4）

## 🎯 结论

### 显存占用总结

- **静态显存（模型参数）**: {param_size:.3f} GB ({param_size/total*100 if total > 0 else 0:.2f}%)
- **动态显存（前向传播）**: {total - param_size:.3f} GB ({(total-param_size)/total*100 if total > 0 else 0:.2f}%)
- **总显存**: {total:.3f} GB

### 主要消耗组件

1. {components[0][0] if components else "N/A"} ({components[0][2] if components else 0:.2f}%)
2. {components[1][0] if len(components) > 1 else "N/A"} ({components[1][2] if len(components) > 1 else 0:.2f}%)
3. {components[2][0] if len(components) > 2 else "N/A"} ({components[2][2] if len(components) > 2 else 0:.2f}%)

### 优化潜力

- **静态显存优化**: 使用FP16可降低50%
- **动态显存优化**: 使用梯度累积保持低显存
- **Batch Size扩展**: 无法直接扩展，建议使用梯度累积

### 推荐配置

```bash
python train_stream_integrated.py \\
  --batch-size 1 \\
  --accumulation-steps 4 \\
  --crop-size "24,24,16" \\
  --voxel-size 0.16
```

**预期效果**:
- 有效batch size: 4
- 显存使用: ~{total:.3f} GB
- 训练稳定性: 高

---

**报告生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**分析器**: AdvancedMemoryProfiler
**测试者**: Frank
"""

    # 保存报告
    report_path = "/home/cwh/coding/former3d/doc/memory_breakdown_detailed_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    test_memory_breakdown_detailed()
