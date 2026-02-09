#!/usr/bin/env python3
"""
显存分析工具
精确测量batch_size=1时各个组件的显存占用
"""

import os
import sys
import torch
import time
import logging
from typing import Dict, List, Tuple, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
from memory_manager import MemoryManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryProfiler:
    """显存分析器"""

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
            torch.cuda.reset_peak_memory_stats()

    def set_label(self, label: str) -> None:
        """
        设置当前标签

        Args:
            label: 标签名称
        """
        self.current_label = label

    def snapshot(self, label: str = None) -> Dict:
        """
        记录当前显存状态

        Args:
            label: 快照标签（如果未设置，使用当前标签）

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

        snapshot = {
            'label': label,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
            'peak_allocated_gb': max(allocated, max_allocated)
        }

        self.snapshots.append(snapshot)
        return snapshot

    def get_memory_difference(self, snapshot1: Dict, snapshot2: Dict) -> Dict:
        """
        计算两个快照之间的显存差异

        Args:
            snapshot1: 第一个快照
            snapshot2: 第二个快照

        Returns:
            显存差异字典
        """
        return {
            'label': f"{snapshot1['label']} -> {snapshot2['label']}",
            'allocated_diff_gb': snapshot2['allocated_gb'] - snapshot1['allocated_gb'],
            'reserved_diff_gb': snapshot2['reserved_gb'] - snapshot1['reserved_gb'],
        }

    def print_snapshots(self) -> None:
        """打印所有快照"""
        print("\n" + "=" * 80)
        print("显存快照")
        print("=" * 80)
        print(f"{'标签':<40} {'已分配':<12} {'已保留':<12} {'峰值':<12}")
        print("-" * 80)

        for i, snap in enumerate(self.snapshots):
            print(f"{snap['label']:<40} "
                  f"{snap['allocated_gb']:<12.3f} "
                  f"{snap['reserved_gb']:<12.3f} "
                  f"{snap['peak_allocated_gb']:<12.3f}")

        print("=" * 80)

    def print_differences(self) -> None:
        """打印所有快照之间的差异"""
        print("\n" + "=" * 80)
        print("显存差异分析")
        print("=" * 80)
        print(f"{'组件':<40} {'已分配增量':<15} {'比例':<10}")
        print("-" * 80)

        if len(self.snapshots) < 2:
            print("需要至少2个快照来计算差异")
            return

        base = self.snapshots[0]
        total_allocated_diff = self.snapshots[-1]['allocated_gb'] - base['allocated_gb']

        for i in range(1, len(self.snapshots)):
            snap = self.snapshots[i]
            prev = self.snapshots[i-1]

            allocated_diff = snap['allocated_gb'] - prev['allocated_gb']
            reserved_diff = snap['reserved_gb'] - prev['reserved_gb']

            if allocated_diff > 0.001:  # 忽略小于1MB的差异
                # 计算占总显存的比例
                if total_allocated_diff > 0:
                    percentage = (allocated_diff / total_allocated_diff) * 100
                    print(f"{snap['label']:<40} "
                          f"{allocated_diff:<15.3f} "
                          f"{percentage:<10.2f}%")

        print("=" * 80)
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

            if allocated_diff > 0.001:  # 忽略小于1MB的差异
                components[snap['label']] = allocated_diff

        # 计算总显存
        total_allocated_diff = sum(components.values())

        # 计算比例（创建新字典避免迭代时修改）
        result = {}
        for label, memory in components.items():
            result[label] = memory
            result[f"{label}_percentage"] = (memory / total_allocated_diff) * 100 if total_allocated_diff > 0 else 0

        result['total'] = total_allocated_diff

        return result


def test_memory_breakdown():
    """测试batch_size=1时的显存分布"""

    print("=" * 80)
    print("Batch Size 1 显存分布分析")
    print("=" * 80)

    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过显存测试")
        return

    print(f"\n设备: {device}")
    print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 创建显存分析器
    profiler = MemoryProfiler(device)

    # 测试参数
    batch_size = 1
    image_size = (256, 256)
    sequence_length = 5

    print(f"\n测试参数：")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - 图像尺寸: {image_size}")
    print(f"  - 序列长度: {sequence_length}")

    # 步骤1：初始化
    print("\n" + "=" * 80)
    print("步骤1: 初始化")
    print("=" * 80)

    torch.cuda.empty_cache()
    profiler.reset()
    profiler.set_label("初始化")
    profiler.snapshot()

    # 步骤2：创建模型
    print("\n步骤2: 创建模型")
    print("-" * 80)

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

    profiler.set_label("模型创建")
    profiler.snapshot()

    # 步骤3：创建输入数据
    print("\n步骤3: 创建输入数据")
    print("-" * 80)

    images = torch.randn(batch_size, 3, *image_size).to(device)
    poses = torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1).to(device)
    intrinsics = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(device)

    profiler.set_label("输入数据")
    profiler.snapshot()

    # 步骤4：第一帧（包含2D和3D编码器）
    print("\n步骤4: 第一帧（包含2D和3D编码器）")
    print("-" * 80)

    with torch.no_grad():
        output, state = model.forward_single_frame(
            images, poses, intrinsics, reset_state=True
        )

    profiler.set_label("第一帧（2D+3D编码器）")
    profiler.snapshot()

    # 步骤6：第二帧
    print("\n步骤6: 第二帧（有历史）")
    print("-" * 80)

    with torch.no_grad():
        output, state = model.forward_single_frame(
            images, poses, intrinsics, reset_state=False
        )

    profiler.set_label("3D编码器（第2帧）")
    profiler.snapshot()

    # 步骤7：流式融合
    print("\n步骤7: 流式融合")
    print("-" * 80)

    # 流式融合已经在forward_single_frame中自动调用
    # 这里只是标记一下
    profiler.set_label("流式融合（包含在第2帧中）")
    profiler.snapshot()

    # 步骤8：历史状态
    print("\n步骤8: 历史状态存储")
    print("-" * 80)

    # 分析历史状态的大小
    state_size = 0
    if state is not None:
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state_size += value.element_size() * value.numel() / 1024**3
                print(f"  {key}: {value.shape}, {value.dtype} = {value.element_size() * value.numel() / 1024**2:.2f} MB")

    profiler.set_label("历史状态（~{:.3f} GB）".format(state_size))
    profiler.snapshot()

    # 打印结果
    profiler.print_snapshots()
    profiler.print_differences()

    # 分析组件
    print("\n" + "=" * 80)
    print("组件显存占用分析")
    print("=" * 80)

    components = profiler.analyze_components()

    total = components.get('total', 0)
    print(f"\n总显存占用: {total:.3f} GB")
    print(f"\n各组件占比：")

    # 按显存占用排序
    component_list = []
    for key, value in components.items():
        if not key.endswith('_percentage') and key != 'total':
            component_list.append((key, value, components.get(f"{key}_percentage", 0)))

    component_list.sort(key=lambda x: -x[1])  # 按显存占用降序排序

    print(f"\n{'组件':<40} {'显存(GB)':<12} {'占比':<10}")
    print("-" * 80)
    for name, memory, percentage in component_list:
        print(f"{name:<40} {memory:<12.3f} {percentage:<10.2f}%")

    print("=" * 80)

    # 生成报告
    generate_report(component_list, total, device)


def generate_report(components: List[Tuple[str, float, float]], total: float, device: torch.device) -> None:
    """
    生成显存分析报告

    Args:
        components: 组件列表 [(名称, 显存, 占比)]
        total: 总显存
        device: 计算设备
    """
    report = f"""# Batch Size 1 显存分析报告

## 📊 测试信息

**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**设备**: {device}
**Batch Size**: 1
**模型配置**: attn_heads=1, attn_layers=1, voxel_size=0.16, crop_size=(24,24,16)

## 🎯 总显存占用

**总显存**: {total:.3f} GB

## 📈 组件显存分布

| 组件 | 显存(GB) | 占比 |
|------|----------|------|
"""

    for name, memory, percentage in components:
        report += f"| {name} | {memory:.3f} | {percentage:.2f}% |\n"

    report += f"""

## 🔍 关键发现

### 1. 主要显存消耗组件

1. **{components[0][0]}** ({components[0][2]:.2f}%)
   - 显存占用: {components[0][1]:.3f} GB
   - 原因: 需要详细分析

"""

    if len(components) > 1:
        report += f"""2. **{components[1][0]}** ({components[1][2]:.2f}%)
   - 显存占用: {components[1][1]:.3f} GB

3. **{components[2][0]}** ({components[2][2]:.2f}%)
   - 显存占用: {components[2][1]:.3f} GB

"""

    report += f"""### 2. 显存使用模式

- **静态显存**: 需要测量
- **动态显存**: 需要测量
- **峰值显存**: 需要测量

### 3. 优化建议

1. **最大消耗组件优化**: {components[0][0]}
   - 建议降低占用
   - 优化方法待确定

2. **次大消耗组件优化**: {components[1][0] if len(components) > 1 else "N/A"}
   - 建议降低占用
   - 优化方法待确定

3. **整体优化**
   - 使用梯度累积
   - 减小模型配置
   - 优化注意力计算

## 🎯 结论

总显存占用: {total:.3f} GB
最大组件: {components[0][0]} ({components[0][2]:.2f}%)
优化潜力: 需要评估

---

**报告生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**分析器**: MemoryProfiler
"""

    # 保存报告
    report_path = "/home/cwh/coding/former3d/doc/memory_breakdown_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    import gc
    test_memory_breakdown()
