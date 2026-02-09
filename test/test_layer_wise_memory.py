#!/usr/bin/env python3
"""
逐层显存监控工具
监控batch_size=4时网络各个层的显存占用增长情况
"""

import os
import sys
import torch
import time
import logging
from typing import Dict, List, Tuple, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LayerWiseMemoryMonitor:
    """逐层显存监控器"""

    def __init__(self, device: torch.device, batch_size: int = 4):
        """
        初始化监控器

        Args:
            device: 计算设备
            batch_size: 批次大小
        """
        self.device = device
        self.batch_size = batch_size
        self.snapshots: List[Dict] = []

    def monitor_forward_pass(self, model, images, poses, intrinsics) -> Dict:
        """
        监控前向传播过程中各个层的显存占用

        Args:
            model: 模型
            images: 输入图像
            poses: 位姿
            intrinsics: 内参

        Returns:
            监控结果字典
        """
        print(f"\n{'='*80}")
        print(f"逐层显存监控 - Batch Size = {self.batch_size}")
        print(f"{'='*80}")

        # 清理显存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect()

        # 记录模型参数显存
        model_params_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        print(f"模型参数显存: {model_params_memory:.3f} GB")

        # 创建输入数据
        input_memory = (images.element_size() * images.numel() +
                       poses.element_size() * poses.numel() +
                       intrinsics.element_size() * intrinsics.numel()) / 1024**3
        print(f"输入数据显存: {input_memory:.3f} GB")

        results = {
            'batch_size': self.batch_size,
            'model_params_memory': model_params_memory,
            'input_memory': input_memory,
            'layers': []
        }

        # 开始监控
        print(f"\n开始逐层监控...")

        with torch.no_grad():
            # 1. 3D编码器 - 第一帧（包含2D和3D）
            try:
                before_allocated = torch.cuda.memory_allocated() / 1024**3
                before_reserved = torch.cuda.memory_reserved() / 1024**3
                torch.cuda.reset_peak_memory_stats()

                output, state = model.forward_single_frame(
                    images, poses, intrinsics, reset_state=True
                )

                after_allocated = torch.cuda.memory_allocated() / 1024**3
                after_reserved = torch.cuda.memory_reserved() / 1024**3
                peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

                result = {
                    'label': "第1帧（2D+3D编码器）",
                    'before_allocated_gb': before_allocated,
                    'before_reserved_gb': before_reserved,
                    'after_allocated_gb': after_allocated,
                    'after_reserved_gb': after_reserved,
                    'peak_allocated_gb': peak_allocated,
                    'peak_reserved_gb': peak_reserved,
                    'allocated_diff_gb': after_allocated - before_allocated,
                    'peak_diff_gb': peak_allocated - before_allocated,
                }
                results['layers'].append(result)
                print(f"  第1帧: 峰值 {result['peak_allocated_gb']:.3f} GB, "
                      f"增量 {result['peak_diff_gb']:.3f} GB")
            except Exception as e:
                print(f"  第1帧失败: {e}")
                import traceback
                traceback.print_exc()
                return results  # 返回已收集的数据

            # 2. 3D编码器 - 第二帧（包含流式融合）
            try:
                before_allocated = torch.cuda.memory_allocated() / 1024**3
                before_reserved = torch.cuda.memory_reserved() / 1024**3
                torch.cuda.reset_peak_memory_stats()

                output, state = model.forward_single_frame(
                    images, poses, intrinsics, reset_state=False
                )

                after_allocated = torch.cuda.memory_allocated() / 1024**3
                after_reserved = torch.cuda.memory_reserved() / 1024**3
                peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

                result = {
                    'label': "第2帧（包含流式融合）",
                    'before_allocated_gb': before_allocated,
                    'before_reserved_gb': before_reserved,
                    'after_allocated_gb': after_allocated,
                    'after_reserved_gb': after_reserved,
                    'peak_allocated_gb': peak_allocated,
                    'peak_reserved_gb': peak_reserved,
                    'allocated_diff_gb': after_allocated - before_allocated,
                    'peak_diff_gb': peak_allocated - before_allocated,
                }
                results['layers'].append(result)
                print(f"  第2帧: 峰值 {result['peak_allocated_gb']:.3f} GB, "
                      f"增量 {result['peak_diff_gb']:.3f} GB")
            except Exception as e:
                print(f"  第2帧失败: {e}")
                import traceback
                traceback.print_exc()

            # 3. 3D编码器 - 第三帧
            try:
                before_allocated = torch.cuda.memory_allocated() / 1024**3
                before_reserved = torch.cuda.memory_reserved() / 1024**3
                torch.cuda.reset_peak_memory_stats()

                output, state = model.forward_single_frame(
                    images, poses, intrinsics, reset_state=False
                )

                after_allocated = torch.cuda.memory_allocated() / 1024**3
                after_reserved = torch.cuda.memory_reserved() / 1024**3
                peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
                peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

                result = {
                    'label': "第3帧（包含流式融合）",
                    'before_allocated_gb': before_allocated,
                    'before_reserved_gb': before_reserved,
                    'after_allocated_gb': after_allocated,
                    'after_reserved_gb': after_reserved,
                    'peak_allocated_gb': peak_allocated,
                    'peak_reserved_gb': peak_reserved,
                    'allocated_diff_gb': after_allocated - before_allocated,
                    'peak_diff_gb': peak_allocated - before_allocated,
                }
                results['layers'].append(result)
                print(f"  第3帧: 峰值 {result['peak_allocated_gb']:.3f} GB, "
                      f"增量 {result['peak_diff_gb']:.3f} GB")
            except Exception as e:
                print(f"  第3帧失败: {e}")
                import traceback
                traceback.print_exc()

        return results

    def analyze_layers(self, results: Dict) -> None:
        """
        分析各层的显存占用

        Args:
            results: 监控结果
        """
        print(f"\n{'='*80}")
        print("逐层显存分析")
        print(f"{'='*80}")

        if len(results['layers']) == 0:
            print("⚠️  没有成功监控任何层")
            return

        print(f"\n{'层':<30} {'峰值显存':<15} {'显存增量':<15} {'占比':<10}")
        print("-" * 80)

        total_peak_diff = 0
        for layer in results['layers']:
            peak_diff = layer['peak_diff_gb']
            total_peak_diff += peak_diff

        for layer in results['layers']:
            peak_allocated = layer['peak_allocated_gb']
            peak_diff = layer['peak_diff_gb']
            percentage = (peak_diff / total_peak_diff * 100) if total_peak_diff > 0 else 0

            print(f"{layer['label']:<30} "
                  f"{peak_allocated:<15.3f} "
                  f"{peak_diff:<15.3f} "
                  f"{percentage:<10.2f}%")

        print("-" * 80)
        print(f"{'总计':<30} {'':<15} {total_peak_diff:<15.3f} {'':<10}")

        # 分析
        print(f"\n关键发现:")
        print(f"  总显存增量: {total_peak_diff:.3f} GB")
        print(f"  最大增量层: {max(results['layers'], key=lambda x: x['peak_diff_gb'])['label']}")
        print(f"  最大峰值: {max(l['peak_allocated_gb'] for l in results['layers']):.3f} GB")

    def generate_report(self, results: Dict) -> None:
        """
        生成逐层显存分析报告

        Args:
            results: 监控结果
        """
        if len(results['layers']) == 0:
            print("⚠️  没有成功监控任何层，无法生成报告")
            return

        total_peak_diff = sum(l['peak_diff_gb'] for l in results['layers'])

        report = f"""# Batch Size {results['batch_size']} 逐层显存分析报告

## 📊 测试信息

**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**设备**: {self.device}
**总显存**: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB
**Batch Size**: {results['batch_size']}
**模型参数显存**: {results['model_params_memory']:.3f} GB
**输入数据显存**: {results['input_memory']:.3f} GB

## 🎯 逐层显存分布

| 层 | 峰值显存(GB) | 显存增量(GB) | 占比 |
|------|-------------|-------------|------|
"""

        for layer in results['layers']:
            peak_allocated = layer['peak_allocated_gb']
            peak_diff = layer['peak_diff_gb']
            percentage = (peak_diff / total_peak_diff * 100) if total_peak_diff > 0 else 0

            report += f"| {layer['label']} | {peak_allocated:.3f} | {peak_diff:.3f} | {percentage:.2f}% |\n"

        report += f"""
**总计**: {total_peak_diff:.3f} GB

## 📈 显存增长分析

### 层级分解

"""

        for i, layer in enumerate(results['layers']):
            peak_allocated = layer['peak_allocated_gb']
            peak_diff = layer['peak_diff_gb']
            percentage = (peak_diff / total_peak_diff * 100) if total_peak_diff > 0 else 0

            report += f"""#### {i+1}. {layer['label']}

- **峰值显存**: {peak_allocated:.3f} GB
- **显存增量**: {peak_diff:.3f} GB
- **占比**: {percentage:.2f}%

"""

        report += f"""## 🔍 关键发现

### 1. 最大显存消耗层

**{max(results['layers'], key=lambda x: x['peak_diff_gb'])['label']}**
- 显存增量: {max(l['peak_diff_gb'] for l in results['layers']):.3f} GB
- 占比: {(max(l['peak_diff_gb'] for l in results['layers']) / total_peak_diff * 100):.2f}%

### 2. 层级显存增长模式

"""

        for i, layer in enumerate(results['layers'][:-1]):
            current = layer['peak_diff_gb']
            next_layer = results['layers'][i+1]['peak_diff_gb']
            ratio = next_layer / current if current > 0 else float('inf')
            report += f"- {layer['label']} → {results['layers'][i+1]['label']}: {ratio:.2f}x\n"

        report += f"""### 3. 与Batch Size 1的对比

基于之前的测试结果（batch_size=1）：

| 层 | Batch 1 | Batch 4 | 倍数 |
|------|---------|---------|------|
"""

        # 假设batch_size=1的显存
        batch1_peaks = {
            '2D编码器': 0.007,  # 估算
            '3D编码器（第1帧）': 0.131,
            '3D编码器（第2帧）': 0.209,
        }

        for layer in results['layers']:
            batch1_peak = batch1_peaks.get(layer['label'], 0)
            batch4_peak = layer['peak_allocated_gb']
            ratio = batch4_peak / batch1_peak if batch1_peak > 0 else float('inf')
            report += f"| {layer['label']} | {batch1_peak:.3f} | {batch4_peak:.3f} | {ratio:.2f}x |\n"

        report += f"""
## 💡 优化建议

### 立即可行（高优先级）

1. **优化最大显存消耗层**
   - 层: {max(results['layers'], key=lambda x: x['peak_diff_gb'])['label']}
   - 建议: 实施checkpointing或分块计算

2. **使用梯度累积**
   - batch_size=1 + accumulation_steps=4
   - 有效batch size=4
   - 显存使用更低

### 中期优化（中优先级）

3. **实施分块计算**
   - 对显存消耗大的层实施分块
   - 降低峰值显存

4. **使用混合精度训练**
   - FP16降低显存50%
   - 可支持更大的batch_size

## 🎯 结论

### 显存增长总结

- **模型参数**: {results['model_params_memory']:.3f} GB
- **输入数据**: {results['input_memory']:.3f} GB
- **前向传播增量**: {total_peak_diff:.3f} GB
- **总峰值显存**: {max(l['peak_allocated_gb'] for l in results['layers']):.3f} GB

### 层级占比

"""

        for i, layer in enumerate(results['layers']):
            percentage = (layer['peak_diff_gb'] / total_peak_diff * 100) if total_peak_diff > 0 else 0
            report += f"{i+1}. {layer['label']}: {percentage:.2f}%\n"

        report += f"""

### 达到9 GB显存占用的预估

基于batch_size=4的峰值显存 {max(l['peak_allocated_gb'] for l in results['layers']):.3f} GB：

- **batch_size=8 预估**: {max(l['peak_allocated_gb'] for l in results['layers']) * 2:.3f} GB
- **batch_size=16 预估**: {max(l['peak_allocated_gb'] for l in results['layers']) * 4:.3f} GB
- **batch_size=32 预估**: {max(l['peak_allocated_gb'] for l in results['layers']) * 8:.3f} GB

---

**报告生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**监控器**: LayerWiseMemoryMonitor
**测试者**: Frank
"""

        # 保存报告
        report_path = f"/home/cwh/coding/former3d/doc/layer_wise_memory_batch_{results['batch_size']}.md"
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

    # 导入模型
    from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated

    # 测试batch_size=4
    batch_size = 4

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

    # 创建输入数据
    image_size = (256, 256)
    images = torch.randn(batch_size, 3, *image_size).to(device)
    poses = torch.eye(4).unsqueeze(0).expand(batch_size, -1, -1).to(device)
    intrinsics = torch.eye(3).unsqueeze(0).expand(batch_size, -1, -1).to(device)

    print(f"\n测试参数：")
    print(f"  Batch Size: {batch_size}")
    print(f"  图像尺寸: {image_size}")

    # 创建监控器
    monitor = LayerWiseMemoryMonitor(device, batch_size)

    # 监控前向传播
    try:
        results = monitor.monitor_forward_pass(model, images, poses, intrinsics)
    except Exception as e:
        print(f"\n⚠️  监控失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 分析结果
    monitor.analyze_layers(results)

    # 生成报告
    monitor.generate_report(results)


if __name__ == "__main__":
    main()
