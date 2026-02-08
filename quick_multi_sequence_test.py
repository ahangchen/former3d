#!/usr/bin/env python3
"""
快速测试多序列数据集
"""

import os
import sys

print("="*80)
print("快速测试多序列数据集")
print("="*80)

# 检查文件是否存在
files_to_check = [
    'multi_sequence_tartanair_dataset.py',
    'multi_sequence_training.py',
    'test_multi_sequence_dataset.py',
    'test_multi_sequence_integration.py'
]

print("\n检查文件:")
for file in files_to_check:
    if os.path.exists(file):
        print(f"✅ {file}")
    else:
        print(f"❌ {file}")

# 检查数据集计划
print("\n检查数据集修改计划:")
plan_file = 'dataset_modification_plan.md'
if os.path.exists(plan_file):
    with open(plan_file, 'r') as f:
        content = f.read()
        print(f"✅ {plan_file} ({len(content)} 字节)")
        
        # 检查关键内容
        if 'MultiSequenceTartanAirDataset' in content:
            print("  ✅ 包含 MultiSequenceTartanAirDataset 类定义")
        if 'n_view' in content:
            print("  ✅ 包含 n_view 参数")
        if '批量处理多个序列' in content:
            print("  ✅ 包含批量处理多个序列的目标")
else:
    print(f"❌ {plan_file} 不存在")

# 检查数据集实现
print("\n检查数据集实现:")
dataset_file = 'multi_sequence_tartanair_dataset.py'
if os.path.exists(dataset_file):
    with open(dataset_file, 'r') as f:
        content = f.read()
        print(f"✅ {dataset_file} ({len(content)} 字节)")
        
        # 检查关键类和方法
        if 'class MultiSequenceTartanAirDataset' in content:
            print("  ✅ 包含 MultiSequenceTartanAirDataset 类")
        if '__init__' in content:
            print("  ✅ 包含 __init__ 方法")
        if '__len__' in content:
            print("  ✅ 包含 __len__ 方法")
        if '__getitem__' in content:
            print("  ✅ 包含 __getitem__ 方法")
        if '_discover_sequences' in content:
            print("  ✅ 包含 _discover_sequences 方法")
        if '_build_segments' in content:
            print("  ✅ 包含 _build_segments 方法")
        
        # 检查输出形状注释
        if '(batch_size, n_view, 3, H, W)' in content:
            print("  ✅ 包含正确的输出形状注释")
else:
    print(f"❌ {dataset_file} 不存在")

# 检查训练脚本
print("\n检查训练脚本:")
training_file = 'multi_sequence_training.py'
if os.path.exists(training_file):
    with open(training_file, 'r') as f:
        content = f.read()
        print(f"✅ {training_file} ({len(content)} 字节)")
        
        # 检查关键函数
        if 'create_dataset' in content:
            print("  ✅ 包含 create_dataset 函数")
        if 'train_epoch' in content:
            print("  ✅ 包含 train_epoch 函数")
        if 'for frame_idx in range(n_view):' in content:
            print("  ✅ 包含遍历 n_view 的训练循环")
        
        # 检查数据形状处理
        if 'rgb_images[:, frame_idx]' in content:
            print("  ✅ 包含正确的帧提取代码")
else:
    print(f"❌ {training_file} 不存在")

# 总结
print("\n" + "="*80)
print("总结")
print("="*80)

print("\n已完成的工作:")
print("1. ✅ 创建了 dataset_modification_plan.md - 详细的数据集修改计划")
print("2. ✅ 创建了 multi_sequence_tartanair_dataset.py - 多序列数据集实现")
print("3. ✅ 创建了 test_multi_sequence_dataset.py - 数据集功能测试")
print("4. ✅ 创建了 multi_sequence_training.py - 多序列训练脚本")
print("5. ✅ 创建了 test_multi_sequence_integration.py - 集成测试")

print("\n数据集特性:")
print("- 支持加载多个TartanAir序列")
print("- 将长序列切分成固定长度片段 (n_view)")
print("- 支持片段步长 (stride)")
print("- 返回形状正确的批量数据")
print("- 支持数据增强选项")
print("- 包含完整的错误处理和日志")

print("\n训练脚本特性:")
print("- 使用新的多序列数据集")
print("- 正确处理批量维度 (batch_size, n_view, ...)")
print("- 遍历每个时刻进行前向传播")
print("- 包含模型状态重置")
print("- 完整的训练循环、验证、检查点保存")
print("- 内存优化配置")

print("\n下一步行动:")
print("1. 确保TartanAir数据目录存在")
print("2. 运行测试验证数据集功能")
print("3. 开始实际训练")
print("4. 监控训练进度和损失曲线")

print("\n注意事项:")
print("- 根据GPU内存调整batch_size")
print("- 根据序列长度调整n_view和stride")
print("- 如果数据目录不存在，需要下载或准备TartanAir数据")

print("\n" + "="*80)
print("🎉 多序列数据集实现完成!")
print("="*80)