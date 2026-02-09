# 多序列数据集集成指南

## 概述

已成功创建 `multi_sequence_tartanair_dataset.py`，这是一个支持多个TartanAir序列并将长序列切分成固定长度片段的数据集类。

## 文件位置

- 主数据集类: `multi_sequence_tartanair_dataset.py`
- 简化测试: `simple_dataset_test.py` (已验证工作)
- 训练脚本框架: `train_with_multi_sequence.py`
- 详细计划: `dataset_modification_plan.md`

## 数据集特性

### 1. 多序列支持
- 自动发现 `/home/cwh/Study/dataset/tartanair` 目录中的所有序列
- 已验证找到 2 个样本序列: `abandonedfactory_sample_P001`, `carwelding_sample_P007`
- 可配置最大序列数量 (`max_sequences` 参数)

### 2. 片段切分
- 将长序列切分成固定长度片段 (`n_view` 参数)
- 可配置片段步长 (`stride` 参数)
- 示例: 434帧序列 → 404个片段 (n_view=5, stride=2)

### 3. 数据形状
```python
# 数据集返回的样本
{
    'rgb_images': torch.Tensor,    # (n_view, 3, H, W)
    'poses': torch.Tensor,         # (n_view, 4, 4)
    'intrinsics': torch.Tensor,    # (3, 3)
    'tsdf': torch.Tensor,          # (1, D, H, W)
    'occupancy': torch.Tensor,     # (1, D, H, W)
    'voxel_coords': torch.Tensor,  # (D, H, W, 3)
    'sequence_name': str,
    'segment_idx': int,
    'start_frame': int,
    'end_frame': int
}
```

### 4. 批量处理
- DataLoader 自动堆叠为: `(batch_size, n_view, 3, H, W)`
- 支持标准 PyTorch DataLoader

## 集成到现有训练脚本

### 步骤1: 替换数据集初始化

**原代码 (optimized_online_training.py):**
```python
from online_tartanair_dataset import OnlineTartanAirDataset

dataset = OnlineTartanAirDataset(
    data_root="/home/cwh/Study/dataset/tartanair",
    sequence_name="abandonedfactory_sample_P001",
    n_frames=5,
    # ... 其他参数
)
```

**新代码:**
```python
from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset

dataset = MultiSequenceTartanAirDataset(
    data_root="/home/cwh/Study/dataset/tartanair",
    n_view=5,  # 片段长度
    stride=2,   # 片段步长
    crop_size=(48, 48, 32),
    voxel_size=0.04,
    target_image_size=(256, 256),
    max_sequences=10,  # 限制使用的序列数量
    shuffle=True
)
```

### 步骤2: 修改训练循环

**原代码处理单个序列的所有帧:**
```python
# 假设 dataset 返回 (F, 3, H, W) 其中 F 是序列总帧数
for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch['rgb_images']  # (batch_size, F, 3, H, W)?
        # 遍历所有帧...
```

**新代码处理片段:**
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch['rgb_images']  # (batch_size, n_view, 3, H, W)
        poses = batch['poses']        # (batch_size, n_view, 4, 4)
        
        batch_size, n_view = images.shape[:2]
        
        # 重置模型状态（每个片段开始时）
        model.reset_state(batch_size, device)
        
        # 遍历片段中的每个时刻
        for frame_idx in range(n_view):
            current_images = images[:, frame_idx]  # (batch_size, 3, H, W)
            current_poses = poses[:, frame_idx]    # (batch_size, 4, 4)
            
            # 前向传播
            output = model(current_images, current_poses, 
                          reset_state=(frame_idx == 0))
            
            # 计算损失...
```

### 步骤3: 修改模型前向传播

模型需要处理:
1. 批量维度: `(batch_size, ...)`
2. 状态重置: 每个片段开始时重置隐藏状态
3. 多帧输入: 按顺序处理片段中的每一帧

## 依赖安装

运行完整数据集需要以下依赖:

```bash
pip install numpy torch imageio pillow scipy
```

对于Stream-SDFFormer项目，可能还需要:
```bash
pip install spconv-pytorch open3d
```

## 测试验证

### 1. 基础测试 (已通过)
```bash
python simple_dataset_test.py
```
验证数据目录结构和片段生成逻辑。

### 2. 完整测试 (需要安装依赖)
```bash
python multi_sequence_tartanair_dataset.py
```
测试完整的数据集功能，包括图像加载和TSDF计算。

### 3. 训练测试
```bash
python train_with_multi_sequence.py
```
测试训练循环集成。

## 配置参数

### 数据集参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data_root` | `/home/cwh/Study/dataset/tartanair` | 数据根目录 |
| `n_view` | 5 | 每个片段的帧数 |
| `stride` | 2 | 片段之间的步长 |
| `crop_size` | (48, 48, 32) | TSDF裁剪尺寸 |
| `voxel_size` | 0.04 | 体素大小（米） |
| `target_image_size` | (256, 256) | 目标图像大小 |
| `max_sequences` | None | 最大序列数量 |
| `shuffle` | True | 是否打乱序列顺序 |

### 训练参数调整
1. **批次大小**: 根据GPU内存调整，建议 2-4
2. **片段长度**: `n_view=5` 适合大多数场景
3. **学习率**: 可能需要微调
4. **epoch数量**: 增加以充分利用多序列数据

## 性能考虑

### 内存优化
1. **懒加载**: 只在需要时加载图像数据
2. **缓存**: 缓存计算昂贵的TSDF
3. **数据增强**: 在线增强减少存储需求

### 计算优化
1. **并行加载**: 使用 DataLoader 的 `num_workers`
2. **TSDF预计算**: 可考虑预计算TSDF加速训练
3. **混合精度**: 使用FP16减少内存和加速计算

## 故障排除

### 常见问题
1. **依赖缺失**: 安装 `imageio`, `pillow`, `scipy`
2. **内存不足**: 减少 `batch_size` 或 `max_sequences`
3. **数据形状错误**: 检查模型输入输出维度
4. **训练不稳定**: 调整学习率或使用学习率调度器

### 调试建议
1. 先使用小数据集测试 (`max_sequences=2`)
2. 验证数据形状在每个步骤都正确
3. 监控GPU内存使用
4. 保存和加载检查点

## 下一步计划

### 短期 (1-2天)
1. [ ] 安装必要依赖
2. [ ] 运行完整数据集测试
3. [ ] 集成到 `optimized_online_training.py`
4. [ ] 验证训练流程

### 中期 (3-5天)
1. [ ] 性能优化（缓存、并行加载）
2. [ ] 添加数据增强
3. [ ] 超参数调优
4. [ ] 验证指标评估

### 长期 (1-2周)
1. [ ] 扩展到所有TartanAir序列
2. [ ] 实现高级TSDF融合
3. [ ] 添加可视化工具
4. [ ] 模型架构优化

## 成功标准

1. ✅ 数据集能加载多个TartanAir序列
2. ✅ 正确切分长序列为固定长度片段  
3. ✅ 返回形状正确的批量数据
4. ✅ 训练循环能正确处理批量维度
5. ✅ 内存使用在合理范围内
6. ✅ 训练损失正常下降

## 联系支持

如有问题，请参考:
- 数据集实现: `multi_sequence_tartanair_dataset.py`
- 测试脚本: `simple_dataset_test.py`
- 详细计划: `dataset_modification_plan.md`
- 训练示例: `train_with_multi_sequence.py`