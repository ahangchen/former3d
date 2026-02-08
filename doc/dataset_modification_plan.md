# 数据集修改计划

## 当前问题分析

### 1. 数据集实现问题
当前 `OnlineTartanAirDataset` 实现存在以下问题：
- 只加载单个序列（`abandonedfactory_sample_P001`）
- 返回形状为 `(F, C, H, W)` 的图像和 `(F, 4, 4)` 的位姿
- 没有将长序列切分成固定长度的片段
- 不支持批量处理多个序列

### 2. 训练循环问题
当前训练循环：
- 假设每个样本包含多个帧（F帧）
- 在 `train_epoch` 中遍历每个帧进行处理
- 但数据集返回的是整个序列的所有帧，而不是按片段组织

## 修改目标

### 1. 数据集修改目标
1. **支持多个序列**：加载TartanAir目录中的所有序列
2. **序列切分**：将长图像序列切分成固定长度 `n_view` 的片段
3. **批量组织**：每个样本返回形状为 `(batch_size, n_view, C, H, W)` 的图像
4. **对应位姿**：每个样本返回形状为 `(batch_size, n_view, 4, 4)` 的位姿
5. **其他数据**：TSDF、占用网格等也按相同方式组织

### 2. 训练循环修改目标
1. **片段遍历**：每次forward时遍历 `n_view` 个时刻
2. **批量处理**：正确处理批量维度
3. **内存优化**：保持内存优化特性

## 详细修改方案

### 第1步：修改数据集类

#### 1.1 修改 `__init__` 方法
- 移除 `sequence_name` 参数，改为加载所有序列
- 添加 `n_view` 参数控制片段长度
- 添加 `stride` 参数控制片段之间的步长
- 收集所有序列的路径信息

#### 1.2 修改 `_load_data_paths` 方法
- 遍历TartanAir目录，找到所有序列
- 为每个序列加载RGB、深度、位姿文件
- 计算每个序列可生成的片段数量

#### 1.3 修改 `__len__` 方法
- 返回总片段数量，而不是固定为1

#### 1.4 修改 `__getitem__` 方法
- 根据索引计算对应的序列和起始帧
- 加载连续 `n_view` 帧的图像和位姿
- 在线计算TSDF（基于片段）
- 返回形状正确的张量

### 第2步：修改训练循环

#### 2.1 修改 `train_epoch` 函数
- 正确处理批量维度：`(batch_size, n_view, ...)`
- 遍历 `n_view` 个时刻进行前向传播
- 保持梯度累积和内存优化

#### 2.2 修改模型输入处理
- 确保模型能处理 `(batch_size, n_view, ...)` 形状的输入
- 正确处理状态重置（每个片段开始时重置）

### 第3步：创建新的数据集类

创建 `MultiSequenceTartanAirDataset` 类，继承或重构现有功能：

```python
class MultiSequenceTartanAirDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        n_view: int = 5,
        stride: int = 1,
        crop_size: tuple = (48, 48, 32),
        voxel_size: float = 0.04,
        target_image_size: tuple = (256, 256),
        max_depth: float = 10.0,
        truncation_margin: float = 0.2,
        augment: bool = False,
        max_sequences: int = None  # 限制使用的序列数量
    ):
        # 初始化参数
        # 加载所有序列信息
        # 计算总片段数
```

## 实施步骤

### 阶段1：创建新的数据集类
1. 创建 `multi_sequence_tartanair_dataset.py` 文件
2. 实现序列发现和片段生成逻辑
3. 测试数据加载和形状正确性

### 阶段2：修改训练脚本
1. 更新 `optimized_online_training.py` 使用新数据集
2. 修改训练循环处理批量维度
3. 测试训练流程

### 阶段3：验证和优化
1. 验证数据形状正确性
2. 测试内存使用情况
3. 优化性能（缓存、并行加载等）

## 预期输出形状

### 数据集返回的样本：
```python
{
    'rgb_images': torch.Tensor,    # (batch_size, n_view, 3, H, W)
    'poses': torch.Tensor,         # (batch_size, n_view, 4, 4)
    'intrinsics': torch.Tensor,    # (batch_size, 3, 3) 或 (3, 3)
    'tsdf': torch.Tensor,          # (batch_size, D, H, W)
    'occupancy': torch.Tensor,     # (batch_size, D, H, W)
    'voxel_coords': torch.Tensor,  # (batch_size, D, H, W, 3)
    # ... 其他字段
}
```

### 训练循环处理：
```python
# 批次数据形状
images: (batch_size, n_view, 3, H, W)
poses: (batch_size, n_view, 4, 4)

# 遍历每个时刻
for frame_idx in range(n_view):
    current_images = images[:, frame_idx]  # (batch_size, 3, H, W)
    current_poses = poses[:, frame_idx]    # (batch_size, 4, 4)
    # 前向传播...
```

## 风险评估

### 技术风险：
1. **内存增加**：同时加载多个序列可能增加内存使用
   - 缓解：使用懒加载、缓存策略
2. **计算复杂度**：在线TSDF计算可能变慢
   - 缓解：优化算法、使用缓存
3. **数据不平衡**：不同序列长度不同
   - 缓解：统一采样策略

### 时间估计：
- 阶段1：2-3小时
- 阶段2：1-2小时
- 阶段3：1-2小时
- 总计：4-7小时

## 成功标准

1. ✅ 数据集能加载多个TartanAir序列
2. ✅ 正确切分长序列为固定长度片段
3. ✅ 返回形状正确的批量数据
4. ✅ 训练循环能正确处理批量维度
5. ✅ 内存使用在合理范围内
6. ✅ 训练损失正常下降

## 下一步行动

1. 立即开始实施阶段1：创建新的数据集类
2. 编写测试验证数据形状
3. 逐步集成到训练流程中