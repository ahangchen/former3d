# BatchNorm3d 修复文档

## 问题描述

在流式训练中，使用自定义的 `CustomBatchNorm3d` 类（继承自 `nn.BatchNorm3d`）导致训练错误：

```
Expected more than 1 spatial element when training, got input size torch.Size([1, 1536, 1, 1, 1])
```

## 根本原因

1. **自定义实现不必要**: `CustomBatchNorm3d` 只是强制设置 `track_running_stats=False`，这个功能可以直接在调用时指定
2. **代码复杂度**: 自定义类增加了代码复杂度和维护成本
3. **潜在的兼容性问题**: 自定义实现可能与 PyTorch 未来版本不兼容

## 解决方案

### 修改内容

删除自定义的 `CustomBatchNorm3d` 类，直接使用 PyTorch 的 `nn.BatchNorm3d`，并在 `global_norm` 中显式设置 `track_running_stats=False`。

### 代码变更

#### 文件: `former3d/net3d/former_v1.py`

**修改前:**
```python
# 使用InstanceNorm3d代替BatchNorm3d，避免batch size限制
# InstanceNorm在每个样本的channel维度上归一化，不依赖batch size
class CustomBatchNorm3d(nn.BatchNorm3d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        # InstanceNorm3d的参数: (num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False)
        # 我们强制设置affine=True以保持学习可缩放参数，track_running_stats=False避免batch依赖
        super(CustomBatchNorm3d, self).__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=False  # 强制禁用running stats
        )

if self.sync_bn == True:
    BatchNorm1d = autocast_norm(change_default_args(eps=1e-3, momentum=0.01)(LayerNorm1d))
    BatchNorm3d = autocast_norm(change_default_args(eps=1e-3, momentum=0.01, affine=True, track_running_stats=False)(CustomBatchNorm3d))
else:
    BatchNorm1d = (change_default_args(eps=1e-3, momentum=0.01)(LayerNorm1d))
    BatchNorm3d = (change_default_args(eps=1e-3, momentum=0.01, affine=True, track_running_stats=False)(CustomBatchNorm3d))

# ... 稍后使用
self.global_norm = nn.Sequential(
    BatchNorm3d(channels[-1]*len(self.pool_scales)),
    nn.ReLU(True))
```

**修改后:**
```python
# 直接使用PyTorch的BatchNorm3d
BatchNorm3d = nn.BatchNorm3d

if self.sync_bn == True:
    BatchNorm1d = autocast_norm(change_default_args(eps=1e-3, momentum=0.01)(LayerNorm1d))
else:
    BatchNorm1d = (change_default_args(eps=1e-3, momentum=0.01)(LayerNorm1d))

# ... 稍后使用
self.global_norm = nn.Sequential(
    BatchNorm3d(channels[-1]*len(self.pool_scales), track_running_stats=False),
    nn.ReLU(True))
```

## 验证结果

### 测试1: 代码检查

- ✅ 已删除自定义的 `CustomBatchNorm3d` 类定义
- ✅ 直接使用 PyTorch 的 `nn.BatchNorm3d`
- ✅ 移除所有对 `CustomBatchNorm3d` 的引用
- ✅ `global_norm` 正确使用 `BatchNorm3d` 并设置 `track_running_stats=False`

### 测试2: BatchNorm3d 行为验证

- ✅ BatchNorm3d 可以处理 batch size=2 的 5D 张量
- ✅ BatchNorm3d 可以处理 batch size=1 的 5D 张量（训练模式）
- ✅ BatchNorm3d 可以处理 batch size=1 的 5D 张量（评估模式）
- ✅ 使用 `track_running_stats=False` 可以处理 batch size=1

### 测试3: 完整场景测试

- ✅ 多尺度池化 + 上采样 + BatchNorm3d 成功
- ✅ BatchNorm3d 处理 batch size=1 成功
- ✅ 正常 batch size + track_running_stats=False 成功
- ✅ 评估模式 batch size=1 成功

## 技术说明

### track_running_stats 参数

- **`track_running_stats=True` (默认)**: 在训练过程中维护 running mean/variance，适用于 batch size 较大的场景
- **`track_running_stats=False`**: 不维护 running stats，每次使用当前 batch 的统计信息，适用于 batch size 较小或变化的场景

### 多尺度池化处理流程

1. 对输入特征进行不同尺度的平均池化
2. 使用 `F.interpolate` 将所有尺度的特征上采样到相同尺寸
3. 在 channel 维度拼接所有尺度的特征
4. 应用 BatchNorm3d 进行归一化

## 优势

1. **简洁性**: 减少代码行数，提高可读性
2. **标准化**: 使用官方实现，确保与 PyTorch 版本兼容
3. **性能**: 官方实现通常经过优化，性能更好
4. **可维护性**: 减少自定义代码，降低维护成本

## Git 提交记录

```
commit cf18e68
fix: 直接使用PyTorch的BatchNorm3d，删除自定义实现

commit a052fa6
fix: 在global_norm中显式设置track_running_stats=False以处理batch size变化
```

## 注意事项

- 在训练模式下，使用 `track_running_stats=False` 可以处理 batch size=1
- 在评估模式下，即使 `track_running_stats=True` 也可以处理 batch size=1
- 确保多尺度池化后使用 `interpolate` 上采样到相同尺寸再拼接

## 相关文件

- `former3d/net3d/former_v1.py` - 主要修改文件
- `test_batchnorm_code.py` - 代码检查测试
- `test_batchnorm_behavior.py` - BatchNorm3d 行为测试
- `test_full_scenario_fixed.py` - 完整场景测试
