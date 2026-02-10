# Batch Size 4 双 GPU 训练修复计划

## 问题分析

### 1. 主要错误
在 `former3d/net3d/former_v1.py` 的第 315 行：
```python
pool_norm = self.global_norm(pool)
```

- `pool` 是 5D 张量：`[batch, C, D, H, W]`
- `global_norm` 使用 `BatchNorm1d`（在第 242 行定义）
- `BatchNorm1d` 只接受 2D 或 3D 输入，不支持 5D 张量

### 2. 错误堆栈
```
ValueError: expected 2D or 3D input (got 5D input)
  File "former3d/net3d/former_v1.py", line 285, in forward
    outputs = inputs.replace_feature(torch.cat([inputs.features, self.global_norm(inputs_dense)], dim=1))
```

### 3. 代码重复问题
- `global_norm` 在 `__init__` 方法的第 242 行定义
- 在 `forward` 方法的第 237-242 行又重新定义了一次
- 这是错误的，应该在 `__init__` 中只定义一次

## 修复方案

### 方案 1: 将 BatchNorm1d 改为 BatchNorm3d（推荐）

**优点：**
- 直接解决 5D 张量问题
- 保持代码结构不变
- BatchNorm3d 适合处理 3D 卷积特征

**修改位置：**
1. `former3d/net3d/former_v1.py` 第 242 行
   ```python
   # 修改前
   self.global_norm = nn.Sequential(
           BatchNorm1d(channels[-1]*len(self.pool_scales)),
           nn.ReLU(True)
   )

   # 修改后
   self.global_norm = nn.Sequential(
           BatchNorm3d(channels[-1]*len(self.pool_scales)),
           nn.ReLU(True)
   )
   ```

2. 删除 `forward` 方法中的重复定义（第 237-242 行）

### 方案 2: 重构 global_avg 逻辑（备选）

如果方案 1 不能解决问题，可以：
- 不在 dense 张量上应用 BatchNorm
- 改为在 sparse 特征上应用 BatchNorm1d
- 需要重构整个 global_avg 部分

## 训练配置

### 目标配置
- Batch size: 4
- 双 GPU: `--multi-gpu` 或 `--gpu-ids 0 1`
- 每个GPU 实际 batch size: 2 (4 / 2)
- Crop size: 根据显存调整，建议 `'8,8,6'` 或更小
- Voxel size: `0.16` 或 `0.20`

### 训练命令
```bash
conda activate former3d
cd /home/cwh/coding/former3d
python train_stream_integrated.py \
  --multi-gpu \
  --batch-size 4 \
  --epochs 10 \
  --learning-rate 1e-4 \
  --voxel-size 0.16 \
  --crop-size 8,8,6 \
  --sequence-length 5 \
  --num-workers 2 \
  --cleanup-freq 10 \
  --memory-threshold 8.0
```

### 显存优化
- 使用 `cleanup_freq=10` 定期清理显存
- 监控显存使用情况
- 如果显存不足，可以：
  - 减小 `crop_size` 到 `'6,6,4'`
  - 减小 `sequence_length` 到 3
  - 增加 `cleanup_freq` 到 5

## 测试计划

### 1. 单元测试
- 测试 BatchNorm3d 对 5D 张量的处理
- 验证前向传播不会报错

### 2. 集成测试
- 使用小数据集测试 batch size 4 + 双 GPU
- 验证训练循环正常运行
- 检查 loss 是否正常下降

### 3. 压力测试
- 完整训练一个 epoch
- 验证显存使用情况
- 检查是否有 CUDA out of memory 错误

## 实施步骤

1. **代码修复**
   - 修改 `former3d/net3d/former_v1.py`
   - 将 BatchNorm1d 改为 BatchNorm3d
   - 删除重复的 global_norm 定义

2. **本地测试**
   - 运行小规模测试验证修复
   - 检查错误是否解决

3. **提交修复**
   - git commit 修改
   - 清理临时文件
   - git commit clean state

4. **训练测试**
   - 运行 batch size 4 + 双 GPU 训练
   - 监控显存和训练进度

## 预期结果

- ✅ 不再出现 5D 张量错误
- ✅ Batch size 4 在双 GPU 上正常工作
- ✅ 显存使用合理（每 GPU < 8GB）
- ✅ 训练速度提升（相比 batch size 2 单 GPU）
