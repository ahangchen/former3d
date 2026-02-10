# BatchNorm 修复总结

## 修复内容

### 1. 问题诊断
**原始错误：**
```
ValueError: expected 2D or 3D input (got 5D input)
  File "former3d/net3d/former_v1.py", line 315, in forward
    pool_norm = self.global_norm(pool)
```

**根本原因：**
- `global_norm` 使用 `BatchNorm1d`，只支持 2D/3D 张量
- 但 `pool` 是 5D 张量 `[batch, num_scales*C, D, H, W]`
- 导致维度不匹配错误

### 2. 修复方案

#### 2.1 将 BatchNorm1d 改为 BatchNorm3d
**文件：** `former3d/net3d/former_v1.py`

**修改前（第 242 行）：**
```python
self.global_norm = nn.Sequential(
    BatchNorm1d(channels[-1]*len(self.pool_scales)),
    nn.ReLU(True)
)
```

**修改后：**
```python
self.global_norm = nn.Sequential(
    BatchNorm3d(channels[-1]*len(self.pool_scales)),
    nn.ReLU(True)
)
```

#### 2.2 删除重复的 global_norm 定义
**问题：** `forward` 方法中重复定义了 `global_norm`
**解决：** 删除 forward 方法中的重复定义

#### 2.3 重写 global_avg 逻辑
**问题：** 之前的 global_avg 逻辑混乱，有重复的代码块
**解决：** 重写为清晰的逻辑：
1. 从 `feats[-1]` 获取输入
2. 转换为 dense tensor
3. 多尺度池化
4. 应用 BatchNorm3d
5. 将原始特征和池化特征拼接
6. 转换回 sparse tensor

**关键代码：**
```python
# 拼接原始特征和池化特征: [N_valid, C*(1+num_scales)]
concatenated_features = torch.cat([original_features, pooled_features], dim=1)
```

## 测试结果

### 1. BatchNorm 测试
✅ **通过** - test_batchnorm3d_fix.py
- 成功完成 batch size 4 的前向传播
- 没有出现 BatchNorm 维度错误
- 输出无 NaN

### 2. 训练测试
⚠️ **显存不足** - 但 BatchNorm 修复成功
- 没有出现 BatchNorm 或 5D 张量错误
- 错误仅为 CUDA out of memory（显存配置问题）
- 说明 BatchNorm 修复有效

## 训练配置建议

### 当前显存限制
- GPU 0: 9.91 GB 总容量，已使用 8.83 GB
- batch size 4 时，每个 GPU 实际 batch size 为 2
- 在 crop_size=(8,8,6), voxel_size=0.16 下显存不足

### 推荐配置

#### 配置 1: 更小 crop size（推荐）
```bash
python train_stream_integrated.py \
    --multi-gpu \
    --batch-size 4 \
    --voxel-size 0.16 \
    --crop-size 6,6,4 \
    --sequence-length 3
```

#### 配置 2: 更大 voxel size
```bash
python train_stream_integrated.py \
    --multi-gpu \
    --batch-size 4 \
    --voxel-size 0.20 \
    --crop-size 8,8,6 \
    --sequence-length 5
```

#### 配置 3: 单 GPU batch size 2
```bash
python train_stream_integrated.py \
    --batch-size 2 \
    --voxel-size 0.16 \
    --crop-size 8,8,6 \
    --sequence-length 5
```

## 提交记录

1. **54dcd59** - 修复BatchNorm问题：将BatchNorm1d改为BatchNorm3d
2. **6a8e8dd** - 修复global_avg逻辑，正确拼接原始特征和池化特征

## 后续工作

- [ ] 调整显存配置，完成 batch size 4 + 双 GPU 训练
- [ ] 验证训练收敛性
- [ ] 对比 batch size 2 和 batch size 4 的训练效果

## 文件变更

- `former3d/net3d/former_v1.py` - 修复 BatchNorm 维度问题
- `doc/batch_size_fix_plan.md` - 修复计划文档
- `test_batchnorm3d_fix.py` - BatchNorm 测试脚本
- `test_train_batch4_multi_gpu.sh` - 训练测试脚本
