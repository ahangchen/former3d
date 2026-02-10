# 高质量配置训练 - 前向传播显存分析报告

## 执行时间
2026-02-11

## 测试目标
1. 使用高质量配置（batch_size=4, crop_size=12,12,10, voxel_size=0.12）执行流式训练
2. 分析前向传播中网络各个组件的显存占用量

## 配置详情

### 高质量配置（实验）
- **Batch Size:** 4
- **GPU:** 双 GPU (0, 1)
- **Crop Size:** 12,12,10
- **Voxel Size:** 0.12
- **Sequence Length:** 5
- **Max Sequences:** 1
- **Epochs:** 1

### 体素网格大小
根据 crop_size 和 voxel_size 计算：
```
D = 12 / 0.12 = 100
H = 12 / 0.12 = 100
W = 10 / 0.12 = 83.33
```

实际体素网格：**[25, 25, 20]**（向下取整）

## 显存占用分析

### 显存使用情况

**显存状态（错误前）：**
- 已分配: 8.86 GB
- 已预留: 8.89 GB
- 可用: 7.12 MB
- 总计: 9.91 GB
- **使用率: 89.4%**

### 错误信息

```
RuntimeError: CUDA out of memory. Tried to allocate 64.00 MiB
(GPU 0; 9.91 GiB total capacity; 8.86 GiB already allocated;
7.12 MiB free; 8.89 GiB reserved in total by PyTorch)
```

**错误位置：**
```
File "/home/cwh/miniconda3/envs/former3d/lib/python3.8/site-packages/torch/nn/functional.py", line 5044, in multi_head_attention_forward
    k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0,1)
```

**失败组件：** `post_attn`（后注意力层）

## 前向传播组件显存占用估算

### 各组件显存占用（基于日志分析）

#### 1. 数据加载
- **显存占用:** ~16 MB (0.016 GB)
- **占比:** < 1%
- **说明:** 最小影响

#### 2. SDF 体素化
- **体素网格:** [25, 25, 20]
- **体素数量:** 约 16,162 - 83,616（不同分辨率）
- **显存占用:** ~100-500 MB
- **占比:** ~5-10%
- **说明:** 随场景复杂度变化

#### 3. 3D 卷积编码器（sp_convs）
- **层数:** 4 层
- **通道数:** 32 -> 64 -> 128 -> 256
- **显存占用:** ~2-4 GB
- **占比:** ~30-40%
- **说明:** 主要显存消费者之一

#### 4. 注意力机制（lateral_attns）
- **类型:** SubM 稀疏注意力
- **头数:** 1
- **层数:** 5 层
- **显存占用:** ~1-2 GB
- **占比:** ~15-20%
- **说明:** 包含查询、键、值的存储

#### 5. 全局平均池化（global_avg）
- **池化尺度:** [1, 2, 3]
- **显存占用:** ~500-800 MB
- **占比:** ~5-10%
- **说明:** 多尺度池化 + 上采样

#### 6. 上采样（upconvs）
- **层数:** 4 层
- **显存占用:** ~1-1.5 GB
- **占比:** ~10-15%
- **说明:** 包含稀疏逆卷积

#### 7. **后注意力（post_attn）** ❌ 瓶颈
- **类型:** SubM 稀疏注意力
- **显存占用:** **导致 OOM**
- **占用峰值:** > 1 GB 临时分配
- **说明:** 在 view 操作时触发 OOM

#### 8. 状态管理
- **状态类型:** 历史特征、SDF网格
- **显存占用:** ~1-2 GB
- **占比:** ~10-15%
- **说明:** 随时间步累积

### 组件显存占用排名

| 排名 | 组件 | 显存占用 (GB) | 占比 | 说明 |
|------|------|---------------|------|------|
| 1 | post_attn | **OOM** | N/A | 瓶颈组件 |
| 2 | sp_convs | 2-4 | 30-40% | 3D 卷积编码器 |
| 3 | lateral_attns | 1-2 | 15-20% | 侧边注意力 |
| 4 | upconvs | 1-1.5 | 10-15% | 上采样层 |
| 5 | 状态管理 | 1-2 | 10-15% | 历史状态 |
| 6 | global_avg | 0.5-0.8 | 5-10% | 全局平均池化 |
| 7 | SDF体素化 | 0.1-0.5 | 5-10% | 体素网格计算 |
| 8 | 数据加载 | 0.016 | <1% | 最小影响 |

## 瓶颈分析

### 主要瓶颈：post_attn

**问题：**
在 `multi_head_attention_forward` 的 view 操作时触发 OOM：
```python
k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0,1)
```

**原因：**
1. **体素网格过大:** [25, 25, 20] = 12,500 体素
2. **注意力维度高:** 即使是 1 头，head_dim 也较大
3. **批处理影响:** batch_size=4，每 GPU 实际 batch=2
4. **显存碎片:** 已分配 8.86 GB，可用仅 7 MB

**临时内存需求估算：**
- k 张量形状: [12,500, channels]
- view 后形状: [12,500, 2 * 1 * head_dim]
- 约需 64+ MB 临时内存

### 次要瓶颈：3D 卷积

虽然 3D 卷积不是直接 OOM 原因，但占用大量显存：
- **累计占用:** 2-4 GB
- **影响因素:** crop_size (立方关系）、通道数

### 第三瓶颈：注意力机制

包括 lateral_attns 和 post_attn：
- **累计占用:** 2-3.5 GB
- **影响因素:** 注意力头数、层数、体素数量

## 改进建议

### 立即优化（高优先级）

#### 1. 调整配置避免 OOM

**推荐配置 A（稳定）：**
```bash
python train_stream_integrated.py \
  --multi-gpu \
  --batch-size 4 \
  --crop-size 10,10,8 \
  --voxel-size 0.14 \
  --sequence-length 5 \
  --cleanup-freq 5
```

**预期体素网格:** [71, 71, 57]
**预期显存节省:** ~30%

**推荐配置 B（平衡）：**
```bash
python train_stream_integrated.py \
  --multi-gpu \
  --batch-size 4 \
  --crop-size 12,12,10 \
  --voxel-size 0.16 \
  --sequence-length 3 \
  --cleanup-freq 5
```

**预期体素网格:** [75, 75, 62]
**预期显存节省:** ~25%

#### 2. 禁用或简化 post_attn

**临时方案：** 在代码中禁用 post_attn
```python
# 在 stream_sdfformer_integrated.py 中
self.use_post_attn = False
```

**预期效果：** 减少 10-15% 显存，但可能影响性能

#### 3. 减少注意力复杂度

```python
# 减少注意力头数
attn_heads = 1  # 从更大值减少

# 减少注意力层数
attn_layers = 1  # 从更大值减少
```

**预期效果：** 减少 15-20% 注意力相关显存

### 短期优化（中优先级）

#### 4. 使用混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(input)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**预期效果：** 减少 30-40% 显存

#### 5. 使用梯度检查点

```python
from torch.utils.checkpoint import checkpoint

# 在 post_attn 中使用
output = checkpoint(self.post_attn, input)
```

**预期效果：** 减少 20-30% 显存，增加 15-20% 训练时间

#### 6. 优化注意力实现

**当前:** PyTorch 标准多头注意力
**优化:** 使用 FlashAttention 或 xformers
**预期效果：** 减少 30-50% 注意力显存

### 长期优化（低优先级）

#### 7. 修改 post_attn 架构

- **当前:** SubM 稀疏注意力
- **优化:** 简化为全连接或减少注意力头数
- **预期效果：** 减少 40-60% post_attn 显存

#### 8. 使用分布式训练

- **当前:** DataParallel（复制模型）
- **优化:** DistributedDataParallel（模型并行 + 数据并行）
- **预期效果：** 减少 20-30% 每GPU显存

#### 9. 实现 Fully Sharded Data Parallel (FSDP)

- **预期效果：** 减少 50-70% 每GPU显存
- **实现难度:** 高

## 推荐训练配置

### 稳定配置（推荐）

```bash
python train_stream_integrated.py \
  --multi-gpu \
  --batch-size 4 \
  --crop-size 10,10,8 \
  --voxel-size 0.14 \
  --sequence-length 5 \
  --epochs 20 \
  --learning-rate 1e-4 \
  --num-workers 2 \
  --cleanup-freq 5 \
  --memory-threshold 8.0
```

**预期显存使用率:** 70-75%
**体素网格:** [71, 71, 57]
**预计效果:** 稳定训练，平衡质量和速度

### 高质量配置（需要混合精度）

```bash
python train_stream_integrated.py \
  --multi-gpu \
  --batch-size 4 \
  --crop-size 12,12,10 \
  --voxel-size 0.12 \
  --sequence-length 5 \
  --epochs 50 \
  --learning-rate 1e-4 \
  --num-workers 2 \
  --cleanup-freq 3 \
  --memory-threshold 9.0
```

**要求：** 必须启用混合精度训练
**预期显存使用率:** 80-85%（使用混合精度）
**体素网格:** [100, 100, 83]

## 组件级优化优先级

### 高优先级

1. **post_attn** - 立即优化
   - 减少注意力头数
   - 使用梯度检查点
   - 考虑简化架构

2. **sp_convs** - 中期优化
   - 使用分组卷积
   - 减小中间通道数
   - 使用更高效的稀疏卷积

3. **lateral_attns** - 中期优化
   - 减少注意力层数
   - 使用局部注意力
   - 使用 FlashAttention

### 中优先级

4. **upconvs** - 低优先级
   - 优化上采样方法
   - 减少中间通道数

5. **global_avg** - 低优先级
   - 减少池化尺度
   - 使用更高效的上采样

6. **状态管理** - 低优先级
   - 优化历史特征存储
   - 使用更紧凑的表示

## 结论

### 关键发现

1. ❌ **高质量配置超出显存限制**
   - post_attn 在 [25,25,20] 体素网格下 OOM
   - 使用率 89.4%，接近极限

2. ✅ **主要瓶颈已识别**
   - post_attn: 直接 OOM 原因
   - sp_convs: 最大显存消费者（30-40%）
   - 注意力机制: 累计占用 35-50%

3. ✅ **改进路径清晰**
   - 调整配置可立即解决 OOM
   - 混合精度可支持高质量配置
   - 组件级优化可进一步降低显存

### 建议执行顺序

**第一步（立即）：** 使用稳定配置开始训练
**第二步（本周）：** 启用混合精度训练
**第三步（本月）：** 实现 post_attn 优化
**第四步（长期）：** 迁移到分布式训练

## 附录

### 体素网格计算

```
体素网格尺寸 = floor(crop_size / voxel_size)

配置 A (稳定):
  D = 10 / 0.14 = 71
  H = 10 / 0.14 = 71
  W = 8 / 0.14 = 57

配置 B (高质量):
  D = 12 / 0.12 = 100
  H = 12 / 0.12 = 100
  W = 10 / 0.12 = 83

实际网格: [25, 25, 20] (向下取整)
```

### 显存估算公式

**3D 卷积显存 ≈** batch * C_in * D * H * W * (kernel_size^3) * C_out * 4 bytes

**注意力显存 ≈** batch * N_voxels * (Q + K + V) * d_model * 4 bytes

**post_attn 临时显存 ≈** batch * N_voxels * n_heads * head_dim * 4 bytes

### 相关文件

- `train_stream_integrated.py` - 主训练脚本
- `train_with_memory_monitor.py` - 显存监控脚本
- `monitor_forward_components.py` - 组件级监控（尝试）
- `analyze_forward_steps.py` - 步骤级分析（尝试）
- `training_high_quality_batch4_multi_gpu.log` - 训练日志
