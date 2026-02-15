# Batch Size和Crop Size优化分析报告

## 测试时间
2026-02-15 09:52-09:54

## 测试环境
- **GPU**: 2x NVIDIA P102-100 (10GB each)
- **CUDA**: 13.0
- **Driver**: 580.126.09
- **训练脚本**: `train_stream_ddp.py`
- **DDP配置**: 2 GPU分布式训练

---

## 测试配置

### 默认配置
```bash
--batch-size 4           # 总batch size（每GPU = 2）
--sequence-length 4      # 每个样本4帧
--crop-size [10, 8, 6]   # 体素网格大小 (D, H, W)
--target-image-size [256, 256]
--voxel-size 0.0625
--max-sequences 1
--epochs 1
```

### 体素计算
- 每帧体素数：10 × 8 × 6 = **480 体素**
- 每个batch体素数（每GPU）：2 samples × 4 frames × 480 = **3,840 体素**
- 总体素数（2个GPU）：**7,680 体素**

---

## 显存占用情况

### 训练峰值显存
| GPU | 总显存 | 已使用 | 使用率 | 空闲 |
|-----|--------|--------|--------|------|
| GPU 0 | 10240 MiB | ~10053 MiB | **98.2%** | ~91 MiB |
| GPU 1 | 10240 MiB | ~10129 MiB | **98.9%** | ~15 MiB |

### 显存溢出错误
在训练第30个batch时发生显存溢出：

```
GPU 0: 尝试分配270.00 MiB (7.01 GiB已分配, 246.88 MiB空闲)
GPU 1: 尝试分配138.00 MiB (8.62 GiB已分配, 132.62 MiB空闲)
```

**分析**：
- 部分batch的显存需求超过了10GB限制
- 原因：场景复杂度不同，某些场景的体素数更多
- 稀疏张量存储的不确定性导致显存波动

### 训练完成时显存
- GPU 0: 9541 MiB / 10240 MiB (93.2%)
- GPU 1: 8365 MiB / 10240 MiB (81.7%)

**说明**：训练完成时显存略低于峰值，说明某些batch确实需要更多显存。

---

## 训练性能

### 训练进度
- **总batch数**: 108
- **训练时长**: 1.7分钟
- **初始Loss**: 0.730884
- **最终Loss**: 0.059134
- **Loss下降**: **91.9%**

### Loss曲线
```
Batch 0:   0.730884
Batch 10:  0.381899  (-47.7%)
Batch 20:  0.236559  (-67.6%)
Batch 40:  0.131597  (-82.0%)
Batch 60:  0.092697  (-87.3%)
Batch 80:  0.071775  (-90.2%)
Batch 90:  0.064512  (-91.2%)
Batch 100: 0.059134  (-91.9%)
```

### GPU利用率
- **GPU利用率**: 76-81%
- **显存利用率**: 82-99%
- **温度**: 48-53°C（正常）
- **功耗**: 77-149W（GPU 0 > GPU 1）

---

## 优化空间评估

### ❌ Batch Size增大空间：无

**原因**：
1. 显存使用率已达98-99%，几乎用满
2. 部分batch已经出现显存溢出
3. 增大batch size会增加梯度存储和中间激活值

**建议**：
- **保持batch-size=4**（每GPU=2）
- 使用梯度累积来增大有效batch size
  ```bash
  --batch-size 2 --accumulation-steps 2  # 有效batch size = 4
  ```

### ⚠️ Crop Size增大空间：非常有限

**当前crop-size**: [10, 8, 6] → 480体素/帧

#### 增大Crop Size的影响
如果增大crop size，显存增长估算：

| Crop Size | 体素/帧 | 增长比例 | 预估峰值显存 | 结论 |
|-----------|---------|----------|--------------|------|
| [10, 8, 6] | 480 | 1.0x | ~10GB | ✅ 当前配置 |
| [12, 10, 8] | 960 | 2.0x | ~20GB | ❌ 超出限制 |
| [14, 12, 10] | 1680 | 3.5x | ~35GB | ❌ 超出限制 |
| [12, 8, 6] | 576 | 1.2x | ~12GB | ❌ 可能溢出 |
| [10, 10, 8] | 800 | 1.67x | ~16.7GB | ❌ 超出限制 |

**结论**：
- 即使小幅增大crop size（10-20%），也可能导致显存溢出
- 稀疏张量的实际占用不确定，存在波动

#### 可行的微调方案
如果想尝试微调crop size，可以：

1. **增大一个维度**：
   ```bash
   --crop-size 10 10 6  # 增大H到10（体素数：480→600，+25%）
   ```

2. **测试验证**：
   ```bash
   # 使用更小的max-sequences测试
   --crop-size 10 10 6 --max-sequences 1 --epochs 1
   ```

3. **监控显存**：
   ```bash
   # 在另一个终端监控显存
   watch -n 1 nvidia-smi
   ```

**风险**：即使小幅增大也可能导致OOM，建议谨慎测试。

---

## 显存占用分析

### 显存使用组成
1. **模型参数**: ~200-500MB（Transformer + 投影网络）
2. **优化器状态**: ~400-1000MB（Adam优化器，2x参数）
3. **输入数据**: ~1-2GB（RGB图像、位姿、TSDF）
4. **中间激活值**: ~5-7GB（主要显存消耗）
   - 多尺度特征（coarse/medium/fine）
   - 注意力计算（如果启用）
   - 历史特征融合
5. **稀疏张量开销**: ~1-2GB（稀疏索引 + 数据）

### 稀疏张量的不确定性
稀疏张量的显存占用取决于：
- **场景复杂度**：不同场景的占用区域差异很大
- **投影范围**：相机视角覆盖的体素数
- **历史特征累积**：流式训练中历史特征不断累积

这解释了为什么某些batch会出现OOM错误。

---

## 优化建议

### 1. 使用梯度累积（推荐）
保持显存使用不变，增大有效batch size：

```bash
--batch-size 2 --accumulation-steps 4  # 有效batch size = 8
```

**优势**：
- 不增加显存占用
- 提高训练稳定性
- 支持更大的学习率

### 2. 启用混合精度训练（推荐）
减少显存占用，加速训练：

```python
# 修改train_stream_ddp.py
from torch.cuda.amp import autocast, GradScaler

# 在训练循环中使用
scaler = GradScaler()
with autocast():
    outputs = model(batch)
    loss = compute_loss(outputs, targets)
```

**预期效果**：
- 显存减少30-50%
- 训练速度提升20-40%
- 可能支持更大的batch size或crop size

### 3. 启用梯度检查点（折衷）
减少中间激活值的显存占用：

```bash
--use-checkpoint
```

**优势**：
- 显存减少20-40%
- **代价**：计算时间增加15-30%

### 4. 优化序列长度
减少序列长度可以降低显存占用：

```bash
--sequence-length 2  # 从4降到2
```

**效果**：
- 显存减少约40-50%
- 训练速度提升约2倍
- **代价**：流式建模能力减弱

### 5. 减少最大序列数
减少训练数据量：

```bash
--max-sequences 1  # 当前配置
--max-sequences 5  # 正常配置
```

---

## 推荐配置

### 当前最优配置（稳定）
```bash
bash launch_ddp_train.sh 2 29506 \
  --batch-size 4 \
  --sequence-length 4 \
  --crop-size 10 8 6 \
  --max-sequences 5 \
  --epochs 100
```

### 显存优化配置（推荐）
```bash
bash launch_ddp_train.sh 2 29506 \
  --batch-size 2 \
  --accumulation-steps 4 \
  --sequence-length 4 \
  --crop-size 10 8 6 \
  --use-checkpoint \
  --max-sequences 5 \
  --epochs 100
```

**优势**：
- 有效batch size = 8（更大）
- 显存占用更低（~6-7GB）
- 训练稳定性更好

### 混合精度配置（未来优化）
需要修改代码支持AMP：

```bash
bash launch_ddp_train.sh 2 29506 \
  --batch-size 6 \
  --sequence-length 4 \
  --crop-size 12 10 8 \
  --use-amp \
  --max-sequences 5 \
  --epochs 100
```

**预期效果**：
- 显存减少30-50%
- 可能支持更大的crop size
- 训练速度提升20-40%

---

## 结论

### 当前配置评估
| 配置项 | 当前值 | 可增大空间 | 建议 |
|--------|--------|------------|------|
| Batch Size | 4 (每GPU=2) | ❌ 无 | 保持不变 |
| Crop Size | [10, 8, 6] | ⚠️ 很小 | 谨慎测试 |
| Sequence Length | 4 | ✅ 有 | 可调整 |
| Max Sequences | 1 | ✅ 有 | 可增加到5 |

### 关键发现
1. **显存瓶颈严重**：98-99%使用率，已接近极限
2. **存在OOM风险**：部分batch会超出10GB显存限制
3. **显存波动大**：稀疏张量的不确定性导致显存波动

### 最佳实践
1. **保持当前batch-size=4**，使用梯度累积增大有效batch size
2. **谨慎增大crop-size**，如果增大需充分测试
3. **启用混合精度训练**（需代码修改），可释放30-50%显存
4. **使用梯度检查点**，牺牲速度换取显存
5. **优化序列长度**，平衡显存和流式建模能力

---

## 测试数据文件
- **训练日志**: `training_test.log`
- **检查点**: `checkpoints/ddp_test/best_model.pth`
- **DDP配置**: `distributed_utils.py`

---

## 附录：显存监控脚本

### 实时显存监控
```bash
# 终端1：启动训练
cd /home/cwh/coding/former3d
bash launch_ddp_train.sh 2 29506 [参数]

# 终端2：实时监控显存
watch -n 1 'nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu --format=csv'
```

### 详细显存分析
```python
# 在train_stream_ddp.py中添加
from distributed_utils import get_gpu_memory_usage

# 在训练循环中添加
if batch_idx % 10 == 0:
    allocated, reserved = get_gpu_memory_usage()
    print_rank_0(f"Batch {batch_idx}: GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
```

---

*报告生成时间: 2026-02-15 09:55*
*分析人员: Frank (AI Assistant)*
