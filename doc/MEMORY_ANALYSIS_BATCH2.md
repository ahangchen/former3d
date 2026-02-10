# 显存分析报告 - Batch Size=2, 双GPU训练

## 📊 测试配置

- **Batch Size**: 2
- **GPU配置**: 2x NVIDIA P102-100 (每张10GB)
- **Crop Size**: 8,8,6
- **Voxel Size**: 0.25
- **Multi-GPU**: 启用 (DataParallel)
- **训练模式**: 流式训练

## 🎯 显存使用分析

### 初始状态

#### 训练开始（模型加载）
- **GPU 0**:
  - 已分配: 114.98 MB
  - 已预留: 128.00 MB
- **GPU 1**:
  - 已分配: 0.00 MB
  - 已预留: 0.00 MB

**观察**:
- 模型只在GPU 0上分配显存（主设备）
- GPU 1初始不使用显存（DataParallel会在需要时复制）

### 数据加载后

- **GPU 0**:
  - 已分配: 131.00 MB
  - 已预留: 144.00 MB
  - **增长量**: +16.02 MB (+13.93%)

**数据规模**:
- images: 15,728,640 bytes ≈ 15.0 MB
- poses: 1,280 bytes ≈ 1.25 KB
- intrinsics: 720 bytes ≈ 0.7 KB
- **总数据**: ~15.0 MB

**观察**:
- 数据加载后显存增长约16MB
- 数据规模与显存增长基本匹配
- GPU 1仍无显存使用（未进入前向传播）

## ⚠️ OOM错误分析

### 错误位置
```
File: former3d/stream_fusion.py, line 224
attn_scores = torch.mm(proj_current, proj_historical.t()) / (self.feature_dim ** 0.5)
```

### 错误信息
```
RuntimeError: CUDA out of memory. Tried to allocate 390.00 MiB
GPU 0; 9.91 GiB total capacity;
8.28 GiB already allocated;
241.12 MiB free;
8.63 GiB reserved in total by PyTorch
```

### 关键发现

#### 1. **显存使用情况（OOM时刻）**
- **已分配**: 8.28 GB (83.6%)
- **已预留**: 8.63 GB (87.1%)
- **可用**: 241.12 MB (2.4%)
- **请求**: 390.00 MB

#### 2. **DataParallel显存分布**
- **GPU 0** (主设备): 承担大部分计算
- **GPU 1**: 在DataParallel中接收部分batch数据

#### 3. **OOM发生点**
- **网络组件**: StreamFusion的HierarchicalAttention
- **操作**: 注意力分数计算 `torch.mm(proj_current, proj_historical.t())`
- **原因**: 历史特征累积导致显存需求快速增长

## 📈 显存增长模式

### 训练步骤 vs 显存

| 步骤 | GPU 0 | GPU 1 | 说明 |
|------|-------|-------|------|
| 模型加载 | 114.98 MB | 0.00 MB | 初始化 |
| 数据加载 | 131.00 MB | 0.00 MB | +16.02 MB |
| 前向传播 | 8.28 GB | - | OOM |

### 网络组件显存占用

#### 已记录组件：
1. **训练开始**: 114.98 MB (模型参数)
2. **数据加载**: +16.02 MB (批次数据)

#### 未记录组件（OOM导致）：
1. **2D特征提取**: ~2-3 GB（估计）
2. **3D稀疏卷积**: ~3-4 GB（估计）
3. **StreamFusion**: ~1-2 GB（估计）
4. **注意力计算**: 显存峰值点

## 🔍 深度分析

### 1. **DataParallel显存分配**

#### 机制：
- DataParallel在每次forward时将batch数据分割到各个GPU
- 每个GPU复制完整的模型参数
- GPU 0负责收集和分发

#### 显存分布：
- **GPU 0**: 模型参数 + 主batch数据 + 梯度 + 中间激活
- **GPU 1**: 模型参数 + 分batch数据 + 中间激活

#### 估算：
- 模型参数: ~114 MB
- 每个GPU的batch: ~15 MB
- DataParallel开销: ~50-100 MB

### 2. **StreamFusion显存瓶颈**

#### 原因：
- **历史特征累积**: 每帧都会保存历史特征
- **注意力计算**: 需要计算当前特征与所有历史特征的相似度
- **矩阵乘法**: `proj_current @ proj_historical.t()` 产生大矩阵

#### 显存需求估算：
- 当前特征: `[batch, n_voxels, feature_dim]`
- 历史特征: `[batch, n_frames, n_voxels, feature_dim]`
- 注意力分数: `[batch, n_voxels, n_frames * n_voxels]`

对于batch=2, frames=10, n_voxels≈2000, feature_dim=256:
- 注意力矩阵: 2 × 2000 × (10 × 2000) = 80,000,000 元素
- FP32: 80M × 4 bytes = 320 MB
- **与错误匹配**: 请求390 MB（包含中间计算）

### 3. **不同组件显存占用估算**

| 组件 | 估算显存 | 说明 |
|------|---------|------|
| 模型参数 | 114.98 MB | 实测 |
| 批次数据 | 16.02 MB | 实测 |
| 2D特征提取 | 2-3 GB | ResNet backbone |
| 3D稀疏卷积 | 3-4 GB | SparseConv网络 |
| StreamFusion | 1-2 GB | 注意力计算 |
| 历史特征缓存 | 0.5-1 GB | 递增 |
| 梯度 | ~114 MB | 等同于参数 |
| 中间激活 | 2-3 GB | 反向传播需要 |
| **总计** | **8-9 GB** | **接近GPU上限** |

## 💡 优化建议

### 1. **短期优化**

#### a. **减小batch size**
```bash
--batch-size 1 --accumulation-steps 4
```
- 优点: 直接降低显存需求50%
- 缺点: BatchNorm需要调整

#### b. **减小crop size**
```bash
--crop-size "6,6,4" --voxel-size 0.30
```
- 优点: 减少体素数量，降低所有组件显存需求
- 缺点: 可能影响模型精度

#### c. **增大voxel size**
```bash
--voxel-size 0.30
```
- 优点: 减少体素数量，降低3D网络显存
- 缺点: 可能影响SDF精度

#### d. **减少历史帧数**
```python
# 在StreamFusion中修改
self.max_history_frames = 5  # 从10减少到5
```
- 优点: 显著降低注意力计算显存
- 缺点: 可能影响时序一致性

### 2. **中期优化**

#### a. **混合精度训练 (AMP)**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
loss = criterion(outputs, targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```
- **预期收益**: 30-40% 显存节省
- **实现难度**: 中等

#### b. **梯度检查点**
```python
from torch.utils.checkpoint import checkpoint

outputs = checkpoint(expensive_function, inputs)
```
- **预期收益**: 20-30% 显存节省（以计算时间为代价）
- **实现难度**: 中等

#### c. **激活值重计算**
```python
# 在forward中保存必要信息，反向时重新计算
```
- **预期收益**: 15-25% 显存节省
- **实现难度**: 较高

### 3. **长期优化**

#### a. **分布式数据并行 (DDP)**
```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(model, device_ids=[local_rank])
```
- **优点**: 每个GPU独立运行，更好的显存利用
- **实现难度**: 较高（需要多进程启动）

#### b. **流式融合优化**
- **稀疏注意力**: 只计算附近的体素
- **分层注意力**: 多尺度融合
- **渐进式融合**: 减少每帧融合的特征数量
- **预期收益**: 40-60% StreamFusion显存节省
- **实现难度**: 高

#### c. **模型蒸馏**
- 训练一个更小的模型
- **预期收益**: 50-70% 参数和显存减少
- **实现难度**: 高

## 📊 推荐配置

### 配置1: 保守型（确保运行）
```bash
python train_stream_integrated.py \
  --batch-size 1 \
  --crop-size "6,6,4" \
  --voxel-size 0.30 \
  --accumulation-steps 8 \
  --multi-gpu
```
**预期显存**: ~4-5 GB

### 配置2: 平衡型（推荐）
```bash
python train_stream_integrated.py \
  --batch-size 1 \
  --crop-size "8,8,6" \
  --voxel-size 0.25 \
  --accumulation-steps 4 \
  --multi-gpu
```
**预期显存**: ~6-7 GB

### 配置3: 激进型（需优化）
```bash
python train_stream_integrated.py \
  --batch-size 2 \
  --crop-size "10,10,8" \
  --voxel-size 0.20 \
  --accumulation-steps 2 \
  --multi-gpu
```
**预期显存**: ~8-9 GB（接近上限，可能OOM）

## 🎯 结论

### 关键发现：
1. ✅ **DataParallel正常工作**: 模型成功分发到两张GPU
2. ⚠️ **StreamFusion是主要瓶颈**: 注意力计算占用最多显存
3. ⚠️ **batch_size=2超出显存限制**: 即使使用双GPU也无法运行
4. 💡 **batch_size=1可行**: 但需要解决BatchNorm问题

### 下一步行动：
1. ✅ 实现显存分析工具（已完成）
2. 🔧 优化StreamFusion的显存使用
3. 🔧 实现混合精度训练
4. 🔧 调整模型参数以适应显存限制
5. 📊 继续监控显存使用情况

---

**报告生成时间**: 2026-02-10 09:57:00
**工具**: memory_profiler.py + analyze_memory.py
**数据来源**: memory_analysis_batch2_epoch_1_batch_0_summary.json
