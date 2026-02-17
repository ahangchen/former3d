# 流式训练显存分析报告
## 训练配置
- **日期**: 2026-02-12 20:38
- **模型**: Former3D Stream SDFFormer
- **GPU**: 2x NVIDIA P102-100 (10GB each)
- **Batch Size**: 2 per GPU (total=4)
- **Sequence Length**: 10 frames
- **Crop Size**: 10x8x6 (depth×height×width)
- **Lightweight Mode**: Enabled
- **Attention Layers**: 0 (disabled)
- **Fusion Radius**: 0 (disabled)

## 显存占用分析

### GPU 0 (主GPU)
| 阶段 | 已分配(MB) | 已预留(MB) | 占用率 |
|------|-----------|-----------|--------|
| 训练开始 | 78.05 | 80.00 | 0.8% |
| 数据加载 | 5227.24 | 8826.00 | 89.1% |
| 前向传播 | 5639.37 | 8826.00 | 89.1% |
| 损失计算 | 5617.54 | 8826.00 | 89.1% |
| 反向传播 | 5226.97 | 8826.00 | 89.1% |
| **峰值** | **8793.72** | **8826.00** | **89.1%** |

### GPU 1 (从GPU)
| 阶段 | 已分配(MB) | 已预留(MB) |
|------|-----------|-----------|
| 所有阶段 | 0.00 | 0.00 |

### 各帧显存占用
| 帧 | 最小(MB) | 最大(MB) | 平均(MB) |
|----|---------|---------|---------|
| Frame 0 | 4060.08 | 5639.37 | 4849.73 |
| Frame 1 | 4060.08 | 5639.37 | 4849.73 |
| Frame 2 | 4060.08 | 5639.37 | 4849.73 |
| ... | ... | ... | ... |
| Frame 9 | 3192.90 | 5639.37 | 4533.69 |

## 问题诊断

### 1. DataParallel未生效
- GPU 1完全没有被使用（0 MB显存占用）
- 训练代码使用`model.module.forward_sequence`绕过了DataParallel的自动batch分发
- 所有计算都集中在GPU 0上

### 2. 显存接近极限
- GPU 0峰值显存占用：8793.72 MB / 9910 MB ≈ 89.1%
- 在Batch 2时OOM（CUDA out of memory）
- 错误信息："Tried to allocate 2.00 MiB but only 3.12 MiB free"

### 3. 流式训练特性
- 前向传播需要序列性处理10帧
- 每帧都保存历史状态（sparse_indices, dense_grids等）
- Lightweight模式减少了dense_grids的显存占用

## 显存瓶颈分析

### 各部分显存占用估算

| 组件 | 估算显存 | 占比 |
|------|---------|------|
| 模型参数 | ~300 MB | 3.4% |
| 梯度 | ~300 MB | 3.4% |
| 优化器状态 | ~900 MB | 10.2% |
| 图像输入 (batch=2, seq=10) | ~150 MB | 1.7% |
| 2D特征 (MnasMulti) | ~800 MB | 9.1% |
| SparseConv输入 | ~1200 MB | 13.6% |
| 3D特征 (3级分辨率) | ~1500 MB | 17.0% |
| 历史状态 (lightweight) | ~1000 MB | 11.4% |
| 中间激活 | ~2500 MB | 28.4% |
| **总计** | **~8650 MB** | **~98%** |

## 优化建议

### 立即可行的方案

1. **减小Crop Size**
   - 当前：10x8x6 → 480体素
   - 建议：8x6x4 → 192体素
   - 预计节省：~2-3 GB显存

2. **减小Batch Size**
   - 当前：2 → 建议：1
   - 预计节省：~2 GB显存
   - 注意：需要确保LayerNorm1d正常工作

3. **减小Sequence Length**
   - 当前：10 → 建议：5
   - 预计节省：~1.5 GB显存

4. **启用Gradient Checkpointing**
   - 重计算前向传播的中间激活
   - 以计算换显存
   - 预计节省：~1.5 GB显存

### 长期优化方案

1. **使用DistributedDataParallel (DDP)**
   - 替代DataParallel
   - 每个进程使用一张GPU
   - 需要修改训练脚本启动方式

2. **混合精度训练 (AMP)**
   - 启用FP16/FP32混合精度
   - 预计节省：~40-50%显存
   - 需要处理可能的数值精度问题

3. **模型架构优化**
   - 减少3D特征维度（96→64→32）
   - 减少网络深度
   - 使用更轻量的Backbone（如MobileNet）

4. **显存高效的数据结构**
   - 使用更紧凑的数据类型（float16）
   - 实现自定义SparseTensor格式

## 推荐配置

### 配置1：极限显存优化
```bash
BATCH_SIZE=1
SEQUENCE_LENGTH=5
CROP_SIZE="6,6,4"
ATTN_LAYERS=0
FUSION_RADIUS=0
USE_LIGHTWEIGHT=true
```

### 配置2：平衡方案（推荐）
```bash
BATCH_SIZE=1
SEQUENCE_LENGTH=8
CROP_SIZE="8,6,4"
ATTN_LAYERS=0
FUSION_RADIUS=0.5
USE_LIGHTWEIGHT=true
```

### 配置3：DDP方案（需要较大改动）
```bash
# 使用torchrun启动
torchrun --nproc_per_node=2 train_stream_integrated_ddp.py \
    --batch-size 2 \
    --sequence-length 10 \
    --crop-size 10,8,6 \
    --use-ddp
```

## 结论

当前使用DataParallel的双GPU训练配置无法充分利用两张GPU的显存，GPU 1完全空闲。建议：

1. **短期**：使用配置1或2，单GPU训练，减小batch size和crop size
2. **中期**：实现DDP方案，真正利用两张GPU
3. **长期**：优化模型架构和训练流程，降低显存需求

---

**生成时间**: 2026-02-12 20:38
**日志文件**: logs/optimized_2gpu_20260212_203818/train.log
**显存分析**: logs/memory_profile_epoch_1_batch_2_summary.json
