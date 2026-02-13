# DistributedDataParallel (DDP) 实现方案

## 📋 项目概述

**目标**: 使用PyTorch DistributedDataParallel实现单机多卡训练，替代DataParallel方案

**优势**:
- ✅ 更高效的GPU利用率
- ✅ 支持多进程并行，减少Python GIL限制
- ✅ 更好的梯度同步性能
- ✅ 支持更复杂的数据类型（包括SparseConvTensor）
- ✅ 更好的扩展性，易于迁移到多机多卡

---

## 🔧 技术方案

### 1. 分布式环境初始化

使用 `torch.distributed` 初始化分布式环境：
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()
```

### 2. 数据加载器适配

使用 `DistributedSampler` 确保每个进程处理不同的数据子集：
```python
from torch.utils.data.distributed import DistributedSampler

def create_dataloader(dataset, batch_size, num_workers=4):
    """创建分布式数据加载器"""
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader, sampler
```

### 3. 模型包装

使用DDP包装模型：
```python
model = StreamSDFFormerIntegrated(...)
model = model.cuda(local_rank)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

### 4. 训练循环修改

修改训练循环以支持分布式：
- 在每个epoch开始时设置sampler的epoch
- 只在rank 0时保存检查点
- 使用 `dist.all_reduce()` 同步指标

---

## 📝 实现步骤

### Phase 1: 分布式辅助函数
- [ ] 创建 `distributed_utils.py` - 分布式初始化和清理函数
- [ ] 实现分布式数据加载器包装
- [ ] 实现分布式指标同步函数

### Phase 2: 训练脚本改造
- [ ] 创建 `train_stream_ddp.py` - DDP版本训练脚本
- [ ] 修改训练循环以支持DDP
- [ ] 实现分布式检查点保存/加载

### Phase 3: 测试验证
- [ ] 单GPU测试
- [ ] 双GPU测试
- [ ] 验证训练稳定性
- [ ] 对比DDP vs DataParallel性能

---

## 🎯 关键改进点

### 1. 更好的多GPU利用率
```
DataParallel:
- 单进程多线程
- 受Python GIL限制
- GPU间通信效率较低

DistributedDataParallel:
- 多进程并行
- 每个进程独立
- 使用高效的NCCL通信
```

### 2. 复杂数据类型支持
```
DataParallel问题:
- SparseConvTensor无法被gather
- 需要自定义gather函数
- 限制返回类型

DDP优势:
- 每个进程独立计算
- 只同步梯度，不同步输出
- 支持任意返回类型
```

### 3. 训练性能
```
DataParallel:
- 每个batch需要CPU-GPU数据传输
- 前向传播后需要gather
- 反向传播需要scatter

DDP:
- 每个进程直接使用本地GPU
- 只在反向传播时同步梯度
- 减少CPU-GPU通信开销
```

---

## 🚀 使用方法

### 启动命令

```bash
# 单机双卡训练
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    train_stream_ddp.py \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --epochs 10
```

### 环境变量

DDP会自动设置以下环境变量：
- `RANK`: 全局进程rank
- `WORLD_SIZE`: 总进程数
- `LOCAL_RANK`: 本地进程rank（用于GPU选择）

---

## 📊 预期效果

| 指标 | DataParallel | DDP | 改进 |
|------|--------------|-----|------|
| 训练速度 | 基准 | 1.5-2x | ⬆️ 50-100% |
| GPU利用率 | 60-70% | 85-95% | ⬆️ 20-30% |
| 内存占用 | 较高 | 较低 | ⬇️ 20-30% |
| 扩展性 | 单机 | 多机 | ✅ |
| 复杂数据类型 | 有限 | 完全支持 | ✅ |

---

## ⚠️ 注意事项

1. **Batch Size**: 每个GPU的batch size = 总batch size / num_gpus
2. **Learning Rate**: 可能需要调整学习率（线性缩放规则）
3. **检查点**: 只在rank 0时保存检查点
4. **日志**: 只在rank 0时打印日志
5. **评估**: 评估时禁用sampler的shuffle

---

## 🔗 参考资料

- PyTorch DDP官方文档: https://pytorch.org/docs/stable/ddp.html
- DDP最佳实践: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html