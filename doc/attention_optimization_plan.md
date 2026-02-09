# 阶段3：注意力计算优化计划

## 📋 目标

通过优化注意力计算减少显存占用，降低流式训练的额外显存开销。

## 🎯 预期效果

| 优化项 | 当前显存 | 优化后显存 | 降低幅度 |
|--------|----------|------------|----------|
| checkpointing | ~8.73 GB | ~6.5 GB | ↓ 25% |
| 分块注意力 | ~6.5 GB | ~5.5 GB | ↓ 15% |

---

## 3.1 Checkpointing优化

### 问题分析
多头注意力计算中需要保存大量的中间变量：
- Q, K, V 投影结果
- 注意力分数矩阵 [N_historical, N_current]
- 注意力权重矩阵
- 上下文聚合结果

这些中间变量在反向传播时需要，导致显存占用高。

### 解决方案
使用 `torch.utils.checkpoint` 只在反向传播时重新计算中间结果，前向传播时不保存。

### 实现步骤

#### Step 1: 分析现有注意力代码
- 检查 `former3d/stream_fusion.py` 中的 `StreamCrossAttention` 类
- 识别需要包装的注意力计算函数

#### Step 2: 创建checkpointing包装函数
```python
from torch.utils.checkpoint import checkpoint

def _compute_attention_checkpointed(self, q, k, v, mask):
    """可被checkpoint的注意力计算函数"""
    # 注意力计算代码（从原forward函数中提取）
    scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v)
    return output
```

#### Step 3: 修改forward方法
```python
def forward(self, current_feats, historical_feats, current_coords, historical_coords):
    # ... 准备Q, K, V ...

    # 使用checkpointing
    output = checkpoint(
        self._compute_attention_checkpointed,
        q, k, v, mask,
        use_reentrant=False
    )

    # ... 后续处理 ...
    return output
```

#### Step 4: 测试数值精度
- 创建测试用例对比checkpointing前后的数值差异
- 确保精度损失在可接受范围内（< 1e-6）

### 测试计划

#### 测试1: 数值精度对比
**文件**: `test/test_checkpointing_precision.py`

**目标**: 验证checkpointing不影响数值精度

**步骤**:
1. 创建相同的输入数据
2. 分别使用原始方法和checkpointing方法计算
3. 对比输出差异（应该 < 1e-6）

#### 测试2: 显存占用对比
**文件**: `test/test_checkpointing_memory.py`

**目标**: 验证checkpointing降低显存占用

**步骤**:
1. 使用原始方法计算，记录峰值显存
2. 使用checkpointing方法计算，记录峰值显存
3. 对比显存降低幅度（应该 > 20%）

---

## 3.2 分块注意力计算

### 问题分析
当历史体素数量很大时（如N=10000），注意力分数矩阵为[10000, 1000]，占用大量显存。

### 解决方案
分块计算注意力，每次只计算一个chunk的注意力分数。

### 实现步骤

#### Step 1: 实现分块计算函数
```python
def _compute_chunked_attention(self, q, k, v, mask, chunk_size=256):
    """分块计算注意力"""
    N_current = q.shape[0]
    N_historical = k.shape[0]

    outputs = []

    for i in range(0, N_current, chunk_size):
        end_i = min(i + chunk_size, N_current)
        chunk_q = q[i:end_i]  # [chunk_size, head_dim]

        chunk_outputs = []
        for j in range(0, N_historical, chunk_size):
            end_j = min(j + chunk_size, N_historical)
            chunk_k = k[j:end_j]  # [chunk_size, head_dim]
            chunk_v = v[j:end_j]  # [chunk_size, head_dim]
            chunk_mask = mask[i:end_i, j:end_j] if mask is not None else None

            # 计算分块注意力
            chunk_output = self._compute_attention(chunk_q, chunk_k, chunk_v, chunk_mask)
            chunk_outputs.append(chunk_output)

        # 合并历史特征分块
        outputs.append(torch.cat(chunk_outputs, dim=1))

    return torch.cat(outputs, dim=0)
```

#### Step 2: 添加chunk_size参数
```python
def __init__(self, ..., chunk_size=256):
    self.chunk_size = chunk_size
```

#### Step 3: 测试性能和显存
- 对比分块计算前后的显存占用
- 对比计算速度（分块可能稍慢）

### 测试计划

#### 测试1: 数值正确性验证
**文件**: `test/test_chunked_attention_correctness.py`

**目标**: 验证分块计算结果与原始方法一致

**步骤**:
1. 创建相同的输入数据
2. 分别使用原始方法和分块方法计算
3. 对比输出差异（应该 < 1e-5，因为浮点运算顺序可能不同）

#### 测试2: 性能对比
**文件**: `test/test_chunked_attention_performance.py`

**目标**: 验证分块计算降低显存并保持合理速度

**步骤**:
1. 使用原始方法计算，记录显存和时间
2. 使用分块方法计算，记录显存和时间
3. 验证显存降低（应该 > 10%）
4. 验证速度损失 < 50%

---

## 🧪 测试用例设计

### 测试用例1: Checkpointing数值精度
```python
def test_checkpointing_precision():
    """测试checkpointing不影响数值精度"""
    # 创建输入
    current_feats = torch.randn(1000, 128)
    historical_feats = torch.randn(5000, 128)
    current_coords = torch.randn(1000, 3)
    historical_coords = torch.randn(5000, 3)

    # 原始计算
    model_original = StreamCrossAttention(feature_dim=128, use_checkpoint=False)
    output_original = model_original(current_feats, historical_feats, current_coords, historical_coords)

    # Checkpointing计算
    model_checkpointed = StreamCrossAttention(feature_dim=128, use_checkpoint=True)
    model_checkpointed.load_state_dict(model_original.state_dict())
    output_checkpointed = model_checkpointed(current_feats, historical_feats, current_coords, historical_coords)

    # 对比差异
    diff = torch.abs(output_original - output_checkpointed).max().item()
    assert diff < 1e-6, f"数值差异过大: {diff}"
```

### 测试用例2: 分块注意力正确性
```python
def test_chunked_attention_correctness():
    """测试分块注意力计算正确"""
    # 创建输入
    current_feats = torch.randn(1000, 128)
    historical_feats = torch.randn(5000, 128)
    current_coords = torch.randn(1000, 3)
    historical_coords = torch.randn(5000, 3)

    # 原始计算
    model_original = StreamCrossAttention(feature_dim=128, chunk_size=None)
    output_original = model_original(current_feats, historical_feats, current_coords, historical_coords)

    # 分块计算
    model_chunked = StreamCrossAttention(feature_dim=128, chunk_size=256)
    model_chunked.load_state_dict(model_original.state_dict())
    output_chunked = model_chunked(current_feats, historical_feats, current_coords, historical_coords)

    # 对比差异
    diff = torch.abs(output_original - output_chunked).max().item()
    assert diff < 1e-5, f"数值差异过大: {diff}"
```

### 测试用例3: 显存占用对比
```python
def test_memory_usage():
    """测试显存占用"""
    # 创建大输入
    current_feats = torch.randn(2000, 128, device='cuda')
    historical_feats = torch.randn(10000, 128, device='cuda')

    # 基线模型
    model_baseline = StreamCrossAttention(feature_dim=128, use_checkpoint=False, chunk_size=None).cuda()

    torch.cuda.reset_peak_memory_stats()
    _ = model_baseline(current_feats, historical_feats, ...)
    memory_baseline = torch.cuda.max_memory_allocated()

    # 优化模型
    model_optimized = StreamCrossAttention(feature_dim=128, use_checkpoint=True, chunk_size=256).cuda()

    torch.cuda.reset_peak_memory_stats()
    _ = model_optimized(current_feats, historical_feats, ...)
    memory_optimized = torch.cuda.max_memory_allocated()

    reduction = (memory_baseline - memory_optimized) / memory_baseline
    assert reduction > 0.25, f"显存降低不足: {reduction:.2%}"
```

---

## 📅 实施时间表

| 任务 | 预计时间 | 优先级 | 风险 |
|------|----------|--------|------|
| Step 1: 分析现有代码 | 30分钟 | 高 | 低 |
| Step 2: 实现checkpointing | 1小时 | 高 | 中 |
| Step 3: 编写测试用例 | 1小时 | 高 | 低 |
| Step 4: 实现分块注意力 | 1小时 | 中 | 低 |
| Step 5: 集成测试 | 1小时 | 中 | 中 |
| **总计** | **4.5小时** | - | - |

---

## ✅ 成功标准

1. ✅ Checkpointing数值精度 < 1e-6
2. ✅ 分块注意力数值精度 < 1e-5
3. ✅ 显存降低 > 25%
4. ✅ 速度损失 < 50%
5. ✅ 所有测试用例通过
6. ✅ 代码已提交到git

---

## 🚨 风险和应对

### 风险1: Checkpointing数值精度问题
**应对**: 如果精度 > 1e-6，调整checkpointing配置或禁用

### 风险2: 分块计算速度损失过大
**应对**: 调整chunk_size参数，找到显存和速度的平衡点

### 风险3: 与现有代码不兼容
**应对**: 保持向后兼容，使用参数控制是否启用新功能

---

**创建时间**: 2026-02-09 19:00
**预计完成时间**: 2026-02-09 23:30
**状态**: 待执行
