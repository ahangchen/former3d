# CUDA OOM 问题分析

## 问题描述

在训练过程中出现 CUDA 显存不足错误：

```
RuntimeError: CUDA out of memory. Tried to allocate 6.66 GiB
Error location: torch.cdist(cur_points_norm, transformed_points, p=2)
```

## 问题原因

### 根本原因
在 `_historical_state_project_sparse` 函数中，使用 `torch.cdist` 计算当前点和历史点的距离矩阵：

```python
dists = torch.cdist(cur_points_norm, transformed_points, p=2)  # [N_cur, N_hist]
```

当稀疏点数量很大时：
- 当前点数 N_cur = 30,283
- 历史点数 N_hist = 60,016
- 距离矩阵大小 = 30,283 × 60,016 × 4 bytes ≈ **7.3 GB**

### 触发条件
1. **场景空间大**：某些场景的稀疏点数量特别多
2. **历史积累**：随着训练进行，历史帧的稀疏点可能累积
3. **Batch处理**：DDP训练时每个GPU都需要分配完整距离矩阵

## 可能的解决方案

### 方案A：分批计算距离（推荐）

**思路**：将距离矩阵计算分成多个小块，避免一次性分配大矩阵

```python
def compute_nearest_neighbor_batched(cur_points, hist_points, batch_size=1000):
    """分批计算最近邻，避免OOM"""
    N_cur = cur_points.shape[0]
    N_hist = hist_points.shape[0]
    device = cur_points.device

    nearest_indices = torch.zeros(N_cur, dtype=torch.long, device=device)
    nearest_dists = torch.zeros(N_cur, device=device)

    # 分批处理当前点
    for i in range(0, N_cur, batch_size):
        end_i = min(i + batch_size, N_cur)
        cur_batch = cur_points[i:end_i]  # [batch_size, 3]

        # 计算当前batch与所有历史点的距离
        dists_batch = torch.cdist(cur_batch, hist_points, p=2)  # [batch_size, N_hist]

        # 找到最近邻
        nearest_dists_batch, nearest_indices_batch = torch.min(dists_batch, dim=1)

        nearest_indices[i:end_i] = nearest_indices_batch
        nearest_dists[i:end_i] = nearest_dists_batch

    return nearest_indices, nearest_dists
```

**优点**：
- 内存占用可控：`batch_size × N_hist`
- 训练不会OOM崩溃
- 精度完全一致

**缺点**：
- 计算时间略有增加（分批循环）

### 方案B：使用KD树或Ball Tree

**思路**：使用scikit-learn的KD树进行快速近邻搜索

```python
from sklearn.neighbors import NearestNeighbors

# 在CPU上构建KD树
nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(hist_points.cpu().numpy())
distances, indices = nbrs.kneighbors(cur_points.cpu().numpy())

# 转回GPU
nearest_indices = torch.from_numpy(indices).long().to(device)
nearest_dists = torch.from_numpy(distances).to(device)
```

**优点**：
- 内存效率极高
- 计算速度快（O(log N) vs O(N)）

**缺点**：
- 需要CPU-GPU数据传输
- 依赖scikit-learn

### 方案C：采样历史点

**思路**：当历史点数过多时，随机采样一部分

```python
max_hist_points = 20000
if hist_points.shape[0] > max_hist_points:
    indices = torch.randperm(hist_points.shape[0])[:max_hist_points]
    hist_points = hist_points[indices]
    hist_features = hist_features[indices]
    hist_logits = hist_logits[indices]
```

**优点**：
- 简单直接
- 内存可控

**缺点**：
- 可能丢失重要历史信息
- 融合精度下降

### 方案D：使用CUDA优化的近邻库

**思路**：使用 FAISS 或其他GPU优化的近邻搜索库

```python
import faiss

# 构建GPU索引
res = faiss.StandardGpuResources()
index = faiss.IndexFlatL2(3)  # 3D坐标
index = faiss.index_cpu_to_gpu(res, 0, index)
index.add(hist_points)

# 搜索最近邻
dists, indices = index.search(cur_points, k=1)
```

**优点**：
- GPU加速，速度快
- 内存效率高
- 工业级解决方案

**缺点**：
- 需要安装FAISS
- 增加依赖

## 推荐实施方案

**优先级排序**：

1. **短期（立即实施）**：方案A（分批计算）
   - 简单可靠，无需额外依赖
   - 可以立即解决OOM问题

2. **中期（性能优化）**：方案D（FAISS）
   - 如果分批计算影响性能，再升级到FAISS

3. **长期（架构优化）**：结合方案C（智能采样）
   - 基于场景复杂度自适应采样历史点

## 实施计划

1. 修改 `_historical_state_project_sparse` 函数
2. 实现分批距离计算
3. 测试验证精度不变
4. 提交代码
5. 重新运行完整训练

---
生成时间：2026-02-20
