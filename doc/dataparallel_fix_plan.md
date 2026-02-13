# DataParallel修复计划

## 问题描述

### 当前问题
1. **Batch Size异常**: 训练过程中batch size在某些情况下会变成1，导致BatchNorm报错 "Expected more than 1 value per channel when training, got input size torch.Size([1, 512])"
2. **DataParallel未生效**: GPU 1显存占用为0MB，所有计算集中在GPU 0上
3. **状态共享问题**: 历史状态（`historical_state`, `historical_pose`, `historical_intrinsics`）是实例变量，在batch维度上共享，导致不同batch sample的状态互相干扰

### 根本原因
1. `MultiGPUStreamTrainer`的`forward_sequence`只保留了GPU 0的状态：`merged_states = gpu_states[0]`
2. 模型的状态存储为实例变量，不支持batch-wise状态管理
3. `forward_sequence`在序列迭代时，状态在batch维度上共享，而不是独立管理

## 修复目标

### 主要目标
1. **实现真正的DataParallel**: 让两个GPU都参与计算，平均分配显存和计算负载
2. **Batch-wise状态管理**: 每个batch sample有独立的历史状态，互相不干扰
3. **保持Batch Size=4**: 在双GPU配置下，每个GPU处理batch size=2，总batch size=4

### 次要目标
1. 保持流式训练的序列依赖性
2. 保持现有API兼容性
3. 不引入额外的显存开销

## 修复方案

### 方案概述
采用**Batch-wise状态管理** + **DataParallel包装**的方案：

1. 将状态从实例变量改为张量存储，支持batch维度
2. 在forward_sequence中正确处理batch维度
3. 使用标准DataParallel自动分发batch到不同GPU
4. 修复MultiGPUStreamTrainer，使其成为可选方案

### 详细设计

#### Phase 1: 重构状态管理

**修改文件**: `former3d/stream_sdfformer_integrated.py`

**当前状态存储**:
```python
# 实例变量，不支持batch-wise
self.historical_state = None
self.historical_pose = None
self.historical_intrinsics = None
```

**目标状态存储**:
```python
# 注册为buffer，支持batch-wise
self.register_buffer('historical_state', None)  # List[Dict] or Dict[str, torch.Tensor]
self.register_buffer('historical_pose', None)    # [B, 4, 4]
self.register_buffer('historical_intrinsics', None)  # [B, 3, 3]
```

**关键变更**:
1. 状态存储从实例变量改为buffer
2. 状态支持batch维度
3. 提供`reset_batch_states(indices)`方法，只重置指定batch sample的状态

#### Phase 2: 修改forward_sequence

**修改文件**: `former3d/stream_sdfformer_integrated.py`

**当前逻辑**:
```python
def forward_sequence(self, images, poses, intrinsics, reset_state=True):
    # 重置所有batch的状态
    if reset_state:
        self.historical_state = None
        self.historical_pose = None
        self.historical_intrinsics = None

    # 遍历序列
    for t in range(n_view):
        # 所有batch共享状态
        output_t, state_t = self.forward_single_frame(...)
        outputs.append(output_t)
```

**目标逻辑**:
```python
def forward_sequence(self, images, poses, intrinsics, reset_state=True):
    batch_size = images.shape[0]
    n_view = images.shape[1]

    # 初始化batch-wise状态
    if reset_state or self.historical_state is None:
        self._init_batch_states(batch_size)

    # 遍历序列
    for t in range(n_view):
        # 提取第t帧（保持batch维度）
        images_t = images[:, t:t+1]  # (batch, 1, 3, H, W)
        poses_t = poses[:, t:t+1]    # (batch, 1, 4, 4)
        intrinsics_t = intrinsics[:, t:t+1]  # (batch, 1, 3, 3)

        # 处理整个batch，每个sample使用自己的状态
        output_t, state_t = self.forward_single_frame_batch(
            images_t, poses_t, intrinsics_t
        )
        outputs.append(output_t)
```

**关键变更**:
1. `forward_single_frame`改为`forward_single_frame_batch`
2. 状态提取和更新支持batch维度
3. 避免序列迭代时的状态混乱

#### Phase 3: 修改forward_single_frame

**修改文件**: `former3d/stream_sdfformer_integrated.py`

**当前逻辑**:
```python
def forward_single_frame(self, images, poses, intrinsics, reset_state=False):
    # 所有batch共享状态
    if reset_state:
        self.historical_state = None
        self.historical_pose = None

    # 提取历史特征（整个batch共享）
    if self.historical_state is not None:
        historical_features = self.extract_historical_features(
            self.historical_state, poses
        )

    # 更新状态（覆盖整个batch）
    new_state = self._create_new_state(output, poses, current_voxel_indices)
    self.historical_state = new_state
    self.historical_pose = poses.detach().clone()
```

**目标逻辑**:
```python
def forward_single_frame_batch(self, images, poses, intrinsics):
    batch_size = images.shape[0]

    # 提取每个batch的历史特征
    historical_features_list = []
    for b in range(batch_size):
        if self.historical_state[b] is not None:
            feat = self.extract_historical_features(
                self.historical_state[b],
                poses[b:b+1]  # (1, 4, 4)
            )
            historical_features_list.append(feat)
        else:
            historical_features_list.append(None)

    # 聚合历史特征
    historical_features = self._aggregate_batch_features(
        historical_features_list, poses
    )

    # 更新每个batch的状态
    new_states = []
    for b in range(batch_size):
        new_state = self._create_new_state_for_batch(
            output[b], poses[b:b+1], current_voxel_indices[b]
        )
        new_states.append(new_state)
        self.historical_state[b] = new_state
        self.historical_pose[b] = poses[b].detach()
```

**关键变更**:
1. 循环处理每个batch sample
2. 独立提取和更新每个sample的状态
3. 避免状态共享导致的混乱

#### Phase 4: 修改特征聚合

**新增方法**: `_aggregate_batch_features`

```python
def _aggregate_batch_features(self,
                               historical_features_list: List[Optional[Dict]],
                               poses: torch.Tensor) -> Optional[Dict]:
    """
    聚合batch的历史特征

    Args:
        historical_features_list: 每个batch的历史特征列表
        poses: 当前位姿 (B, 4, 4)

    Returns:
        聚合后的历史特征字典
    """
    # 如果所有batch都没有历史特征，返回None
    if all(f is None for f in historical_features_list):
        return None

    # 聚合各batch的特征
    aggregated = {}
    for key in ['dense_grids', 'sparse_indices', 'spatial_shapes', 'resolutions']:
        values = [f[key] for f in historical_features_list if f is not None]
        if values:
            # 沿batch维度拼接
            aggregated[key] = torch.cat(values, dim=0)
        else:
            aggregated[key] = None

    return aggregated
```

#### Phase 5: 修改训练脚本

**修改文件**: `train_stream_integrated.py`

**当前逻辑**:
```python
if len(gpu_ids) > 1:
    logger.info(f"启用多GPU流式训练，使用GPU: {gpu_ids}")
    try:
        from multi_gpu_stream_trainer import MultiGPUStreamTrainer
        model = MultiGPUStreamTrainer(model, gpu_ids)
    except ImportError:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)

# 在训练循环中
if hasattr(model, 'module'):
    outputs, states = model.module.forward_sequence(...)
else:
    outputs, states = model.forward_sequence(...)
```

**目标逻辑**:
```python
if len(gpu_ids) > 1:
    logger.info(f"启用多GPU流式训练，使用GPU: {gpu_ids}")
    # 直接使用DataParallel，自动分发batch
    model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    logger.info(f"✅ 模型已包装为DataParallel")

# 在训练循环中
# 直接调用model(images)，DataParallel会自动分发
# 前向传播
outputs = model(images, poses, intrinsics)
# DataParallel会自动处理batch分发和结果合并
```

**关键变更**:
1. 移除MultiGPUStreamTrainer的依赖
2. 使用标准DataParallel
3. 修改forward接口，使其兼容DataParallel

#### Phase 6: 测试和验证

**测试用例**:
1. 单GPU，batch_size=4
2. 双GPU，batch_size=4（每个GPU=2）
3. 双GPU，batch_size=8（每个GPU=4）

**验证指标**:
1. GPU显存占用：两个GPU都应占用显存
2. Batch Norm错误：不应出现batch size=1的错误
3. 训练收敛性：损失应正常下降
4. 序列一致性：流式训练应保持序列依赖性

## 实施步骤

### Step 1: 创建测试用例
- 文件: `test/test_dataparallel_batch_states.py`
- 测试batch-wise状态管理
- 测试DataParallel包装

### Step 2: 实现状态管理重构
- 修改`stream_sdfformer_integrated.py`
- 实现batch-wise状态存储
- 实现`_init_batch_states`

### Step 3: 实现forward_sequence修改
- 修改`forward_sequence`
- 实现`forward_single_frame_batch`
- 实现`_aggregate_batch_features`

### Step 4: 修改训练脚本
- 修改`train_stream_integrated.py`
- 使用标准DataParallel

### Step 5: 测试和调试
- 运行测试用例
- 调试显存和batch size问题
- 验证训练正确性

### Step 6: 文档和提交
- 更新文档
- 提交代码

## 风险评估

### 高风险
1. **状态同步问题**: batch-wise状态可能导致GPU之间的状态不一致
2. **显存开销**: batch-wise状态可能增加显存占用
3. **向后兼容性**: API变更可能影响现有代码

### 缓解措施
1. 实现状态检查点机制
2. 监控显存使用情况
3. 保持向后兼容的API

## 预期效果

### 显存分配（双GPU，batch_size=4）
| GPU | 当前 | 目标 |
|-----|------|------|
| GPU 0 | 8793 MB (89%) | ~5000 MB (50%) |
| GPU 1 | 0 MB (0%) | ~5000 MB (50%) |

### 训练稳定性
- 消除BatchNorm batch size=1错误
- 保持序列依赖性
- 支持更大batch size

### 性能
- 吞吐量提升约2倍（双GPU并行）
- 训练时间减半

## 时间估算
- Phase 1: 2小时
- Phase 2: 3小时
- Phase 3: 4小时
- Phase 4: 2小时
- Phase 5: 2小时
- Phase 6: 3小时

**总计**: ~16小时（2个工作日）

## 参考资料
- PyTorch DataParallel文档
- Batch-wise状态管理最佳实践
- SparseConv3D批处理规范
