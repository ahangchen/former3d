# StreamSDFFormer 代码重构计划

## 📋 概述

本计划旨在修复代码中的三个核心问题：
1. MultiSequenceTartanAirDataset的数据shape问题
2. StreamSDFFormerIntegrated的历史特征创建逻辑错误
3. 训练循环的低效batch处理

**原则**：不新增任何模型、数据集或训练循环代码，直接在现有代码上修改。

---

## 🔴 问题1：MultiSequenceTartanAirDataset的shape问题

### 问题描述

当前实现的数据shape不符合PyTorch的自动batch组包机制。

**问题根源**：
- 图像和pose的shape没有正确设计
- 导致torch无法正确组batch
- 训练循环不得不手动遍历batch_idx

### 修复目标

**正确的数据shape设计**：

| 字段 | 当前shape | 目标shape | 组batch后 |
|------|----------|----------|-----------|
| 图像 | `(n_view, 3, h, w)` | `(1, n_view, 3, h, w)` | `(batch, n_view, 3, h, w)` |
| pose | `(n_view, 4, 4)` | `(1, n_view, 4, 4)` | `(batch, n_view, 4, 4)` |
| 内参 | `(n_view, 3, 3)` | `(1, n_view, 3, 3)` | `(batch, n_view, 3, 3)` |
| tsdf | `(n_voxel, 1)` | `(1, n_voxel, 1)` | `(batch, n_voxel, 1)` |
| occ | `(n_voxel, 1)` | `(1, n_voxel, 1)` | `(batch, n_voxel, 1)` |

**关键点**：
- `n_view`：序列片段长度
- `batch`：PyTorch自动添加的批次维度
- 所有张量都需要在最前面添加一个维度`1`

### 修复步骤

#### 步骤1.1：修改MultiSequenceTartanAirDataset的__getitem__

**文件**：`former3d/dataset/multi_sequence_tartanair.py`

**需要修改的字段**：

```python
def __getitem__(self, idx):
    # 原有逻辑...

    # 修改图像shape
    # images: (n_view, 3, h, w) -> (1, n_view, 3, h, w)
    images = images.unsqueeze(0)

    # 修改pose shape
    # poses: (n_view, 4, 4) -> (1, n_view, 4, 4)
    poses = poses.unsqueeze(0)

    # 修改内参shape
    # intrinsics: (n_view, 3, 3) -> (1, n_view, 3, 3)
    intrinsics = intrinsics.unsqueeze(0)

    # 修改tsdf shape（如果存在）
    if 'tsdf' in data:
        tsdf = tsdf.unsqueeze(0)  # (1, n_voxel, 1)

    # 修改occ shape（如果存在）
    if 'occ' in data:
        occ = occ.unsqueeze(0)  # (1, n_voxel, 1)

    return {
        'images': images,
        'poses': poses,
        'intrinsics': intrinsics,
        'tsdf': tsdf,
        'occ': occ,
        # ... 其他字段
    }
```

#### 步骤1.2：修改collate_fn（如果有）

**文件**：`former3d/dataset/multi_sequence_tartanair.py`

如果存在自定义的`collate_fn`，需要确保正确处理添加的维度。

**预期行为**：
- PyTorch的默认`collate_fn`会自动在第0维度进行stack
- 从`(1, ...)`变为`(batch, ...)`

#### 步骤1.3：验证DataLoader输出

**测试代码**（新建测试脚本）：

```python
# test/test_dataset_shape.py
from torch.utils.data import DataLoader
from former3d.dataset.multi_sequence_tartanair import MultiSequenceTartanAirDataset

dataset = MultiSequenceTartanAirDataset(...)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

for batch in dataloader:
    print("Images shape:", batch['images'].shape)  # 应该是 (batch, n_view, 3, h, w)
    print("Poses shape:", batch['poses'].shape)    # 应该是 (batch, n_view, 4, 4)
    print("Intrinsics shape:", batch['intrinsics'].shape)  # 应该是 (batch, n_view, 3, 3)
    break
```

---

## 🔴 问题2：StreamSDFFormerIntegrated的历史特征创建逻辑错误

### 问题描述

当前的`_create_new_state`函数是一个完全不正确的简化版本：
- 历史特征全都是随机的
- 跟网络推理结果根本没关系
- 没有正确实现流式融合的核心逻辑

### 修复目标

实现正确的历史特征创建逻辑：

**核心思想**：
1. 根据历史时刻和当前时刻的pose差异
2. 计算历史多尺度特征和当前多尺度特征的映射关系
3. 使用`grid_sample`将历史特征搬运到当前坐标系
4. 注意：历史特征是**multi scale feature**，不是occ或tsdf结果

### 修复步骤

#### 步骤2.1：理解正确的流式融合逻辑

**文件**：`former3d/stream_sdfformer_integrated.py`

**核心概念**：

```python
# 历史时刻 t-1
- 历史pose: pose_prev (4, 4)
- 历史特征: features_prev [多尺度，例如 [N_prev_1, C], [N_prev_2, C], ...]

# 当前时刻 t
- 当前pose: pose_curr (4, 4)
- 当前体素坐标: voxel_coords_curr (N_curr, 3)

# 目标
# 将历史特征搬运到当前体素坐标系下
```

**关键步骤**：
1. 计算历史时刻到当前时刻的变换矩阵
2. 将当前体素坐标变换到历史时刻的坐标系
3. 使用`grid_sample`在历史特征上采样
4. 对所有多尺度特征重复上述过程

#### 步骤2.2：实现正确的_create_new_state

**文件**：`former3d/stream_sdfformer_integrated.py`

**伪代码**：

```python
def _create_new_state(self, state_prev, pose_curr, voxel_coords_curr):
    """
    创建新的历史状态

    Args:
        state_prev: 上一时刻的状态字典
        pose_curr: 当前时刻的pose (batch, 4, 4)
        voxel_coords_curr: 当前体素坐标 (N, 3)

    Returns:
        新的状态字典
    """
    if state_prev is None:
        # 第一帧，初始化空状态
        return {
            'features': None,  # 多尺度特征列表
            'pose_prev': None,
            'voxel_coords_prev': None,
        }

    # 提取历史信息
    features_prev = state_prev['features']  # 多尺度特征列表
    pose_prev = state_prev['pose_prev']    # (4, 4)
    voxel_coords_prev = state_prev['voxel_coords_prev']  # (N_prev, 3)

    # 计算变换矩阵：历史 -> 当前
    # pose_curr = T_curr_to_world * pose_curr_to_cam
    # pose_prev = T_prev_to_world * pose_prev_to_cam
    # T_curr_to_prev = pose_curr * inv(pose_prev)
    pose_curr_to_prev = pose_curr @ torch.inverse(pose_prev)  # (4, 4)

    # 将当前体素坐标变换到历史时刻的坐标系
    # voxel_coords_curr (N, 3) -> (N, 4) -> (N, 4) -> (N, 3)
    coords_ones = torch.ones((voxel_coords_curr.shape[0], 4),
                              device=voxel_coords_curr.device,
                              dtype=voxel_coords_curr.dtype)
    coords_4d = torch.cat([voxel_coords_curr, coords_ones[:, 3:]], dim=1)  # (N, 4)
    coords_in_prev = (pose_curr_to_prev @ coords_4d.T).T[:, :3]  # (N, 3)

    # 对每个尺度进行特征搬运
    features_curr_list = []
    for scale_idx, features_prev_scale in enumerate(features_prev):
        # features_prev_scale: (N_prev, C)

        # 将体素坐标归一化到[-1, 1]范围（用于grid_sample）
        # 需要知道历史体素网格的边界
        grid = self._normalize_coords_to_grid(
            coords_in_prev,
            voxel_coords_prev,
            scale_idx
        )  # (N, 2) 或 (N, 3)

        # 使用grid_sample采样历史特征
        features_curr_scale = F.grid_sample(
            features_prev_scale.view(1, C, H, W, D),  # 如果是体素特征
            grid.view(1, N, 1, 1, 1),  # (1, N, 1, 1, 1)
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # (1, C, N, 1, 1, 1)

        # 重新形状
        features_curr_scale = features_curr_scale.squeeze()  # (N, C)

        features_curr_list.append(features_curr_scale)

    # 返回新状态
    return {
        'features': features_curr_list,
        'pose_prev': pose_curr.clone(),
        'voxel_coords_prev': voxel_coords_curr.clone(),
    }
```

#### 步骤2.3：实现辅助函数

**需要实现的关键函数**：

```python
def _normalize_coords_to_grid(self, coords, voxel_coords_ref, scale_idx):
    """
    将体素坐标归一化到grid_sample需要的[-1, 1]范围

    Args:
        coords: 需要归一化的坐标 (N, 3)
        voxel_coords_ref: 参考体素坐标 (N_ref, 3)
        scale_idx: 尺度索引

    Returns:
        grid: 归一化后的坐标 (N, 3)
    """
    # 计算参考体素网格的边界
    min_coords = voxel_coords_ref.min(dim=0)[0]  # (3,)
    max_coords = voxel_coords_ref.max(dim=0)[0]  # (3,)

    # 归一化到[-1, 1]
    grid = 2.0 * (coords - min_coords) / (max_coords - min_coords) - 1.0

    return grid
```

#### 步骤2.4：修改forward_single_frame中的状态更新逻辑

**文件**：`former3d/stream_sdfformer_integrated.py`

**关键修改**：

```python
def forward_single_frame(self, images, poses, intrinsics, reset_state=False):
    # ... 前面的逻辑 ...

    # 获取当前体素坐标
    voxel_coords_curr = ...  # (N, 3)

    # 创建新状态（使用正确的逻辑）
    if reset_state or self.state is None:
        new_state = None
    else:
        new_state = self._create_new_state(
            self.state,
            poses[0],  # 取batch中第一个pose（假设batch内的pose相同）
            voxel_coords_curr
        )

    # 使用新状态进行融合
    if new_state is not None and new_state['features'] is not None:
        # 流式融合：当前特征 + 历史特征
        fused_features = self._fuse_features(
            current_features,
            new_state['features']
        )
    else:
        fused_features = current_features

    # 更新状态
    self.state = new_state

    # ... 后面的逻辑 ...
```

#### 步骤2.5：测试历史特征创建

**测试代码**（新建测试脚本）：

```python
# test/test_feature_warping.py
import torch
from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated

model = StreamSDFFormerIntegrated(...).to('cuda')
model.eval()

# 模拟两帧数据
images_t1 = torch.randn(1, 1, 3, 256, 256).to('cuda')
poses_t1 = torch.eye(4).unsqueeze(0).unsqueeze(0).to('cuda')

images_t2 = torch.randn(1, 1, 3, 256, 256).to('cuda')
poses_t2 = torch.eye(4).unsqueeze(0).unsqueeze(0).to('cuda')

# 第一帧
output_t1, state_t1 = model.forward_single_frame(
    images_t1, poses_t1, intrinsics, reset_state=True
)

# 第二帧（使用历史状态）
output_t2, state_t2 = model.forward_single_frame(
    images_t2, poses_t2, intrinsics, reset_state=False
)

# 验证历史特征不是随机的
assert state_t2 is not None
assert state_t2['features'] is not None

# 验证历史特征与前一帧的输出相关
#（这里需要根据实际实现添加具体的验证逻辑）
```

---

## 🔴 问题3：训练循环的低效batch处理

### 问题描述

当前训练循环遍历batch_idx，导致batch维度无法并行处理，效率极低。

### 修复目标

移除训练循环中的batch_idx遍历，让模型的forward_sequence和forward_single_frame并行处理batch维度。

### 修复步骤

#### 步骤3.1：修改train_epoch_stream函数

**文件**：`former3d/train_stream_integrated.py`

**当前问题**：

```python
# 当前的低效实现
for batch_idx, batch in enumerate(dataloader):
    # 遍历batch中的每个样本
    for i in range(batch['images'].shape[0]):
        # 手动处理每个样本
        image = batch['images'][i]  # (n_view, 3, h, w)
        pose = batch['poses'][i]    # (n_view, 4, 4)
        # ...
```

**修复后的实现**：

```python
# 修复后的高效实现
for batch_idx, batch in enumerate(dataloader):
    # batch的shape已经是 (batch, n_view, 3, h, w)
    # 直接将整个batch喂给模型
    images = batch['images']  # (batch, n_view, 3, h, w)
    poses = batch['poses']    # (batch, n_view, 4, 4)
    intrinsics = batch['intrinsics']  # (batch, n_view, 3, 3)

    # 调用模型的forward_sequence，内部处理序列
    outputs, states = model.forward_sequence(
        images, poses, intrinsics
    )  # 输出已经包含了batch维度

    # 计算损失（batch维度已经正确）
    loss = compute_loss(outputs, batch['targets'])

    # 反向传播
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

#### 步骤3.2：确保forward_sequence正确处理batch维度

**文件**：`former3d/stream_sdfformer_integrated.py`

**当前问题**：

```python
# 当前的实现可能没有正确处理batch
def forward_sequence(self, images, poses, intrinsics):
    # images: (batch, n_view, 3, h, w)
    # 可能错误：只处理第一个样本
    image = images[0]  # 错误！
```

**修复后的实现**：

```python
# 修复后的实现
def forward_sequence(self, images, poses, intrinsics):
    """
    流式处理序列数据

    Args:
        images: (batch, n_view, 3, h, w)
        poses: (batch, n_view, 4, 4)
        intrinsics: (batch, n_view, 3, 3)

    Returns:
        outputs: (batch, n_view, ...)
        states: 状态列表，每个元素对应一帧
    """
    batch_size, n_view, C, H, W = images.shape

    outputs = []
    states = []

    # 重置状态
    self.reset_state()

    # 遍历序列中的每一帧
    for t in range(n_view):
        # 提取第t帧的数据（保留batch维度）
        images_t = images[:, t:t+1]  # (batch, 1, 3, h, w)
        poses_t = poses[:, t:t+1]    # (batch, 1, 4, 4)
        intrinsics_t = intrinsics[:, t:t+1]  # (batch, 1, 3, 3)

        # 调用forward_single_frame，处理batch维度
        output_t, state_t = self.forward_single_frame(
            images_t, poses_t, intrinsics_t,
            reset_state=(t == 0)
        )  # output_t的shape是 (batch, 1, ...)

        outputs.append(output_t)
        states.append(state_t)

    # 堆叠输出
    outputs = torch.cat(outputs, dim=1)  # (batch, n_view, ...)

    return outputs, states
```

#### 步骤3.3：确保forward_single_frame正确处理batch维度

**文件**：`former3d/stream_sdfformer_integrated.py`

**当前问题**：

```python
# 当前的实现可能没有正确处理batch
def forward_single_frame(self, images, poses, intrinsics, reset_state=False):
    # images: (batch, 1, 3, h, w)
    # 可能错误：只处理第一个样本
    image = images[0, 0]  # 错误！
```

**修复后的实现**：

```python
# 修复后的实现
def forward_single_frame(self, images, poses, intrinsics, reset_state=False):
    """
    处理单帧数据（支持batch）

    Args:
        images: (batch, 1, 3, h, w)
        poses: (batch, 1, 4, 4)
        intrinsics: (batch, 1, 3, 3)
        reset_state: 是否重置状态

    Returns:
        output: (batch, 1, ...)
        state: 状态字典
    """
    batch_size = images.shape[0]

    # 去掉序列维度（因为只有1帧）
    images = images.squeeze(1)  # (batch, 3, h, w)
    poses = poses.squeeze(1)    # (batch, 4, 4)
    intrinsics = intrinsics.squeeze(1)  # (batch, 3, 3)

    # 2D特征提取（支持batch）
    features_2d = self.extract_features(images)  # (batch, C2D, h2d, w2d)

    # 3D体素生成（支持batch）
    voxel_data = self.generate_voxels(features_2d, poses, intrinsics)  # (batch, N_voxel, ...)

    # 3D编码器（支持batch）
    features_3d = self.net3d(voxel_data)  # (batch, N_voxel, C3D)

    # ... 后面的逻辑都需要支持batch ...

    # 恢复序列维度
    output = output.unsqueeze(1)  # (batch, 1, ...)

    return output, state
```

#### 步骤3.4：测试训练循环

**测试代码**（新建测试脚本）：

```python
# test/test_training_batch.py
from torch.utils.data import DataLoader
from former3d.dataset.multi_sequence_tartanair import MultiSequenceTartanAirDataset
from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated

# 创建数据集和DataLoader
dataset = MultiSequenceTartanAirDataset(...)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# 创建模型
model = StreamSDFFormerIntegrated(...).to('cuda')

# 测试一个batch
for batch in dataloader:
    images = batch['images']  # (batch, n_view, 3, h, w)
    poses = batch['poses']    # (batch, n_view, 4, 4)
    intrinsics = batch['intrinsics']  # (batch, n_view, 3, 3)

    # 调用forward_sequence
    outputs, states = model.forward_sequence(images, poses, intrinsics)

    # 验证输出shape
    assert outputs.shape[0] == batch.shape[0]  # batch维度正确
    assert outputs.shape[1] == batch.shape[1]  # n_view维度正确

    print("测试通过！")
    break
```

---

## 📅 实施计划

### 阶段1：数据shape修复（问题1）

**预计时间**：1-2小时

**任务清单**：
- [ ] 1.1 修改MultiSequenceTartanAirDataset的__getitem__
- [ ] 1.2 修改collate_fn（如果有）
- [ ] 1.3 验证DataLoader输出shape正确
- [ ] 1.4 编写测试脚本test/test_dataset_shape.py

**验证标准**：
- DataLoader输出的shape正确：
  - images: (batch, n_view, 3, h, w)
  - poses: (batch, n_view, 4, 4)
  - intrinsics: (batch, n_view, 3, 3)

### 阶段2：历史特征创建逻辑修复（问题2）

**预计时间**：3-4小时

**任务清单**：
- [ ] 2.1 理解正确的流式融合逻辑
- [ ] 2.2 实现正确的_create_new_state函数
- [ ] 2.3 实现辅助函数_normalize_coords_to_grid
- [ ] 2.4 修改forward_single_frame中的状态更新逻辑
- [ ] 2.5 编写测试脚本test/test_feature_warping.py

**验证标准**：
- 历史特征不是随机的
- 历史特征与前一帧的输出相关
- 特征搬运的逻辑正确（使用grid_sample）

### 阶段3：训练循环batch并行修复（问题3）

**预计时间**：2-3小时

**任务清单**：
- [ ] 3.1 修改train_epoch_stream函数
- [ ] 3.2 确保forward_sequence正确处理batch维度
- [ ] 3.3 确保forward_single_frame正确处理batch维度
- [ ] 3.4 编写测试脚本test/test_training_batch.py

**验证标准**：
- 训练循环不再遍历batch_idx
- forward_sequence和forward_single_frame正确处理batch维度
- 训练速度提升（预期2-3倍）

### 阶段4：集成测试

**预计时间**：1-2小时

**任务清单**：
- [ ] 4.1 运行完整的训练脚本
- [ ] 4.2 验证训练收敛
- [ ] 4.3 验证显存使用合理
- [ ] 4.4 编写集成测试脚本

**验证标准**：
- 训练可以正常进行
- 损失正常下降
- 显存使用符合预期
- 无OOM错误

---

## 📝 注意事项

### 1. 不要新增代码

**原则**：只修改现有代码，不新增任何：
- ❌ 新的模型类
- ❌ 新的数据集类
- ❌ 新的训练循环函数
- ✅ 修改现有函数的逻辑

### 2. 保持向后兼容

**原则**：修改后确保：
- 现有的配置文件仍然可用
- 现有的训练参数仍然有效
- 现有的评估脚本仍然可以运行

### 3. 分步验证

**原则**：
- 每个阶段完成后进行验证
- 不要等到最后才发现问题
- 使用测试脚本验证每个修改

### 4. 性能关注

**原则**：
- 确保修改后性能不下降
- 预期训练速度提升2-3倍
- 预期显存使用合理

---

## 🎯 最终目标

修复后的代码应该：

1. **数据shape正确**：
   - DataLoader输出shape符合PyTorch规范
   - 可以正确组batch

2. **历史特征正确**：
   - 使用正确的特征搬运逻辑
   - 使用grid_sample进行特征对齐
   - 历史特征不是随机的

3. **训练高效**：
   - 不再遍历batch_idx
   - batch维度正确并行处理
   - 训练速度提升2-3倍

---

## 📋 文件清单

**需要修改的文件**：
1. `former3d/dataset/multi_sequence_tartanair.py` - 数据shape
2. `former3d/stream_sdfformer_integrated.py` - 历史特征创建逻辑
3. `former3d/train_stream_integrated.py` - 训练循环batch处理

**需要新建的测试文件**：
1. `test/test_dataset_shape.py` - 数据shape测试
2. `test/test_feature_warping.py` - 特征搬运测试
3. `test/test_training_batch.py` - 训练batch处理测试

---

**计划制定时间**：2026-02-10 01:00
**预计开始实施**：等待用户审核后开始
**预计完成时间**：审核通过后7-9小时
