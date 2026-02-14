# PoseAwareStreamSdfFormer 实现计划

## 项目概述

参考 `StreamSdfformerIntegrated` 类，创建一个新的流式融合类 `PoseAwareStreamSdfFormer`，用于基于Pose投影的历史特征和SDF融合。

## 类结构设计

```python
class PoseAwareStreamSdfFormer(SDFFormer):
    """
    基于Pose感知的流式SDF融合类

    功能:
    1. 保存历史稀疏fine级别特征和SDF
    2. 使用Pose将历史信息投影到当前帧
    3. 融合历史信息和当前信息
    """
```

## 接口定义

### __init__
```python
def __init__(self,
             attn_heads: int,
             attn_layers: int,
             use_proj_occ: bool,
             voxel_size: float = 0.04,
             fusion_local_radius: float = 3.0,
             crop_size: Tuple[int, int, int] = (48, 96, 96),
             use_checkpoint: bool = False):
```

参数与 `StreamSdfformerIntegrated` 保持一致。

### forward_single_frame
```python
def forward_single_frame(self,
                        images: torch.Tensor,
                        poses: torch.Tensor,
                        intrinsics: torch.Tensor,
                        reset_state: bool = False,
                        origin: Optional[torch.Tensor] = None) -> Tuple[Dict, Dict]:
```

输入:
- images: [B, 3, H, W] 或 [B, 1, 3, H, W]
- poses: [B, 4, 4] 或 [B, 1, 4, 4]
- intrinsics: [B, 3, 3] 或 [B, 1, 3, 3]
- reset_state: 是否重置历史状态
- origin: [B, 3] 原点坐标

输出:
- output: 当前帧输出字典
- new_state: 新的历史状态字典

### forward_sequence
```python
def forward_sequence(self,
                     images: torch.Tensor,
                     poses: torch.Tensor,
                     intrinsics: torch.Tensor,
                     reset_state: bool = True) -> Tuple[torch.Tensor, List[Dict]]:
```

输入:
- images: [B, N, 3, H, W]
- poses: [B, N, 4, 4]
- intrinsics: [B, N, 3, 3]
- reset_state: 是否重置状态

输出:
- outputs_cat: 合并的输出序列
- states: 状态列表

## 实现任务分解

### 任务一：初始化和历史信息保存

#### 1.1 添加历史信息存储属性
在 `__init__` 中添加:
```python
# 历史信息存储
self.historical_state = None         # 历史特征和SDF状态
self.historical_pose = None          # [B, 4, 4] 历史Pose
self.historical_intrinsics = None     # [B, 3, 3] 历史内参
self.historical_3d_points = None     # [N, 3] 历史3D坐标
```

#### 1.2 实现 _record_state 方法
```python
def _record_state(self,
                 output: Dict,
                 current_pose: torch.Tensor,
                 current_intrinsics: torch.Tensor,
                 current_3d_points: torch.Tensor):
    """
    记录当前帧的状态到历史信息

    Args:
        output: 当前帧输出字典
        current_pose: [B, 4, 4] 当前Pose
        current_intrinsics: [B, 3, 3] 当前内参
        current_3d_points: [N, 3] 当前3D点坐标
    """
    # 保存sparse的fine级别feature和SDF
    # 更新历史信息，避免显存泄露
```

关键点:
- 从 output 中提取 fine 级别的 sparse feature 和 SDF
- 保存到 historical_state（使用 detach 和 clone 避免显存泄露）
- 更新 historical_pose, historical_intrinsics, historical_3d_points
- 确保没有梯度引用（使用 .detach()）

### 任务二：历史信息投影

#### 2.1 实现 _historical_state_project 方法
```python
def _historical_state_project(self,
                             current_pose: torch.Tensor,
                             current_voxel_indices: torch.Tensor):
    """
    将历史状态投影到当前帧坐标系

    Args:
        current_pose: [B, 4, 4] 当前帧Pose
        current_voxel_indices: [N, 4] 当前帧体素索引

    Returns:
        projected_features: [N, C] 投影的特征
        projected_sdfs: [N, 1] 投影的SDF
    """
    # 1. 计算相对位姿 T_ch
    # 2. 变换历史3D点到当前坐标系
    # 3. 过滤超出范围的点
    # 4. 使用grid_sample搬运到dense 3D空间
```

关键步骤:
1. **计算相对位姿**:
   ```python
   T_ch = current_pose @ torch.inverse(self.historical_pose)
   ```

2. **变换历史3D点**:
   ```python
   # historical_3d_points: [N, 3]
   # 转换为齐次坐标，应用T_ch，转回3D
   ```

3. **过滤超出范围的点**:
   ```python
   # 检查点是否在当前帧3D输出范围内
   # 只保留有效点
   ```

4. **GridSample搬运**:
   ```python
   # 使用F.grid_sample从稀疏点采样到dense网格
   # 得到 self.project_features 和 self.project_sdfs
   ```

### 任务三：forward_single_frame 完整链路

#### 3.1 判断是否有历史信息
```python
if self.historical_state is None:
    # 第一帧：调用super().forward()
    # rgb_imgs维度: [B, 1, 3, H, W]
else:
    # 有历史信息：执行融合
```

#### 3.2 融合逻辑（有历史信息时）
```python
# 1. 调用 _historical_state_project 投影历史信息
projected_features, projected_sdfs = self._historical_state_project(...)

# 2. 获取当前帧dense fine 3D feature
current_fine_features = output['voxel_outputs']['fine']  # SparseConvTensor

# 3. 转换为dense grid
current_fine_dense = self._sparse_to_dense_grid(current_fine_features, batch_size)

# 4. Concat历史投影特征和当前特征
concat_features = torch.cat([
    projected_features,      # [B, C_hist, D, H, W]
    current_fine_dense,      # [B, C_curr, D, H, W]
    projected_sdfs           # [B, 1, D, H, W]
], dim=1)

# 5. 两层3D卷积融合
fusion_features = self.fusion_3d(concat_features)  # [B, 128, D, H, W]

# 6. 使用super().net3d处理融合后的特征
# 注意：需要转换为SparseConvTensor格式
```

#### 3.3 更新历史信息
```python
# 调用 _record_state
self._record_state(output, current_pose, current_intrinsics, current_3d_points)
```

### 任务四：forward_sequence 实现

参考 `StreamSdfformerIntegrated.forward_sequence`，逐帧调用 `forward_single_frame`。

关键点:
- 遍历序列中的每一帧
- 对每帧调用 forward_single_frame
- 收集输出和状态
- 合并输出序列

## 技术细节

### 显存管理
1. **避免显存泄露**:
   - 使用 `.detach().clone()` 保存历史信息
   - 不要保留计算图中的中间变量

2. **Sparse到Dense转换**:
   - 仅在必要时转换
   - 及时释放不需要的中间变量

### GridSample 使用
```python
# 归一化坐标到[-1, 1]
normalized_coords = ...  # [N, 3]

# 调整shape: [B, 1, 1, N, 3]
grid = normalized_coords.view(1, 1, 1, -1, 3)

# 采样
sampled = F.grid_sample(
    dense_grid,  # [B, C, D, H, W]
    grid,
    mode='bilinear',
    padding_mode='zeros',
    align_corners=False
)  # [B, C, 1, 1, N]
```

### 3D卷积融合网络
```python
self.fusion_3d = nn.Sequential(
    nn.Conv3d(in_channels, 128, kernel_size=3, padding=1),
    nn.BatchNorm3d(128),
    nn.ReLU(),
    nn.Conv3d(128, 128, kernel_size=1),
    nn.ReLU()
)
```

## 测试计划

### 单元测试
1. **_record_state 测试**:
   - 验证历史信息正确保存
   - 验证显存无泄露

2. **_historical_state_project 测试**:
   - 验证Pose变换正确性
   - 验证grid_sample采样结果
   - 验证范围过滤逻辑

3. **forward_single_frame 测试**:
   - 测试第一帧（无历史信息）
   - 测试后续帧（有历史信息）
   - 验证融合前后shape

### 集成测试
1. **forward_sequence 测试**:
   - 测试完整序列推理
   - 验证状态传递正确

2. **训练集成测试**:
   - 在 train_stream_ddp.py 中测试
   - 验证前向传播和反向传播

## 实现步骤

1. ✅ 编写实现计划文档
2. ⬜ 创建文件 `former3d/pose_aware_stream_sdfformer.py`
3. ⬜ 实现任务一：初始化和 _record_state
4. ⬜ 实现任务二：_historical_state_project
5. ⬜ 实现任务三：forward_single_frame
6. ⬜ 实现任务四：forward_sequence
7. ⬜ 编写测试用例
8. ⬜ 执行测试并修复问题
9. ⬜ 在 train_stream_ddp.py 中替换模型引用
10. ⬜ 最终测试和验证

## 文件结构

```
former3d/
├── pose_aware_stream_sdfformer.py  # 新文件
├── sdfformer.py                    # 基类
└── stream_sdfformer_integrated.py  # 参考实现

test/
├── test_pose_aware_stream.py      # 测试文件

train_stream_ddp.py                 # 修改此文件
```

## 注意事项

1. **接口兼容性**: 保持与 `StreamSdfformerIntegrated` 相同的接口定义
2. **显存优化**: 注意显存使用，避免不必要的转换和拷贝
3. **错误处理**: 添加适当的错误检查和边界条件处理
4. **代码复用**: 尽可能复用 `SDFFormer` 和 `StreamSdfformerIntegrated` 中的代码
