# Pose-Aware特征投影实现计划

## 任务目标

实现选项A的完整版本：使用Pose对SDF和Feature进行投影，然后在融合时concat投影后的历史特征、当前特征和SDF，最后使用3D卷积融合。

## 当前状态分析

1. **_create_new_state**（line 889）：保存多尺度特征（dense_grids）和SDF（sdf_grid），但没有做pose投影
2. **_apply_stream_fusion**（line 701）：期望从historical_features中获取'projected_features'，但当前_create_new_state没有生成它
3. **forward_sequence**（line 1251）：在循环内部重复创建historical_projector和fusion_3d（错误）

## 需要完成的任务

### Task 1: 创建可复用的Pose-Aware投影类

创建一个`PoseAwareFeatureProjector`类，负责：
- 接收历史多尺度特征、历史SDF、历史pose、当前pose、当前体素索引
- 使用pose变换将历史特征投影到当前坐标系
- 返回投影后的多尺度特征和SDF

**文件位置**：`former3d/pose_aware_projection.py`

### Task 2: 修改_create_new_state方法

- 添加`current_voxel_indices`参数
- 在保存dense_grids和sdf_grid后，调用`PoseAwareFeatureProjector`进行投影
- 将投影结果保存到state['projected_features']

### Task 3: 修复forward_sequence方法

- 将`historical_projector`和`fusion_3d`的创建移到`__init__`方法中
- 在调用`_create_new_state`时，从output中提取`current_voxel_indices`并传入

### Task 4: 完善_apply_stream_fusion方法

- 从`historical_features['projected_features']`中获取投影后的特征
- Extract fine级别特征和SDF
- Concat: 投影的fine + 当前features + 投影的SDF
- 使用3D卷积融合
- 返回融合后的特征

### Task 5: 测试

- 创建测试用例验证投影逻辑
- 运行训练验证端到端流程

## 实现细节

### Pose-Aware投影类设计

```python
class PoseAwareFeatureProjector:
    def __init__(self, voxel_size: float):
        self.voxel_size = voxel_size

    def project(self,
               historical_features: Dict,  # 包含dense_grids, sparse_indices等
               historical_pose: torch.Tensor,
               current_pose: torch.Tensor,
               current_voxel_indices: torch.Tensor) -> Dict:
        """
        Args:
            historical_features: 历史特征字典
                - dense_grids: {resname: [B, C, D, H, W]}
                - sparse_indices: {resname: [N, 4]}
                - sdf_grid: [B, 1, D, H, W]
                - sdf_indices: [N, 4]
            historical_pose: [B, 4, 4]
            current_pose: [B, 4, 4]
            current_voxel_indices: [N, 4] (x, y, z, batch_idx)

        Returns:
            projected_features: {
                'fine': [N, C_fine],  # 投影到当前坐标系的fine特征
                'coarse': [N, C_coarse],
                'medium': [N, C_medium],
                'sdf': [N, 1]  # 投影到当前坐标系的SDF
            }
        """
```

### 投影逻辑

1. **计算变换矩阵**：`T_ch = current_pose @ inverse(historical_pose)`
2. **提取历史体素索引**：从`historical_features['sparse_indices']['fine']`等
3. **变换到世界坐标**：`world_coords = historical_voxel_indices[:, :3] * voxel_size`
4. **变换到当前体素坐标**：
   - 应用`T_ch`到世界坐标
   - 转换回体素坐标：`current_voxel_coords = transformed_world_coords / voxel_size`
5. **使用Grid Sample**：从`dense_grids`中采样
6. **重复上述步骤**：对每个分辨率级别和SDF
7. **返回结果**：投影后的特征字典

### _apply_stream_fusion融合逻辑

```python
# 1. 获取投影特征
projected = historical_features['projected_features']

# 2. 提取当前特征
current_feats = current_features['features']  # [N, 128]

# 3. 提取投影的fine和SDF
projected_fine = projected['fine']  # [N, C_fine]
projected_sdf = projected['sdf']  # [N, 1]

# 4. 维度对齐（如果需要）
if projected_fine.shape[1] != current_feats.shape[1]:
    projected_fine = self._feat_aligner(projected_fine)

# 5. Concat: [N, C_fine + 128 + 1]
concat_features = torch.cat([projected_fine, current_feats, projected_sdf], dim=1)

# 6. 添加空间维度用于3D卷积
concat_features = concat_features.unsqueeze(1).unsqueeze(2)  # [N, C, 1, 1]
concat_features = concat_features.permute(1, 0, 2, 3)  # [C, N, 1, 1]

# 7. 3D卷积融合
fused = self.fusion_3d(concat_features)
fused = fused.permute(1, 0, 2, 3).squeeze(-1).squeeze(-1).squeeze(-1)  # [N, 128]
```

## 测试计划

1. **单元测试**：测试PoseAwareFeatureProjector的投影逻辑
2. **集成测试**：测试_create_new_state是否正确生成projected_features
3. **端到端测试**：运行训练验证整个流程

## 实施顺序

1. ✅ Task 1: 创建PoseAwareFeatureProjector
2. ✅ Task 2: 修改_create_new_state
3. ✅ Task 3: 修复forward_sequence和__init__
4. ✅ Task 4: 完善_apply_stream_fusion
5. ✅ Task 5: 测试
