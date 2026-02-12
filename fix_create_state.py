#!/usr/bin/env python3
"""
修改_create_new_state.py
添加current_voxel_indices参数和预投影逻辑
"""

import re

# 读取文件
with open('former3d/stream_sdfformer_integrated.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 修改方法签名
old_signature = r'def _create_new_state\(self, output: Dict, current_pose: torch\.Tensor\) -> Dict:'
new_signature = 'def _create_new_state(self, output: Dict, current_pose: torch.Tensor, current_voxel_indices: Optional[torch.Tensor] = None) -> Dict:'

content = re.sub(old_signature, new_signature, content)

print("✅ 步骤2.1: 方法签名已修改")

# 2. 在返回前添加预投影逻辑
# 找到return new_state前的位置
marker = '        # 为了向后兼容，添加一个默认的coords和batch_inds'
insert_code = '''
        # Phase 1 改进：预投影历史特征到当前坐标系
        projected_features = None
        
        if current_voxel_indices is not None and len(dense_grids) > 0:
            print("[Phase 1 改进] 开始预投影历史特征到当前坐标系")
            
            # 使用fine分辨率的dense_grid作为参考
            fine_dense_grid = dense_grids.get('fine', None)
            
            if fine_dense_grid is not None:
                # 构建历史特征字典
                historical_features_for_projection = {
                    'dense_grids': dense_grids,
                    'sparse_indices': sparse_indices,
                    'spatial_shapes': spatial_shapes,
                    'resolutions': resolutions,
                    'sdf_grid': sdf_grid,
                    'sdf_indices': sdf_indices,
                    'sdf_spatial_shape': sdf_spatial_shape,
                    'sdf_resolution': sdf_resolution
                }
                
                # 预投影到当前坐标系
                try:
                    projected_features = self.historical_projector.project_all(
                        historical_features=historical_features_for_projection,
                        current_voxel_indices=current_voxel_indices,
                        historical_pose=current_pose,  # 注意：第一次时historical_pose == current_pose
                        current_pose=current_pose
                    )
                    
                    print(f"[Phase 1 改进] 预投影完成: {projected_features.get('num_points', 0)}个点")
                
                except Exception as e:
                    print(f"[Phase 1 改进] 预投影失败: {e}")
                    import traceback
                    traceback.print_exc()
                    projected_features = None
'''

content = content.replace(marker, marker + insert_code)

print("✅ 步骤2.2: 预投影逻辑已添加")

# 3. 修改return new_state，添加projected_features
old_return = '        return new_state'
new_return = '''        # Phase 1 改进：添加预投影的特征
        if projected_features is not None:
            new_state['projected_features'] = projected_features
        
        return new_state'''

content = content.replace(old_return, new_return)

print("✅ 步骤2.3: projected_features已添加到返回值")

# 写入文件
with open('former3d/stream_sdfformer_integrated.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ _create_new_state方法修改完成")
