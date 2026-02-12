#!/usr/bin/env python3
"""
快速实施选项A - 预投影特征 + concat + 3D卷积融合
"""
import re
import sys

def apply_option_a():
    # 读取文件
    with open('former3d/stream_sdfformer_integrated.py', 'r', encoding='utf-8') as f:
        content = f.read()

    print("✓ 读取文件")

    # 1. 添加导入
    if 'from former3d.stream_projection import HistoricalFeatureProjector' not in content:
        content = content.replace(
            'from former3d.stream_fusion_concat import StreamConcatFusion\n',
            'from former3d.stream_fusion_concat import StreamConcatFusion\nfrom former3d.stream_projection import HistoricalFeatureProjector\n'
        )
        print("✓ 添加导入")

    # 2. 在__init__中添加3D卷积融合网络
    pattern1 = r'(self\.lightweight_state_mode = True  # 默认启用轻量级模式)'
    replacement1 = r'''self.lightweight_state_mode = True  # 默认启用轻量级模式
        
        # 流式投影器（预投影历史特征）
        self.historical_projector = HistoricalFeatureProjector(voxel_size=voxel_size)
        
        # 3D卷积融合网络（用于融合历史和当前特征）
        self.fusion_3d = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),  # 输入：历史+当前+SDF
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=1),
            nn.ReLU()
        )
        self.fusion_3d_enabled = True'''
    
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        print("✓ 添加3D卷积融合网络")
    else:
        print("✗ 未找到插入位置1")
        return False

    # 3. 修改_create_new_state调用处
    pattern2 = r'(new_state = self\._create_new_state\(output, poses\))'
    replacement2 = r'''        # 从voxel_outputs中提取当前体素索引（用于预投影）
        current_voxel_indices = None
        if 'voxel_outputs' in output and 'fine' in output['voxel_outputs']:
            fine_output = output['voxel_outputs']['fine']
            if hasattr(fine_output, 'indices'):
                current_voxel_indices = fine_output.indices  # [N, 4]
                print(f"[StreamFusion] 提取当前体素索引: {current_voxel_indices.shape}")
        
        new_state = self._create_new_state(output, poses, current_voxel_indices)'''
    
    if re.search(pattern2, content):
        content = re.sub(pattern2, replacement2, content, count=1)
        print("✓ 修改_create_new_state调用处")
    else:
        print("✗ 未找到插入位置2")
        return False

    # 4. 修改_create_new_state方法签名和添加预投影逻辑
    pattern3 = r'(def _create_new_state\(self, output: Dict, current_pose: torch\.Tensor\) -> Dict:.*?return new_state)'
    def replacer3(match):
        return match.group(1) + '''

        # Phase 1 改进：预投影历史特征到当前坐标系
        projected_features = None
        
        if current_voxel_indices is not None and len(dense_grids) > 0:
            print("[Phase 1 改进] 开始预投影历史特征到当前坐标系")
            
            fine_dense_grid = dense_grids.get('fine', None)
            
            if fine_dense_grid is not None:
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
                
                try:
                    projected_features = self.historical_projector.project_all(
                        historical_features=historical_features_for_projection,
                        current_voxel_indices=current_voxel_indices,
                        historical_pose=current_pose,
                        current_pose=current_pose
                    )
                    print(f"[Phase 1 改进] 预投影完成: {projected_features.get('num_points', 0)}个点")
                except Exception as e:
                    print(f"[Phase 1 改进] 预投影失败: {e}")
                    projected_features = None
        
        # 添加预投影特征
        if projected_features is not None:
            new_state['projected_features'] = projected_features
        
        return new_state'''

    if re.search(pattern3, content, re.DOTALL):
        content = re.sub(pattern3, replacer3, content, flags=re.DOTALL)
        print("✓ 修改_create_new_state方法签名和添加预投影逻辑")
    else:
        print("✗ 未找到_create_new_state方法")
        return False

    # 5. 修改_create_legacy_state方法签名
    pattern4 = r'(def _create_legacy_state\(self, output: Dict, current_pose: torch\.Tensor\) -> Dict:)'
    replacement4 = r'''def _create_legacy_state(self, output: Dict, current_pose: torch.Tensor, current_voxel_indices: Optional[torch.Tensor] = None) -> Dict:'''

    if re.search(pattern4, content):
        content = re.sub(pattern4, replacement4, content, count=1)
        print("✓ 修改_create_legacy_state方法签名")
    else:
        print("✗ 未找到_create_legacy_state方法")
        return False

    # 6. 在_create_legacy_state末尾添加预投影逻辑
    pattern5 = r'(# 注意：新的流式融合应该使用dense_grids而不是这些兼容字段\n    device = current_pose\.device)'
    replacement5 = r'''        # 注意：新的流式融合应该使用dense_grids而不是这些兼容字段
        device = current_pose.device'''
    
    # 在旧的兼容代码块之后添加预投影逻辑
    pattern6 = r'(# 为了向后兼容，添加一个默认的coords和batch_inds\n        # 注意：新的流式融合应该使用dense_grids而不是这些兼容字段\n        device = current_pose\.device\n        \n        for resname in dense_grids:\n            print\(f"  \{resname\}: 密集网格\{dense_grids\[resname\]\.shape\}"\)\n        \n        return new_state)'
    replacement6 = r'''        # 为了向后兼容，添加一个默认的coords和batch_inds
        device = current_pose.device
        
        for resname in dense_grids:
            print(f"  {resname}: 密集网格{dense_grids[resname].shape}")
        
        # Phase 1 改进：添加预投影的特征
        if projected_features is not None:
            new_state['projected_features'] = projected_features
        
        return new_state'''

    if re.search(pattern6, content, re.DOTALL):
        content = re.sub(pattern6, replacement6, content, flags=re.DOTALL)
        print("✓ 在_create_legacy_state末尾添加预投影逻辑")
    else:
        print("✗ 未找到插入位置6")
        return False

    # 保存文件
    with open('former3d/stream_sdfformer_integrated.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✓ 保存文件")
    return True

if __name__ == "__main__":
    success = apply_option_a()
    sys.exit(0 if success else 1)
