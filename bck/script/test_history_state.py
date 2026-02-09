#!/usr/bin/env python
"""测试历史状态创建的最小化测试"""

import torch
import torch.nn as nn

# 模拟SparseConvTensor
class MockSparseConvTensor:
    def __init__(self, features, indices):
        self.features = features
        self.indices = indices
        self.spatial_shape = [192, 384, 384]  # 模拟空间形状

def test_history_state_creation():
    """测试历史状态创建"""
    print("=== 测试历史状态创建 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模拟数据
    batch_size = 1
    num_voxels = 1000
    
    # 模拟特征和索引
    features = torch.randn(num_voxels, 1, device=device)  # [N, 1]
    indices = torch.randint(0, 100, (num_voxels, 4), device=device)  # [N, 4]
    indices[:, 3] = torch.randint(0, batch_size, (num_voxels,), device=device)  # batch_idx
    
    # 创建模拟输出
    fine_output = MockSparseConvTensor(features, indices)
    
    output = {
        'voxel_outputs': {
            'coarse': fine_output,
            'medium': fine_output,
            'fine': fine_output
        },
        'proj_occ_logits': {},
        'bp_data': {},
        'sdf': features,
        'occupancy': torch.sigmoid(features)
    }
    
    print(f"\n模拟输出结构:")
    print(f"  voxel_outputs键: {list(output['voxel_outputs'].keys())}")
    print(f"  fine_output类型: {type(output['voxel_outputs']['fine'])}")
    print(f"  hasattr(features): {hasattr(output['voxel_outputs']['fine'], 'features')}")
    print(f"  hasattr(indices): {hasattr(output['voxel_outputs']['fine'], 'indices')}")
    
    # 测试_create_new_state逻辑
    class TestModel:
        def __init__(self):
            self.resolutions = {'fine': 0.03125, 'coarse': 0.125}
            self.crop_size = (48, 96, 96)
            self.feature_expansion = nn.Linear(1, 128).to(device)
            self.feature_compression = nn.Linear(128, 1).to(device)
        
        def _create_new_state(self, output, current_pose):
            batch_size = current_pose.shape[0]
            device = current_pose.device
            
            # 尝试从输出中提取真实的体素数据
            if 'voxel_outputs' in output and 'fine' in output['voxel_outputs']:
                fine_output = output['voxel_outputs']['fine']
                
                # 调试信息
                print(f"\n调试信息:")
                print(f"  fine_output类型 = {type(fine_output)}")
                print(f"  hasattr(features) = {hasattr(fine_output, 'features')}")
                print(f"  hasattr(indices) = {hasattr(fine_output, 'indices')}")
                
                if hasattr(fine_output, 'features') and hasattr(fine_output, 'indices'):
                    # 使用真实的体素数据
                    features = fine_output.features  # [N, 1]
                    indices = fine_output.indices    # [N, 4] (x, y, z, batch_idx)
                    print(f"  features形状 = {features.shape}")
                    print(f"  indices形状 = {indices.shape}")
                    
                    # 扩展特征维度（从1到128）
                    if features.shape[1] == 1 and hasattr(self, 'feature_expansion'):
                        features = self.feature_expansion(features)  # [N, 128]
                        print(f"  扩展后features形状 = {features.shape}")
                    
                    # 提取坐标和批次索引
                    coords = indices[:, :3].float() * self.resolutions['fine']
                    batch_inds = indices[:, 3].long()
                    
                    # 提取SDF和占用（如果可用）
                    sdf = output.get('sdf', None)
                    occupancy = output.get('occupancy', None)
                    
                    new_state = {
                        'features': features,
                        'sdf': sdf,
                        'occupancy': occupancy,
                        'coords': coords,
                        'batch_inds': batch_inds,
                        'num_voxels': features.shape[0],
                        'pose': current_pose.detach().clone(),
                        'output': output,
                        'original_features': fine_output.features
                    }
                    
                    print(f"\n✅ 成功创建真实历史状态")
                    print(f"  体素数: {new_state['num_voxels']}")
                    print(f"  特征维度: {new_state['features'].shape[1]}")
                    return new_state
            
            # 如果无法提取真实数据，使用简化版本
            print("\n⚠️ 使用简化的历史状态创建")
            return None
    
    # 创建测试模型
    model = TestModel()
    
    # 创建当前位姿
    current_pose = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 测试历史状态创建
    state = model._create_new_state(output, current_pose)
    
    if state is not None:
        print(f"\n✅ 历史状态创建测试通过")
        print(f"状态键: {list(state.keys())}")
    else:
        print(f"\n❌ 历史状态创建失败")

if __name__ == "__main__":
    test_history_state_creation()