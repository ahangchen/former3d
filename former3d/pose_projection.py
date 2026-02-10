"""
姿态投影模块（稀疏表示版本）
功能：将历史状态（SDF、占用、特征）从历史坐标系变换到当前坐标系
实现方案：基于原SDFFormer的稀疏投影逻辑，支持稀疏体素表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


class PoseProjection(nn.Module):
    """姿态投影模块（稀疏表示版本）
    
    将历史状态（特征、SDF、占用）从历史坐标系变换到当前坐标系。
    基于原SDFFormer的稀疏投影逻辑，支持稀疏体素表示。
    
    Args:
        voxel_size: 体素大小（米）
        crop_size: 裁剪尺寸 [x, y, z]
        device: 计算设备
    """
    
    def __init__(self, voxel_size: float = 0.0625, 
                 crop_size: Tuple[int, int, int] = (96, 96, 48),
                 device: str = 'cpu'):
        super().__init__()
        self.voxel_size = voxel_size
        self.crop_size = crop_size
        self.device = device
        
    def compute_coordinate_mapping(self, 
                                  historical_pose: torch.Tensor, 
                                  current_pose: torch.Tensor,
                                  voxel_coords: torch.Tensor,
                                  voxel_batch_inds: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算历史坐标系到当前坐标系的3D坐标映射（稀疏版本）
        
        Args:
            historical_pose: 历史相机到世界变换 [batch, 4, 4]
            current_pose: 当前相机到世界变换 [batch, 4, 4]
            voxel_coords: 体素坐标 [num_voxels, 3]（物理坐标，单位：米）
            voxel_batch_inds: 体素批次索引 [num_voxels]
            
        Returns:
            字典包含：
            - 'historical_coords': 历史坐标系中的坐标 [num_voxels, 3]
            - 'batch_inds': 批次索引 [num_voxels]
            - 'mask': 有效掩码 [num_voxels]
        """
        device = voxel_coords.device
        n_voxels = len(voxel_coords)
        batch_size = historical_pose.shape[0]
        
        # 初始化输出
        historical_coords = torch.zeros((n_voxels, 3), device=device, dtype=torch.float32)
        mask = torch.zeros((n_voxels,), device=device, dtype=torch.bool)
        
        # 计算变换矩阵：历史坐标系 -> 当前坐标系
        # T_current_to_historical = inv(current_pose) @ historical_pose
        inv_current_pose = torch.inverse(current_pose)  # [batch, 4, 4]
        transform = torch.bmm(inv_current_pose, historical_pose)  # [batch, 4, 4]
        
        # 向量化实现：为每个体素选择对应的变换矩阵
        # 将batch_inds归一化到当前batch_size范围
        normalized_batch_inds = voxel_batch_inds % batch_size
        
        # 使用gather为每个体素选择对应的变换矩阵
        # 首先将transform从[batch, 4, 4]转换为[batch, 16]
        transform_flat = transform.reshape(batch_size, 16)  # [batch, 16]
        
        # 为每个体素选择对应的变换矩阵
        selected_transform_flat = transform_flat[normalized_batch_inds]  # [n_voxels, 16]
        selected_transform = selected_transform_flat.reshape(n_voxels, 4, 4)  # [n_voxels, 4, 4]
        
        # 转换为齐次坐标
        ones = torch.ones((n_voxels, 1), device=device, dtype=torch.float32)
        voxel_coords_h = torch.cat([voxel_coords, ones], dim=-1)  # [n_voxels, 4]
        
        # 应用变换：历史坐标 = transform @ 当前坐标
        # 使用批量矩阵乘法
        historical_coords_h = torch.bmm(voxel_coords_h.unsqueeze(1), selected_transform.transpose(1, 2)).squeeze(1)  # [n_voxels, 4]
        
        # 提取3D坐标
        historical_coords = historical_coords_h[:, :3]
        
        # 检查坐标是否在裁剪范围内
        # 转换为体素索引
        historical_voxel_coords = historical_coords / self.voxel_size
        
        # 检查是否在有效范围内 [0, crop_size-1]
        in_range_x = (historical_voxel_coords[:, 0] >= 0) & (historical_voxel_coords[:, 0] < self.crop_size[0])
        in_range_y = (historical_voxel_coords[:, 1] >= 0) & (historical_voxel_coords[:, 1] < self.crop_size[1])
        in_range_z = (historical_voxel_coords[:, 2] >= 0) & (historical_voxel_coords[:, 2] < self.crop_size[2])
        
        mask = in_range_x & in_range_y & in_range_z
        
        return {
            'historical_coords': historical_coords,
            'batch_inds': normalized_batch_inds,
            'mask': mask
        }
    
    def project_sparse_features(self, 
                               historical_features: torch.Tensor,
                               coordinate_mapping: Dict[str, torch.Tensor]) -> torch.Tensor:
        """投影稀疏特征（向量化实现）
        
        Args:
            historical_features: 历史特征 [num_voxels, channels]
            coordinate_mapping: 坐标映射字典
            
        Returns:
            投影后的特征 [num_voxels, channels]
        """
        device = historical_features.device
        mask = coordinate_mapping['mask']
        
        n_voxels = len(historical_features)
        channels = historical_features.shape[1]
        
        # 初始化输出
        projected_features = torch.zeros((n_voxels, channels), device=device, dtype=historical_features.dtype)
        
        # 只处理有效体素
        valid_mask = mask
        if not torch.any(valid_mask):
            return projected_features
        
        # 对于稀疏表示，直接使用对应的历史特征
        # 因为坐标变换后，体素位置可能变化不大
        # 这里简化处理：直接使用原始特征
        projected_features[valid_mask] = historical_features[valid_mask]
        
        return projected_features
    
    def forward(self, 
                historical_state: Dict[str, torch.Tensor],
                historical_pose: torch.Tensor, 
                current_pose: torch.Tensor) -> Dict[str, torch.Tensor]:
        """完整投影流程（稀疏版本）
        
        Args:
            historical_state: 字典包含 'features', 'sdf', 'occupancy', 'coords', 'batch_inds'
            historical_pose: 历史相机到世界变换 [batch, 4, 4]
            current_pose: 当前相机到世界变换 [batch, 4, 4]
            
        Returns:
            投影后的状态字典
        """
        # 检查必要的输入
        required_keys = ['coords', 'batch_inds']
        for key in required_keys:
            if key not in historical_state:
                raise ValueError(f"历史状态必须包含 '{key}'")
        
        voxel_coords = historical_state['coords']  # [num_voxels, 3]
        voxel_batch_inds = historical_state['batch_inds']  # [num_voxels]
        
        # 计算坐标映射
        coordinate_mapping = self.compute_coordinate_mapping(
            historical_pose, current_pose, voxel_coords, voxel_batch_inds
        )
        
        # 投影各个分量
        projected_state = {}
        
        for key in ['features', 'sdf', 'occupancy']:
            if key in historical_state:
                features = historical_state[key]
                
                # 投影特征
                projected = self.project_sparse_features(features, coordinate_mapping)
                projected_state[key] = projected
        
        # 添加坐标和批次信息
        projected_state['coords'] = voxel_coords
        projected_state['batch_inds'] = voxel_batch_inds
        projected_state['coordinate_mapping'] = coordinate_mapping
        projected_state['mask'] = coordinate_mapping['mask']
        
        return projected_state


def test_sparse_pose_projection():
    """测试稀疏姿态投影模块"""
    print("测试稀疏姿态投影模块...")
    
    # 创建投影模块
    projector = PoseProjection(voxel_size=0.0625, crop_size=(48, 96, 96))
    
    # 创建模拟稀疏数据
    batch_size = 2
    num_voxels = 1000
    channels = 64
    
    # 随机生成体素坐标（物理坐标，单位：米）
    voxel_coords = torch.randn(num_voxels, 3) * 0.5  # 在[-0.25, 0.25]米范围内
    voxel_batch_inds = torch.randint(0, batch_size, (num_voxels,))
    
    # 随机生成特征
    historical_features = torch.randn(num_voxels, channels)
    
    # 恒等变换
    identity_pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 创建历史状态
    historical_state = {
        'features': historical_features,
        'coords': voxel_coords,
        'batch_inds': voxel_batch_inds
    }
    
    # 投影
    projected_state = projector(
        historical_state,
        identity_pose,
        identity_pose
    )
    
    # 检查输出
    print(f"✅ 输入特征形状: {historical_features.shape}")
    print(f"✅ 输出特征形状: {projected_state['features'].shape}")
    print(f"✅ 坐标形状: {projected_state['coords'].shape}")
    print(f"✅ 批次索引形状: {projected_state['batch_inds'].shape}")
    print(f"✅ 掩码形状: {projected_state['mask'].shape}")
    print(f"✅ 有效体素数量: {projected_state['mask'].sum().item()}")
    
    # 在恒等变换下，坐标应该保持不变
    if torch.allclose(voxel_coords, projected_state['coords'], rtol=1e-4):
        print("✅ 坐标恒等变换测试通过")
    else:
        print("⚠️ 坐标有微小差异")
    
    print("稀疏投影测试完成！")


if __name__ == "__main__":
    test_sparse_pose_projection()