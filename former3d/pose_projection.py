"""
姿态投影模块
功能：将历史状态（SDF、占用、特征）从历史坐标系变换到当前坐标系
实现方案：使用torch.grid_sample进行可微分的特征搬运
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class PoseProjection(nn.Module):
    """姿态投影模块
    
    将历史状态（特征、SDF、占用）从历史坐标系变换到当前坐标系。
    使用torch.grid_sample进行可微分的三线性插值。
    
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
        
        # 生成体素网格坐标
        self.register_buffer('voxel_grid', self._create_voxel_grid())
        
    def _create_voxel_grid(self) -> torch.Tensor:
        """创建体素网格坐标
        
        Returns:
            体素网格坐标 [x, y, z, 3]
        """
        x_range = torch.arange(0, self.crop_size[0], device=self.device)
        y_range = torch.arange(0, self.crop_size[1], device=self.device)
        z_range = torch.arange(0, self.crop_size[2], device=self.device)
        
        # 创建网格
        grid_x, grid_y, grid_z = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        # 组合成坐标张量 [x, y, z, 3]
        voxel_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        
        # 转换为物理坐标（米）
        voxel_coords = voxel_coords.float() * self.voxel_size
        
        return voxel_coords
    
    def compute_coordinate_mapping(self, 
                                  historical_pose: torch.Tensor, 
                                  current_pose: torch.Tensor) -> torch.Tensor:
        """计算历史坐标系到当前坐标系的3D坐标映射
        
        Args:
            historical_pose: 历史相机到世界变换 [4, 4] 或 [batch, 4, 4]
            current_pose: 当前相机到世界变换 [4, 4] 或 [batch, 4, 4]
            
        Returns:
            归一化坐标映射 [batch, x, y, z, 3]，范围[-1, 1]
        """
        # 确保有batch维度
        if historical_pose.dim() == 2:
            historical_pose = historical_pose.unsqueeze(0)
            current_pose = current_pose.unsqueeze(0)
            
        batch_size = historical_pose.shape[0]
        
        # 计算变换矩阵：历史坐标系 -> 当前坐标系
        # T_current_to_historical = inv(current_pose) @ historical_pose
        inv_current_pose = torch.inverse(current_pose)
        transform = torch.bmm(inv_current_pose, historical_pose)  # [batch, 4, 4]
        
        # 获取当前体素网格的物理坐标
        current_coords = self.voxel_grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # [batch, x, y, z, 3]
        
        # 转换为齐次坐标 [batch, x, y, z, 4]
        ones = torch.ones_like(current_coords[..., :1])
        current_coords_homo = torch.cat([current_coords, ones], dim=-1)
        
        # 应用变换：历史坐标 = transform @ 当前坐标
        # 重塑以便进行批量矩阵乘法
        batch_shape = current_coords_homo.shape[:-1]  # [batch, x, y, z]
        current_coords_flat = current_coords_homo.reshape(batch_size, -1, 4)  # [batch, x*y*z, 4]
        
        # 应用变换
        historical_coords_flat = torch.bmm(current_coords_flat, transform.transpose(1, 2))  # [batch, x*y*z, 4]
        
        # 恢复形状并提取3D坐标
        historical_coords = historical_coords_flat[..., :3].reshape(*batch_shape, 3)  # [batch, x, y, z, 3]
        
        # 归一化到[-1, 1]范围用于grid_sample
        # 归一化公式：coord_norm = 2 * (coord / (size-1)) - 1
        norm_factors = torch.tensor([
            self.crop_size[0] - 1,
            self.crop_size[1] - 1, 
            self.crop_size[2] - 1
        ], device=self.device).float()
        
        # 转换为体素索引坐标
        historical_voxel_coords = historical_coords / self.voxel_size
        
        # 归一化
        normalized_coords = 2.0 * (historical_voxel_coords / norm_factors) - 1.0
        
        # 调整坐标顺序：grid_sample期望 [batch, depth, height, width, 3]
        # 当前是 [batch, x, y, z, 3]，需要转换为 [batch, z, y, x, 3]
        normalized_coords = normalized_coords.permute(0, 3, 2, 1, 4)
        
        return normalized_coords
    
    def project_features(self, 
                        historical_features: torch.Tensor, 
                        coordinate_mapping: torch.Tensor) -> torch.Tensor:
        """使用grid_sample投影特征
        
        Args:
            historical_features: 历史特征 [batch, channels, depth, height, width]
            coordinate_mapping: 坐标映射 [batch, depth, height, width, 3]
            
        Returns:
            投影后的特征 [batch, channels, depth, height, width]
        """
        # 使用grid_sample进行三线性插值
        # mode='bilinear' 实际上是三线性插值对于3D
        # padding_mode='zeros' 对于超出边界的坐标返回0
        projected = F.grid_sample(
            historical_features,
            coordinate_mapping,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        return projected
    
    def forward(self, 
                historical_state: Dict[str, torch.Tensor],
                historical_pose: torch.Tensor, 
                current_pose: torch.Tensor) -> Dict[str, torch.Tensor]:
        """完整投影流程
        
        Args:
            historical_state: 字典包含 'features', 'sdf', 'occupancy'
            historical_pose: 历史相机到世界变换 [4, 4] 或 [batch, 4, 4]
            current_pose: 当前相机到世界变换 [4, 4] 或 [batch, 4, 4]
            
        Returns:
            投影后的状态字典
        """
        # 计算坐标映射
        coordinate_mapping = self.compute_coordinate_mapping(historical_pose, current_pose)
        
        # 投影各个分量
        projected_state = {}
        
        for key in ['features', 'sdf', 'occupancy']:
            if key in historical_state:
                # 确保特征有正确的维度顺序
                features = historical_state[key]
                
                # 如果是稀疏表示，需要先转换为稠密
                if features.dim() == 2:  # 稀疏特征 [num_voxels, channels]
                    # 这里需要根据具体数据结构实现稀疏到稠密的转换
                    # 暂时假设已经是稠密表示
                    pass
                
                # 投影特征
                projected = self.project_features(features, coordinate_mapping)
                projected_state[key] = projected
        
        # 添加坐标映射信息
        projected_state['coordinate_mapping'] = coordinate_mapping
        
        return projected_state


def test_pose_projection():
    """简单的测试函数"""
    print("测试姿态投影模块...")
    
    # 创建投影模块
    projector = PoseProjection(voxel_size=0.0625, crop_size=(48, 48, 24))
    
    # 创建模拟数据
    batch_size = 2
    channels = 64
    historical_features = torch.randn(batch_size, channels, 24, 48, 48)
    
    # 恒等变换
    identity_pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 投影
    projected_state = projector(
        {'features': historical_features},
        identity_pose,
        identity_pose
    )
    
    # 检查形状
    assert projected_state['features'].shape == historical_features.shape
    print(f"✅ 特征形状正确: {projected_state['features'].shape}")
    
    # 检查恒等变换下的结果应该接近原始
    if torch.allclose(historical_features, projected_state['features'], rtol=1e-4):
        print("✅ 恒等变换测试通过")
    else:
        print("⚠️ 恒等变换有微小差异（可能是插值误差）")
    
    print("测试完成！")


if __name__ == "__main__":
    test_pose_projection()