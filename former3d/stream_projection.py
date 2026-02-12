"""
流式投影模块
用于在_create_new_state时预投影历史特征到当前坐标系
然后在_apply_stream_fusion时直接使用投影后的特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class HistoricalFeatureProjector(nn.Module):
    """历史特征投影器
    
    功能：在_create_new_state时将历史多尺度特征和SDF投影到当前坐标系
    然后在_apply_stream_fusion时直接使用投影后的特征进行融合
    """
    
    def __init__(self, voxel_size: float = 0.0625):
        """
        初始化投影器
        
        Args:
            voxel_size: 体素大小（米）
        """
        super().__init__()
        self.voxel_size = voxel_size
    
    def compute_transform(self, historical_pose: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
        """
        计算从历史pose到当前pose的变换矩阵
        
        Args:
            historical_pose: [B, 4, 4] 历史pose
            current_pose: [B, 4, 4] 当前pose
            
        Returns:
            T_ch: [B, 4, 4] 从历史到当前pose的变换
        """
        T_cw = current_pose  # [B, 4, 4]
        T_hw = historical_pose  # [B, 4, 4]
        T_hw_inv = torch.inverse(T_hw)  # [B, 4, 4]
        T_ch = torch.bmm(T_cw, T_hw_inv)  # [B, 4, 4]
        return T_ch
    
    def project_dense_grid(self,
                        dense_grid: torch.Tensor,
                        current_voxel_indices: torch.Tensor,
                        T_ch: torch.Tensor,
                        resolution: float) -> torch.Tensor:
        """
        投影密集网格到当前坐标系
        
        Args:
            dense_grid: [B, C, D, H, W] 历史密集网格
            current_voxel_indices: [N, 4] 当前体素索引 (b, x, y, z)
            T_ch: [B, 4, 4] 变换矩阵
            resolution: 体素分辨率
            
        Returns:
            projected: [N, C] 投影到当前体素的特征
        """
        device = dense_grid.device
        batch_size = dense_grid.shape[0]
        num_channels = dense_grid.shape[1]
        num_points = current_voxel_indices.shape[0]
        
        # 获取batch索引
        batch_indices = current_voxel_indices[:, 0].long()  # [N]
        
        # 获取当前体素坐标（体素索引转世界坐标）
        current_coords_voxel = current_voxel_indices[:, 1:4].float()  # [N, 3]
        current_coords_world = current_coords_voxel * self.voxel_size  # [N, 3]
        
        # 添加齐次坐标
        ones = torch.ones(num_points, 1, device=device, dtype=current_coords_world.dtype)
        current_coords_homo = torch.cat([current_coords_world, ones], dim=1)  # [N, 4]
        
        # 选择对应的变换矩阵
        T_ch_batch = T_ch[batch_indices]  # [N, 4, 4]
        
        # 变换到历史坐标系（T_ch的逆）
        T_ch_inv = torch.inverse(T_ch_batch)  # [N, 4, 4]
        historical_coords_homo = torch.bmm(T_ch_inv, current_coords_homo.unsqueeze(-1)).squeeze(-1)  # [N, 4]
        historical_coords = historical_coords_homo[:, :3]  # [N, 3]
        
        # 转换为体素坐标
        historical_coords_voxel = historical_coords / resolution  # [N, 3]
        
        # 归一化到[-1, 1]
        D, H, W = dense_grid.shape[2:5]
        
        # 避免除零
        D = max(D, 1)
        H = max(H, 1)
        W = max(W, 1)
        
        x_norm = (historical_coords_voxel[:, 0] / (D - 1)) * 2 - 1
        y_norm = (historical_coords_voxel[:, 1] / (H - 1)) * 2 - 1
        z_norm = (historical_coords_voxel[:, 2] / (W - 1)) * 2 - 1
        
        normalized_coords = torch.stack([x_norm, y_norm, z_norm], dim=1)  # [N, 3]
        
        # 裁剪到有效范围
        normalized_coords = torch.clamp(normalized_coords, -1.0, 1.0)
        
        # 调整为grid_sample格式：[B, C, 1, 1, N]
        grid = normalized_coords.view(1, 1, 1, num_points, 3)
        grid = grid.expand(batch_size, -1, -1, -1, -1)  # [B, 1, 1, N, 3]
        
        # 调整dense_grid格式：[B, C, D, H, W] -> [B, C, D, H, W]（已经是）
        
        # 采样
        try:
            sampled = F.grid_sample(
                dense_grid,  # [B, C, D, H, W]
                grid,  # [B, 1, 1, N, 3]
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False
            )  # [B, C, 1, 1, N]
            
            # 根据batch索引提取
            projected = []
            for b in range(batch_size):
                mask = batch_indices == b
                if mask.any():
                    projected_b = sampled[b, :, 0, 0, mask].permute(2, 1, 0)  # [N_b, C]
                    projected.append(projected_b)
                else:
                    projected.append(torch.zeros(
                        (0, num_channels),
                        device=device,
                        dtype=dense_grid.dtype
                    ))
            
            projected = torch.cat(projected, dim=0)  # [N, C]
            
        except Exception as e:
            print(f"[HistoricalFeatureProjector] grid_sample失败: {e}")
            projected = torch.zeros(num_points, num_channels, device=device, dtype=dense_grid.dtype)
        
        return projected
    
    def project_all(self,
                   historical_features: Dict,
                   current_voxel_indices: torch.Tensor,
                   historical_pose: torch.Tensor,
                   current_pose: torch.Tensor) -> Dict:
        """
        投影所有历史特征（多尺度特征 + SDF）到当前坐标系
        
        Args:
            historical_features: 历史特征字典（包含dense_grids和SDF）
            current_voxel_indices: [N, 4] 当前体素索引
            historical_pose: [B, 4, 4] 历史pose
            current_pose: [B, 4, 4] 当前pose
            
        Returns:
            projected_features: {
                'coarse': [N, C_coarse],
                'medium': [N, C_medium],
                'fine': [N, C_fine],
                'sdf': [N, 1],
                'num_points': N
            }
        """
        # 计算变换
        T_ch = self.compute_transform(historical_pose, current_pose)  # [B, 4, 4]
        
        projected_features = {'num_points': current_voxel_indices.shape[0]}
        
        # 投影多尺度特征
        if 'dense_grids' in historical_features:
            for resname in ['coarse', 'medium', 'fine']:
                if resname in historical_features['dense_grids']:
                    dense_grid = historical_features['dense_grids'][resname]  # [B, C, D, H, W]
                    resolution = historical_features['resolutions'][resname]
                    
                    # 投影
                    projected = self.project_dense_grid(
                        dense_grid,
                        current_voxel_indices,
                        T_ch,
                        resolution
                    )  # [N, C]
                    
                    projected_features[resname] = projected
                    print(f"[StreamProjection] {resname} 投影完成: {projected.shape}")
        
        # 投影SDF
        if 'sdf_grid' in historical_features:
            sdf_grid = historical_features['sdf_grid']  # [B, 1, D, H, W]
            sdf_resolution = historical_features['sdf_resolution']
            
            projected_sdf = self.project_dense_grid(
                sdf_grid,
                current_voxel_indices,
                T_ch,
                sdf_resolution
            )  # [N, 1]
            
            projected_features['sdf'] = projected_sdf
            print(f"[StreamProjection] SDF 投影完成: {projected_sdf.shape}")
        
        return projected_features
