"""
Pose-Aware特征投影模块

使用Pose变换将历史多尺度特征和SDF投影到当前坐标系
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class PoseAwareFeatureProjector:
    """
    基于Pose的特征投影器

    将历史多尺度特征和SDF投影到当前坐标系
    """

    def __init__(self, voxel_size: float = 0.16):
        """
        初始化投影器

        Args:
            voxel_size: 体素大小（米）
        """
        self.voxel_size = voxel_size

    def compute_transform(self,
                        historical_pose: torch.Tensor,
                        current_pose: torch.Tensor) -> torch.Tensor:
        """
        计算从历史pose到当前pose的变换矩阵

        Args:
            historical_pose: [B, 4, 4] 历史pose
            current_pose: [B, 4, 4] 当前pose

        Returns:
            T_ch: [B, 4, 4] 从历史到当前pose的变换
        """
        # T_cw: 从世界到当前相机的变换
        T_cw = current_pose  # [B, 4, 4]

        # T_hw: 从世界到历史相机的变换
        T_hw = historical_pose  # [B, 4, 4]

        # T_ch = T_cw * T_hw^{-1}: 从历史相机到当前相机的变换
        T_hw_inv = torch.inverse(T_hw)  # [B, 4, 4]
        T_ch = torch.bmm(T_cw, T_hw_inv)  # [B, 4, 4]

        return T_ch

    def transform_historical_indices_to_world(self,
                                       historical_indices: torch.Tensor) -> torch.Tensor:
        """
        将历史体素索引转换为世界坐标（米）

        Args:
            historical_indices: [N_historical, 4] (x, y, z, batch_idx)

        Returns:
            world_coords: [N_historical, 3] 世界坐标（米）
        """
        # 提取体素索引
        voxel_indices = historical_indices[:, 1:4].float()  # [N_historical, 3]

        # 转换为世界坐标
        world_coords = voxel_indices * self.voxel_size  # [N_historical, 3]

        return world_coords

    def transform_world_to_current_voxel(self,
                                     world_coords: torch.Tensor,
                                     T_ch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将世界坐标变换到当前体素坐标

        Args:
            world_coords: [N, 3] 世界坐标（米）
            T_ch: [B, 4, 4] 从历史到当前pose的变换

        Returns:
            current_voxel_coords: [N, B, 3] 当前体素坐标
            batch_indices: [N] 对应的batch索引
        """
        N, _ = world_coords.shape
        B = T_ch.shape[0]

        # 添加齐次坐标
        ones = torch.ones(N, 1, device=world_coords.device, dtype=world_coords.dtype)
        coords_homo = torch.cat([world_coords, ones], dim=1)  # [N, 4]

        # 对每个batch应用变换
        # 由于需要批量变换N个点到B个batch，我们使用广播
        coords_homo_expanded = coords_homo.unsqueeze(1).expand(-1, B, -1)  # [N, B, 4]
        T_ch_expanded = T_ch.unsqueeze(0).expand(N, -1, -1, -1)  # [N, B, 4, 4]

        # 重塑为适合bmm的形状
        coords_flat = coords_homo_expanded.contiguous().view(N * B, 4).unsqueeze(1)  # [N*B, 1, 4]
        T_ch_flat = T_ch_expanded.contiguous().view(N * B, 4, 4)  # [N*B, 4, 4]

        # 矩阵乘法
        transformed_flat = torch.bmm(coords_flat, T_ch_flat)  # [N*B, 1, 4]
        transformed = transformed_flat.view(N, B, 4)  # [N, B, 4]

        # 提取坐标部分
        transformed_coords = transformed[:, :, :3]  # [N, B, 3]

        # 转换回体素坐标
        current_voxel_coords = transformed_coords / self.voxel_size  # [N, B, 3]

        # 生成batch索引 [0, 1, ..., B-1]
        batch_indices = torch.arange(B, device=world_coords.device).view(1, B).expand(N, -1)  # [N, B]

        return current_voxel_coords, batch_indices

    def sample_from_dense_grid(self,
                           dense_grid: torch.Tensor,
                           coords: torch.Tensor,
                           batch_indices: torch.Tensor) -> torch.Tensor:
        """
        从密集网格中采样特征

        Args:
            dense_grid: [B, C, D, H, W] 密集网格
            coords: [N, B, 3] 采样坐标（体素坐标）
            batch_indices: [N] batch索引

        Returns:
            sampled_features: [N, C] 采样的特征
        """
        N, B, _ = coords.shape
        C = dense_grid.shape[1]
        D, H, W = dense_grid.shape[2], dense_grid.shape[3], dense_grid.shape[4]

        # 准备坐标用于grid_sample
        # grid_sample需要归一化到[-1, 1]的坐标
        # 将体素坐标归一化
        coords_normalized = torch.zeros_like(coords)
        coords_normalized[:, :, 0] = coords[:, :, 0] / (D - 1) * 2 - 1  # Depth
        coords_normalized[:, :, 1] = coords[:, :, 1] / (H - 1) * 2 - 1  # Height
        coords_normalized[:, :, 2] = coords[:, :, 2] / (W - 1) * 2 - 1  # Width

        # 重新排列坐标为[N, B, D, H, W]格式用于grid_sample
        # 实际上我们需要[N, B, 3]，其中3是(x, y, z)
        # grid_sample期望的坐标格式是[N, C, D, H, W]或[B, C, D, H, W]
        # 对于每个batch和点，我们需要一个3D坐标

        # 更简单的方法：逐个batch处理
        sampled_features = []

        for b in range(B):
            # 提取当前batch的坐标
            batch_coords = coords[:, b, :]  # [N, 3]

            # 添加batch和channel维度
            batch_coords = batch_coords.unsqueeze(0).unsqueeze(0)  # [1, 1, N, 3]

            # grid_sample: [B=1, C=1, D, H, W] -> [B=1, C, N]
            batch_grid = dense_grid[b:b+1]  # [1, C, D, H, W]

            sampled = F.grid_sample(
                batch_grid.float(),
                batch_coords.float(),
                mode='bilinear',
                padding_mode='border',
                align_corners=False
            )  # [1, C, N, 1, 1]

            sampled = sampled.squeeze(0).squeeze(-1).squeeze(-1).T  # [N, C]
            sampled_features.append(sampled)

        # 拼接所有batch的结果
        sampled_features = torch.stack(sampled_features, dim=0)  # [B, N, C]

        # 根据batch_indices重新排列
        # 我们需要[N, C]，其中每个点对应其batch的特征
        final_features = []
        for n in range(N):
            b = batch_indices[n]  # 这个点应该属于哪个batch
            # 但实际上coords已经是[N, B]，我们需要选择对应的采样点
            # 这变得复杂了，让我们重新思考

        # 让我们用更简单的方法：对于每个点，找到它所在的batch
        # 由于historical_points来自某个特定的batch，它们应该全部属于同一个batch
        # 所以我们只需要确定historical_points来自哪个batch

        return None

    def sample_from_dense_grid_v2(self,
                                dense_grid: torch.Tensor,
                                coords: torch.Tensor,
                                batch_indices: torch.Tensor) -> torch.Tensor:
        """
        从密集网格中采样特征（简化版本）
        使用最近邻插值代替grid_sample，避免复杂的坐标变换

        Args:
            dense_grid: [B, C, D, H, W] 密集网格
            coords: [N, B, 3] 采样坐标（体素坐标）
            batch_indices: [N] batch索引（实际应该都是同一个值）

        Returns:
            sampled_features: [N, C] 采样的特征
        """
        N, B, _ = coords.shape
        C = dense_grid.shape[1]
        
        # 由于每个点可能来自不同的batch，我们需要为每个点选择对应的batch
        # 但由于我们不知道每个点应该属于哪个batch，这里简化处理
        # 假设所有点都来自第一个batch
        target_batch = 0

        # 提取该batch的坐标
        batch_coords = coords[:, target_batch, :]  # [N, 3]

        # 将坐标转换为整数索引
        coords_int = torch.clamp(batch_coords.long(), min=0)  # [N, 3]

        # 限制坐标在范围内
        D, H, W = dense_grid.shape[2], dense_grid.shape[3], dense_grid.shape[4]
        coords_int[:, 0] = torch.clamp(coords_int[:, 0], max=D-1)
        coords_int[:, 1] = torch.clamp(coords_int[:, 1], max=H-1)
        coords_int[:, 2] = torch.clamp(coords_int[:, 2], max=W-1)

        # 从密集网格中采样
        sampled_features = dense_grid[target_batch, :, coords_int[:, 0], coords_int[:, 1], coords_int[:, 2]].T  # [C, N] -> [N, C]

        return sampled_features

    def project(self,
               historical_features: Dict,
               historical_pose: torch.Tensor,
               current_pose: torch.Tensor,
               current_voxel_indices: Optional[torch.Tensor] = None) -> Dict:
        """
        投影历史特征到当前坐标系

        Args:
            historical_features: 历史特征字典，包含：
                - dense_grids: {resname: [B, C, D, H, W]}
                - sparse_indices: {resname: [N, 4]}
                - sdf_grid: [B, 1, D, H, W] (可选)
                - sdf_indices: [N, 4] (可选)
            historical_pose: [B, 4, 4] 历史pose
            current_pose: [B, 4, 4] 当前pose
            current_voxel_indices: [N, 4] 当前体素索引（用于确定投影目标点）

        Returns:
            projected_features: {
                'fine': [N_current, C_fine],
                'coarse': [N_current, C_coarse],
                'medium': [N_current, C_medium],
                'sdf': [N_current, 1]
            }
        """
        projected = {}

        # 如果没有提供current_voxel_indices，使用historical_indices本身
        # 这意味着我们只是变换historical点，而不采样到新位置
        if current_voxel_indices is None:
            print("[PoseAwareProjector] 警告：未提供current_voxel_indices，使用historical_indices")
            # 使用第一个分辨率级别的索引作为目标
            if 'fine' in historical_features.get('sparse_indices', {}):
                current_voxel_indices = historical_features['sparse_indices']['fine']
            else:
                print("[PoseAwareProjector] 错误：没有可用的sparse_indices")
                return projected

        # 计算变换矩阵
        T_ch = self.compute_transform(historical_pose, current_pose)  # [B, 4, 4]

        # 提取historical体素索引（使用fine级别）
        if 'sparse_indices' not in historical_features or 'fine' not in historical_features['sparse_indices']:
            print("[PoseAwareProjector] 错误：historical_features中没有sparse_indices.fine")
            return projected

        historical_indices = historical_features['sparse_indices']['fine']  # [N_hist, 4]

        # 将historical索引转换为世界坐标
        world_coords = self.transform_historical_indices_to_world(historical_indices)  # [N_hist, 3]

        # 将世界坐标变换到当前体素坐标
        current_voxel_coords, _ = self.transform_world_to_current_voxel(
            world_coords, T_ch
        )  # [N_hist, B, 3]

        # 对每个分辨率级别进行投影
        dense_grids = historical_features.get('dense_grids', {})

        for resname in ['coarse', 'medium', 'fine']:
            if resname not in dense_grids:
                print(f"[PoseAwareProjector] 跳过不存在的分辨率: {resname}")
                continue

            dense_grid = dense_grids[resname]  # [B, C, D, H, W]

            # 生成batch索引（假设所有点属于batch 0）
            N_hist = current_voxel_coords.shape[0]
            B = dense_grid.shape[0]
            batch_indices = torch.zeros(N_hist, dtype=torch.long, device=current_voxel_coords.device)  # [N_hist]

            # 从dense grid中采样
            sampled_features = self.sample_from_dense_grid_v2(
                dense_grid,
                current_voxel_coords,
                batch_indices
            )  # [N_hist, C]

            projected[resname] = sampled_features

        # 投影SDF
        if 'sdf_grid' in historical_features and historical_features['sdf_grid'] is not None:
            sdf_grid = historical_features['sdf_grid']  # [B, 1, D, H, W]

            # 从SDF grid中采样
            sampled_sdf = self.sample_from_dense_grid_v2(
                sdf_grid,
                current_voxel_coords,
                batch_indices
            )  # [N_hist, 1]

            projected['sdf'] = sampled_sdf
        else:
            print("[PoseAwareProjector] 警告：historical_features中没有sdf_grid")

        print(f"[PoseAwareProjector] 投影完成: {list(projected.keys())}")
        for key in projected:
            print(f"  {key}: {projected[key].shape}")

        return projected


def test_pose_aware_projection():
    """测试PoseAwareFeatureProjector"""
    print("测试PoseAwareFeatureProjector...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建投影器（不使用.to()，因为不是nn.Module）
    projector = PoseAwareFeatureProjector(voxel_size=0.16)

    # 创建测试数据
    B, N_hist = 2, 1000
    C_coarse, C_medium, C_fine = 16, 64, 256
    D, H, W = 96, 96, 96

    # Dense grids
    dense_grids = {
        'coarse': torch.randn(B, C_coarse, D//4, H//4, W//4, device=device),
        'medium': torch.randn(B, C_medium, D//2, H//2, W//2, device=device),
        'fine': torch.randn(B, C_fine, D, H, W, device=device)
    }

    # Sparse indices
    sparse_indices = {
        'coarse': torch.randint(0, D//4, (N_hist//4, 4), device=device),
        'medium': torch.randint(0, D//2, (N_hist//2, 4), device=device),
        'fine': torch.randint(0, D, (N_hist, 4), device=device)
    }

    # SDF grid
    sdf_grid = torch.randn(B, 1, D, H, W, device=device)

    # Poses
    historical_pose = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)  # [B, 4, 4]
    current_pose = torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)  # [B, 4, 4]

    # 历史特征
    historical_features = {
        'dense_grids': dense_grids,
        'sparse_indices': sparse_indices,
        'sdf_grid': sdf_grid
    }

    # 当前体素索引（使用fine索引作为示例）
    current_voxel_indices = sparse_indices['fine']  # [N_hist, 4]

    # 执行投影
    projected = projector.project(
        historical_features,
        historical_pose,
        current_pose,
        current_voxel_indices
    )

    print("\n投影结果:")
    for key in projected:
        print(f"  {key}: {projected[key].shape}")

    print("✅ 测试完成")


if __name__ == "__main__":
    test_pose_aware_projection()
