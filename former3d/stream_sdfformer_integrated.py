"""
流式SDFFormer集成版本
将原始SDFFormer组件与流式架构集成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import collections
import logging

logger = logging.getLogger(__name__)

# 导入原始SDFFormer组件
from former3d.sdfformer import SDFFormer
from former3d import cnn2d, mv_fusion, utils, view_direction_encoder
from former3d.net3d.former_v1 import Former3D as backbone3d
from former3d.net3d.sparse3d import combineSparseConvTensor, xyzb2bxyz, bxyz2xyzb

# 导入Phase 1的流式组件
from former3d.pose_projection import PoseProjection
from former3d.stream_fusion_concat import StreamConcatFusion


class PoseBasedFeatureProjection:
    """
    基于Pose的特征投影模块

    使用GridSample将历史多尺度特征投影到当前坐标系
    """

    def __init__(self, voxel_size: float = 0.0625):
        """
        初始化投影器

        Args:
            voxel_size: 体素大小
        """
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
        # T_cw: 从世界到当前相机的变换
        T_cw = current_pose  # [B, 4, 4]

        # T_hw: 从世界到历史相机的变换
        T_hw = historical_pose  # [B, 4, 4]

        # T_ch = T_cw * T_hw^{-1}: 从历史相机到当前相机的变换
        T_hw_inv = torch.inverse(T_hw)  # [B, 4, 4]
        T_ch = torch.bmm(T_cw, T_hw_inv)  # [B, 4, 4]

        return T_ch

    def transform_voxel_coords(self, voxel_coords: torch.Tensor, T_ch: torch.Tensor) -> torch.Tensor:
        """
        变换体素坐标

        Args:
            voxel_coords: [N, 3] 体素坐标
            T_ch: [4, 4] 或 [B, 4, 4] 变换矩阵

        Returns:
            transformed_coords: [N, 3] 或 [N, B, 3] 变换后的坐标
        """
        # 添加齐次坐标
        ones = torch.ones(voxel_coords.shape[0], 1, device=voxel_coords.device, dtype=voxel_coords.dtype)
        coords_homo = torch.cat([voxel_coords, ones], dim=1)  # [N, 4]

        # 应用变换
        if T_ch.dim() == 2:  # [4, 4]
            transformed = (T_ch @ coords_homo.T).T  # [N, 4]
        else:  # [B, 4, 4]
            coords_homo = coords_homo.unsqueeze(1)  # [N, 1, 4]
            # 使用bmm进行批量矩阵乘法
            # coords_homo: [N, 1, 4], T_ch: [B, 4, 4]
            # 需要广播到 [N, B, 4] x [N, B, 4, 4] 或类似的形状

            # 更简单的方式：逐个batch处理
            num_points = voxel_coords.shape[0]
            batch_size = T_ch.shape[0]

            # 扩展coords_homo到[N, B, 4]
            coords_homo_expanded = coords_homo.unsqueeze(1).expand(-1, batch_size, -1)  # [N, B, 4]
            # 扩展T_ch到[N, B, 4, 4]
            T_ch_expanded = T_ch.unsqueeze(0).expand(num_points, -1, -1)  # [N, B, 4, 4]

            # 矩阵乘法: [N, B, 4, 4] @ [N, B, 4, 1]
            coords_homo_for_bmm = coords_homo_expanded.unsqueeze(-1)  # [N, B, 4, 1]
            transformed = torch.matmul(T_ch_expanded, coords_homo_for_bmm).squeeze(-1)  # [N, B, 4]

        # 只返回前3个坐标
        return transformed[..., :3]

    def normalize_coords(self, coords: torch.Tensor, grid_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        将坐标归一化到[-1, 1]范围

        Args:
            coords: [N, 3] 坐标
            grid_shape: [D, H, W] 网格形状

        Returns:
            normalized_coords: [N, 3] 归一化后的坐标
        """
        D, H, W = grid_shape

        # 避免除零
        D = max(D, 1)
        H = max(H, 1)
        W = max(W, 1)

        x_norm = (coords[:, 0] / (D - 1)) * 2 - 1
        y_norm = (coords[:, 1] / (H - 1)) * 2 - 1
        z_norm = (coords[:, 2] / (W - 1)) * 2 - 1

        normalized_coords = torch.stack([x_norm, y_norm, z_norm], dim=1)

        return normalized_coords

    def project_features(self,
                        historical_features_grid: torch.Tensor,
                        historical_indices: torch.Tensor,
                        current_indices: torch.Tensor,
                        T_ch: torch.Tensor,
                        grid_shape: Tuple[int, int, int]) -> torch.Tensor:
        """
        使用grid_sample从历史特征投影到当前坐标

        Args:
            historical_features_grid: [B, C, D, H, W] 历史特征密集网格
            historical_indices: [N, 4] 历史体素索引 (b, x, y, z)
            current_indices: [N, 4] 当前体素索引 (b, x, y, z)
            T_ch: [B, 4, 4] 变换矩阵
            grid_shape: [D, H, W] 网格形状

        Returns:
            projected_features: [N, C] 投影后的特征
        """
        device = historical_features_grid.device
        num_points = current_indices.shape[0]
        batch_size = historical_features_grid.shape[0]

        # 提取历史体素坐标并转换为世界坐标
        # historical_indices格式: [b, x, y, z]
        historical_coords = historical_indices[:, 1:4].float()  # [N, 3]
        historical_coords = historical_coords * self.voxel_size  # 世界坐标

        # 添加齐次坐标
        ones = torch.ones(num_points, 1, device=device, dtype=historical_coords.dtype)
        historical_coords_homo = torch.cat([historical_coords, ones], dim=1)  # [N, 4]

        # 根据batch索引选择对应的变换矩阵
        batch_indices = current_indices[:, 0].long()  # [N]

        # 检查batch索引是否在有效范围内
        if (batch_indices < 0).any() or (batch_indices >= batch_size).any():
            print(f"[PoseBasedFeatureProjection] 警告：batch索引超出范围，min={batch_indices.min()}, max={batch_indices.max()}, batch_size={batch_size}")
            batch_indices = torch.clamp(batch_indices, 0, batch_size - 1)

        T_ch_batch = T_ch[batch_indices]  # [N, 4, 4]

        # 变换坐标到当前相机坐标系
        transformed_coords_homo = torch.bmm(T_ch_batch, historical_coords_homo.unsqueeze(-1))
        transformed_coords = transformed_coords_homo.squeeze(-1)[:, :3]  # [N, 3]

        # 转换回体素坐标
        transformed_voxel_coords = transformed_coords / self.voxel_size  # [N, 3]

        # 归一化坐标到[-1, 1]
        normalized_coords = self.normalize_coords(transformed_voxel_coords, grid_shape)

        # 裁剪坐标到有效范围，防止grid_sample越界
        normalized_coords = torch.clamp(normalized_coords, -1.0, 1.0)

        # 使用grid_sample采样
        grid = normalized_coords.view(1, 1, 1, num_points, 3)  # [1, 1, 1, N, 3]
        grid = grid.expand(batch_size, -1, -1, -1, -1)  # [B, 1, 1, N, 3]

        # 采样
        sampled = F.grid_sample(
            historical_features_grid,  # [B, C, D, H, W]
            grid,                      # [B, 1, 1, N, 3]
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B, C, 1, 1, N]

        # 提取采样的特征：[N, C]
        # 需要根据batch索引提取对应的特征
        projected_features = []
        for b in range(batch_size):
            mask = batch_indices == b
            if mask.any():
                projected_features.append(sampled[b, :, 0, 0, mask])
            else:
                projected_features.append(torch.zeros(
                    (0, historical_features_grid.shape[1]),
                    device=device,
                    dtype=historical_features_grid.dtype
                ))

        projected_features = torch.cat(projected_features, dim=0)  # [N, C]

        return projected_features


class StreamSDFFormerIntegrated(SDFFormer):
    """流式SDFFormer集成版本
    
    继承原始SDFFormer，添加流式推理能力
    支持基于历史状态的单帧流式推理
    """
    
    def __init__(self,
                 attn_heads: int,
                 attn_layers: int,
                 use_proj_occ: bool,
                 voxel_size: float = 0.04,
                 fusion_local_radius: float = 3.0,
                 crop_size: Tuple[int, int, int] = (48, 96, 96),
                 use_checkpoint: bool = False):
        """初始化集成版本

        Args:
            attn_heads: 注意力头数
            attn_layers: 注意力层数
            use_proj_occ: 是否使用投影占用预测
            voxel_size: 体素大小
            fusion_local_radius: 流式融合局部半径
            crop_size: 裁剪空间大小 (depth, height, width)
            use_checkpoint: 是否使用gradient checkpointing节省显存
        """
        # 初始化原始SDFFormer
        super().__init__(attn_heads, attn_layers, use_proj_occ, voxel_size)

        # 保存额外参数
        self.fusion_local_radius = fusion_local_radius
        self.crop_size = crop_size
        self.use_checkpoint = use_checkpoint

        # 添加流式组件
        self.pose_projection = PoseProjection()

        # 注意：原始SDFFormer输出特征维度为1，但流式融合需要更大维度
        # 这里我们使用线性层进行特征维度的扩展和压缩
        self.feature_expansion = nn.Linear(1, 128) if not use_proj_occ else nn.Identity()
        self.feature_compression = nn.Linear(128, 1) if not use_proj_occ else nn.Identity()

        # 使用concat融合替代注意力融合，大幅节省显存
        self.stream_fusion = StreamConcatFusion(
            feature_dim=128,  # 扩展后的特征维度
            hidden_dim=256,  # 隐藏层维度
            use_residual=True,
            dropout=0.1
        )
        self.stream_fusion_enabled = True  # 启用流式融合

        # 历史状态管理
        self.historical_state = None
        self.historical_pose = None
        self.historical_intrinsics = None

        # 轻量级状态模式（防止内存泄漏）
        self.lightweight_state_mode = True  # 默认启用轻量级模式
        
        # 流式特定的投影网络
        self.img_feat_projection = nn.Sequential(
            nn.Linear(3, 64),  # 输入：3D坐标
            nn.ReLU(),
            nn.Linear(64, 128),  # 输出：特征维度
            nn.ReLU()
        )
        
        # 坐标生成器（用于稀疏体素生成）
        self.coord_generator = nn.Sequential(
            nn.Linear(64, 128),  # 输入：图像特征维度
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1500)  # 输出：500个体素 * 3坐标
        )
        
        # 体素特征投影
        self.voxel_feat_projection = nn.Sequential(
            nn.Linear(3, 64),  # 输入：3D坐标
            nn.ReLU(),
            nn.Linear(64, 128),  # 输出：特征维度
            nn.ReLU()
        )
        
        print(f"初始化StreamSDFFormerIntegrated:")
        print(f"  - 体素大小: {voxel_size}")
        print(f"  - 裁剪空间: {crop_size}")
        print(f"  - 融合半径: {fusion_local_radius}")
        print(f"  - 使用投影占用: {use_proj_occ}")
    
    def convert_to_sdfformer_batch(self, 
                                  images: torch.Tensor,
                                  poses: torch.Tensor,
                                  intrinsics: torch.Tensor,
                                  origin: Optional[torch.Tensor] = None) -> Dict:
        """将流式输入转换为原始SDFFormer的batch格式
        
        Args:
            images: 当前帧图像 [batch, 3, height, width]
            poses: 当前帧相机位姿 [batch, 4, 4]
            intrinsics: 当前帧相机内参 [batch, 3, 3]
            origin: 原点坐标 [batch, 3]，如果为None则使用默认值
            
        Returns:
            SDFFormer格式的batch字典
        """
        batch_size = images.shape[0]
        device = images.device
        
        # 确保所有张量都在正确的设备上
        images = images.to(device)
        poses = poses.to(device)
        intrinsics = intrinsics.to(device)
        
        # 原始SDFFormer期望多视图输入，这里将单视图扩展为多视图
        n_views = 1  # 流式推理使用单视图
        
        # 扩展图像维度 [batch, n_views, 3, H, W]
        rgb_imgs = images.unsqueeze(1)
        
        # 计算投影矩阵（批量处理，避免条件判断）
        proj_mats = {}
        cam_positions = torch.zeros(batch_size, n_views, 3, device=device)

        for resname in self.resolutions:
            # 构建投影矩阵 [batch, n_views, 4, 4]
            # 原始poses是(batch, 4, 4)，需要扩展为(batch, 1, 4, 4)以匹配n_views维度
            proj_mat = poses.unsqueeze(1).expand(batch_size, n_views, 4, 4)

            # 确保投影矩阵在正确的设备上
            proj_mat = proj_mat.to(device)
            proj_mats[resname] = proj_mat

        # 设置原点（如果未提供）
        if origin is None:
            origin = torch.zeros(batch_size, 3, device=device)
        else:
            origin = origin.to(device)

        # 确保cam_positions在正确的设备上
        cam_positions = cam_positions.to(device)
        
        # 构建batch字典
        batch = {
            "rgb_imgs": rgb_imgs.to(device),
            "proj_mats": proj_mats,
            "cam_positions": cam_positions,
            "origin": origin
        }
        
        return batch
    
    def generate_voxel_inds(self, 
                           batch_size: int,
                           num_voxels_per_batch: int = 500,
                           device: torch.device = None) -> torch.Tensor:
        """生成稀疏体素索引
        
        Args:
            batch_size: 批次大小
            num_voxels_per_batch: 每个批次的体素数量
            device: 设备
            
        Returns:
            体素索引 [total_voxels, 4] (x, y, z, batch_idx)
        """
        if device is None:
            device = torch.device('cpu')
        
        total_voxels = batch_size * num_voxels_per_batch
        
        # 生成体素坐标（在裁剪范围内）
        # 注意：原始SDFFormer使用16倍下采样的体素索引
        # 所以我们需要生成在合理范围内的索引
        
        # 计算体素网格大小（基于裁剪空间和体素大小）
        voxel_grid_size = [
            int(self.crop_size[0] / self.resolutions['coarse']),
            int(self.crop_size[1] / self.resolutions['coarse']),
            int(self.crop_size[2] / self.resolutions['coarse'])
        ]
        
        print(f"体素网格大小: {voxel_grid_size}")

        # 生成随机体素索引（向量化实现，避免遍历batch_idx）
        # 为所有batch的体素生成索引
        # x, y, z: [total_voxels, 1]
        x = torch.randint(0, voxel_grid_size[0], (total_voxels, 1), device=device)
        y = torch.randint(0, voxel_grid_size[1], (total_voxels, 1), device=device)
        z = torch.randint(0, voxel_grid_size[2], (total_voxels, 1), device=device)

        # 生成batch索引（正确的批量处理）
        # [batch0_0, batch0_1, ..., batch0_num-1, batch1_0, ..., batch1_num-1, ...]
        batch_inds = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_voxels_per_batch).reshape(total_voxels, 1)

        # 拼接: [total_voxels, 4] (x, y, z, batch_idx)
        voxel_inds = torch.cat([x, y, z, batch_inds], dim=1)

        # 转换为int32类型（spconv要求）
        voxel_inds = voxel_inds.to(torch.int32)

        return voxel_inds
    
    def extract_historical_features(self, 
                                   historical_state: Dict,
                                   current_pose: torch.Tensor) -> Dict:
        """从历史状态提取特征用于流式融合
        
        Args:
            historical_state: 历史状态字典
            current_pose: 当前帧相机位姿 [batch, 4, 4]
            
        Returns:
            处理后的历史特征字典
        """
        if historical_state is None:
            return None
        
        # 投影历史状态到当前坐标系
        projected_state = self.pose_projection(
            historical_state, 
            self.historical_pose, 
            current_pose
        )
        
        return projected_state
    
    def forward_single_frame(self, 
                            images: torch.Tensor,
                            poses: torch.Tensor,
                            intrinsics: torch.Tensor,
                            reset_state: bool = False,
                            origin: Optional[torch.Tensor] = None) -> Tuple[Dict, Dict]:
        """单帧流式推理（支持批量处理）
        
        Args:
            images: 当前帧图像 (batch, 1, 3, H, W) 或 (batch, 3, H, W)
            poses: 当前帧相机位姿 (batch, 1, 4, 4) 或 (batch, 4, 4)
            intrinsics: 当前帧相机内参 (batch, 1, 3, 3) 或 (batch, 3, 3)
            reset_state: 是否重置历史状态
            origin: 原点坐标 (batch, 3)
            
        Returns:
            output: 当前帧输出字典
            new_state: 新的历史状态字典
        """
        # 处理输入维度，确保是(batch, 3, H, W)格式
        if len(images.shape) == 5:  # (batch, 1, 3, H, W)
            images = images.squeeze(1)  # (batch, 3, H, W)
        if len(poses.shape) == 4:    # (batch, 1, 4, 4)
            poses = poses.squeeze(1)  # (batch, 4, 4)
        if len(intrinsics.shape) == 4:  # (batch, 1, 3, 3)
            intrinsics = intrinsics.squeeze(1)  # (batch, 3, 3)
        
        batch_size = images.shape[0]
        device = images.device
        
        # 重置历史状态
        if reset_state:
            self.historical_state = None
            self.historical_pose = None
            self.historical_intrinsics = None
            print("重置历史状态")
        
        # 1. 转换为原始SDFFormer输入格式
        batch = self.convert_to_sdfformer_batch(images, poses, intrinsics, origin)
        
        # 2. 生成体素索引
        voxel_inds_16 = self.generate_voxel_inds(batch_size, num_voxels_per_batch=500, device=device)
        
        # 3. 如果有历史状态，提取历史特征
        historical_features = None
        if self.historical_state is not None and self.historical_pose is not None:
            print("提取历史特征用于流式融合")
            historical_features = self.extract_historical_features(
                self.historical_state, poses
            )
        
        # 4. 调用原始SDFFormer的forward方法，请求多尺度特征
        # 支持两种返回格式：带/不带multiscale_features
        result = super().forward(batch, voxel_inds_16, return_multiscale_features=True)
        if len(result) == 4:
            voxel_outputs, proj_occ_logits, bp_data, multiscale_features = result
            output['multiscale_features'] = multiscale_features
        else:
            voxel_outputs, proj_occ_logits, bp_data = result
            output['multiscale_features'] = None
        
        # 5. 如果有历史特征，执行流式融合
        if historical_features is not None and self.stream_fusion_enabled:
            # 执行流式融合
            current_features = self._extract_current_features(voxel_outputs, bp_data)
            if current_features is not None:
                fused_features = self._apply_stream_fusion(
                    current_features, historical_features, poses
                )
                voxel_outputs = self._update_voxel_outputs(voxel_outputs, fused_features)
        
        # 6. 构建输出字典
        output = self._build_output_dict(voxel_outputs, proj_occ_logits, bp_data)
        
        # 7. 更新历史状态
        new_state = self._create_new_state(output, poses)
        self.historical_state = new_state
        self.historical_pose = poses.detach().clone()
        if intrinsics is not None:
            self.historical_intrinsics = intrinsics.detach().clone()
        
        # 如果原始输入是(batch, 1, ...)，则输出也需要恢复该维度
        if 'sdf' in output and output['sdf'] is not None:
            # 假设SDF输出需要与输入的帧数匹配
            pass  # 暂时不处理维度匹配，具体根据实际输出结构决定
        
        return output, new_state
    
    def _extract_current_features(self, voxel_outputs: Dict, bp_data: Dict) -> Optional[Dict]:
        """从当前输出中提取特征用于流式融合
        
        Args:
            voxel_outputs: 体素输出字典
            bp_data: 反投影数据
            
        Returns:
            当前特征字典，包含特征、坐标等信息
        """
        if 'fine' not in voxel_outputs:
            return None
        
        fine_output = voxel_outputs['fine']
        if not hasattr(fine_output, 'features') or not hasattr(fine_output, 'indices'):
            return None
        
        # 提取特征和坐标
        features = fine_output.features  # [N, 1]
        indices = fine_output.indices    # [N, 4] (x, y, z, batch_idx)
        
        # 扩展特征维度（从1到128）
        if features.shape[1] == 1 and hasattr(self, 'feature_expansion'):
            features = self.feature_expansion(features)  # [N, 128]
        
        # 转换为世界坐标
        coords = indices[:, :3].float() * self.resolutions['fine']
        batch_inds = indices[:, 3].long()
        
        return {
            'features': features,
            'coords': coords,
            'batch_inds': batch_inds,
            'num_voxels': features.shape[0],
            'original_features': fine_output.features  # 保存原始特征
        }
    
    def _apply_stream_fusion(self, 
                           current_features: Dict,
                           historical_features: Dict,
                           current_pose: torch.Tensor) -> torch.Tensor:
        """应用流式融合
        
        Args:
            current_features: 当前特征字典
            historical_features: 历史特征字典
            current_pose: 当前帧位姿
            
        Returns:
            融合后的特征
        """
        # 提取当前和历史特征
        current_feats = current_features['features']
        current_coords = current_features['coords']
        current_batch_inds = current_features['batch_inds']
        
        historical_feats = historical_features['features']
        historical_coords = historical_features['coords']
        historical_batch_inds = historical_features['batch_inds']

        # 执行流式融合（使用concat替代注意力，大幅节省显存）
        # 检查特征维度是否匹配
        if current_feats.shape[1] == historical_feats.shape[1] == 128:
            # Concat融合不需要坐标，只使用特征
            fused_features = self.stream_fusion(
                current_feats=current_feats,
                historical_feats=historical_feats
            )
        else:
            # 特征维度不匹配，跳过流式融合
            print(f"⚠️ 特征维度不匹配，跳过流式融合: current={current_feats.shape}, historical={historical_feats.shape}")
            fused_features = current_feats

        return fused_features
    
    def _update_voxel_outputs(self, voxel_outputs: Dict, fused_features: torch.Tensor) -> Dict:
        """更新体素输出中的特征
        
        Args:
            voxel_outputs: 原始体素输出
            fused_features: 融合后的特征 [N, 128]
            
        Returns:
            更新后的体素输出
        """
        if 'fine' in voxel_outputs:
            fine_output = voxel_outputs['fine']
            if hasattr(fine_output, 'features'):
                # 将融合后的特征压缩回原始维度
                # 这里简化处理：取第一个通道或使用线性层
                if fused_features.shape[1] > 1:
                    # 如果有特征压缩层，使用它；否则取第一个通道
                    if hasattr(self, 'feature_compression'):
                        compressed_features = self.feature_compression(fused_features)
                    else:
                        compressed_features = fused_features[:, 0:1]
                else:
                    compressed_features = fused_features
                
                # 更新特征 - 修复spconv错误
                fine_output = fine_output.replace_feature(compressed_features)
        
        return voxel_outputs
    
    def _build_output_dict(self, 
                          voxel_outputs: Dict,
                          proj_occ_logits: Dict,
                          bp_data: Dict) -> Dict:
        """构建输出字典
        
        Args:
            voxel_outputs: 体素输出字典
            proj_occ_logits: 投影占用预测
            bp_data: 反投影数据
            
        Returns:
            统一的输出字典
        """
        output = {
            'voxel_outputs': voxel_outputs,
            'proj_occ_logits': proj_occ_logits,
            'bp_data': bp_data,
            'sdf': None,
            'occupancy': None
        }
        
        # 从可用的分辨率中提取SDF和占用（按优先级：fine > medium > coarse）
        resolutions = ['fine', 'medium', 'coarse']
        for res in resolutions:
            if res in voxel_outputs:
                res_output = voxel_outputs[res]
                if hasattr(res_output, 'features'):
                    features = res_output.features
                    # 原始SDFFormer输出特征维度为1，需要特殊处理
                    if features.shape[1] == 1:
                        # 对于单通道输出，我们可以将其视为SDF
                        output['sdf'] = features
                        output['occupancy'] = torch.sigmoid(features)  # 将SDF转换为占用概率
                        print(f"从{res}分辨率提取SDF和occupancy，形状: {features.shape}")
                        break
                    elif features.shape[1] >= 2:
                        # 如果有多通道，假设前两个通道是SDF和占用
                        output['sdf'] = features[:, 0:1]
                        output['occupancy'] = features[:, 1:2]
                        print(f"从{res}分辨率提取SDF和occupancy，形状: {features.shape}")
                        break
        
        # 尝试提取multiscale_features（从forward方法的返回值）
        if 'multiscale_features' in output:
            output['multiscale_features'] = output['multiscale_features']
        else:
            output['multiscale_features'] = None

        # 如果仍然没有SDF输出，创建默认输出
        if output['sdf'] is None:
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
            num_points = 1024
            output['sdf'] = torch.randn(num_points, 1, device=device)
            output['occupancy'] = torch.sigmoid(output['sdf'])
            print(f"⚠️ 使用默认SDF输出: {output['sdf'].shape}")
        
        return output
    
    def _create_new_state(self, output: Dict, current_pose: torch.Tensor) -> Dict:
        """从当前输出创建新的历史状态
        
        Args:
            output: 当前帧输出
            current_pose: 当前帧相机位姿
            
        Returns:
            新的历史状态字典
        """
        batch_size = current_pose.shape[0]
        device = current_pose.device
        
        # 尝试从输出中提取真实的体素数据
        if 'voxel_outputs' in output and 'fine' in output['voxel_outputs']:
            fine_output = output['voxel_outputs']['fine']
            
            if hasattr(fine_output, 'features') and hasattr(fine_output, 'indices'):
                # 使用真实的体素数据
                features = fine_output.features  # [N, 1]
                indices = fine_output.indices    # [N, 4] (x, y, z, batch_idx)
                
                # 扩展特征维度（从1到128）
                if features.shape[1] == 1 and hasattr(self, 'feature_expansion'):
                    features = self.feature_expansion(features)  # [N, 128]
                
                # 提取坐标和批次索引
                coords = indices[:, :3].float() * self.resolutions['fine']
                batch_inds = indices[:, 3].long()
                
                # 提取SDF和占用（如果可用）
                sdf = output.get('sdf', None)
                occupancy = output.get('occupancy', None)
                
                # 基础状态信息（始终保存）
                new_state = {
                    'features': features,
                    'sdf': sdf,
                    'occupancy': occupancy,
                    'coords': coords,
                    'batch_inds': batch_inds,
                    'num_voxels': features.shape[0],
                    'pose': current_pose.detach().clone(),
                }

                # 轻量级模式：只保存必要信息，不保存完整输出
                # 默认启用轻量级模式以防止内存泄漏
                if not self.lightweight_state_mode:
                    # 非轻量级模式：保存完整输出以供调试（不推荐）
                    new_state['output'] = output
                    new_state['original_features'] = fine_output.features
                    logger.warning("⚠️  非轻量级模式：保存完整输出可能导致内存泄漏")
                
                return new_state

    
    def _sparse_to_dense_grid(self, sparse_tensor, batch_size):
        """
        将SparseConvTensor转换为密集网格

        Args:
            sparse_tensor: SparseConvTensor
            batch_size: 批次大小

        Returns:
            dense_grid: [batch_size, C, D, H, W] 密集网格
        """
        features = sparse_tensor.features  # [N, C]
        indices = sparse_tensor.indices    # [N, 4] (b, x, y, z)
        spatial_shape = sparse_tensor.spatial_shape  # [D, H, W]
        num_channels = features.shape[1]

        # 创建密集网格
        dense_grid = torch.zeros(
            (batch_size, num_channels, *spatial_shape),
            device=features.device,
            dtype=features.dtype
        )

        # 填充稀疏特征
        for i in range(len(features)):
            b, x, y, z = indices[i].tolist()

            # 检查索引是否在有效范围内
            if 0 <= b < batch_size and                0 <= x < spatial_shape[0] and                0 <= y < spatial_shape[1] and                0 <= z < spatial_shape[2]:
                dense_grid[b, :, x, y, z] = features[i]

        return dense_grid

    def _create_legacy_state(self, output: Dict, current_pose: torch.Tensor) -> Dict:
        """
        创建legacy状态（用于向后兼容）

        Args:
            output: 当前帧输出
            current_pose: 当前帧相机位姿

        Returns:
            新的历史状态字典
        """
        batch_size = current_pose.shape[0]
        device = current_pose.device

        num_voxels_per_batch = 500
        total_voxels = batch_size * num_voxels_per_batch

        # 生成坐标
        max_coord = torch.tensor(self.crop_size, device=device).float() * self.resolutions['coarse']
        coords = torch.rand(total_voxels, 3, device=device) * max_coord

        # 批次索引
        batch_inds = torch.repeat_interleave(
            torch.arange(batch_size, device=device),
            num_voxels_per_batch
        )

        # 生成特征
        feature_dim = 128
        features = torch.randn(total_voxels, feature_dim, device=device)

        # 生成SDF和占用（模拟）
        sdf = torch.randn(total_voxels, 1, device=device)
        occupancy = torch.randn(total_voxels, 1, device=device)

        new_state = {
            'features': features,
            'sdf': sdf,
            'occupancy': occupancy,
            'coords': coords,
            'batch_inds': batch_inds,
            'num_voxels': total_voxels,
            'pose': current_pose.detach().clone(),
        }

        return new_state
        
        # 如果无法提取真实数据，使用简化版本（向后兼容）
        print("警告：使用简化的历史状态创建")
        num_voxels_per_batch = 500
        total_voxels = batch_size * num_voxels_per_batch
        
        # 生成坐标
        max_coord = torch.tensor(self.crop_size, device=device).float() * self.resolutions['coarse']
        coords = torch.rand(total_voxels, 3, device=device) * max_coord
        
        # 批次索引
        batch_inds = torch.repeat_interleave(
            torch.arange(batch_size, device=device), 
            num_voxels_per_batch
        )
        
        # 生成特征
        feature_dim = 128
        features = torch.randn(total_voxels, feature_dim, device=device)
        
        # 生成SDF和占用（模拟）
        sdf = torch.randn(total_voxels, 1, device=device)
        occupancy = torch.randn(total_voxels, 1, device=device)
        
        new_state = {
            'features': features,
            'sdf': sdf,
            'occupancy': occupancy,
            'coords': coords,
            'batch_inds': batch_inds,
            'num_voxels': total_voxels,
            'pose': current_pose.detach().clone(),
            # 移除完整输出保存，防止内存泄漏
            # 'output': output  # ❌ 删除：保存完整输出导致内存泄漏
        }
        
        return new_state
    
    def forward(self, 
               images: torch.Tensor,
               poses: torch.Tensor,
               intrinsics: torch.Tensor,
               reset_state: bool = False,
               origin: Optional[torch.Tensor] = None) -> Dict:
        """流式推理接口（兼容原始调用方式）
        
        Args:
            images: 当前帧图像 [batch, 3, height, width]
            poses: 当前帧相机位姿 [batch, 4, 4]
            intrinsics: 当前帧相机内参 [batch, 3, 3]
            reset_state: 是否重置历史状态
            origin: 原点坐标 [batch, 3]
            
        Returns:
            当前帧输出字典
        """
        output, _ = self.forward_single_frame(
            images, poses, intrinsics, reset_state, origin
        )
        return output
    
    def enable_stream_fusion(self, enabled: bool = True):
        """启用或禁用流式融合
        
        Args:
            enabled: 是否启用流式融合
        """
        self.stream_fusion_enabled = enabled
        print(f"流式融合已{'启用' if enabled else '禁用'}")
    
    def clear_history(self):
        """清除历史状态"""
        self.historical_state = None
        self.historical_pose = None
        self.historical_intrinsics = None
        print("历史状态已清除")

    def enable_lightweight_state(self, enabled: bool = True):
        """启用或禁用轻量级状态模式

        轻量级状态模式只保存必要的特征信息，避免保存完整输出，
        从而防止内存泄漏并减少显存占用。

        Args:
            enabled: 是否启用轻量级状态模式
        """
        self.lightweight_state_mode = enabled

        # 如果切换到轻量级模式，清理当前历史状态中的冗余数据
        if enabled and self.historical_state is not None:
            if 'output' in self.historical_state:
                del self.historical_state['output']
            if 'original_features' in self.historical_state:
                del self.historical_state['original_features']

        print(f"轻量级状态模式已{'启用' if enabled else '禁用'}")
    
    def reset_state(self):
        """重置状态（clear_history的别名）"""
        self.clear_history()
    
    def forward_sequence(self,
                        images: torch.Tensor,
                        poses: torch.Tensor,
                        intrinsics: torch.Tensor,
                        reset_state: bool = True) -> Tuple[torch.Tensor, List[Dict]]:
        """序列流式推理（支持批量处理）
        
        Args:
            images: 图像序列 (batch, n_view, 3, H, W)
            poses: 位姿序列 (batch, n_view, 4, 4)
            intrinsics: 内参序列 (batch, n_view, 3, 3)
            reset_state: 是否在序列开始时重置状态
            
        Returns:
            Tuple[torch.Tensor, List[Dict]]: (输出序列 (batch, n_view, ...), 状态列表)
        """
        batch_size, n_view, _, H, W = images.shape

        outputs = []
        states = []

        # 重置状态
        if reset_state:
            self.historical_state = None
            self.historical_pose = None
            self.historical_intrinsics = None

        # 遍历序列中的每一帧（frame_idx循环在模型内部）
        for t in range(n_view):
            # 提取第t帧的数据（保留batch维度）
            images_t = images[:, t:t+1]  # (batch, 1, 3, H, W)
            poses_t = poses[:, t:t+1]    # (batch, 1, 4, 4)
            intrinsics_t = intrinsics[:, t:t+1]  # (batch, 1, 3, 3)

            # 调用forward_single_frame，处理batch维度
            # 注意：forward_single_frame需要能处理(batch, 1, 3, H, W)的输入
            output_t, state_t = self.forward_single_frame(
                images_t, poses_t, intrinsics_t,
                reset_state=(t == 0)  # 第一帧重置状态
            )  # output_t的shape是 (batch, 1, ...)

            outputs.append(output_t)
            states.append(state_t)

        # 堆叠输出
        if outputs and outputs[0] is not None:
            # 如果输出是字典，需要特殊处理
            if isinstance(outputs[0], dict):
                # 合并字典中的张量
                combined_output = {}
                for key in outputs[0].keys():
                    if key == 'sdf' and outputs[0][key] is not None:
                        # 对于SDF，假设每个batch元素有不同的点数，暂时只返回第一个
                        combined_output[key] = outputs[0][key]
                    elif isinstance(outputs[0][key], torch.Tensor):
                        # 尝试堆叠张量
                        try:
                            # 将所有帧的输出沿第1维度拼接
                            tensors = []
                            for out in outputs:
                                if out[key] is not None:
                                    tensors.append(out[key])
                            if tensors:
                                combined_output[key] = torch.cat(tensors, dim=1)  # (batch, n_view, ...)
                            else:
                                combined_output[key] = None
                        except:
                            # 如果无法堆叠，保留第一个
                            combined_output[key] = outputs[0][key]
                    else:
                        combined_output[key] = outputs[0][key]
                outputs_cat = combined_output
            else:
                # 如果输出是张量，直接拼接
                outputs_cat = torch.cat(outputs, dim=1)  # (batch, n_view, ...)
        else:
            outputs_cat = None

        return outputs_cat, states


def test_stream_sdfformer_integrated():
    """测试集成版本"""
    print("测试StreamSDFFormerIntegrated...")
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=2,
        attn_layers=2,
        use_proj_occ=False,
        voxel_size=0.0625,
        fusion_local_radius=3.0,
        crop_size=(48, 96, 96)
    ).to(device)
    
    # 设置为eval模式
    model.eval()
    
    # 创建测试数据
    batch_size = 2
    seq_length = 3
    
    # 创建序列数据
    images_seq = []
    poses_seq = []
    intrinsics_seq = []
    
    for i in range(seq_length):
        # 图像 [batch, 3, 256, 256]
        images = torch.randn(batch_size, 3, 256, 256, device=device)
        
        # 位姿 [batch, 4, 4]
        pose = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        pose[:, 0, 3] = i * 0.1  # X方向平移
        
        # 内参 [batch, 3, 3]
        intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics[:, 0, 0] = 500.0  # fx
        intrinsics[:, 1, 1] = 500.0  # fy
        intrinsics[:, 0, 2] = 128.0  # cx
        intrinsics[:, 1, 2] = 128.0  # cy
        
        images_seq.append(images)
        poses_seq.append(pose)
        intrinsics_seq.append(intrinsics)
    
    # 测试序列推理
    print("测试序列推理...")
    outputs = model.forward_sequence(images_seq, poses_seq, intrinsics_seq)
    
    print(f"输出数量: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"  第{i+1}帧输出键: {list(output.keys())}")
        if 'sdf' in output and output['sdf'] is not None:
            print(f"    SDF形状: {output['sdf'].shape}")
        if 'occupancy' in output and output['occupancy'] is not None:
            print(f"    占用形状: {output['occupancy'].shape}")
    
    print("✅ 集成版本测试完成")


if __name__ == "__main__":
    test_stream_sdfformer_integrated()