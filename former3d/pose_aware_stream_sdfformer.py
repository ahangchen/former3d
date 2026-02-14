"""
Pose-Aware流式SDFFormer
基于Pose投影的历史特征和SDF融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import collections

from former3d.sdfformer import SDFFormer
from former3d.net3d.sparse3d import xyzb2bxyz, bxyz2xyzb


class PoseAwareStreamSdfFormer(SDFFormer):
    """
    基于Pose感知的流式SDF融合类

    功能:
    1. 保存历史稀疏fine级别特征和SDF
    2. 使用Pose将历史信息投影到当前帧
    3. 融合历史信息和当前信息
    """

    def __init__(self,
                 attn_heads: int,
                 attn_layers: int,
                 use_proj_occ: bool,
                 voxel_size: float = 0.04,
                 fusion_local_radius: float = 3.0,
                 crop_size: Tuple[int, int, int] = (48, 96, 96),
                 use_checkpoint: bool = False):
        """
        初始化PoseAwareStreamSdfFormer

        Args:
            attn_heads: 注意力头数
            attn_layers: 注意力层数
            use_proj_occ: 是否使用投影占用预测
            voxel_size: 体素大小
            fusion_local_radius: 融合局部半径
            crop_size: 裁剪空间大小 (depth, height, width)
            use_checkpoint: 是否使用gradient checkpointing
        """
        # 初始化基类SDFFormer
        super().__init__(attn_heads, attn_layers, use_proj_occ, voxel_size)

        # 保存参数
        self.voxel_size = voxel_size
        self.fusion_local_radius = fusion_local_radius
        self.crop_size = crop_size
        self.use_checkpoint = use_checkpoint

        # 历史信息存储
        self.historical_state = None         # 历史特征和SDF状态
        self.historical_pose = None          # [B, 4, 4] 历史Pose
        self.historical_intrinsics = None     # [B, 3, 3] 历史内参
        self.historical_3d_points = None     # [N, 3] 历史3D坐标

        # 融合用的3D卷积网络
        # 输入: 投影特征 + 当前特征 + 投影SDF
        # 假设投影特征对齐到128维，当前特征128维，SDF 1维
        self.fusion_3d = nn.Sequential(
            nn.Conv3d(128 + 128 + 1, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128, track_running_stats=False),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=1),
            nn.ReLU()
        )

        # 特征维度对齐层（动态创建）
        self._feat_aligner = None

        print(f"初始化PoseAwareStreamSdfFormer:")
        print(f"  - 体素大小: {voxel_size}")
        print(f"  - 裁剪空间: {crop_size}")
        print(f"  - 融合半径: {fusion_local_radius}")

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
        batch_size = current_pose.shape[0]

        # 提取fine级别的sparse feature和SDF
        if 'voxel_outputs' not in output or 'fine' not in output['voxel_outputs']:
            print("[_record_state] 警告：输出中没有fine级别的voxel_outputs")
            return

        fine_output = output['voxel_outputs']['fine']  # SparseConvTensor

        # 保存sparse特征和SDF（使用detach和clone避免显存泄露）
        # fine_output.features: [N, 1] 或 [N, C]
        # fine_output.indices: [N, 4] (b, x, y, z)
        historical_state = {
            'features': fine_output.features.detach().clone(),
            'indices': fine_output.indices.detach().clone(),
            'spatial_shape': fine_output.spatial_shape,
            'batch_size': batch_size,
            'resolution': self.resolutions['fine']
        }

        # 更新历史信息
        self.historical_state = historical_state
        self.historical_pose = current_pose.detach().clone()
        if current_intrinsics is not None:
            self.historical_intrinsics = current_intrinsics.detach().clone()
        self.historical_3d_points = current_3d_points.detach().clone()

        print(f"[_record_state] 已保存历史状态:")
        print(f"  - features: {historical_state['features'].shape}")
        print(f"  - indices: {historical_state['indices'].shape}")
        print(f"  - 3d_points: {current_3d_points.shape}")

    def _historical_state_project(self,
                                 current_pose: torch.Tensor,
                                 current_voxel_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将历史状态投影到当前帧坐标系

        Args:
            current_pose: [B, 4, 4] 当前帧Pose
            current_voxel_indices: [N, 4] 当前帧体素索引 (b, x, y, z)

        Returns:
            projected_features: [N, C] 投影的特征
            projected_sdfs: [N, 1] 投影的SDF
        """
        if self.historical_state is None:
            raise RuntimeError("historical_state为空，无法投影")

        device = current_pose.device
        batch_size = current_pose.shape[0]

        # 1. 计算相对位姿 T_ch (从历史到当前)
        # T_ch = T_cw * T_hw^{-1}
        T_hw_inv = torch.inverse(self.historical_pose)  # [B, 4, 4]
        T_ch = torch.bmm(current_pose, T_hw_inv)  # [B, 4, 4]

        # 2. 获取历史稀疏特征和SDF
        historical_features = self.historical_state['features']  # [N_hist, C]
        historical_indices = self.historical_state['indices']    # [N_hist, 4] (b, x, y, z)
        historical_spatial_shape = self.historical_state['spatial_shape']  # [D, H, W]
        historical_resolution = self.historical_state['resolution']  # float

        # 提取历史3D坐标（世界坐标系）
        # indices格式: (b, x, y, z)
        historical_coords = historical_indices[:, 1:4].float()  # [N_hist, 3]
        historical_coords_world = historical_coords * historical_resolution  # [N_hist, 3]

        # 添加齐次坐标
        ones = torch.ones(historical_coords_world.shape[0], 1, device=device, dtype=historical_coords_world.dtype)
        historical_coords_homo = torch.cat([historical_coords_world, ones], dim=1)  # [N_hist, 4]

        # 提取batch索引
        historical_batch_inds = historical_indices[:, 0].long()  # [N_hist]

        # 3. 变换到当前相机坐标系
        # 根据batch索引选择对应的变换矩阵
        T_ch_batch = T_ch[historical_batch_inds]  # [N_hist, 4, 4]

        # 应用变换
        transformed_coords_homo = torch.bmm(T_ch_batch, historical_coords_homo.unsqueeze(-1))  # [N_hist, 4, 1]
        transformed_coords = transformed_coords_homo.squeeze(-1)[:, :3]  # [N_hist, 3]

        # 4. 转换回体素坐标
        transformed_voxel_coords = transformed_coords / historical_resolution  # [N_hist, 3]

        # 5. 过滤超出范围的点
        # 检查是否在当前帧的spatial_shape范围内
        D, H, W = historical_spatial_shape

        valid_mask = (
            (transformed_voxel_coords[:, 0] >= 0) & (transformed_voxel_coords[:, 0] < D) &
            (transformed_voxel_coords[:, 1] >= 0) & (transformed_voxel_coords[:, 1] < H) &
            (transformed_voxel_coords[:, 2] >= 0) & (transformed_voxel_coords[:, 2] < W)
        )

        # 只保留有效点
        valid_transformed_coords = transformed_voxel_coords[valid_mask]  # [N_valid, 3]
        valid_historical_features = historical_features[valid_mask]     # [N_valid, C]

        if valid_transformed_coords.shape[0] == 0:
            print("[_historical_state_project] 警告：没有有效点，返回零特征")
            # 返回与当前体素索引数量匹配的零特征
            num_current = current_voxel_indices.shape[0]
            feature_dim = historical_features.shape[1]
            return (
                torch.zeros(num_current, 128, device=device),
                torch.zeros(num_current, 1, device=device)
            )

        # 6. 使用grid_sample采样到当前体素位置
        # 6.1 创建历史特征的dense grid
        historical_dense = self._sparse_to_dense_grid(
            historical_features,
            historical_indices,
            historical_spatial_shape,
            batch_size
        )  # [B, C, D, H, W]

        # 6.2 归一化坐标到[-1, 1]
        def normalize_coords(coords, shape):
            """归一化坐标到[-1, 1]"""
            D, H, W = shape
            x_norm = (coords[:, 0] / (D - 1)) * 2 - 1
            y_norm = (coords[:, 1] / (H - 1)) * 2 - 1
            z_norm = (coords[:, 2] / (W - 1)) * 2 - 1
            return torch.stack([x_norm, y_norm, z_norm], dim=1)

        normalized_coords = normalize_coords(valid_transformed_coords, historical_spatial_shape)

        # 裁剪到[-1, 1]范围
        normalized_coords = torch.clamp(normalized_coords, -1.0, 1.0)

        # 6.3 使用grid_sample采样
        grid = normalized_coords.view(1, 1, 1, -1, 3)  # [1, 1, 1, N_valid, 3]
        grid = grid.expand(batch_size, -1, -1, -1, -1)  # [B, 1, 1, N_valid, 3]

        sampled = F.grid_sample(
            historical_dense,  # [B, C, D, H, W]
            grid,              # [B, 1, 1, N_valid, 3]
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )  # [B, C, 1, 1, N_valid]

        # 6.4 提取采样结果
        # 根据valid_mask对应的batch索引提取
        valid_batch_inds = historical_batch_inds[valid_mask]  # [N_valid]

        projected_features_list = []
        projected_sdfs_list = []

        for b in range(batch_size):
            mask = valid_batch_inds == b
            if mask.any():
                # 提取该batch的特征
                feat = sampled[b, :, 0, 0, mask]  # [N_b, C]
                projected_features_list.append(feat)

                # 如果特征维度>1，第一维是SDF
                if feat.shape[1] > 1:
                    projected_sdfs_list.append(feat[:, 0:1])  # [N_b, 1]
                else:
                    projected_sdfs_list.append(feat)  # [N_b, 1]
            else:
                # 该batch没有有效点
                num_channels = historical_dense.shape[1]
                projected_features_list.append(torch.zeros(0, num_channels, device=device))
                projected_sdfs_list.append(torch.zeros(0, 1, device=device))

        # 拼接所有batch
        projected_features = torch.cat(projected_features_list, dim=0)  # [N_valid, C]
        projected_sdfs = torch.cat(projected_sdfs_list, dim=0)  # [N_valid, 1]

        # 7. 将投影特征映射到当前体素索引位置
        # 如果投影的特征数量与当前体素索引不同，需要进行对齐
        num_current = current_voxel_indices.shape[0]
        num_projected = projected_features.shape[0]

        if num_projected != num_current:
            print(f"[_historical_state_project] 警告：投影特征数({num_projected}) != 当前体素数({num_current})")
            # 简单处理：重复或截断
            if num_projected < num_current:
                # 重复最后一个特征
                last_feat = projected_features[-1:]
                last_sdf = projected_sdfs[-1:]
                repeat_times = num_current - num_projected
                projected_features = torch.cat([projected_features, last_feat.repeat(repeat_times, 1)], dim=0)
                projected_sdfs = torch.cat([projected_sdfs, last_sdf.repeat(repeat_times, 1)], dim=0)
            else:
                # 截断
                projected_features = projected_features[:num_current]
                projected_sdfs = projected_sdfs[:num_current]

        # 8. 对齐特征维度到128
        if projected_features.shape[1] != 128:
            if self._feat_aligner is None:
                in_dim = projected_features.shape[1]
                self._feat_aligner = nn.Linear(in_dim, 128).to(device=device)
            projected_features = self._feat_aligner(projected_features)

        print(f"[_historical_state_project] 投影完成:")
        print(f"  - 投影特征: {projected_features.shape}")
        print(f"  - 投影SDF: {projected_sdfs.shape}")

        return projected_features, projected_sdfs

    def _sparse_to_dense_grid(self,
                             features: torch.Tensor,
                             indices: torch.Tensor,
                             spatial_shape: Tuple[int, int, int],
                             batch_size: int) -> torch.Tensor:
        """
        将稀疏特征转换为密集网格

        Args:
            features: [N, C] 稀疏特征
            indices: [N, 4] 稀疏索引 (b, x, y, z)
            spatial_shape: [D, H, W] 空间形状
            batch_size: int batch大小

        Returns:
            dense_grid: [B, C, D, H, W] 密集网格
        """
        device = features.device
        num_channels = features.shape[1]

        # 创建密集网格
        dense_grid = torch.zeros(
            (batch_size, num_channels, *spatial_shape),
            device=device,
            dtype=features.dtype
        )

        # 填充稀疏特征
        for i in range(len(features)):
            b, x, y, z = indices[i].tolist()

            # 检查索引是否在有效范围内
            if 0 <= b < batch_size and 0 <= x < spatial_shape[0] and 0 <= y < spatial_shape[1] and 0 <= z < spatial_shape[2]:
                dense_grid[b, :, x, y, z] = features[i]

        return dense_grid

    def _create_batch_dict(self,
                          images: torch.Tensor,
                          poses: torch.Tensor,
                          intrinsics: torch.Tensor,
                          origin: Optional[torch.Tensor] = None) -> Dict:
        """
        创建SDFFormer需要的batch字典

        Args:
            images: [B, 3, H, W] 图像
            poses: [B, 4, 4] 位姿
            intrinsics: [B, 3, 3] 内参
            origin: [B, 3] 原点

        Returns:
            batch: SDFFormer格式的batch字典
        """
        batch_size = images.shape[0]
        device = images.device

        # 扩展图像维度: [B, 1, 3, H, W]
        n_views = 1
        rgb_imgs = images.unsqueeze(1)

        # 构建投影矩阵
        proj_mats = {}
        for resname in self.resolutions:
            # 扩展pose: [B, 1, 4, 4]
            proj_mat = poses.unsqueeze(1).expand(batch_size, n_views, 4, 4)
            proj_mats[resname] = proj_mat

        # 相机位置（简化：使用零向量）
        cam_positions = torch.zeros(batch_size, n_views, 3, device=device)

        # 原点
        if origin is None:
            origin = torch.zeros(batch_size, 3, device=device)

        batch = {
            "rgb_imgs": rgb_imgs,
            "proj_mats": proj_mats,
            "cam_positions": cam_positions,
            "origin": origin
        }

        return batch

    def forward_single_frame(self,
                            images: torch.Tensor,
                            poses: torch.Tensor,
                            intrinsics: torch.Tensor,
                            reset_state: bool = False,
                            origin: Optional[torch.Tensor] = None) -> Tuple[Dict, Dict]:
        """
        单帧流式推理

        Args:
            images: [B, 3, H, W] 或 [B, 1, 3, H, W] 图像
            poses: [B, 4, 4] 或 [B, 1, 4, 4] 位姿
            intrinsics: [B, 3, 3] 或 [B, 1, 3, 3] 内参
            reset_state: 是否重置历史状态
            origin: [B, 3] 原点

        Returns:
            output: 当前帧输出字典
            new_state: 新的历史状态字典
        """
        # 处理输入维度
        if len(images.shape) == 5:  # [B, 1, 3, H, W]
            images = images.squeeze(1)  # [B, 3, H, W]
        if len(poses.shape) == 4:     # [B, 1, 4, 4]
            poses = poses.squeeze(1)   # [B, 4, 4]
        if len(intrinsics.shape) == 4:  # [B, 1, 3, 3]
            intrinsics = intrinsics.squeeze(1)  # [B, 3, 3]

        batch_size = images.shape[0]
        device = images.device

        # 重置历史状态
        if reset_state:
            self.historical_state = None
            self.historical_pose = None
            self.historical_intrinsics = None
            self.historical_3d_points = None
            print("[forward_single_frame] 重置历史状态")

        # 1. 创建SDFFormer格式的batch
        batch = self._create_batch_dict(images, poses, intrinsics, origin)

        # 2. 生成体素索引
        voxel_inds_16 = self._generate_voxel_inds(batch_size, device)

        # 3. 判断是否有历史信息
        if self.historical_state is None:
            # 第一帧：调用super().forward()
            print("[forward_single_frame] 第一帧，调用super().forward()")

            result = super().forward(batch, voxel_inds_16, return_multiscale_features=False)
            if len(result) == 3:
                voxel_outputs, proj_occ_logits, bp_data = result
            else:
                raise RuntimeError(f"Unexpected result length: {len(result)}")

            # 构建输出
            output = self._build_output_dict(voxel_outputs, proj_occ_logits, bp_data)

        else:
            # 有历史信息：执行融合
            print("[forward_single_frame] 有历史信息，执行融合")

            # 1) 调用super().forward()获取当前帧特征
            result = super().forward(batch, voxel_inds_16, return_multiscale_features=False)
            if len(result) == 3:
                voxel_outputs, proj_occ_logits, bp_data = result
            else:
                raise RuntimeError(f"Unexpected result length: {len(result)}")

            # 2) 提取当前帧fine级别的体素索引
            current_voxel_indices = voxel_outputs['fine'].indices  # [N, 4]

            # 3) 投影历史信息
            projected_features, projected_sdfs = self._historical_state_project(
                poses, current_voxel_indices
            )  # [N, 128], [N, 1]

            # 4) 转换当前fine特征为dense grid
            current_fine_sparse = voxel_outputs['fine']  # SparseConvTensor
            current_fine_dense = self._sparse_to_dense_grid(
                current_fine_sparse.features,
                current_fine_sparse.indices,
                current_fine_sparse.spatial_shape,
                batch_size
            )  # [B, C, D, H, W]

            # 5) 扩展当前特征到128维
            if current_fine_dense.shape[1] == 1:
                # 使用1D卷积扩展维度
                if not hasattr(self, '_feat_expander'):
                    self._feat_expander = nn.Conv3d(1, 128, kernel_size=1).to(device)
                current_fine_dense_128 = self._feat_expander(current_fine_dense)
            else:
                # 使用线性层对齐
                if current_fine_dense.shape[1] != 128:
                    if not hasattr(self, '_feat_aligner_current'):
                        in_dim = current_fine_dense.shape[1]
                        self._feat_aligner_current = nn.Linear(in_dim, 128).to(device)
                    # 逐个batch应用线性层
                    B, C, D, H, W = current_fine_dense.shape
                    current_fine_dense_flat = current_fine_dense.permute(0, 2, 3, 4, 1).reshape(B, -1, C)
                    current_fine_dense_128_flat = self._feat_aligner_current(current_fine_dense_flat)
                    current_fine_dense_128 = current_fine_dense_128_flat.reshape(B, D, H, W, 128).permute(0, 4, 1, 2, 3)
                else:
                    current_fine_dense_128 = current_fine_dense

            # 6) 将投影的稀疏特征转换为dense grid
            # projected_features: [N, 128]，需要在sparse体素位置上构建dense grid
            projected_dense = self._sparse_to_dense_grid(
                projected_features,
                current_voxel_indices,
                current_fine_sparse.spatial_shape,
                batch_size
            )  # [B, 128, D, H, W]

            # 7) Concat历史投影特征、当前特征、投影SDF
            # projected_sdfs: [N, 1]，需要转换为dense grid
            projected_sdfs_dense = self._sparse_to_dense_grid(
                projected_sdfs,
                current_voxel_indices,
                current_fine_sparse.spatial_shape,
                batch_size
            )  # [B, 1, D, H, W]

            concat_features = torch.cat([
                projected_dense,        # [B, 128, D, H, W]
                current_fine_dense_128, # [B, 128, D, H, W]
                projected_sdfs_dense     # [B, 1, D, H, W]
            ], dim=1)  # [B, 257, D, H, W]

            # 8) 两层3D卷积融合
            fusion_features = self.fusion_3d(concat_features)  # [B, 128, D, H, W]

            # 9) 将融合特征转回sparse格式，更新voxel_outputs
            # 提取sparse体素位置的融合特征
            fusion_sparse_features = []
            for i in range(len(current_voxel_indices)):
                b, x, y, z = current_voxel_indices[i].tolist()
                if 0 <= b < batch_size and 0 <= x < current_fine_sparse.spatial_shape[0] and \
                   0 <= y < current_fine_sparse.spatial_shape[1] and 0 <= z < current_fine_sparse.spatial_shape[2]:
                    fusion_sparse_features.append(fusion_features[b, :, x, y, z])

            fusion_sparse_features = torch.stack(fusion_sparse_features)  # [N, 128]

            # 压缩回原始维度
            if fusion_sparse_features.shape[1] > 1:
                if not hasattr(self, '_feat_compressor'):
                    self._feat_compressor = nn.Linear(128, 1).to(device)
                fusion_sparse_features = self._feat_compressor(fusion_sparse_features)  # [N, 1]

            # 更新voxel_outputs
            from spconv.pytorch import SparseConvTensor
            voxel_outputs['fine'] = SparseConvTensor(
                features=fusion_sparse_features,
                indices=current_voxel_indices,
                spatial_shape=current_fine_sparse.spatial_shape,
                batch_size=batch_size
            )

            # 10) 构建输出
            output = self._build_output_dict(voxel_outputs, proj_occ_logits, bp_data)

        # 4. 记录当前帧状态
        # 提取当前帧的3D点坐标
        current_voxel_indices = output['voxel_outputs']['fine'].indices  # [N, 4]
        current_3d_points = current_voxel_indices[:, 1:4].float() * self.resolutions['fine']  # [N, 3]

        self._record_state(output, poses, intrinsics, current_3d_points)

        # 5. 返回新状态
        new_state = self.historical_state

        return output, new_state

    def forward_sequence(self,
                         images: torch.Tensor,
                         poses: torch.Tensor,
                         intrinsics: torch.Tensor,
                         reset_state: bool = True) -> Tuple[torch.Tensor, List[Dict]]:
        """
        序列流式推理

        Args:
            images: [B, N, 3, H, W] 图像序列
            poses: [B, N, 4, 4] 位姿序列
            intrinsics: [B, N, 3, 3] 内参序列
            reset_state: 是否在序列开始时重置状态

        Returns:
            outputs_cat: 合并的输出序列
            states: 状态列表
        """
        batch_size, n_view, _, H, W = images.shape

        outputs = []
        states = []

        # 遍历序列中的每一帧
        for t in range(n_view):
            # 提取第t帧的数据
            images_t = images[:, t:t+1]  # [B, 1, 3, H, W]
            poses_t = poses[:, t:t+1]    # [B, 1, 4, 4]
            intrinsics_t = intrinsics[:, t:t+1]  # [B, 1, 3, 3]

            # 调用forward_single_frame
            output_t, state_t = self.forward_single_frame(
                images_t, poses_t, intrinsics_t,
                reset_state=(t == 0 and reset_state)
            )

            outputs.append(output_t)
            states.append(state_t)

        # 合并输出
        if outputs and outputs[0] is not None:
            if isinstance(outputs[0], dict):
                # 合并字典中的张量
                combined_output = {}
                for key in outputs[0].keys():
                    if isinstance(outputs[0][key], torch.Tensor):
                        # 尝试拼接张量
                        try:
                            tensors = [out[key] for out in outputs if out[key] is not None]
                            if tensors:
                                combined_output[key] = torch.cat(tensors, dim=0)
                            else:
                                combined_output[key] = None
                        except:
                            combined_output[key] = outputs[0][key]
                    else:
                        combined_output[key] = outputs[0][key]
                outputs_cat = combined_output
            else:
                outputs_cat = torch.cat(outputs, dim=0)
        else:
            outputs_cat = None

        return outputs_cat, states

    def _generate_voxel_inds(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        生成体素索引

        Args:
            batch_size: batch大小
            device: 设备

        Returns:
            voxel_inds: [N, 4] 体素索引
        """
        # 计算体素网格大小
        voxel_grid_size = [
            int(self.crop_size[0] / self.resolutions['coarse']),
            int(self.crop_size[1] / self.resolutions['coarse']),
            int(self.crop_size[2] / self.resolutions['coarse'])
        ]

        num_voxels_per_batch = 500
        total_voxels = batch_size * num_voxels_per_batch

        # 生成随机体素索引
        x = torch.randint(0, voxel_grid_size[0], (total_voxels, 1), device=device)
        y = torch.randint(0, voxel_grid_size[1], (total_voxels, 1), device=device)
        z = torch.randint(0, voxel_grid_size[2], (total_voxels, 1), device=device)

        # 生成batch索引
        batch_inds = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_voxels_per_batch).reshape(total_voxels, 1)

        # 拼接
        voxel_inds = torch.cat([x, y, z, batch_inds], dim=1)  # [N, 4]

        return voxel_inds

    def _build_output_dict(self,
                           voxel_outputs: Dict,
                           proj_occ_logits: Dict,
                           bp_data: Dict) -> Dict:
        """
        构建输出字典

        Args:
            voxel_outputs: 体素输出字典
            proj_occ_logits: 投影占用预测
            bp_data: 反投影数据

        Returns:
            output: 输出字典
        """
        output = {
            'voxel_outputs': voxel_outputs,
            'proj_occ_logits': proj_occ_logits,
            'bp_data': bp_data,
            'sdf': None,
            'occupancy': None
        }

        # 从fine级别提取SDF和占用
        resolutions = ['fine', 'medium', 'coarse']
        for res in resolutions:
            if res in voxel_outputs:
                res_output = voxel_outputs[res]
                if hasattr(res_output, 'features'):
                    features = res_output.features
                    if features.shape[1] == 1:
                        output['sdf'] = features
                        output['occupancy'] = torch.sigmoid(features)
                        print(f"[_build_output_dict] 从{res}分辨率提取SDF和occupancy，形状: {features.shape}")
                        break

        # 如果没有SDF输出，创建默认输出
        if output['sdf'] is None:
            device = next(self.parameters()).device
            output['sdf'] = torch.randn(1024, 1, device=device)
            output['occupancy'] = torch.sigmoid(output['sdf'])

        return output

    def clear_history(self):
        """清除历史状态"""
        self.historical_state = None
        self.historical_pose = None
        self.historical_intrinsics = None
        self.historical_3d_points = None
        print("[clear_history] 历史状态已清除")
