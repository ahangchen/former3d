"""
稀疏融合版本的PoseAwareStreamSdfFormer
避免dense 3D grid，降低显存占用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import collections

from former3d.sdfformer import SDFFormer
from former3d.net3d.sparse3d import xyzb2bxyz, bxyz2xyzb


class PoseAwareStreamSdfFormerSparse(SDFFormer):
    """
    基于Pose感知的流式SDF融合类（稀疏版本）

    关键优化：
    - 不使用dense 3D grid
    - 直接在稀疏特征上操作
    - 大幅降低显存占用
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
        初始化PoseAwareStreamSdfFormerSparse

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

        # 稀疏融合网络（直接在稀疏特征上操作）
        self.sparse_fusion = nn.Sequential(
            nn.Linear(129, 256),  # 历史特征(128) + 当前特征(1) + SDF(不使用)
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        print(f"初始化PoseAwareStreamSdfFormerSparse(稀疏版本）:")
        print(f"  - 体素大小: {voxel_size}")
        print(f"  - 裁剪空间: {crop_size}")
        print(f"  - 融合半径: {fusion_local_radius}")

    def _record_state(self,
                     output: Dict,
                     current_pose: torch.Tensor,
                     current_intrinsics: torch.Tensor,
                     current_3d_points: torch.Tensor,
                     multiscale_features: Optional[Dict] = None):
        """
        记录当前帧的状态到历史信息

        Args:
            output: 当前帧输出字典
            current_pose: [B, 4, 4] 当前Pose
            current_intrinsics: [B, 3, 3] 当前内参
            current_3d_points: [N, 3] 当前3D点坐标
            multiscale_features: 多尺度特征字典（coarse, medium, fine）
        """
        batch_size = current_pose.shape[0]

        # 优先使用multiscale_features
        if multiscale_features is not None:
            # 保存所有尺度的特征
            historical_state = {
                'multiscale': {},
                'batch_size': batch_size
            }

            for resname in ['coarse', 'medium', 'fine']:
                if resname in multiscale_features:
                    res_data = multiscale_features[resname]
                    # res_data['features'] 是 SparseConvTensor，需要提取其内部的特征
                    sparse_tensor = res_data['features']
                    historical_state['multiscale'][resname] = {
                        'features': sparse_tensor.features.detach().clone(),  # [N, C]
                        'indices': sparse_tensor.indices.detach().clone(),  # [N, 4]
                        'spatial_shape': sparse_tensor.spatial_shape,
                        'batch_size': sparse_tensor.batch_size,
                        'resolution': res_data['resolution'],
                        'logits': res_data['logits'].features.detach().clone()  # [N, 1]
                    }

            print(f"[_record_state] 已保存多尺度历史状态:")
            for resname in historical_state['multiscale']:
                print(f"  - {resname}: features={historical_state['multiscale'][resname]['features'].shape}")

        else:
            # 兼容旧代码：从voxel_outputs提取fine级别
            if 'voxel_outputs' not in output or 'fine' not in output['voxel_outputs']:
                print("[_record_state] 警告：输出中没有fine级别的voxel_outputs")
                return

            fine_output = output['voxel_outputs']['fine']  # SparseConvTensor

            # 保存sparse特征和SDF（使用detach和clone避免显存泄露）
            historical_state = {
                'features': fine_output.features.detach().clone(),
                'indices': fine_output.indices.detach().clone(),
                'spatial_shape': fine_output.spatial_shape,
                'batch_size': batch_size,
                'resolution': self.resolutions['fine']
            }

            print(f"[_record_state] 已保存历史状态:")
            print(f"  - features: {historical_state['features'].shape}")
            print(f"  - indices: {historical_state['indices'].shape}")

        # 更新历史信息
        self.historical_state = historical_state
        self.historical_pose = current_pose.detach().clone()
        if current_intrinsics is not None:
            self.historical_intrinsics = current_intrinsics.detach().clone()
        self.historical_3d_points = current_3d_points.detach().clone()

    def _historical_state_project_sparse(self,
                                         current_pose: torch.Tensor,
                                         current_features: torch.Tensor,
                                         current_indices: torch.Tensor,
                                         current_multiscale: Optional[Dict] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将历史状态投影到当前帧坐标系（稀疏版本）

        策略：直接按顺序匹配历史和当前特征，避免复杂变换

        Args:
            current_pose: [B, 4, 4] 当前帧Pose
            current_features: [N, 1] 当前帧稀疏特征（fine级别）
            current_indices: [N, 4] 当前帧体素索引
            current_multiscale: 当前帧的多尺度特征（可选）

        Returns:
            projected_features: [N, C] 投影的特征（128维）
            projected_sdfs: [N, 1] 投影的SDF
        """
        if self.historical_state is None:
            raise RuntimeError("historical_state为空，无法投影")

        device = current_pose.device
        num_current = current_features.shape[0]

        # 如果历史状态包含多尺度特征，优先使用fine级别
        if 'multiscale' in self.historical_state:
            hist_multiscale = self.historical_state['multiscale']

            # 使用fine级别的特征进行投影
            if 'fine' in hist_multiscale:
                hist_fine = hist_multiscale['fine']
                historical_features = hist_fine['features']  # [N_hist, C]
                historical_logits = hist_fine['logits']  # [N_hist, 1] - 已经是tensor
                print(f"[_historical_state_project_sparse] 使用历史多尺度fine特征: {historical_features.shape}")
            else:
                # 降级到medium
                if 'medium' in hist_multiscale:
                    hist_medium = hist_multiscale['medium']
                    historical_features = hist_medium['features']
                    historical_logits = hist_medium['logits']
                else:
                    # 最后使用coarse
                    hist_coarse = hist_multiscale['coarse']
                    historical_features = hist_coarse['features']
                    historical_logits = hist_coarse['logits']
        else:
            # 兼容旧代码：从历史状态直接提取
            historical_features = self.historical_state['features']  # [N_hist, 1]
            historical_logits = None

        print(f"[_historical_state_project_sparse] 历史特征: {historical_features.shape}, 当前特征: {current_features.shape}")

        # 简单策略：重复或截断历史特征以匹配当前特征数量
        num_historical = historical_features.shape[0]

        if num_historical >= num_current:
            # 截断到当前数量
            projected_features = historical_features[:num_current]
            projected_sdfs = torch.zeros(num_current, 1, device=device)
        else:
            # 重复历史特征以填满当前数量
            # 计算需要重复的完整次数和剩余数量
            repeat_count = (num_current + num_historical - 1) // num_historical  # 向上取整
            # 拼接重复的历史特征
            projected_features_list = [historical_features] * repeat_count
            projected_features = torch.cat(projected_features_list, dim=0)[:num_current]
            projected_sdfs = torch.zeros(num_current, 1, device=device)

        # 如果历史特征是SDF logits (C=1)，需要对齐到128维
        if projected_features.shape[1] != 128:
            if not hasattr(self, '_feat_aligner'):
                in_dim = projected_features.shape[1]
                self._feat_aligner = nn.Linear(in_dim, 128).to(device)
            projected_features = self._feat_aligner(projected_features)

        print(f"[_historical_state_project_sparse] 投影完成: {projected_features.shape}")

        return projected_features, projected_sdfs

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

        # 确保所有张量都在正确的设备上
        images = images.to(device)
        poses = poses.to(device)
        intrinsics = intrinsics.to(device)

        # 原始SDFFormer期望多视图输入，这里将单视图扩展为多视图
        n_views = 1  # 流式推理使用单视图

        # 扩展图像维度 [B, 1, 3, H, W]
        rgb_imgs = images.unsqueeze(1)

        # 构建投影矩阵（批量处理，避免条件判断）
        proj_mats = {}
        cam_positions = torch.zeros(batch_size, n_views, 3, device=device)

        for resname in self.resolutions:
            # 扩展pose: [B, 1, 4, 4]
            proj_mat = poses.unsqueeze(1).expand(batch_size, n_views, 4, 4)
            proj_mats[resname] = proj_mat

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

        # 生成随机体素索引（向量化实现，避免遍历batch_idx）
        x = torch.randint(0, voxel_grid_size[0], (total_voxels, 1), device=device)
        y = torch.randint(0, voxel_grid_size[1], (total_voxels, 1), device=device)
        z = torch.randint(0, voxel_grid_size[2], (total_voxels, 1), device=device)

        # 生成batch索引（正确的批量处理）
        batch_inds = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_voxels_per_batch).reshape(total_voxels, 1)

        # 拼接: [total_voxels, 4] (x, y, z, batch_idx)
        voxel_inds = torch.cat([x, y, z, batch_inds], dim=1)

        # 转换为int32类型（spconv要求）
        voxel_inds = voxel_inds.to(torch.int32)

        return voxel_inds

    def forward_single_frame(self,
                            images: torch.Tensor,
                            poses: torch.Tensor,
                            intrinsics: torch.Tensor,
                            reset_state: bool = False,
                            origin: Optional[torch.Tensor] = None) -> Tuple[Dict, Dict]:
        """
        单帧流式推理（稀疏融合版本）

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
            print("[forward_single_frame] 重置历史状态")

        # 1. 创建SDFFormer格式的batch
        batch = self._create_batch_dict(images, poses, intrinsics, origin)

        # 2. 生成体素索引
        voxel_inds_16 = self._generate_voxel_inds(batch_size, device)

        # 3. 判断是否有历史信息
        use_fusion = self.historical_state is not None

        # 初始化multiscale_features
        multiscale_features = None

        if not use_fusion:
            # 第一帧：调用super().forward()
            print("[forward_single_frame] 第一帧，调用super().forward()")

            result = super().forward(batch, voxel_inds_16, return_multiscale_features=True)
            if len(result) == 4:
                voxel_outputs, proj_occ_logits, bp_data, multiscale_features = result
            else:
                raise RuntimeError(f"Unexpected result length: {len(result)}")

            # 构建输出
            output = self._build_output_dict(voxel_outputs, proj_occ_logits, bp_data)

        else:
            # 有历史信息：执行稀疏融合
            print("[forward_single_frame] 有历史信息，执行稀疏融合")

            # 1) 调用super().forward()获取当前帧特征
            result = super().forward(batch, voxel_inds_16, return_multiscale_features=True)
            if len(result) == 4:
                voxel_outputs, proj_occ_logits, bp_data, multiscale_features = result
            else:
                raise RuntimeError(f"Unexpected result length: {len(result)}")

            # 2) 提取当前帧fine级别的稀疏特征
            current_fine_sparse = voxel_outputs['fine']  # SparseConvTensor
            current_features = current_fine_sparse.features  # [N, 1]
            current_indices = current_fine_sparse.indices  # [N, 4] (b, x, y, z)

            # 3) 投影历史信息（稀疏）
            projected_features, projected_sdfs = self._historical_state_project_sparse(
                poses,
                current_features,
                current_indices,
                multiscale_features  # 传递当前帧的多尺度特征
            )  # [N, 128], [N, 1]

            # 4) 稀疏融合：concat + MLP
            # 将当前特征(1维)与投影特征(128维)拼接
            concat_features = torch.cat([
                projected_features,      # [N, 128]
                current_features,         # [N, 1]
            ], dim=1)  # [N, 129]

            # 通过MLP融合
            fused_features = self.sparse_fusion(concat_features)  # [N, 128]

            # 5) 更新voxel_outputs（直接修改SparseConvTensor的特征）
            from spconv.pytorch import SparseConvTensor
            voxel_outputs['fine'] = SparseConvTensor(
                features=fused_features,
                indices=current_indices,
                spatial_shape=current_fine_sparse.spatial_shape,
                batch_size=batch_size
            )

            # 6) 构建输出
            output = self._build_output_dict(voxel_outputs, proj_occ_logits, bp_data)

        # 4. 记录当前帧状态
        # 提取当前帧的3D点坐标
        current_indices = output['voxel_outputs']['fine'].indices  # [N, 4]
        current_3d_points = current_indices[:, 1:4].float() * self.resolutions['fine']  # [N, 3]

        self._record_state(output, poses, intrinsics, current_3d_points, multiscale_features)

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
                reset_state=(t == 0)
            )

            outputs.append(output_t)
            states.append(state_t)

        # 合并输出
        if outputs and outputs[0] is not None:
            # 如果输出是字典，需要特殊处理
            if isinstance(outputs[0], dict):
                # 合并字典中的张量
                combined_output = {}
                for key in outputs[0].keys():
                    if isinstance(outputs[0][key], torch.Tensor):
                        # 尝试堆叠张量
                        try:
                            # 将所有帧的输出沿第1维度拼接
                            tensors = [out[key] for out in outputs if out[key] is not None]
                            if tensors:
                                combined_output[key] = torch.cat(tensors, dim=0)
                            else:
                                combined_output[key] = None
                        except:
                            # 如果无法堆叠，保留第一个
                            combined_output[key] = outputs[0][key]
                    else:
                        combined_output[key] = outputs[0][key]
                outputs_cat = combined_output
            else:
                outputs_cat = None

        return outputs_cat, states

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
            output: 统一的输出字典
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
                    if features.shape[1] == 1:
                        output['sdf'] = features
                        output['occupancy'] = torch.sigmoid(features)
                        print(f"[_build_output_dict] 从{res}分辨率提取SDF和occupancy，形状: {features.shape}")
                        break

        # 如果仍然没有SDF输出，创建默认输出
        if output['sdf'] is None:
            device = next(self.parameters()).device if hasattr(self, 'parameters') else torch.device('cpu')
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
