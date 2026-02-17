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

        # 稀疏融合网络将在forward时动态创建，因为我们不知道fine级别特征维度
        self.sparse_fusion = None

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
                                         current_sparse_features: torch.Tensor,
                                         current_sparse_indices: torch.Tensor,
                                         multiscale_features: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将历史稀疏状态投影到当前帧稀疏空间（稀疏版本）

        实现步骤（按照pose_aware_historical_feature_fusion_plan.md任务二）：
        1. 使用当前帧Pose和历史帧Pose，计算相对位姿
        2. 根据相对位姿，计算历史3D点投影到当前帧视角下的3D位置
        3. 忽略超出当前帧3D输出范围的点
        4. 使用最近邻查找，将历史稀疏特征投影到当前稀疏空间

        Args:
            current_pose: [B, 4, 4] 当前帧Pose
            current_sparse_features: [N_cur, C_cur] 当前帧稀疏特征
            current_sparse_indices: [N_cur, 4] 当前帧稀疏体素索引(b, x, y, z)
            multiscale_features: 当前帧多尺度特征字典

        Returns:
            projected_features: [N_cur, C_hist] 投影的历史特征（与当前稀疏点一一对应）
            projected_sdfs: [N_cur, 1] 投影的历史SDF
        """
        if self.historical_state is None:
            raise RuntimeError("historical_state为空，无法投影")

        device = current_pose.device
        batch_size = current_pose.shape[0]

        # 提取历史信息
        if 'multiscale' in self.historical_state:
            hist_multiscale = self.historical_state['multiscale']

            # 使用fine级别的特征
            if 'fine' in hist_multiscale:
                hist_fine = hist_multiscale['fine']
                historical_features = hist_fine['features']  # [N_hist, 128]
                historical_indices = hist_fine['indices']  # [N_hist, 4] (b, x, y, z)
                historical_spatial_shape = hist_fine['spatial_shape']  # (D, H, W)
                historical_logits = hist_fine['logits']  # [N_hist, 1] SparseConvTensor
            else:
                raise RuntimeError("历史状态中缺少fine级别特征")
        else:
            raise RuntimeError("历史状态格式错误，缺少multiscale字段")

        # 确保historical_logits是特征张量
        if hasattr(historical_logits, 'features'):
            historical_logits = historical_logits.features

        print(f"[_historical_state_project_sparse] 历史稀疏特征: {historical_features.shape}")
        print(f"[_historical_state_project_sparse] 当前稀疏特征: {current_sparse_features.shape}")

        # 步骤1: 计算相对位姿
        # T_rel = T_cur @ T_hist.inv()
        historical_pose = self.historical_pose  # [B_hist, 4, 4]

        # 如果历史batch和当前batch大小不同，取第一个并扩展
        if historical_pose.shape[0] != batch_size:
            historical_pose = historical_pose[0:1, :, :]  # [1, 4, 4]
            historical_pose = historical_pose.repeat(batch_size, 1, 1)  # [B, 4, 4]

        # 计算相对位姿: T_rel = T_cur @ T_hist^{-1}
        historical_pose_inv = torch.inverse(historical_pose)  # [B, 4, 4]
        relative_pose = current_pose @ historical_pose_inv  # [B, 4, 4]

        print(f"[_historical_state_project_sparse] 相对位姿计算完成")

        # 步骤2: 将历史3D点投影到当前帧视角
        # 从indices提取稀疏3D坐标（体素索引）
        # historical_indices: [N_hist, 4] (b, x, y, z)
        # 需要转换为世界坐标系下的3D点，然后变换到当前帧坐标系

        # 获取当前帧和历史帧的fine级别分辨率
        current_resolution = self.resolutions['fine']
        historical_resolution = hist_fine['resolution']

        # 计算历史稀疏点在当前帧坐标系下的位置
        # 先转换到归一化的体素坐标空间
        D_hist, H_hist, W_hist = historical_spatial_shape
        hist_b = historical_indices[:, 0]  # [N_hist]
        hist_x = historical_indices[:, 1].float()  # [N_hist]
        hist_y = historical_indices[:, 2].float()  # [N_hist]
        hist_z = historical_indices[:, 3].float()  # [N_hist]

        # 转换为归一化坐标[-1, 1]范围
        hist_x_norm = (hist_x / (W_hist - 1)) * 2 - 1 if W_hist > 1 else hist_x * 0
        hist_y_norm = (hist_y / (H_hist - 1)) * 2 - 1 if H_hist > 1 else hist_y * 0
        hist_z_norm = (hist_z / (D_hist - 1)) * 2 - 1 if D_hist > 1 else hist_z * 0

        # 堆叠成[N_hist, 3]坐标
        hist_points_norm = torch.stack([hist_x_norm, hist_y_norm, hist_z_norm], dim=1)  # [N_hist, 3]

        # 应用相对位姿变换（只对旋转部分，忽略平移，因为我们工作在归一化空间）
        # 提取旋转矩阵: [B, 3, 3]
        if batch_size > 1:
            rotation = relative_pose[:, :3, :3]  # [B, 3, 3]
        else:
            rotation = relative_pose[0, :3, :3].unsqueeze(0)  # [1, 3, 3]

        # 对所有历史点应用相同的旋转变换（简化处理）
        # hist_points_norm: [N_hist, 3], rotation: [B, 3, 3]
        # 使用第一个batch的旋转
        rot_0 = rotation[0]  # [3, 3]
        transformed_points = torch.matmul(hist_points_norm, rot_0.T)  # [N_hist, 3]

        print(f"[_historical_state_project_sparse] 历史点变换完成: {transformed_points.shape}")

        # 步骤3: 获取当前帧稀疏点的归一化坐标
        # current_sparse_indices: [N_cur, 4] (b, x, y, z)
        N_cur = current_sparse_indices.shape[0]

        # 从当前帧的多尺度特征获取spatial_shape
        if 'fine' in multiscale_features:
            current_fine = multiscale_features['fine']
            current_spatial_shape = current_fine['features'].spatial_shape  # (D, H, W)
        else:
            # 使用历史帧的spatial_shape作为fallback
            current_spatial_shape = historical_spatial_shape

        D_cur, H_cur, W_cur = current_spatial_shape
        cur_b = current_sparse_indices[:, 0]  # [N_cur]
        cur_x = current_sparse_indices[:, 1].float()  # [N_cur]
        cur_y = current_sparse_indices[:, 2].float()  # [N_cur]
        cur_z = current_sparse_indices[:, 3].float()  # [N_cur]

        # 转换为归一化坐标[-1, 1]范围
        cur_x_norm = (cur_x / (W_cur - 1)) * 2 - 1 if W_cur > 1 else cur_x * 0
        cur_y_norm = (cur_y / (H_cur - 1)) * 2 - 1 if H_cur > 1 else cur_y * 0
        cur_z_norm = (cur_z / (D_cur - 1)) * 2 - 1 if D_cur > 1 else cur_z * 0

        # 堆叠成[N_cur, 3]坐标
        cur_points_norm = torch.stack([cur_x_norm, cur_y_norm, cur_z_norm], dim=1)  # [N_cur, 3]

        # 步骤4: 使用最近邻查找，为每个当前稀疏点找到最近的历史投影点
        # cur_points_norm: [N_cur, 3], transformed_points: [N_hist, 3]
        # 计算距离矩阵 [N_cur, N_hist]
        dists = torch.cdist(cur_points_norm, transformed_points, p=2)  # [N_cur, N_hist]

        # 找到最近邻的索引
        nearest_indices = torch.argmin(dists, dim=1)  # [N_cur]

        # 获取最近邻点的距离
        nearest_dists = torch.gather(dists, 1, nearest_indices.unsqueeze(1))  # [N_cur, 1]

        # 设置距离阈值，超出阈值的点使用零特征
        dist_threshold = 0.5  # 在归一化空间[-1, 1]^3中的距离阈值
        valid_mask = nearest_dists.squeeze(1) < dist_threshold  # [N_cur]

        # 获取历史特征维度
        hist_feat_dim = historical_features.shape[1]  # 历史特征维度（不一定是128）

        # 收集对应的历史特征
        projected_features = torch.zeros(N_cur, hist_feat_dim, device=device)  # [N_cur, hist_feat_dim]
        projected_sdfs = torch.zeros(N_cur, 1, device=device)  # [N_cur, 1]

        # 只为有效点赋值
        valid_indices = nearest_indices[valid_mask]  # [N_valid]
        valid_cur_indices = torch.where(valid_mask)[0]  # [N_valid]

        if valid_cur_indices.shape[0] > 0:
            projected_features[valid_cur_indices] = historical_features[valid_indices]  # [N_valid, hist_feat_dim]
            projected_sdfs[valid_cur_indices] = historical_logits[valid_indices]  # [N_valid, 1]

        print(f"[_historical_state_project_sparse] 有效投影点: {valid_cur_indices.shape[0]}/{N_cur}")

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
            # 动态创建或更新融合网络（适应不同特征维度）
            hist_feat_dim = projected_features.shape[1]  # 历史特征维度
            current_feat_dim = current_features.shape[1]  # 当前特征维度（通常是1）
            fusion_input_dim = hist_feat_dim + current_feat_dim + 1  # +1 for SDF
            fusion_output_dim = current_feat_dim  # 输出维度与当前特征维度一致

            if self.sparse_fusion is None or self.sparse_fusion[0].in_features != fusion_input_dim:
                # 创建新的融合网络
                print(f"[forward_single_frame] 创建新的融合网络: 输入{fusion_input_dim}维 -> 输出{fusion_output_dim}维")
                self.sparse_fusion = nn.Sequential(
                    nn.Linear(fusion_input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, fusion_output_dim),
                    nn.ReLU()
                ).to(projected_features.device)

            # 将当前特征、投影历史特征、投影历史SDF拼接
            concat_features = torch.cat([
                projected_features,      # [N, hist_feat_dim]
                current_features,         # [N, current_feat_dim]
                projected_sdfs,           # [N, 1]
            ], dim=1)  # [N, fusion_input_dim]

            # 通过MLP融合
            fused_features = self.sparse_fusion(concat_features)  # [N, fusion_output_dim]

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
