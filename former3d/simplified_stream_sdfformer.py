"""
简化版StreamSDFFormer - 专注于单帧训练
移除复杂的流式融合逻辑，简化输入接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import collections

# 导入原始SDFFormer组件
from former3d.sdfformer import SDFFormer
from former3d import cnn2d, mv_fusion, utils, view_direction_encoder
from former3d.net3d.former_v1 import Former3D as backbone3d
from former3d.net3d.sparse3d import combineSparseConvTensor, xyzb2bxyz, bxyz2xyzb


class SimplifiedStreamSDFFormer(SDFFormer):
    """简化版StreamSDFFormer - 专注于单帧训练
    
    继承原始SDFFormer，但简化输入接口以接受单视图数据
    移除流式融合和历史状态管理，专注于单帧推理
    """
    
    def __init__(self, 
                 attn_heads: int, 
                 attn_layers: int, 
                 use_proj_occ: bool, 
                 voxel_size: float = 0.08,
                 crop_size: Tuple[int, int, int] = (48, 96, 96),
                 image_size: Tuple[int, int] = (256, 256)):
        """初始化简化版本
        
        Args:
            attn_heads: 注意力头数
            attn_layers: 注意力层数
            use_proj_occ: 是否使用投影占用预测
            voxel_size: 体素大小
            crop_size: 裁剪空间大小 (depth, height, width)
            image_size: 输入图像大小 (height, width)
        """
        # 初始化原始SDFFormer
        super().__init__(attn_heads, attn_layers, use_proj_occ, voxel_size)
        
        # 保存额外参数
        self.crop_size = crop_size
        self.image_size = image_size
        
        # 计算特征图大小（基于CNN2D的下采样）
        # MnasMulti下采样率为：coarse: 16x, medium: 8x, fine: 4x
        self.feat_sizes = {
            'coarse': (image_size[0] // 16, image_size[1] // 16),
            'medium': (image_size[0] // 8, image_size[1] // 8),
            'fine': (image_size[0] // 4, image_size[1] // 4)
        }
        
        print(f"初始化SimplifiedStreamSDFFormer:")
        print(f"  - 体素大小: {voxel_size}")
        print(f"  - 裁剪空间: {crop_size}")
        print(f"  - 图像大小: {image_size}")
        print(f"  - 特征图大小: {self.feat_sizes}")
        print(f"  - 使用投影占用: {use_proj_occ}")
    
    def convert_to_sdfformer_batch(self, 
                                  images: torch.Tensor,
                                  poses: torch.Tensor,
                                  intrinsics: torch.Tensor,
                                  origin: Optional[torch.Tensor] = None) -> Dict:
        """将单视图输入转换为原始SDFFormer的batch格式
        
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
        
        # 原始SDFFormer期望多视图输入，这里将单视图扩展为多视图
        n_views = 1  # 简化版使用单视图
        
        # 扩展图像维度 [batch, n_views, 3, H, W]
        rgb_imgs = images.unsqueeze(1)
        
        # 计算投影矩阵 - 简化版本，使用相同的投影矩阵
        proj_mats = {}
        cam_positions = torch.zeros(batch_size, n_views, 3, device=device)
        
        for resname in self.resolutions:
            # 构建投影矩阵 [batch, n_views, 4, 4]
            proj_mat = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_views, 1, 1)
            
            # 设置旋转和平移
            proj_mat[:, :, :3, :3] = poses[:, :3, :3].unsqueeze(1)
            proj_mat[:, :, :3, 3] = poses[:, :3, 3].unsqueeze(1)
            
            # 添加内参缩放（根据分辨率调整）
            # 注意：这里简化处理，实际需要根据特征图大小调整内参
            scale_factor = self.image_size[0] / self.feat_sizes[resname][0]
            scaled_intrinsics = intrinsics.clone()
            scaled_intrinsics[:, 0, 0] = intrinsics[:, 0, 0] / scale_factor  # fx
            scaled_intrinsics[:, 1, 1] = intrinsics[:, 1, 1] / scale_factor  # fy
            scaled_intrinsics[:, 0, 2] = intrinsics[:, 0, 2] / scale_factor  # cx
            scaled_intrinsics[:, 1, 2] = intrinsics[:, 1, 2] / scale_factor  # cy
            
            # 将内参合并到投影矩阵中
            proj_mat[:, :, :3, :3] = proj_mat[:, :, :3, :3] @ scaled_intrinsics.unsqueeze(1)
            
            proj_mats[resname] = proj_mat
        
        # 设置原点（如果未提供）
        if origin is None:
            # 使用相机位置作为原点
            origin = poses[:, :3, 3].clone()
        
        # 构建batch字典
        batch = {
            "rgb_imgs": rgb_imgs,
            "proj_mats": proj_mats,
            "cam_positions": cam_positions,
            "origin": origin
        }
        
        return batch
    
    def generate_voxel_inds(self, 
                           batch_size: int,
                           num_voxels_per_batch: int = 2000,
                           device: torch.device = None) -> torch.Tensor:
        """生成稀疏体素索引 - 简化版本
        
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
        
        # 计算体素网格大小（基于裁剪空间和体素大小）
        # 使用最粗分辨率（coarse）的体素网格
        voxel_grid_size = [
            int(self.crop_size[0] / self.resolutions['coarse']),
            int(self.crop_size[1] / self.resolutions['coarse']),
            int(self.crop_size[2] / self.resolutions['coarse'])
        ]
        
        # 生成体素索引
        voxel_inds_list = []
        
        for batch_idx in range(batch_size):
            # 为每个批次生成体素索引
            # 使用均匀分布生成体素坐标
            x_coords = torch.randint(0, voxel_grid_size[0], (num_voxels_per_batch,), device=device)
            y_coords = torch.randint(0, voxel_grid_size[1], (num_voxels_per_batch,), device=device)
            z_coords = torch.randint(0, voxel_grid_size[2], (num_voxels_per_batch,), device=device)
            batch_coords = torch.full((num_voxels_per_batch,), batch_idx, device=device)
            
            # 组合坐标
            batch_inds = torch.stack([x_coords, y_coords, z_coords, batch_coords], dim=1)
            voxel_inds_list.append(batch_inds)
        
        voxel_inds = torch.cat(voxel_inds_list, dim=0)
        
        # 转换为int32类型（spconv要求）
        voxel_inds = voxel_inds.to(torch.int32)
        
        return voxel_inds
    
    def forward_single_frame(self, 
                            images: torch.Tensor,
                            poses: torch.Tensor,
                            intrinsics: torch.Tensor,
                            num_voxels: int = 2000,
                            origin: Optional[torch.Tensor] = None) -> Dict:
        """单帧推理 - 简化版本
        
        Args:
            images: 当前帧图像 [batch, 3, height, width]
            poses: 当前帧相机位姿 [batch, 4, 4]
            intrinsics: 当前帧相机内参 [batch, 3, 3]
            num_voxels: 每个批次的体素数量
            origin: 原点坐标 [batch, 3]
            
        Returns:
            输出字典，包含SDF和占用预测
        """
        batch_size = images.shape[0]
        device = images.device
        
        # 1. 转换为原始SDFFormer输入格式
        batch = self.convert_to_sdfformer_batch(images, poses, intrinsics, origin)
        
        # 2. 生成体素索引
        voxel_inds_16 = self.generate_voxel_inds(batch_size, num_voxels_per_batch=num_voxels, device=device)
        
        # 3. 调用原始SDFFormer的forward方法
        voxel_outputs, proj_occ_logits, bp_data = super().forward(batch, voxel_inds_16)
        
        # 4. 构建输出字典
        output = self._build_output_dict(voxel_outputs, proj_occ_logits, bp_data)
        
        return output
    
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
            'occupancy': None,
            'coarse_sdf': None,
            'medium_sdf': None,
            'fine_sdf': None
        }
        
        # 从所有分辨率的输出提取SDF
        for resname in ['coarse', 'medium', 'fine']:
            if resname in voxel_outputs:
                res_output = voxel_outputs[resname]
                if hasattr(res_output, 'features'):
                    features = res_output.features
                    if features.shape[1] == 1:
                        output[f'{resname}_sdf'] = features
        
        # 从最细分辨率的输出提取SDF和占用
        if 'fine' in voxel_outputs:
            fine_output = voxel_outputs['fine']
            if hasattr(fine_output, 'features'):
                features = fine_output.features
                if features.shape[1] == 1:
                    output['sdf'] = features
                    # 将SDF转换为占用概率（使用sigmoid）
                    output['occupancy'] = torch.sigmoid(features)
        
        return output
    
    def forward(self, 
               images: torch.Tensor,
               poses: torch.Tensor,
               intrinsics: torch.Tensor,
               num_voxels: int = 2000,
               origin: Optional[torch.Tensor] = None) -> Dict:
        """前向传播接口
        
        Args:
            images: 当前帧图像 [batch, 3, height, width]
            poses: 当前帧相机位姿 [batch, 4, 4]
            intrinsics: 当前帧相机内参 [batch, 3, 3]
            num_voxels: 每个批次的体素数量
            origin: 原点坐标 [batch, 3]
            
        Returns:
            输出字典
        """
        return self.forward_single_frame(images, poses, intrinsics, num_voxels, origin)
    
    def compute_loss(self, 
                    output: Dict,
                    target_points: torch.Tensor,
                    target_sdf: torch.Tensor,
                    weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """计算损失函数
        
        Args:
            output: 模型输出字典
            target_points: 目标点坐标 [N, 3]
            target_sdf: 目标SDF值 [N, 1]
            weights: 每个点的权重 [N, 1]
            
        Returns:
            loss: 总损失
            loss_dict: 各分量损失字典
        """
        if output['sdf'] is None:
            return torch.tensor(0.0, device=target_sdf.device), {}
        
        # 获取预测的SDF
        pred_sdf = output['sdf']
        
        # 确保形状匹配
        if pred_sdf.shape[0] != target_sdf.shape[0]:
            # 如果数量不匹配，使用最近邻插值
            # 这里简化处理：取前N个预测
            N = min(pred_sdf.shape[0], target_sdf.shape[0])
            pred_sdf = pred_sdf[:N]
            target_sdf = target_sdf[:N]
            if weights is not None:
                weights = weights[:N]
        
        # 计算L1损失（对SDF回归更鲁棒）
        if weights is not None:
            loss = torch.mean(weights * torch.abs(pred_sdf - target_sdf))
        else:
            loss = torch.mean(torch.abs(pred_sdf - target_sdf))
        
        # 计算其他损失分量
        loss_dict = {
            'sdf_l1_loss': loss.item(),
            'pred_mean': pred_sdf.mean().item(),
            'pred_std': pred_sdf.std().item(),
            'target_mean': target_sdf.mean().item(),
            'target_std': target_sdf.std().item()
        }
        
        return loss, loss_dict


def test_simplified_stream_sdfformer():
    """测试简化版本"""
    print("测试SimplifiedStreamSDFFormer...")
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimplifiedStreamSDFFormer(
        attn_heads=2,
        attn_layers=2,
        use_proj_occ=False,
        voxel_size=0.08,
        crop_size=(48, 96, 96),
        image_size=(256, 256)
    ).to(device)
    
    # 设置为eval模式
    model.eval()
    
    # 创建测试数据
    batch_size = 2
    
    # 图像 [batch, 3, 256, 256]
    images = torch.randn(batch_size, 3, 256, 256, device=device)
    
    # 位姿 [batch, 4, 4]
    poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    poses[:, 0, 3] = 1.0  # X方向平移
    
    # 内参 [batch, 3, 3]
    intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics[:, 0, 0] = 320.0  # fx
    intrinsics[:, 1, 1] = 320.0  # fy
    intrinsics[:, 0, 2] = 128.0  # cx
    intrinsics[:, 1, 2] = 128.0  # cy
    
    # 测试前向传播
    print("测试前向传播...")
    with torch.no_grad():
        output = model(images, poses, intrinsics, num_voxels=1000)
    
    print(f"输出键: {list(output.keys())}")
    for key, value in output.items():
        if value is not None and hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
        elif key in ['coarse_sdf', 'medium_sdf', 'fine_sdf', 'sdf', 'occupancy']:
            if value is not None:
                print(f"  {key}: {value.shape}")
    
    # 测试损失计算
    print("\n测试损失计算...")
    target_points = torch.randn(500, 3, device=device)
    target_sdf = torch.randn(500, 1, device=device)
    
    loss, loss_dict = model.compute_loss(output, target_points, target_sdf)
    print(f"损失: {loss.item():.6f}")
    print(f"损失字典: {loss_dict}")
    
    print("✅ 简化版本测试完成")


if __name__ == "__main__":
    test_simplified_stream_sdfformer()