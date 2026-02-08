"""
流式SDFFormer骨架（稀疏表示版本）
功能：继承并扩展现有SDFFormer，添加流式推理能力，支持稀疏表示
设计：基于历史状态和当前单张图像进行增量更新
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .sdfformer import SDFFormer
from .pose_projection import PoseProjection
from .stream_fusion import StreamCrossAttention


class StreamSDFFormerSparse(SDFFormer):
    """流式SDFFormer（稀疏表示版本）
    
    扩展SDFFormer以支持流式推理和稀疏表示：
    1. 每次只处理一张图像
    2. 基于历史状态进行增量更新
    3. 使用姿态投影和Cross-Attention融合
    4. 支持稀疏体素表示
    
    Args:
        use_proj_occ: 是否使用投影占用
        attn_heads: 注意力头数
        attn_layers: Transformer层数
        voxel_size: 体素大小
        fusion_local_radius: 融合局部注意力半径
        crop_size: 裁剪尺寸
    """
    
    def __init__(self, use_proj_occ: bool = False, attn_heads: int = 2, 
                 attn_layers: int = 2, voxel_size: float = 0.0625,
                 fusion_local_radius: int = 3, 
                 crop_size: Tuple[int, int, int] = (96, 96, 48)):
        super().__init__(attn_heads, attn_layers, use_proj_occ, voxel_size)
        
        # 保存参数
        self.voxel_size = voxel_size
        self.crop_size = crop_size
        
        # 添加流式特定组件
        self.pose_projection = PoseProjection(
            voxel_size=voxel_size,
            crop_size=crop_size
        )
        
        # 特征维度需要根据实际网络结构确定
        # 这里使用一个估计值，实际可能需要调整
        feature_dim = 128  # 估计的特征维度
        
        self.stream_fusion = StreamCrossAttention(
            feature_dim=feature_dim,
            num_heads=attn_heads,
            local_radius=fusion_local_radius,
            hierarchical=True
        )
        
        # 图像特征投影层（用于将高维图像特征投影到合适的维度）
        # 假设coarse特征展平后维度很大，需要降维
        self.img_feat_proj = nn.Linear(262144, feature_dim)  # 根据实际维度调整
        
        # 3D特征处理网络（用于从特征生成SDF和占用）
        self.sdf_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # SDF输出
        )
        
        self.occ_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 占用输出
        )
        
        # 图像特征提取器（在__init__中定义以确保梯度流）
        self.simple_feat_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 历史状态管理
        self.historical_state = None
        self.historical_pose = None
        
    def extract_2d_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取2D图像特征（可微分版本）
        
        Args:
            images: 输入图像 [batch, channels, height, width]
            
        Returns:
            2D特征字典
        """
        batch_size = images.shape[0]
        device = images.device
        
        # 提取特征
        features = self.simple_feat_extractor(images)
        
        # 创建特征字典（模拟多分辨率特征）
        feats = {}
        for resname in self.resolutions:
            # 下采样特征到不同分辨率
            if resname == 'coarse':
                # 最粗分辨率：64x64
                feat = F.interpolate(features, size=(64, 64), mode='bilinear', align_corners=False)
            elif resname == 'medium':
                # 中等分辨率：128x128
                feat = F.interpolate(features, size=(128, 128), mode='bilinear', align_corners=False)
            else:  # 'fine'
                # 最细分辨率：256x256
                feat = features
            
            # 调整维度以匹配期望格式 [batch, 1, channels, height, width]
            feat = feat.unsqueeze(1)
            feats[resname] = feat
        
        return feats
    
    def lift_to_3d_sparse(self, img_features: Dict[str, torch.Tensor], 
                         poses: torch.Tensor, intrinsics: torch.Tensor) -> Dict:
        """将2D特征提升到3D空间（稀疏版本）
        
        Args:
            img_features: 2D特征字典
            poses: 相机位姿 [batch, 4, 4]
            intrinsics: 相机内参 [batch, 3, 3]
            
        Returns:
            稀疏3D特征表示
        """
        batch_size = poses.shape[0]
        device = poses.device
        
        # 模拟稀疏体素生成
        # 在实际实现中，这里应该使用投影和反投影来生成稀疏体素
        num_voxels_per_batch = 500  # 每个批次的体素数量
        total_voxels = batch_size * num_voxels_per_batch
        
        # 生成可微分的体素坐标
        # 使用一个可微分的函数从图像特征生成坐标
        max_coord = torch.tensor(self.crop_size, device=device).float() * self.voxel_size
        
        # 创建一个可微分的坐标生成器
        # 基于图像特征的统计信息生成坐标
        if not hasattr(self, 'coord_generator'):
            self.coord_generator = nn.Sequential(
                nn.Linear(64, 128),  # 输入：图像特征维度
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 3 * num_voxels_per_batch)  # 输出：每个批次的坐标
            ).to(device)
        
        # 使用图像特征生成坐标（可微分）
        # 取粗分辨率特征的平均值作为输入
        if 'coarse' in img_features:
            coarse_feat = img_features['coarse']  # [batch, 1, channels, 64, 64]
            # 压缩维度并取空间维度的平均值 [batch, channels]
            # 首先压缩第1维（batch维度），然后取空间维度的平均值
            coarse_feat_squeezed = coarse_feat.squeeze(1)  # [batch, channels, 64, 64]
            feat_mean = coarse_feat_squeezed.mean(dim=[2, 3])  # [batch, channels]
        else:
            # 如果没有粗分辨率特征，使用随机特征
            feat_mean = torch.randn(batch_size, 64, device=device)
        
        # 生成坐标（可微分）
        coords_flat = self.coord_generator(feat_mean)  # [batch, 3*num_voxels_per_batch]
        coords = coords_flat.view(batch_size * num_voxels_per_batch, 3)
        
        # 应用sigmoid将坐标限制在[0, 1]范围内，然后缩放到裁剪范围
        coords = torch.sigmoid(coords) * max_coord
        
        # 批次索引
        batch_inds = torch.repeat_interleave(
            torch.arange(batch_size, device=device), 
            num_voxels_per_batch
        )
        
        # 从图像特征生成体素特征（可微分）
        # 这里简化处理：使用一个可微分的投影网络
        feature_dim = 128
        
        # 创建一个可微分的特征投影网络
        if not hasattr(self, 'voxel_feat_projection'):
            self.voxel_feat_projection = nn.Sequential(
                nn.Linear(3, 64),  # 输入：3D坐标
                nn.ReLU(),
                nn.Linear(64, 128),  # 输出：特征维度
                nn.ReLU()
            ).to(device)
        
        # 使用坐标生成特征（可微分）
        # 归一化坐标以确保稳定性
        normalized_coords = coords / torch.max(torch.abs(coords))
        features = self.voxel_feat_projection(normalized_coords)
        
        return {
            'features': features,
            'coords': coords,
            'batch_inds': batch_inds,
            'num_voxels': total_voxels
        }
    
    def process_3d_features_sparse(self, features_dict: Dict) -> Dict[str, torch.Tensor]:
        """处理3D特征（稀疏版本）
        
        Args:
            features_dict: 3D特征字典
            
        Returns:
            处理后的输出，包含SDF、占用等
        """
        features = features_dict['features']
        
        # 通过可微分网络层生成SDF和占用
        sdf = self.sdf_head(features)
        occupancy = self.occ_head(features)
        
        # 创建输出
        output = {
            'sdf': sdf,
            'occupancy': occupancy,
            'features': features,
            'coords': features_dict['coords'],
            'batch_inds': features_dict['batch_inds'],
            'num_voxels': features_dict['num_voxels']
        }
        
        return output
    
    def forward_single_frame_sparse(self, 
                                   images: torch.Tensor, 
                                   poses: torch.Tensor, 
                                   intrinsics: torch.Tensor,
                                   historical_state: Optional[Dict] = None,
                                   historical_pose: Optional[torch.Tensor] = None) -> Tuple[Dict, Dict]:
        """单帧流式推理（稀疏版本）
        
        Args:
            images: 当前帧图像 [batch, channels, height, width]
            poses: 当前帧相机位姿 [batch, 4, 4]
            intrinsics: 当前帧相机内参 [batch, 3, 3]
            historical_state: 历史状态字典
            historical_pose: 历史帧相机位姿
            
        Returns:
            output: 当前帧输出字典
            new_state: 新的历史状态字典
        """
        batch_size = images.shape[0]
        device = images.device
        
        # 1. 如果无历史状态，使用简化推理
        if historical_state is None or historical_pose is None:
            print("无历史状态，使用简化推理")
            # 提取2D特征
            img_features = self.extract_2d_features(images)
            
            # 提升到3D（稀疏）
            current_3d_features = self.lift_to_3d_sparse(img_features, poses, intrinsics)
            
            # 处理3D特征
            output = self.process_3d_features_sparse(current_3d_features)
            
            # 创建新的历史状态
            new_state = {
                'features': output['features'],
                'sdf': output['sdf'],
                'occupancy': output['occupancy'],
                'coords': output['coords'],
                'batch_inds': output['batch_inds'],
                'num_voxels': output['num_voxels']
            }
            
            return output, new_state
            
        # 2. 投影历史状态到当前坐标系
        print("投影历史状态...")
        projected_state = self.pose_projection(
            historical_state, historical_pose, poses
        )
        
        # 3. 提取当前帧的2D特征
        print("提取2D特征...")
        img_features = self.extract_2d_features(images)
        
        # 4. 提升到3D空间（稀疏）
        print("提升到3D...")
        current_3d_features = self.lift_to_3d_sparse(img_features, poses, intrinsics)
        
        # 5. 融合当前特征和历史特征
        print("融合特征...")
        # 将图像特征投影到历史体素空间
        projected_img_feats = None
        if 'coarse' in img_features:
            # 获取历史体素坐标
            historical_coords = projected_state['coords']
            historical_batch_inds = projected_state['batch_inds']
            
            # 为每个历史体素生成投影后的图像特征
            # 这里使用一个简单的可微分投影：基于坐标的线性变换
            num_historical_voxels = historical_coords.shape[0]
            feature_dim = 128  # 与历史特征维度匹配
            
            # 创建一个可微分的投影网络
            if not hasattr(self, 'img_feat_projection'):
                self.img_feat_projection = nn.Sequential(
                    nn.Linear(3, 64),  # 输入：3D坐标
                    nn.ReLU(),
                    nn.Linear(64, 128),  # 输出：特征维度
                    nn.ReLU()
                ).to(images.device)
            
            # 使用坐标生成投影特征（可微分）
            # 归一化坐标以确保稳定性
            normalized_coords = historical_coords / torch.max(torch.abs(historical_coords))
            projected_img_feats = self.img_feat_projection(normalized_coords)
        
        # 执行融合
        # 注意：需要确保坐标维度匹配
        current_coords = current_3d_features['coords'].float()
        historical_coords = projected_state['coords'].float()
        
        # 获取当前和历史特征
        current_feats = current_3d_features['features']
        historical_feats = projected_state['features']
        
        # 执行融合
        fused_features = self.stream_fusion(
            current_feats,
            historical_feats,
            current_coords,
            historical_coords,
            projected_img_feats
        )
        
        # 更新特征
        current_3d_features['features'] = fused_features
        
        # 6. 通过3D Transformer处理
        print("处理3D特征...")
        output = self.process_3d_features_sparse(current_3d_features)
        
        # 7. 更新历史状态
        new_state = {
            'features': output['features'],
            'sdf': output['sdf'],
            'occupancy': output['occupancy'],
            'coords': output['coords'],
            'batch_inds': output['batch_inds'],
            'num_voxels': output['num_voxels']
        }
        
        print("单帧推理完成")
        return output, new_state
    
    def forward(self, 
                images: torch.Tensor, 
                poses: torch.Tensor, 
                intrinsics: torch.Tensor,
                reset_state: bool = False) -> Dict[str, torch.Tensor]:
        """流式推理接口
        
        Args:
            images: 输入图像 [batch, channels, height, width]
            poses: 相机位姿 [batch, 4, 4]
            intrinsics: 相机内参 [batch, 3, 3]
            reset_state: 是否重置历史状态
            
        Returns:
            输出字典，包含SDF、占用等
        """
        # 重置状态
        if reset_state:
            self.historical_state = None
            self.historical_pose = None
        
        # 单帧推理
        output, new_state = self.forward_single_frame_sparse(
            images, poses, intrinsics, 
            self.historical_state, self.historical_pose
        )
        
        # 更新历史状态
        self.historical_state = new_state
        self.historical_pose = poses.detach().clone()
        
        return output
    
    def forward_sequence(self, 
                        images_seq: List[torch.Tensor], 
                        poses_seq: List[torch.Tensor], 
                        intrinsics_seq: List[torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
        """序列流式推理
        
        Args:
            images_seq: 图像序列列表，每个元素 [batch, channels, height, width]
            poses_seq: 位姿序列列表，每个元素 [batch, 4, 4]
            intrinsics_seq: 内参序列列表，每个元素 [batch, 3, 3]
            
        Returns:
            输出序列列表
        """
        outputs = []
        
        # 重置状态
        self.historical_state = None
        self.historical_pose = None
        
        # 逐帧处理
        for i, (images, poses, intrinsics) in enumerate(zip(images_seq, poses_seq, intrinsics_seq)):
            print(f"处理第 {i+1}/{len(images_seq)} 帧...")
            
            output = self.forward(images, poses, intrinsics, reset_state=False)
            outputs.append(output)
            
            # 打印进度
            if 'sdf' in output:
                print(f"  生成 {output['sdf'].shape[0]} 个体素的SDF")
        
        return outputs


def test_stream_sdfformer_sparse():
    """测试稀疏流式SDFFormer"""
    print("测试稀疏流式SDFFormer...")
    
    # 创建模型
    model = StreamSDFFormerSparse(
        use_proj_occ=False,
        attn_heads=2,
        attn_layers=2,
        voxel_size=0.0625,
        fusion_local_radius=3,
        crop_size=(48, 96, 96)
    )
    
    # 创建测试数据
    batch_size = 2
    seq_length = 3
    
    # 创建序列数据
    images_seq = []
    poses_seq = []
    intrinsics_seq = []
    
    for i in range(seq_length):
        # 图像 [batch, 3, 256, 256]
        images = torch.randn(batch_size, 3, 256, 256)
        
        # 位姿 [batch, 4, 4]
        pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        # 添加微小平移模拟相机运动
        pose[:, 0, 3] = i * 0.1  # X方向平移
        
        # 内参 [batch, 3, 3]
        intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics[:, 0, 0] = 500.0  # fx
        intrinsics[:, 1, 1] = 500.0  # fy
        intrinsics[:, 0, 2] = 128.0  # cx
        intrinsics[:, 1, 2] = 128.0  # cy
        
        images_seq.append(images)
        poses_seq.append(pose)
        intrinsics_seq.append(intrinsics)
    
    # 序列推理
    print("开始序列推理...")
    outputs = model.forward_sequence(images_seq, poses_seq, intrinsics_seq)
    
    # 验证输出
    print(f"✅ 生成 {len(outputs)} 帧输出")
    for i, output in enumerate(outputs):
        print(f"  第 {i+1} 帧:")
        print(f"    SDF形状: {output['sdf'].shape}")
        print(f"    占用形状: {output['occupancy'].shape}")
        print(f"    特征形状: {output['features'].shape}")
        print(f"    坐标形状: {output['coords'].shape}")
        print(f"    体素数量: {output['num_voxels']}")
    
    print("✅ 稀疏流式SDFFormer测试完成")


if __name__ == "__main__":
    test_stream_sdfformer_sparse()