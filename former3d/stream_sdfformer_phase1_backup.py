"""
流式SDFFormer骨架
功能：继承并扩展现有SDFFormer，添加流式推理能力
设计：基于历史状态和当前单张图像进行增量更新
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .sdfformer import SDFFormer
from .pose_projection import PoseProjection
from .stream_fusion import StreamCrossAttention


class StreamSDFFormer(SDFFormer):
    """流式SDFFormer
    
    扩展SDFFormer以支持流式推理：
    1. 每次只处理一张图像
    2. 基于历史状态进行增量更新
    3. 使用姿态投影和Cross-Attention融合
    
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
        estimated_feature_dim = 128
        
        self.stream_fusion = StreamCrossAttention(
            feature_dim=estimated_feature_dim,
            num_heads=attn_heads,
            local_radius=fusion_local_radius,
            hierarchical=True,
            dropout=0.1
        )
        
        # 状态管理
        self.historical_state = None
        self.historical_pose = None
        
        # 状态初始化标志
        self._state_initialized = False
        
    def reset_state(self):
        """重置历史状态"""
        self.historical_state = None
        self.historical_pose = None
        self._state_initialized = False
        print("历史状态已重置")
        
    def initialize_state(self, batch_size: int, device: torch.device):
        """初始化历史状态
        
        Args:
            batch_size: 批量大小
            device: 计算设备
        """
        # 创建空的初始状态
        # 实际实现中可能需要根据第一帧初始化
        self.historical_state = {
            'features': None,
            'sdf': None,
            'occupancy': None,
            'coords': None
        }
        self._state_initialized = True
        print(f"历史状态已初始化，batch_size={batch_size}, device={device}")
        
    def extract_2d_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """提取2D特征（简化版本）
        
        Args:
            images: 输入图像 [batch, channels, height, width]
            
        Returns:
            2D特征字典
        """
        # 这里简化实现，实际需要调用父类的get_img_feats方法
        # 注意：原始SDFFormer处理多图像，这里需要适配单图像
        batch_size = images.shape[0]
        
        # 创建模拟特征（实际实现需要替换为真正的特征提取）
        feats = {}
        for resname in self.resolutions:
            # 简化：创建随机特征
            # 实际需要根据图像计算
            feat_shape = (batch_size, 1, 64, images.shape[2]//4, images.shape[3]//4)
            feats[resname] = torch.randn(feat_shape, device=images.device)
            
        return feats
    
    def lift_to_3d(self, img_features: Dict[str, torch.Tensor], 
                   poses: torch.Tensor, intrinsics: torch.Tensor) -> Dict:
        """将2D特征提升到3D空间（简化版本）
        
        Args:
            img_features: 2D特征字典
            poses: 相机位姿 [batch, 4, 4]
            intrinsics: 相机内参 [batch, 3, 3]
            
        Returns:
            3D特征表示
        """
        # 这里简化实现，实际需要调用父类的投影和反投影方法
        batch_size = poses.shape[0]
        
        # 创建模拟的3D特征
        # 实际实现需要复杂的投影和反投影操作
        num_voxels = 1000  # 模拟体素数量
        feature_dim = 128  # 模拟特征维度
        
        # 创建稀疏特征表示
        features = torch.randn(batch_size * num_voxels, feature_dim, device=poses.device)
        coords = torch.randint(0, 48, (batch_size * num_voxels, 3), device=poses.device)
        batch_inds = torch.repeat_interleave(
            torch.arange(batch_size, device=poses.device), 
            num_voxels
        )
        
        # 组合坐标和批次索引
        indices = torch.cat([batch_inds.unsqueeze(1), coords], dim=1)
        
        return {
            'features': features,
            'indices': indices,
            'coords': coords,
            'batch_inds': batch_inds
        }
    
    def process_3d_features(self, features_dict: Dict) -> Dict[str, torch.Tensor]:
        """处理3D特征（简化版本）
        
        Args:
            features_dict: 3D特征字典
            
        Returns:
            处理后的输出，包含SDF、占用等
        """
        # 这里简化实现，实际需要调用父类的3D Transformer处理
        batch_size = len(torch.unique(features_dict['batch_inds']))
        
        # 创建模拟输出
        num_voxels = features_dict['features'].shape[0]
        
        output = {
            'sdf': torch.randn(num_voxels, 1, device=features_dict['features'].device),
            'occupancy': torch.randn(num_voxels, 1, device=features_dict['features'].device),
            'features': features_dict['features'],
            'coords': features_dict['coords']
        }
        
        return output
    
    def forward_single_frame(self, 
                            images: torch.Tensor, 
                            poses: torch.Tensor, 
                            intrinsics: torch.Tensor,
                            historical_state: Optional[Dict] = None,
                            historical_pose: Optional[torch.Tensor] = None) -> Tuple[Dict, Dict]:
        """单帧流式推理
        
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
            
            # 提升到3D
            current_3d_features = self.lift_to_3d(img_features, poses, intrinsics)
            
            # 处理3D特征
            output = self.process_3d_features(current_3d_features)
            
            # 创建新的历史状态
            new_state = {
                'features': output['features'],
                'sdf': output['sdf'],
                'occupancy': output['occupancy'],
                'coords': output['coords']
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
        
        # 4. 提升到3D空间
        print("提升到3D...")
        current_3d_features = self.lift_to_3d(img_features, poses, intrinsics)
        
        # 5. 融合当前特征和历史特征
        print("融合特征...")
        # 准备图像特征用于指导融合（如果有）
        img_feat_for_fusion = None
        if 'coarse' in img_features:
            # 使用最粗尺度的图像特征
            img_feat_for_fusion = img_features['coarse'].reshape(batch_size, -1)
        
        # 执行融合
        fused_features = self.stream_fusion(
            current_3d_features['features'],
            projected_state['features'].reshape(-1, projected_state['features'].shape[1]),
            current_3d_features['coords'],
            # 需要从投影状态中提取坐标信息
            # 这里简化处理，实际需要根据投影状态重建坐标
            torch.randn_like(current_3d_features['coords']),
            img_feat_for_fusion
        )
        
        # 更新特征
        current_3d_features['features'] = fused_features
        
        # 6. 通过3D Transformer处理
        print("处理3D特征...")
        output = self.process_3d_features(current_3d_features)
        
        # 7. 更新历史状态
        new_state = {
            'features': output['features'],
            'sdf': output['sdf'],
            'occupancy': output['occupancy'],
            'coords': output['coords']
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
        # 重置状态（如果需要）
        if reset_state:
            self.reset_state()
            
        # 初始化状态（如果未初始化）
        if not self._state_initialized:
            self.initialize_state(images.shape[0], images.device)
            
        # 调用单帧推理
        output, new_state = self.forward_single_frame(
            images, poses, intrinsics,
            self.historical_state,
            self.historical_pose
        )
        
        # 更新历史状态
        self.historical_state = new_state
        self.historical_pose = poses.clone()
        
        return output
    
    def forward_sequence(self, 
                        image_sequence: torch.Tensor, 
                        pose_sequence: torch.Tensor,
                        intrinsics_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """处理图像序列
        
        Args:
            image_sequence: 图像序列 [sequence, batch, channels, height, width]
            pose_sequence: 位姿序列 [sequence, batch, 4, 4]
            intrinsics_sequence: 内参序列 [sequence, batch, 3, 3]
            
        Returns:
            序列输出字典
        """
        sequence_length = image_sequence.shape[0]
        batch_size = image_sequence.shape[1]
        
        # 重置状态
        self.reset_state()
        
        # 存储每帧输出
        sequence_outputs = {
            'sdf': [],
            'occupancy': [],
            'features': []
        }
        
        # 逐帧处理
        for t in range(sequence_length):
            print(f"处理第 {t+1}/{sequence_length} 帧...")
            
            # 获取当前帧
            current_image = image_sequence[t]
            current_pose = pose_sequence[t]
            current_intrinsics = intrinsics_sequence[t]
            
            # 流式推理
            output = self.forward(
                current_image, current_pose, current_intrinsics,
                reset_state=(t == 0)  # 第一帧重置状态
            )
            
            # 存储输出
            for key in sequence_outputs:
                if key in output:
                    sequence_outputs[key].append(output[key])
        
        # 合并序列输出
        for key in sequence_outputs:
            if sequence_outputs[key]:
                sequence_outputs[key] = torch.stack(sequence_outputs[key], dim=0)
        
        return sequence_outputs


def test_stream_sdfformer():
    """简单的测试函数"""
    print("测试流式SDFFormer...")
    
    # 创建模型
    model = StreamSDFFormer(
        use_proj_occ=False,
        attn_heads=2,
        attn_layers=2,
        voxel_size=0.0625,
        fusion_local_radius=3
    )
    
    # 测试1：状态管理
    print("测试状态管理...")
    assert model.historical_state is None
    assert not model._state_initialized
    
    model.initialize_state(batch_size=2, device='cpu')
    assert model._state_initialized
    assert model.historical_state is not None
    print("✅ 状态管理测试通过")
    
    # 测试2：单帧推理（无历史状态）
    print("测试单帧推理（无历史状态）...")
    batch_size = 2
    images = torch.randn(batch_size, 3, 256, 256)
    poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    intrinsics = torch.tensor([[256, 0, 128], [0, 256, 128], [0, 0, 1]]).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 重置状态并推理
    output = model(images, poses, intrinsics, reset_state=True)
    
    # 检查输出
    assert 'sdf' in output
    assert 'occupancy' in output
    assert 'features' in output
    print(f"✅ 单帧推理输出包含所需字段")
    
    # 测试3：多帧序列推理
    print("测试多帧序列推理...")
    sequence_length = 3
    image_sequence = torch.randn(sequence_length, batch_size, 3, 256, 256)
    pose_sequence = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(sequence_length, batch_size, 1, 1)
    
    # 添加一些平移
    for t in range(sequence_length):
        pose_sequence[t, :, :3, 3] = torch.tensor([t * 0.1, 0, 0])
    
    intrinsics_sequence = torch.tensor([[256, 0, 128], [0, 256, 128], [0, 0, 1]]).unsqueeze(0).unsqueeze(0).repeat(sequence_length, batch_size, 1, 1)
    
    # 处理序列
    sequence_output = model.forward_sequence(
        image_sequence, pose_sequence, intrinsics_sequence
    )
    
    # 检查序列输出
    for key in ['sdf', 'occupancy', 'features']:
        if key in sequence_output:
            assert sequence_output[key].shape[0] == sequence_length
            print(f"✅ 序列输出 {key} 形状正确: {sequence_output[key].shape}")
    
    # 测试4：梯度流
    print("测试梯度流...")
    model.reset_state()
    
    images.requires_grad_(True)
    output = model(images, poses, intrinsics, reset_state=True)
    
    loss = output['sdf'].sum() + output['occupancy'].sum()
    loss.backward()
    
    assert images.grad is not None
    assert not torch.all(images.grad == 0)
    print(f"✅ 梯度存在，梯度范数: {torch.norm(images.grad):.6f}")
    
    print("测试完成！")


if __name__ == "__main__":
    test_stream_sdfformer()