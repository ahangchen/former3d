"""
流式SDFFormer - 简化集成版本
功能：简化版本，绕过view_direction_encoder问题，快速测试流式架构
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from .pose_projection import PoseProjection
from .stream_fusion import StreamCrossAttention


class SimpleStreamSDFFormer(nn.Module):
    """简化版本的流式SDFFormer
    
    绕过复杂的view_direction_encoder问题，专注于测试流式架构。
    
    Args:
        feature_dim: 特征维度
        voxel_size: 体素大小
        crop_size: 裁剪尺寸
        fusion_local_radius: 融合局部注意力半径
    """
    
    def __init__(self, feature_dim: int = 128, voxel_size: float = 0.0625,
                 crop_size: Tuple[int, int, int] = (96, 96, 48),
                 fusion_local_radius: int = 3):
        super().__init__()
        
        # 保存参数
        self.feature_dim = feature_dim
        self.voxel_size = voxel_size
        self.crop_size = crop_size
        
        # 模拟的2D特征提取（简化）
        self.fake_2d_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, feature_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 姿态投影模块
        self.pose_projection = PoseProjection(
            voxel_size=voxel_size,
            crop_size=crop_size
        )
        
        # 流式融合模块
        self.stream_fusion = StreamCrossAttention(
            feature_dim=feature_dim,
            num_heads=8,
            local_radius=fusion_local_radius,
            hierarchical=True,
            dropout=0.1
        )
        
        # 模拟的3D处理（简化）
        self.fake_3d_processor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2)  # 输出SDF和占用
        )
        
        # 状态管理
        self.historical_state = None
        self.historical_pose = None
        
        # 初始化体素网格
        self._init_voxel_grid()
        
    def _init_voxel_grid(self):
        """初始化体素网格"""
        x = torch.arange(0, self.crop_size[0])
        y = torch.arange(0, self.crop_size[1])
        z = torch.arange(0, self.crop_size[2])
        
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        
        # 体素坐标 [x, y, z]
        self.voxel_coords = torch.stack([
            grid_x.flatten(),
            grid_y.flatten(), 
            grid_z.flatten()
        ], dim=1).float()
        
        # 体素数量
        self.num_voxels = self.voxel_coords.shape[0]
        
    def reset_state(self):
        """重置历史状态"""
        self.historical_state = None
        self.historical_pose = None
        print("历史状态已重置")
        
    def extract_2d_features(self, image: torch.Tensor) -> torch.Tensor:
        """提取2D特征（简化版本）
        
        Args:
            image: 输入图像 [batch, channels, height, width]
            
        Returns:
            2D特征 [batch, feature_dim, height, width]
        """
        return self.fake_2d_extractor(image)
    
    def lift_to_3d(self, img_features: torch.Tensor, 
                   poses: torch.Tensor) -> torch.Tensor:
        """将2D特征提升到3D空间（简化版本）
        
        Args:
            img_features: 2D特征 [batch, feature_dim, height, width]
            poses: 相机位姿 [batch, 4, 4]
            
        Returns:
            3D特征 [batch * num_voxels, feature_dim]
        """
        batch_size = img_features.shape[0]
        
        # 简化：将2D特征平均池化后复制到所有体素
        pooled_features = torch.mean(img_features, dim=[2, 3])  # [batch, feature_dim]
        
        # 复制到所有体素
        voxel_features = pooled_features.unsqueeze(1).repeat(1, self.num_voxels, 1)
        voxel_features = voxel_features.reshape(batch_size * self.num_voxels, self.feature_dim)
        
        return voxel_features
    
    def process_single_frame(self, 
                            image: torch.Tensor,
                            pose: torch.Tensor,
                            historical_state: Optional[Dict] = None,
                            historical_pose: Optional[torch.Tensor] = None) -> Tuple[Dict, Dict]:
        """处理单帧（简化版本）
        
        Args:
            image: 当前帧图像 [batch, channels, height, width]
            pose: 当前帧相机位姿 [batch, 4, 4]
            historical_state: 历史状态
            historical_pose: 历史位姿
            
        Returns:
            output: 输出字典
            new_state: 新的历史状态
        """
        batch_size = image.shape[0]
        device = image.device
        
        # 1. 提取2D特征
        img_features = self.extract_2d_features(image)
        
        # 2. 提升到3D
        current_3d_features = self.lift_to_3d(img_features, pose)
        
        # 3. 获取体素坐标
        voxel_coords = self.voxel_coords.to(device)
        voxel_coords = voxel_coords.unsqueeze(0).repeat(batch_size, 1, 1)
        voxel_coords = voxel_coords.reshape(batch_size * self.num_voxels, 3)
        
        # 4. 如果没有历史状态，直接处理
        if historical_state is None or historical_pose is None:
            print("无历史状态，直接处理")
            
            # 通过3D处理器
            processed_features = self.fake_3d_processor(current_3d_features)
            
            # 分离SDF和占用
            sdf = processed_features[:, 0:1]
            occupancy = torch.sigmoid(processed_features[:, 1:2])
            
            output = {
                'sdf': sdf,
                'occupancy': occupancy,
                'features': current_3d_features,
                'coords': voxel_coords
            }
            
            # 创建新的历史状态
            new_state = {
                'features': current_3d_features.detach(),
                'sdf': sdf.detach(),
                'occupancy': occupancy.detach(),
                'coords': voxel_coords.detach()
            }
            
            return output, new_state
        
        # 5. 有历史状态：流式处理
        print("有历史状态，使用流式处理")
        
        # 投影历史状态
        projected_state = self.pose_projection(
            historical_state, historical_pose, pose
        )
        
        # 提取投影后的特征
        projected_features = projected_state['features']
        projected_features_flat = projected_features.reshape(
            batch_size * self.num_voxels, self.feature_dim
        )
        
        # 提取投影后的坐标
        projected_coords = projected_state.get('coords', voxel_coords)
        
        # 流式融合
        fused_features = self.stream_fusion(
            current_3d_features,
            projected_features_flat,
            voxel_coords,
            projected_coords.reshape(batch_size * self.num_voxels, 3)
        )
        
        # 通过3D处理器
        processed_features = self.fake_3d_processor(fused_features)
        
        # 分离SDF和占用
        sdf = processed_features[:, 0:1]
        occupancy = torch.sigmoid(processed_features[:, 1:2])
        
        output = {
            'sdf': sdf,
            'occupancy': occupancy,
            'features': fused_features,
            'coords': voxel_coords
        }
        
        # 创建新的历史状态
        new_state = {
            'features': fused_features.detach(),
            'sdf': sdf.detach(),
            'occupancy': occupancy.detach(),
            'coords': voxel_coords.detach()
        }
        
        return output, new_state
    
    def forward(self, 
                image: torch.Tensor, 
                pose: torch.Tensor,
                reset_state: bool = False) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            image: 输入图像 [batch, channels, height, width]
            pose: 相机位姿 [batch, 4, 4]
            reset_state: 是否重置历史状态
            
        Returns:
            输出字典
        """
        # 重置状态（如果需要）
        if reset_state:
            self.reset_state()
            
        # 处理单帧
        output, new_state = self.process_single_frame(
            image, pose,
            self.historical_state,
            self.historical_pose
        )
        
        # 更新历史状态
        self.historical_state = new_state
        self.historical_pose = pose.clone()
        
        return output
    
    def forward_sequence(self, 
                        image_sequence: torch.Tensor, 
                        pose_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """处理图像序列
        
        Args:
            image_sequence: 图像序列 [sequence, batch, channels, height, width]
            pose_sequence: 位姿序列 [sequence, batch, 4, 4]
            
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
            
            # 流式推理
            output = self.forward(
                current_image, current_pose,
                reset_state=(t == 0)
            )
            
            # 存储输出
            for key in sequence_outputs:
                if key in output:
                    sequence_outputs[key].append(output[key])
        
        # 合并序列输出
        for key in sequence_outputs:
            if sequence_outputs[key]:
                # 堆叠所有帧
                sequence_outputs[key] = torch.stack(sequence_outputs[key], dim=0)
        
        return sequence_outputs


def test_simple_stream_sdfformer():
    """测试简化版本的流式SDFFormer"""
    print("测试简化版本的流式SDFFormer...")
    
    # 创建模型
    model = SimpleStreamSDFFormer(
        feature_dim=128,
        voxel_size=0.0625,
        crop_size=(48, 48, 24),  # 使用较小的尺寸以节省内存
        fusion_local_radius=3
    )
    
    # 测试1：单帧推理
    print("测试单帧推理...")
    batch_size = 2
    image = torch.randn(batch_size, 3, 256, 256)
    pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 重置状态并推理
    output = model(image, pose, reset_state=True)
    
    # 检查输出
    required_fields = ['sdf', 'occupancy', 'features', 'coords']
    for field in required_fields:
        assert field in output
        print(f"✅ 输出包含 {field}: {output[field].shape}")
    
    # 测试2：多帧序列
    print("测试多帧序列...")
    sequence_length = 3
    image_sequence = torch.randn(sequence_length, batch_size, 3, 256, 256)
    pose_sequence = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(sequence_length, batch_size, 1, 1)
    
    # 添加平移
    for t in range(sequence_length):
        pose_sequence[t, :, :3, 3] = torch.tensor([t * 0.1, 0, 0])
    
    # 处理序列
    sequence_output = model.forward_sequence(image_sequence, pose_sequence)
    
    # 检查序列输出
    for key in ['sdf', 'occupancy', 'features']:
        if key in sequence_output:
            assert sequence_output[key].shape[0] == sequence_length
            print(f"✅ 序列输出 {key} 形状正确: {sequence_output[key].shape}")
    
    # 测试3：梯度流
    print("测试梯度流...")
    model.reset_state()
    
    image.requires_grad_(True)
    output = model(image, pose, reset_state=True)
    
    loss = output['sdf'].sum() + output['occupancy'].sum()
    loss.backward()
    
    assert image.grad is not None
    assert not torch.all(image.grad == 0)
    print(f"✅ 梯度存在，梯度范数: {torch.norm(image.grad):.6f}")
    
    print("简化版本测试完成！")


if __name__ == "__main__":
    test_simple_stream_sdfformer()