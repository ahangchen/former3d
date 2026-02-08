"""
TartanAir SDF数据集类
加载TartanAir图像、位姿和SDF真值用于训练StreamSDFFormer
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

class TartanAirSDFDataset(Dataset):
    """
    TartanAir数据集，包含SDF真值
    
    数据格式：
    - 图像：PNG格式，640x480
    - 位姿：pose_left.txt，7列格式 [x, y, z, qx, qy, qz, qw]
    - SDF真值：NPZ文件，包含SDF网格和占用网格
    """
    
    def __init__(self, 
                 data_root: str,
                 sdf_path: str,
                 sequence_name: str = "abandonedfactory_sample_P001",
                 target_size: Tuple[int, int] = (256, 256),
                 max_frames: Optional[int] = None,
                 num_samples: int = 1024,
                 surface_weight: float = 0.4,
                 free_weight: float = 0.3,
                 occ_weight: float = 0.3,
                 truncation_margin: float = 0.1):
        """
        初始化数据集
        
        Args:
            data_root: TartanAir数据根目录
            sdf_path: SDF真值NPZ文件路径
            sequence_name: 序列名称
            target_size: 目标图像大小
            max_frames: 最大帧数（None表示使用所有帧）
            num_samples: 每帧采样点数
            surface_weight: 表面附近采样权重
            free_weight: 自由空间采样权重
            occ_weight: 占用空间采样权重
            truncation_margin: SDF截断边界
        """
        super().__init__()
        
        self.data_root = data_root
        self.sdf_path = sdf_path
        self.sequence_name = sequence_name
        self.target_size = target_size
        self.num_samples = num_samples
        self.surface_weight = surface_weight
        self.free_weight = free_weight
        self.occ_weight = occ_weight
        self.truncation_margin = truncation_margin
        
        # 构建序列路径
        self.sequence_dir = os.path.join(data_root, sequence_name, "P001")
        if not os.path.exists(self.sequence_dir):
            raise FileNotFoundError(f"序列目录不存在: {self.sequence_dir}")
        
        # 加载SDF真值
        self._load_sdf_ground_truth()
        
        # 加载图像和位姿
        self._load_images_and_poses(max_frames)
        
        # 计算采样权重
        self._compute_sampling_weights()
        
        print(f"TartanAir SDF数据集初始化完成:")
        print(f"  序列: {sequence_name}")
        print(f"  帧数: {len(self.image_paths)}")
        print(f"  SDF网格: {self.sdf_grid.shape}")
        print(f"  体素大小: {self.voxel_size:.4f}米")
        print(f"  场景边界: {self.bounds}")
        print(f"  每帧采样点数: {num_samples}")
        
    def _load_sdf_ground_truth(self):
        """加载SDF真值"""
        print(f"加载SDF真值: {self.sdf_path}")
        data = np.load(self.sdf_path)
        
        self.sdf_grid = data['sdf'].astype(np.float32)  # (D, H, W)
        self.occupancy_grid = data['occupancy'].astype(np.float32)
        self.voxel_size = float(data['voxel_size'])
        self.bounds = data['bounds'].astype(np.float32)  # (3, 2)
        self.intrinsics = data['intrinsics'].astype(np.float32)
        
        # 计算网格维度
        self.grid_dims = self.sdf_grid.shape  # (depth, height, width)
        
        # 创建体素坐标网格
        z_coords = np.arange(self.grid_dims[0])
        y_coords = np.arange(self.grid_dims[1])
        x_coords = np.arange(self.grid_dims[2])
        
        self.voxel_coords = np.stack(
            np.meshgrid(z_coords, y_coords, x_coords, indexing='ij'),
            axis=-1
        ).reshape(-1, 3)  # (N, 3)
        
        # 将体素坐标转换为世界坐标
        self.world_coords = self.voxel_coords * self.voxel_size + self.bounds[:, 0].reshape(1, 3)
        
        # 获取对应的SDF值
        self.sdf_values = self.sdf_grid.reshape(-1)
        
        print(f"  SDF网格形状: {self.sdf_grid.shape}")
        print(f"  体素数量: {len(self.sdf_values)}")
        print(f"  有效SDF值: {(self.sdf_values < 1.0).sum()}")
        
    def _load_images_and_poses(self, max_frames: Optional[int]):
        """加载图像和位姿"""
        # 图像目录
        image_dir = os.path.join(self.sequence_dir, "image_left")
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"图像目录不存在: {image_dir}")
        
        # 获取所有图像文件
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        if max_frames is not None:
            image_files = image_files[:max_frames]
        
        self.image_paths = [os.path.join(image_dir, f) for f in image_files]
        
        # 加载位姿
        pose_file = os.path.join(self.sequence_dir, "pose_left.txt")
        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"位姿文件不存在: {pose_file}")
        
        poses = np.loadtxt(pose_file)
        if max_frames is not None:
            poses = poses[:max_frames]
        
        # 转换位姿格式：7列 [x, y, z, qx, qy, qz, qw] -> 4x4矩阵
        self.poses = []
        for i in range(len(poses)):
            pose_7d = poses[i]
            pose_mat = self._pose_7d_to_matrix(pose_7d)
            self.poses.append(pose_mat)
        
        # 验证数量匹配
        if len(self.image_paths) != len(self.poses):
            raise ValueError(f"图像数量({len(self.image_paths)})与位姿数量({len(self.poses)})不匹配")
        
        print(f"  加载图像: {len(self.image_paths)}张")
        print(f"  加载位姿: {len(self.poses)}个")
        
    def _pose_7d_to_matrix(self, pose_7d: np.ndarray) -> np.ndarray:
        """
        将7维位姿 [x, y, z, qx, qy, qz, qw] 转换为4x4矩阵
        
        Args:
            pose_7d: 7维位姿 [x, y, z, qx, qy, qz, qw]
            
        Returns:
            4x4位姿矩阵
        """
        from scipy.spatial.transform import Rotation
        
        # 提取位置和四元数
        position = pose_7d[:3]
        quaternion = pose_7d[3:]  # [qx, qy, qz, qw]
        
        # 确保四元数格式正确
        if quaternion.shape[0] == 4:
            # 转换为 [w, x, y, z] 格式
            quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        
        # 创建旋转矩阵
        rotation = Rotation.from_quat(quaternion)
        rot_mat = rotation.as_matrix()
        
        # 创建4x4变换矩阵
        pose_mat = np.eye(4, dtype=np.float32)
        pose_mat[:3, :3] = rot_mat
        pose_mat[:3, 3] = position
        
        return pose_mat
    
    def _compute_sampling_weights(self):
        """计算采样权重"""
        # 根据SDF值分类点
        sdf_near_zero = np.abs(self.sdf_values) < self.truncation_margin
        sdf_positive = (self.sdf_values >= self.truncation_margin) & (self.sdf_values < 1.0)
        sdf_negative = (self.sdf_values <= -self.truncation_margin) & (self.sdf_values > -1.0)
        
        self.surface_indices = np.where(sdf_near_zero)[0]
        self.free_indices = np.where(sdf_positive)[0]
        self.occ_indices = np.where(sdf_negative)[0]
        
        print(f"  表面点: {len(self.surface_indices)}")
        print(f"  自由空间点: {len(self.free_indices)}")
        print(f"  占用空间点: {len(self.occ_indices)}")
        
        # 计算每类采样数量
        total_weight = self.surface_weight + self.free_weight + self.occ_weight
        self.num_surface = int(self.num_samples * self.surface_weight / total_weight)
        self.num_free = int(self.num_samples * self.free_weight / total_weight)
        self.num_occ = int(self.num_samples * self.occ_weight / total_weight)
        
        # 调整总数
        self.num_surface = min(self.num_surface, len(self.surface_indices))
        self.num_free = min(self.num_free, len(self.free_indices))
        self.num_occ = min(self.num_occ, len(self.occ_indices))
        
        print(f"  采样配置: 表面{self.num_surface}, 自由{self.num_free}, 占用{self.num_occ}")
        
    def _sample_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        从SDF网格中采样点
        
        Returns:
            points: 采样点世界坐标 (N, 3)
            sdf_values: 对应的SDF值 (N,)
        """
        # 随机采样每类点
        surface_sample = np.random.choice(self.surface_indices, self.num_surface, replace=False)
        free_sample = np.random.choice(self.free_indices, self.num_free, replace=False)
        occ_sample = np.random.choice(self.occ_indices, self.num_occ, replace=False)
        
        # 合并采样点
        sample_indices = np.concatenate([surface_sample, free_sample, occ_sample])
        
        # 获取对应的世界坐标和SDF值
        points = self.world_coords[sample_indices]
        sdf_values = self.sdf_values[sample_indices]
        
        return points, sdf_values
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            包含图像、位姿、采样点和SDF真值的字典
        """
        # 加载图像
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # 调整大小
        if image.size != self.target_size:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # 转换为张量并归一化
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        # 获取位姿
        pose = self.poses[idx]
        pose_tensor = torch.from_numpy(pose).float()
        
        # 采样点
        points, sdf_values = self._sample_points()
        points_tensor = torch.from_numpy(points).float()
        sdf_tensor = torch.from_numpy(sdf_values).float()
        
        # 内参矩阵（调整到目标大小）
        original_h, original_w = 480, 640
        target_h, target_w = self.target_size
        
        scale_x = target_w / original_w
        scale_y = target_h / original_h
        
        intrinsics = self.intrinsics.copy()
        intrinsics[0, 0] *= scale_x  # fx
        intrinsics[1, 1] *= scale_y  # fy
        intrinsics[0, 2] *= scale_x  # cx
        intrinsics[1, 2] *= scale_y  # cy
        
        intrinsics_tensor = torch.from_numpy(intrinsics).float()
        
        return {
            'image': image_tensor,          # (3, H, W)
            'pose': pose_tensor,            # (4, 4)
            'intrinsics': intrinsics_tensor, # (3, 3)
            'points': points_tensor,        # (N, 3)
            'sdf_gt': sdf_tensor,           # (N,)
            'frame_idx': idx,
            'sequence_name': self.sequence_name
        }
    
    def get_scene_info(self) -> Dict:
        """获取场景信息"""
        return {
            'bounds': self.bounds,
            'voxel_size': self.voxel_size,
            'grid_dims': self.grid_dims,
            'intrinsics': self.intrinsics,
            'num_frames': len(self),
            'sequence_name': self.sequence_name
        }


def create_tartanair_sdf_dataloader(
    data_root: str,
    sdf_path: str,
    batch_size: int = 1,
    num_workers: int = 0,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    创建TartanAir SDF数据加载器
    
    Args:
        data_root: 数据根目录
        sdf_path: SDF真值路径
        batch_size: 批次大小
        num_workers: 数据加载工作进程数
        **dataset_kwargs: 传递给TartanAirSDFDataset的参数
        
    Returns:
        数据加载器
    """
    from torch.utils.data import DataLoader
    
    dataset = TartanAirSDFDataset(data_root, sdf_path, **dataset_kwargs)
    
    # 自定义collate函数处理可变长度的点
    def collate_fn(batch):
        collated = {}
        for key in batch[0].keys():
            if key == 'points' or key == 'sdf_gt':
                # 对于点和SDF值，保持为列表
                collated[key] = [item[key] for item in batch]
            elif key == 'frame_idx' or key == 'sequence_name':
                collated[key] = [item[key] for item in batch]
            else:
                # 对于其他张量，正常堆叠
                collated[key] = torch.stack([item[key] for item in batch])
        return collated
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试数据集
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    data_root = "/home/cwh/Study/dataset/tartanair"
    sdf_path = "/home/cwh/coding/former3d/tartanair_sdf_output/abandonedfactory_sample_P001_sdf_occ.npz"
    
    try:
        dataset = TartanAirSDFDataset(
            data_root=data_root,
            sdf_path=sdf_path,
            sequence_name="abandonedfactory_sample_P001",
            target_size=(256, 256),
            max_frames=10,
            num_samples=512
        )
        
        print("\n测试数据集:")
        print(f"  数据集大小: {len(dataset)}")
        
        # 获取一个样本
        sample = dataset[0]
        print(f"\n样本信息:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} {value.dtype}")
            else:
                print(f"  {key}: {value}")
        
        # 获取场景信息
        scene_info = dataset.get_scene_info()
        print(f"\n场景信息:")
        for key, value in scene_info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()