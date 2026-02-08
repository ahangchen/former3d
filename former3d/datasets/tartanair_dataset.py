"""
TartanAir流式数据集实现
支持TartanAir格式的流式推理
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from .streaming_dataset import StreamingDataset
from PIL import Image


class TartanAirStreamingDataset(StreamingDataset):
    """
    TartanAir流式数据集
    
    支持TartanAir格式：
    - RGB图像: image_left/*.png
    - 相机位姿: pose_left.txt (所有帧的位姿在一个文件中)
    - 相机内参: 固定的内参矩阵
    - 深度图: depth_left/*.png (可选)
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 sequence_ids: Optional[List[str]] = None,
                 transform: Optional[Callable] = None,
                 load_depth: bool = False,
                 load_sdf: bool = False,
                 max_sequence_length: Optional[int] = None,
                 image_size: Tuple[int, int] = (256, 256),  # TartanAir默认尺寸
                 normalize_images: bool = True,
                 cache_data: bool = False,
                 use_left_camera: bool = True,
                 frame_interval: int = 1):
        """
        初始化TartanAir数据集
        
        Args:
            data_root: TartanAir数据集根目录
            split: 数据集划分 ('train', 'val', 'test')
            sequence_ids: 指定加载的序列ID列表
            transform: 数据增强变换
            load_depth: 是否加载深度图
            load_sdf: 是否加载SDF真值
            max_sequence_length: 最大序列长度
            image_size: 输出图像尺寸 (H, W)
            normalize_images: 是否归一化图像
            cache_data: 是否缓存数据
            use_left_camera: 是否使用左相机（True=左，False=右）
            frame_interval: 帧采样间隔
        """
        self.use_left_camera = use_left_camera
        self.frame_interval = frame_interval
        self.camera_side = 'left' if use_left_camera else 'right'
        
        # TartanAir原始图像尺寸
        self.original_size = (480, 640)  # TartanAir默认尺寸
        
        # TartanAir固定内参（针对左相机，640x480图像）
        # 这些是典型值，实际可能因序列而异
        self.fixed_intrinsic = np.array([
            [320.0, 0.0, 320.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        super().__init__(
            data_root=data_root,
            split=split,
            sequence_ids=sequence_ids,
            transform=transform,
            load_depth=load_depth,
            load_sdf=load_sdf,
            max_sequence_length=max_sequence_length,
            image_size=image_size,
            normalize_images=normalize_images,
            cache_data=cache_data
        )
    
    def _load_dataset(self, sequence_ids: Optional[List[str]]):
        """
        加载TartanAir数据集
        
        TartanAir目录结构:
        data_root/
        ├── abandonedfactory/
        │   ├── Easy/
        │   │   ├── P000/
        │   │   │   ├── image_left/      # 左相机图像 (.png)
        │   │   │   ├── depth_left/      # 左相机深度图 (.png) (可选)
        │   │   │   └── pose_left.txt    # 左相机位姿
        │   │   └── P001/
        │   └── Hard/
        └── carwelding/
        """
        # 获取所有环境
        env_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        
        # 收集所有序列
        all_sequences = []
        for env_dir in env_dirs:
            # 获取难度级别
            for difficulty in ['Easy', 'Hard', 'Normal']:
                difficulty_dir = env_dir / difficulty
                if not difficulty_dir.exists():
                    continue
                
                # 获取所有轨迹
                for traj_dir in difficulty_dir.iterdir():
                    if traj_dir.is_dir():
                        # 序列ID格式: env/difficulty/trajectory
                        seq_id = f"{env_dir.name}/{difficulty}/{traj_dir.name}"
                        all_sequences.append((seq_id, traj_dir))
        
        # 过滤序列
        selected_sequences = []
        if sequence_ids is not None:
            seq_id_map = {seq[0]: seq[1] for seq in all_sequences}
            for seq_id in sequence_ids:
                if seq_id in seq_id_map:
                    selected_sequences.append((seq_id, seq_id_map[seq_id]))
        else:
            selected_sequences = all_sequences
        
        # 加载每个序列的帧
        total_frames = 0
        for seq_id, seq_path in selected_sequences:
            # 检查必要的文件
            image_dir = seq_path / f"image_{self.camera_side}"
            pose_file = seq_path / f"pose_{self.camera_side}.txt"
            
            if not (image_dir.exists() and pose_file.exists()):
                print(f"警告: 序列 {seq_id} 缺少必要文件，跳过")
                continue
            
            # 获取图像文件
            image_files = sorted(image_dir.glob('*.png'))
            if len(image_files) == 0:
                print(f"警告: 序列 {seq_id} 没有图像文件，跳过")
                continue
            
            # 加载位姿文件
            try:
                poses_np = np.loadtxt(pose_file, dtype=np.float32)
                
                # 处理单行情况
                if poses_np.ndim == 1:
                    if poses_np.shape[0] != 16:
                        print(f"警告: 序列 {seq_id} 位姿文件格式错误（单行但不是16个值），跳过")
                        continue
                    poses_np = poses_np.reshape(1, 16)  # 转换为(1, 16)
                
                # 每行是一个4x4矩阵（按行展开的16个值）
                if poses_np.shape[1] != 16:
                    print(f"警告: 序列 {seq_id} 位姿文件格式错误，跳过")
                    continue
                
                num_poses = poses_np.shape[0]
                if num_poses < len(image_files):
                    print(f"警告: 序列 {seq_id} 位姿数量({num_poses})少于图像数量({len(image_files)})")
                    image_files = image_files[:num_poses]
                elif num_poses > len(image_files):
                    print(f"警告: 序列 {seq_id} 位姿数量({num_poses})多于图像数量({len(image_files)})")
            
            except Exception as e:
                print(f"警告: 序列 {seq_id} 位姿文件加载失败: {e}，跳过")
                continue
            
            # 采样帧
            frame_indices = list(range(0, len(image_files), self.frame_interval))
            
            # 限制最大序列长度
            if self.max_sequence_length is not None:
                frame_indices = frame_indices[:self.max_sequence_length]
            
            # 添加帧索引
            start_idx = len(self.frame_indices)
            for frame_idx in frame_indices:
                self.frame_indices.append((seq_id, frame_idx))
            
            end_idx = len(self.frame_indices) - 1
            
            # 保存序列信息
            self.sequence_info[seq_id] = {
                'path': str(seq_path),
                'start_idx': start_idx,
                'end_idx': end_idx,
                'length': len(frame_indices),
                'total_frames': len(image_files),
                'sampled_frames': frame_indices,
                'image_dir': str(image_dir),
                'pose_file': str(pose_file),
                'num_poses': num_poses,
                'poses_np': poses_np  # 缓存所有位姿
            }
            
            total_frames += len(frame_indices)
            print(f"  序列 {seq_id}: {len(frame_indices)} 帧 (总共 {len(image_files)} 帧)")
    
    def _get_image_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取TartanAir图像文件路径"""
        seq_info = self.sequence_info[sequence_id]
        image_dir = Path(seq_info['image_dir'])
        
        # TartanAir图像文件名格式: 000000_left.png, 000001_left.png, ...
        # 或者: 000000.png, 000001.png, ...
        filename_patterns = [
            f"{frame_idx:06d}_{self.camera_side}.png",
            f"{frame_idx:06d}.png",
            f"frame_{frame_idx:06d}.png"
        ]
        
        for pattern in filename_patterns:
            image_path = image_dir / pattern
            if image_path.exists():
                return str(image_path)
        
        raise FileNotFoundError(f"TartanAir图像文件不存在: {image_dir}/{frame_idx:06d}_*.png")
    
    def _get_pose_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取TartanAir位姿文件路径（返回文件路径，但实际从缓存加载）"""
        seq_info = self.sequence_info[sequence_id]
        return seq_info['pose_file']
    
    def _get_intrinsic_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取TartanAir内参文件路径（返回空字符串，使用固定内参）"""
        return ""
    
    def _get_depth_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取TartanAir深度图文件路径"""
        if not self.load_depth:
            return ""
        
        seq_info = self.sequence_info[sequence_id]
        seq_path = Path(seq_info['path'])
        depth_dir = seq_path / f"depth_{self.camera_side}"
        
        if not depth_dir.exists():
            return ""
        
        # TartanAir深度图文件名格式: 000000_left_depth.npy 或 000000.png
        filename_patterns = [
            f"{frame_idx:06d}_{self.camera_side}_depth.npy",
            f"{frame_idx:06d}_depth.npy",
            f"{frame_idx:06d}.png"
        ]
        
        for pattern in filename_patterns:
            depth_path = depth_dir / pattern
            if depth_path.exists():
                return str(depth_path)
        
        return ""
    
    def _get_sdf_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取TartanAir SDF文件路径"""
        if not self.load_sdf:
            return ""
        
        seq_info = self.sequence_info[sequence_id]
        seq_path = Path(seq_info['path'])
        sdf_dir = seq_path / 'sdf'
        
        if not sdf_dir.exists():
            return ""
        
        # 假设SDF文件名为: 000000.npz
        filename = f"{frame_idx:06d}.npz"
        sdf_path = sdf_dir / filename
        
        if not sdf_path.exists():
            return ""
        
        return str(sdf_path)
    
    def _adjust_intrinsic_for_resize(self, intrinsic: torch.Tensor, 
                                    sequence_id: str, frame_idx: int) -> torch.Tensor:
        """
        根据图像尺寸调整TartanAir内参矩阵
        
        TartanAir原始图像尺寸: 480x640
        调整到目标尺寸: self.image_size
        """
        # 使用固定内参
        K_original = self.fixed_intrinsic.copy()
        
        # 计算缩放因子
        scale_h = self.image_size[0] / self.original_size[0]
        scale_w = self.image_size[1] / self.original_size[1]
        
        # 调整内参
        K_scaled = K_original.copy()
        K_scaled[0, 0] *= scale_w  # fx
        K_scaled[1, 1] *= scale_h  # fy
        K_scaled[0, 2] *= scale_w  # cx
        K_scaled[1, 2] *= scale_h  # cy
        
        return torch.from_numpy(K_scaled).float()
    
    def _load_pose(self, sequence_id: str, frame_idx: int) -> torch.Tensor:
        """
        加载TartanAir相机位姿
        
        TartanAir位姿格式: 每行16个值，表示4x4矩阵（按行展开）
        位姿表示相机到世界坐标系的变换
        """
        seq_info = self.sequence_info[sequence_id]
        
        # 从缓存加载位姿
        poses_np = seq_info['poses_np']
        
        if frame_idx >= poses_np.shape[0]:
            raise IndexError(f"帧索引 {frame_idx} 超出位姿数量 {poses_np.shape[0]}")
        
        # 获取该帧的位姿
        pose_flat = poses_np[frame_idx]
        pose_np = pose_flat.reshape(4, 4)
        
        # TartanAir位姿可能需要转换坐标系
        # 这里假设已经是正确的相机到世界变换
        
        pose_tensor = torch.from_numpy(pose_np).float()
        
        if self.cache_data:
            cache_key = f"{sequence_id}_{frame_idx}"
            self.pose_cache[cache_key] = pose_tensor
        
        return pose_tensor
    
    def _load_intrinsic(self, sequence_id: str, frame_idx: int) -> torch.Tensor:
        """
        加载TartanAir相机内参
        
        TartanAir使用固定内参
        """
        # 调整内参以适应图像尺寸
        intrinsic_tensor = self._adjust_intrinsic_for_resize(
            torch.from_numpy(self.fixed_intrinsic).float(),
            sequence_id, frame_idx
        )
        
        if self.cache_data:
            cache_key = f"{sequence_id}_{frame_idx}"
            self.intrinsic_cache[cache_key] = intrinsic_tensor
        
        return intrinsic_tensor
    
    def _load_depth(self, sequence_id: str, frame_idx: int) -> Optional[torch.Tensor]:
        """
        加载TartanAir深度图
        
        TartanAir深度图格式: .npy文件（浮点数）或.png文件（16位）
        """
        if not self.load_depth:
            return None
        
        depth_path = self._get_depth_path(sequence_id, frame_idx)
        if not depth_path or not os.path.exists(depth_path):
            return None
        
        cache_key = f"{sequence_id}_{frame_idx}"
        
        if self.cache_data and cache_key in self.depth_cache:
            return self.depth_cache[cache_key]
        
        try:
            # 根据文件扩展名加载
            if depth_path.endswith('.npy'):
                # .npy文件（浮点数）
                depth_np = np.load(depth_path)
                
                # 调整尺寸
                if self.image_size and depth_np.shape != self.image_size:
                    # 使用OpenCV或PIL调整尺寸
                    from PIL import Image
                    import cv2
                    
                    # 转换为PIL图像调整尺寸
                    depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8)
                    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
                    depth_image = Image.fromarray(depth_uint8)
                    
                    depth_image = depth_image.resize((self.image_size[1], self.image_size[0]), 
                                                    Image.NEAREST)
                    depth_resized = np.array(depth_image, dtype=np.float32)
                    
                    # 恢复原始深度范围
                    depth_np = depth_resized / 255.0 * (depth_np.max() - depth_np.min()) + depth_np.min()
                    
            elif depth_path.endswith('.png'):
                # .png文件（16位）
                depth_image = Image.open(depth_path)
                
                # 调整尺寸
                if self.image_size:
                    depth_image = depth_image.resize((self.image_size[1], self.image_size[0]), 
                                                    Image.NEAREST)
                
                depth_np = np.array(depth_image, dtype=np.float32)
                depth_np = depth_np / 1000.0  # 毫米转米
            else:
                print(f"警告: 不支持的深度图格式: {depth_path}")
                return None
            
            # 转换为torch tensor
            depth_tensor = torch.from_numpy(depth_np).float()
            
            if self.cache_data:
                self.depth_cache[cache_key] = depth_tensor
            
            return depth_tensor
            
        except Exception as e:
            print(f"警告: 加载TartanAir深度图失败: {depth_path}, 错误: {e}")
            return None
    
    @classmethod
    def get_available_environments(cls, data_root: str) -> List[str]:
        """
        获取可用的环境列表
        
        Args:
            data_root: 数据集根目录
            
        Returns:
            环境名称列表
        """
        data_path = Path(data_root)
        if not data_path.exists():
            return []
        
        envs = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
        return envs
    
    @classmethod
    def get_available_sequences(cls, data_root: str, 
                               environment: str = None) -> List[str]:
        """
        获取可用的序列列表
        
        Args:
            data_root: 数据集根目录
            environment: 环境名称（可选）
            
        Returns:
            序列ID列表
        """
        data_path = Path(data_root)
        
        sequences = []
        if environment:
            env_path = data_path / environment
            if not env_path.exists():
                return []
            
            envs = [env_path]
        else:
            envs = [d for d in data_path.iterdir() if d.is_dir()]
        
        for env_dir in envs:
            env_name = env_dir.name
            for difficulty in ['Easy', 'Hard', 'Normal']:
                difficulty_dir = env_dir / difficulty
                if not difficulty_dir.exists():
                    continue
                
                for traj_dir in difficulty_dir.iterdir():
                    if traj_dir.is_dir():
                        seq_id = f"{env_name}/{difficulty}/{traj_dir.name}"
                        sequences.append(seq_id)
        
        return sorted(sequences)