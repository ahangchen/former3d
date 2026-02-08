"""
ScanNet流式数据集实现
支持ScanNet v2格式的流式推理
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from .streaming_dataset import StreamingDataset
from PIL import Image


class ScanNetStreamingDataset(StreamingDataset):
    """
    ScanNet流式数据集
    
    支持ScanNet v2格式：
    - RGB图像: color/*.jpg
    - 相机位姿: pose/*.txt
    - 相机内参: intrinsic/intrinsic_color.txt
    - 深度图: depth/*.png (可选)
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 sequence_ids: Optional[List[str]] = None,
                 transform: Optional[Callable] = None,
                 load_depth: bool = False,
                 load_sdf: bool = False,
                 max_sequence_length: Optional[int] = None,
                 image_size: Tuple[int, int] = (240, 320),  # ScanNet默认尺寸
                 normalize_images: bool = True,
                 cache_data: bool = False,
                 use_sampled_frames: bool = True,
                 frame_interval: int = 1):
        """
        初始化ScanNet数据集
        
        Args:
            data_root: ScanNet数据集根目录
            split: 数据集划分 ('train', 'val', 'test')
            sequence_ids: 指定加载的序列ID列表
            transform: 数据增强变换
            load_depth: 是否加载深度图
            load_sdf: 是否加载SDF真值
            max_sequence_length: 最大序列长度
            image_size: 输出图像尺寸 (H, W)
            normalize_images: 是否归一化图像
            cache_data: 是否缓存数据
            use_sampled_frames: 是否使用采样帧（跳过一些帧）
            frame_interval: 帧采样间隔
        """
        self.use_sampled_frames = use_sampled_frames
        self.frame_interval = frame_interval
        
        # ScanNet原始图像尺寸
        self.original_size = (480, 640)  # ScanNet默认尺寸
        
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
        加载ScanNet数据集
        
        ScanNet目录结构:
        data_root/
        ├── scans/
        │   ├── scene0000_00/
        │   │   ├── color/          # RGB图像 (.jpg)
        │   │   ├── pose/           # 相机位姿 (.txt)
        │   │   ├── intrinsic/      # 相机内参 (.txt)
        │   │   ├── depth/          # 深度图 (.png) (可选)
        │   │   └── ...
        │   └── scene0000_01/
        └── scans_test/             # 测试集
        """
        # 确定扫描目录
        if self.split == 'test':
            scans_dir = self.data_root / 'scans_test'
        else:
            scans_dir = self.data_root / 'scans'
        
        if not scans_dir.exists():
            raise FileNotFoundError(f"ScanNet扫描目录不存在: {scans_dir}")
        
        # 获取所有序列（场景）
        all_sequences = sorted([d.name for d in scans_dir.iterdir() if d.is_dir()])
        
        # 过滤序列
        if sequence_ids is not None:
            sequences = [seq for seq in all_sequences if seq in sequence_ids]
        else:
            sequences = all_sequences
        
        # 加载每个序列的帧
        total_frames = 0
        for seq_id in sequences:
            seq_path = scans_dir / seq_id
            
            # 检查必要的目录
            color_dir = seq_path / 'color'
            pose_dir = seq_path / 'pose'
            intrinsic_dir = seq_path / 'intrinsic'
            
            if not (color_dir.exists() and pose_dir.exists() and intrinsic_dir.exists()):
                print(f"警告: 序列 {seq_id} 缺少必要目录，跳过")
                continue
            
            # 获取颜色图像文件
            color_files = sorted(color_dir.glob('*.jpg'))
            if len(color_files) == 0:
                print(f"警告: 序列 {seq_id} 没有图像文件，跳过")
                continue
            
            # 采样帧
            if self.use_sampled_frames:
                frame_indices = list(range(0, len(color_files), self.frame_interval))
            else:
                frame_indices = list(range(len(color_files)))
            
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
                'total_frames': len(color_files),
                'sampled_frames': frame_indices,
                'color_dir': str(color_dir),
                'pose_dir': str(pose_dir),
                'intrinsic_dir': str(intrinsic_dir)
            }
            
            total_frames += len(frame_indices)
            print(f"  序列 {seq_id}: {len(frame_indices)} 帧 (总共 {len(color_files)} 帧)")
    
    def _get_image_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取ScanNet图像文件路径"""
        seq_info = self.sequence_info[sequence_id]
        color_dir = Path(seq_info['color_dir'])
        
        # ScanNet图像文件名格式: 000000.jpg, 000001.jpg, ...
        filename = f"{frame_idx:06d}.jpg"
        image_path = color_dir / filename
        
        if not image_path.exists():
            # 尝试其他格式
            alt_path = color_dir / f"frame-{frame_idx:06d}.color.jpg"
            if alt_path.exists():
                return str(alt_path)
            else:
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        return str(image_path)
    
    def _get_pose_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取ScanNet位姿文件路径"""
        seq_info = self.sequence_info[sequence_id]
        pose_dir = Path(seq_info['pose_dir'])
        
        # ScanNet位姿文件名格式: 000000.txt, 000001.txt, ...
        filename = f"{frame_idx:06d}.txt"
        pose_path = pose_dir / filename
        
        if not pose_path.exists():
            # 尝试其他格式
            alt_path = pose_dir / f"frame-{frame_idx:06d}.pose.txt"
            if alt_path.exists():
                return str(alt_path)
            else:
                raise FileNotFoundError(f"位姿文件不存在: {pose_path}")
        
        return str(pose_path)
    
    def _get_intrinsic_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取ScanNet内参文件路径"""
        seq_info = self.sequence_info[sequence_id]
        intrinsic_dir = Path(seq_info['intrinsic_dir'])
        
        # ScanNet通常所有帧共享同一个内参文件
        intrinsic_path = intrinsic_dir / "intrinsic_color.txt"
        
        if not intrinsic_path.exists():
            # 尝试其他文件名
            alt_path = intrinsic_dir / "intrinsic.txt"
            if alt_path.exists():
                return str(alt_path)
            else:
                raise FileNotFoundError(f"内参文件不存在: {intrinsic_path}")
        
        return str(intrinsic_path)
    
    def _get_depth_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取ScanNet深度图文件路径"""
        if not self.load_depth:
            return ""
        
        seq_info = self.sequence_info[sequence_id]
        seq_path = Path(seq_info['path'])
        depth_dir = seq_path / 'depth'
        
        if not depth_dir.exists():
            return ""
        
        # ScanNet深度图文件名格式: 000000.png, 000001.png, ...
        filename = f"{frame_idx:06d}.png"
        depth_path = depth_dir / filename
        
        if not depth_path.exists():
            # 尝试其他格式
            alt_path = depth_dir / f"frame-{frame_idx:06d}.depth.png"
            if alt_path.exists():
                return str(alt_path)
            else:
                return ""
        
        return str(depth_path)
    
    def _get_sdf_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取ScanNet SDF文件路径"""
        if not self.load_sdf:
            return ""
        
        seq_info = self.sequence_info[sequence_id]
        seq_path = Path(seq_info['path'])
        sdf_dir = seq_path / 'sdf'
        
        if not sdf_dir.exists():
            return ""
        
        # 假设SDF文件名为: 000000.npz, 000001.npz, ...
        filename = f"{frame_idx:06d}.npz"
        sdf_path = sdf_dir / filename
        
        if not sdf_path.exists():
            return ""
        
        return str(sdf_path)
    
    def _adjust_intrinsic_for_resize(self, intrinsic: torch.Tensor, 
                                    sequence_id: str, frame_idx: int) -> torch.Tensor:
        """
        根据图像尺寸调整ScanNet内参矩阵
        
        ScanNet原始图像尺寸: 480x640
        调整到目标尺寸: self.image_size
        """
        # 原始内参（针对480x640图像）
        K_original = intrinsic.numpy()
        
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
        加载ScanNet相机位姿
        
        ScanNet位姿格式: 4x4矩阵，相机到世界坐标系的变换
        """
        pose_path = self._get_pose_path(sequence_id, frame_idx)
        
        try:
            # 加载4x4矩阵
            pose_np = np.loadtxt(pose_path, dtype=np.float32)
            
            # 确保是4x4矩阵
            if pose_np.shape != (4, 4):
                # 尝试重塑
                if pose_np.size == 16:
                    pose_np = pose_np.reshape(4, 4)
                else:
                    raise ValueError(f"ScanNet位姿矩阵形状错误: {pose_np.shape}")
            
            # ScanNet位姿可能需要转换坐标系
            # 这里假设已经是正确的相机到世界变换
            
            pose_tensor = torch.from_numpy(pose_np).float()
            
            if self.cache_data:
                cache_key = f"{sequence_id}_{frame_idx}"
                self.pose_cache[cache_key] = pose_tensor
            
            return pose_tensor
            
        except Exception as e:
            raise RuntimeError(f"加载ScanNet位姿失败: {pose_path}, 错误: {e}")
    
    def _load_intrinsic(self, sequence_id: str, frame_idx: int) -> torch.Tensor:
        """
        加载ScanNet相机内参
        
        ScanNet内参格式: 3x3矩阵
        """
        intrinsic_path = self._get_intrinsic_path(sequence_id, frame_idx)
        
        try:
            # 加载3x3矩阵
            intrinsic_np = np.loadtxt(intrinsic_path, dtype=np.float32)
            
            # 确保是3x3矩阵
            if intrinsic_np.shape != (3, 3):
                # 尝试重塑
                if intrinsic_np.size == 9:
                    intrinsic_np = intrinsic_np.reshape(3, 3)
                else:
                    raise ValueError(f"ScanNet内参矩阵形状错误: {intrinsic_np.shape}")
            
            intrinsic_tensor = torch.from_numpy(intrinsic_np).float()
            
            # 调整内参以适应图像尺寸
            intrinsic_tensor = self._adjust_intrinsic_for_resize(
                intrinsic_tensor, sequence_id, frame_idx)
            
            if self.cache_data:
                cache_key = f"{sequence_id}_{frame_idx}"
                self.intrinsic_cache[cache_key] = intrinsic_tensor
            
            return intrinsic_tensor
            
        except Exception as e:
            raise RuntimeError(f"加载ScanNet内参失败: {intrinsic_path}, 错误: {e}")
    
    def get_scene_bbox(self, sequence_id: str) -> Optional[torch.Tensor]:
        """
        获取场景的边界框（如果可用）
        
        Args:
            sequence_id: 序列ID
            
        Returns:
            bbox: [2, 3] 最小和最大坐标，或None
        """
        seq_info = self.sequence_info[sequence_id]
        seq_path = Path(seq_info['path'])
        
        # 尝试加载axis_aligned.txt（包含场景边界框）
        bbox_file = seq_path / 'axis_aligned.txt'
        
        if bbox_file.exists():
            try:
                bbox_np = np.loadtxt(bbox_file, dtype=np.float32)
                if bbox_np.shape == (6,):
                    # 格式: min_x min_y min_z max_x max_y max_z
                    bbox_tensor = torch.from_numpy(bbox_np).reshape(2, 3).float()
                    return bbox_tensor
            except:
                pass
        
        return None
    
    @classmethod
    def create_train_val_split(cls, data_root: str, 
                              val_ratio: float = 0.1,
                              random_seed: int = 42) -> Dict[str, List[str]]:
        """
        创建训练/验证划分
        
        Args:
            data_root: 数据集根目录
            val_ratio: 验证集比例
            random_seed: 随机种子
            
        Returns:
            划分字典: {'train': [...], 'val': [...]}
        """
        import random
        
        scans_dir = Path(data_root) / 'scans'
        if not scans_dir.exists():
            raise FileNotFoundError(f"ScanNet扫描目录不存在: {scans_dir}")
        
        # 获取所有序列
        all_sequences = sorted([d.name for d in scans_dir.iterdir() if d.is_dir()])
        
        # 随机划分
        random.seed(random_seed)
        random.shuffle(all_sequences)
        
        val_size = int(len(all_sequences) * val_ratio)
        val_sequences = all_sequences[:val_size]
        train_sequences = all_sequences[val_size:]
        
        return {
            'train': train_sequences,
            'val': val_sequences
        }