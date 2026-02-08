"""
流式推理数据集基类
支持单帧加载，用于StreamSDFFormer训练和推理
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
from collections import defaultdict
import torch.utils.data as data
from PIL import Image
import cv2


class StreamingDataset(data.Dataset):
    """
    流式推理数据集基类
    
    特性：
    1. 按帧加载数据（单帧输入，单帧输出）
    2. 支持序列连续性跟踪
    3. 可选的历史状态管理
    4. 与StreamSDFFormerIntegrated兼容
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 sequence_ids: Optional[List[str]] = None,
                 transform: Optional[Callable] = None,
                 load_depth: bool = False,
                 load_sdf: bool = False,
                 max_sequence_length: Optional[int] = None,
                 image_size: Tuple[int, int] = (128, 128),
                 normalize_images: bool = True,
                 cache_data: bool = False):
        """
        初始化流式数据集
        
        Args:
            data_root: 数据集根目录
            split: 数据集划分 ('train', 'val', 'test')
            sequence_ids: 指定加载的序列ID列表，None表示加载所有序列
            transform: 数据增强变换
            load_depth: 是否加载深度图
            load_sdf: 是否加载SDF真值
            max_sequence_length: 最大序列长度（用于截断长序列）
            image_size: 输出图像尺寸 (H, W)
            normalize_images: 是否对图像进行归一化
            cache_data: 是否缓存数据到内存
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.load_depth = load_depth
        self.load_sdf = load_sdf
        self.max_sequence_length = max_sequence_length
        self.image_size = image_size
        self.normalize_images = normalize_images
        self.cache_data = cache_data
        
        # 数据缓存
        self.image_cache = {}
        self.pose_cache = {}
        self.intrinsic_cache = {}
        self.depth_cache = {} if load_depth else None
        self.sdf_cache = {} if load_sdf else None
        
        # 帧索引管理
        self.frame_indices = []  # 每个元素是 (sequence_id, frame_idx)
        self.sequence_info = {}  # 序列信息 {sequence_id: {start_idx: int, end_idx: int, length: int}}
        
        # 加载数据集
        self._load_dataset(sequence_ids)
        
        print(f"✅ StreamingDataset初始化完成")
        print(f"   数据集: {self.data_root.name}")
        print(f"   划分: {split}")
        print(f"   序列数: {len(self.sequence_info)}")
        print(f"   总帧数: {len(self)}")
        print(f"   加载深度: {load_depth}")
        print(f"   加载SDF: {load_sdf}")
    
    def _load_dataset(self, sequence_ids: Optional[List[str]]):
        """
        加载数据集，创建帧索引
        
        子类需要重写此方法以实现具体的数据集加载逻辑
        """
        raise NotImplementedError("子类必须实现_load_dataset方法")
    
    def _load_image(self, sequence_id: str, frame_idx: int) -> torch.Tensor:
        """
        加载图像
        
        Args:
            sequence_id: 序列ID
            frame_idx: 帧索引
            
        Returns:
            image: [3, H, W] RGB图像，值范围[0, 1]
        """
        cache_key = f"{sequence_id}_{frame_idx}"
        
        if self.cache_data and cache_key in self.image_cache:
            return self.image_cache[cache_key]
        
        # 子类需要实现具体的图像加载逻辑
        image_path = self._get_image_path(sequence_id, frame_idx)
        
        # 加载图像
        try:
            # 使用PIL加载
            image = Image.open(image_path).convert('RGB')
            
            # 调整尺寸
            if self.image_size:
                image = image.resize((self.image_size[1], self.image_size[0]), 
                                    Image.BILINEAR)
            
            # 转换为numpy并归一化
            image_np = np.array(image, dtype=np.float32) / 255.0
            
            # 转换为torch tensor [C, H, W]
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
            
            # 标准化（ImageNet均值标准差）
            if self.normalize_images:
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                image_tensor = (image_tensor - mean) / std
            
            if self.cache_data:
                self.image_cache[cache_key] = image_tensor
            
            return image_tensor
            
        except Exception as e:
            raise RuntimeError(f"加载图像失败: {image_path}, 错误: {e}")
    
    def _load_pose(self, sequence_id: str, frame_idx: int) -> torch.Tensor:
        """
        加载相机位姿
        
        Args:
            sequence_id: 序列ID
            frame_idx: 帧索引
            
        Returns:
            pose: [4, 4] 相机到世界坐标系的变换矩阵
        """
        cache_key = f"{sequence_id}_{frame_idx}"
        
        if self.cache_data and cache_key in self.pose_cache:
            return self.pose_cache[cache_key]
        
        # 子类需要实现具体的位姿加载逻辑
        pose_path = self._get_pose_path(sequence_id, frame_idx)
        
        try:
            # 加载位姿文件（假设是4x4矩阵的文本文件）
            pose_np = np.loadtxt(pose_path, dtype=np.float32)
            
            # 确保是4x4矩阵
            if pose_np.shape != (4, 4):
                raise ValueError(f"位姿矩阵形状错误: {pose_np.shape}, 应为(4, 4)")
            
            pose_tensor = torch.from_numpy(pose_np).float()
            
            if self.cache_data:
                self.pose_cache[cache_key] = pose_tensor
            
            return pose_tensor
            
        except Exception as e:
            raise RuntimeError(f"加载位姿失败: {pose_path}, 错误: {e}")
    
    def _load_intrinsic(self, sequence_id: str, frame_idx: int) -> torch.Tensor:
        """
        加载相机内参
        
        Args:
            sequence_id: 序列ID
            frame_idx: 帧索引
            
        Returns:
            intrinsic: [3, 3] 相机内参矩阵
        """
        cache_key = f"{sequence_id}_{frame_idx}"
        
        if self.cache_data and cache_key in self.intrinsic_cache:
            return self.intrinsic_cache[cache_key]
        
        # 子类需要实现具体的内参加载逻辑
        intrinsic_path = self._get_intrinsic_path(sequence_id, frame_idx)
        
        try:
            # 加载内参文件（假设是3x3矩阵的文本文件）
            intrinsic_np = np.loadtxt(intrinsic_path, dtype=np.float32)
            
            # 确保是3x3矩阵
            if intrinsic_np.shape != (3, 3):
                raise ValueError(f"内参矩阵形状错误: {intrinsic_np.shape}, 应为(3, 3)")
            
            intrinsic_tensor = torch.from_numpy(intrinsic_np).float()
            
            # 根据图像尺寸调整内参
            if self.image_size:
                # 这里需要知道原始图像尺寸来调整内参
                # 子类应该重写此方法以实现正确的内参调整
                intrinsic_tensor = self._adjust_intrinsic_for_resize(
                    intrinsic_tensor, sequence_id, frame_idx)
            
            if self.cache_data:
                self.intrinsic_cache[cache_key] = intrinsic_tensor
            
            return intrinsic_tensor
            
        except Exception as e:
            raise RuntimeError(f"加载内参失败: {intrinsic_path}, 错误: {e}")
    
    def _load_depth(self, sequence_id: str, frame_idx: int) -> Optional[torch.Tensor]:
        """
        加载深度图（可选）
        
        Args:
            sequence_id: 序列ID
            frame_idx: 帧索引
            
        Returns:
            depth: [H, W] 深度图，或None
        """
        if not self.load_depth:
            return None
        
        cache_key = f"{sequence_id}_{frame_idx}"
        
        if self.cache_data and cache_key in self.depth_cache:
            return self.depth_cache[cache_key]
        
        # 子类需要实现具体的深度图加载逻辑
        depth_path = self._get_depth_path(sequence_id, frame_idx)
        
        if not os.path.exists(depth_path):
            return None
        
        try:
            # 加载深度图（假设是16位PNG）
            depth_image = Image.open(depth_path)
            
            # 调整尺寸
            if self.image_size:
                depth_image = depth_image.resize((self.image_size[1], self.image_size[0]), 
                                                Image.NEAREST)
            
            # 转换为numpy
            depth_np = np.array(depth_image, dtype=np.float32)
            
            # 深度图通常需要除以缩放因子（如1000.0对于毫米单位）
            depth_np = depth_np / 1000.0  # 毫米转米
            
            depth_tensor = torch.from_numpy(depth_np).float()
            
            if self.cache_data:
                self.depth_cache[cache_key] = depth_tensor
            
            return depth_tensor
            
        except Exception as e:
            print(f"警告: 加载深度图失败: {depth_path}, 错误: {e}")
            return None
    
    def _load_sdf(self, sequence_id: str, frame_idx: int) -> Optional[Dict]:
        """
        加载SDF真值（可选）
        
        Args:
            sequence_id: 序列ID
            frame_idx: 帧索引
            
        Returns:
            sdf_data: 包含SDF真值的字典，或None
        """
        if not self.load_sdf:
            return None
        
        cache_key = f"{sequence_id}_{frame_idx}"
        
        if self.cache_data and cache_key in self.sdf_cache:
            return self.sdf_cache[cache_key]
        
        # 子类需要实现具体的SDF加载逻辑
        sdf_path = self._get_sdf_path(sequence_id, frame_idx)
        
        if not os.path.exists(sdf_path):
            return None
        
        try:
            # 加载SDF数据（假设是.npz文件）
            sdf_data = np.load(sdf_path)
            
            # 转换为torch tensor
            sdf_dict = {
                'coords': torch.from_numpy(sdf_data['coords']).float(),
                'sdf_values': torch.from_numpy(sdf_data['sdf_values']).float(),
                'occupancy': torch.from_numpy(sdf_data['occupancy']).float() if 'occupancy' in sdf_data else None
            }
            
            if self.cache_data:
                self.sdf_cache[cache_key] = sdf_dict
            
            return sdf_dict
            
        except Exception as e:
            print(f"警告: 加载SDF失败: {sdf_path}, 错误: {e}")
            return None
    
    def _get_image_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取图像文件路径（子类需要实现）"""
        raise NotImplementedError
    
    def _get_pose_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取位姿文件路径（子类需要实现）"""
        raise NotImplementedError
    
    def _get_intrinsic_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取内参文件路径（子类需要实现）"""
        raise NotImplementedError
    
    def _get_depth_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取深度图文件路径（子类需要实现）"""
        raise NotImplementedError
    
    def _get_sdf_path(self, sequence_id: str, frame_idx: int) -> str:
        """获取SDF文件路径（子类需要实现）"""
        raise NotImplementedError
    
    def _adjust_intrinsic_for_resize(self, intrinsic: torch.Tensor, 
                                    sequence_id: str, frame_idx: int) -> torch.Tensor:
        """
        根据图像尺寸调整内参矩阵
        
        Args:
            intrinsic: 原始内参矩阵 [3, 3]
            sequence_id: 序列ID
            frame_idx: 帧索引
            
        Returns:
            调整后的内参矩阵
        """
        # 默认实现：假设图像从原始尺寸缩放到self.image_size
        # 子类应该重写此方法以使用正确的原始尺寸
        return intrinsic
    
    def __len__(self) -> int:
        """返回数据集总帧数"""
        return len(self.frame_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单帧数据
        
        Args:
            idx: 帧索引
            
        Returns:
            data_dict: 包含帧数据的字典
        """
        sequence_id, frame_idx = self.frame_indices[idx]
        
        # 加载数据
        image = self._load_image(sequence_id, frame_idx)
        pose = self._load_pose(sequence_id, frame_idx)
        intrinsic = self._load_intrinsic(sequence_id, frame_idx)
        depth = self._load_depth(sequence_id, frame_idx)
        sdf_data = self._load_sdf(sequence_id, frame_idx)
        
        # 构建数据字典
        data_dict = {
            'image': image,                    # [3, H, W]
            'pose': pose,                      # [4, 4]
            'intrinsics': intrinsic,           # [3, 3]
            'frame_id': frame_idx,             # 标量
            'sequence_id': sequence_id,        # 字符串（在collate_fn中处理）
            'idx': idx,                        # 数据集索引
        }
        
        # 添加可选数据
        if depth is not None:
            data_dict['depth'] = depth         # [H, W]
        
        if sdf_data is not None:
            data_dict.update(sdf_data)         # 包含'coords', 'sdf_values'等
        
        # 应用数据增强
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        
        return data_dict
    
    def get_sequence_info(self, sequence_id: str) -> Dict:
        """
        获取序列信息
        
        Args:
            sequence_id: 序列ID
            
        Returns:
            序列信息字典
        """
        if sequence_id not in self.sequence_info:
            raise ValueError(f"序列 {sequence_id} 不存在")
        
        return self.sequence_info[sequence_id].copy()
    
    def get_frame_range(self, sequence_id: str) -> Tuple[int, int]:
        """
        获取序列的帧范围
        
        Args:
            sequence_id: 序列ID
            
        Returns:
            (start_idx, end_idx): 帧索引范围
        """
        info = self.get_sequence_info(sequence_id)
        return info['start_idx'], info['end_idx']
    
    def get_sequence_length(self, sequence_id: str) -> int:
        """
        获取序列长度
        
        Args:
            sequence_id: 序列ID
            
        Returns:
            序列帧数
        """
        info = self.get_sequence_info(sequence_id)
        return info['length']
    
    def get_all_sequence_ids(self) -> List[str]:
        """
        获取所有序列ID
        
        Returns:
            序列ID列表
        """
        return list(self.sequence_info.keys())
    
    def clear_cache(self):
        """清空数据缓存"""
        self.image_cache.clear()
        self.pose_cache.clear()
        self.intrinsic_cache.clear()
        if self.depth_cache is not None:
            self.depth_cache.clear()
        if self.sdf_cache is not None:
            self.sdf_cache.clear()
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        批处理函数，用于DataLoader
        
        Args:
            batch: 批次数据列表
            
        Returns:
            批处理后的数据字典
        """
        if len(batch) == 0:
            return {}
        
        # 确定哪些键需要批处理
        collated = {}
        
        # 处理可以堆叠的张量
        stackable_keys = ['image', 'pose', 'intrinsics', 'depth', 'idx']
        
        for key in stackable_keys:
            if key in batch[0] and batch[0][key] is not None:
                try:
                    collated[key] = torch.stack([item[key] for item in batch])
                except:
                    # 如果形状不一致，使用列表
                    collated[key] = [item[key] for item in batch]
        
        # 处理frame_id
        if 'frame_id' in batch[0]:
            collated['frame_id'] = torch.tensor(
                [item['frame_id'] for item in batch], dtype=torch.long)
        
        # 处理sequence_id（字符串列表）
        if 'sequence_id' in batch[0]:
            collated['sequence_id'] = [item['sequence_id'] for item in batch]
        
        # 处理SDF数据（稀疏表示，不能直接堆叠）
        sdf_keys = ['coords', 'sdf_values', 'occupancy']
        for key in sdf_keys:
            if key in batch[0] and batch[0][key] is not None:
                # 对于稀疏数据，我们返回列表而不是堆叠
                collated[key] = [item[key] for item in batch]
        
        return collated