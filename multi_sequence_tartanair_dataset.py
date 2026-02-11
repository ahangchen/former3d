#!/usr/bin/env python3
"""
多序列TartanAir数据集
支持加载多个序列并将长序列切分成固定长度的片段
"""

import os
import numpy as np
import torch
import imageio
import glob
from torch.utils.data import Dataset
import spconv.pytorch as spconv
from PIL import Image
from scipy.spatial.transform import Rotation
from typing import List, Tuple, Dict, Optional
import random


class MultiSequenceTartanAirDataset(Dataset):
    """
    多序列TartanAir数据集
    支持加载多个序列并将长序列切分成固定长度的片段
    """
    
    def __init__(
        self,
        data_root: str,
        n_view: int = 5,
        stride: int = 1,
        crop_size: tuple = (48, 48, 32),
        voxel_size: float = 0.04,
        target_image_size: tuple = (256, 256),
        max_depth: float = 10.0,
        truncation_margin: float = 0.2,
        augment: bool = False,
        max_sequences: int = None,
        shuffle: bool = True
    ):
        """
        初始化多序列数据集
        
        Args:
            data_root: TartanAir数据根目录
            n_view: 每个片段的帧数
            stride: 片段之间的步长
            crop_size: 裁剪尺寸（体素单位）
            voxel_size: 体素大小（米）
            target_image_size: 目标图像大小
            max_depth: 最大深度值（米）
            truncation_margin: TSDF截断边界
            augment: 是否使用数据增强
            max_sequences: 最大序列数量（None表示使用所有）
            shuffle: 是否打乱序列顺序
        """
        super().__init__()
        
        self.data_root = data_root
        self.n_view = n_view
        self.stride = stride
        self.crop_size = np.array(crop_size)
        self.voxel_size = voxel_size
        self.target_image_size = target_image_size
        self.max_depth = max_depth
        self.truncation_margin = truncation_margin
        self.augment = augment
        self.shuffle = shuffle
        
        # TartanAir固定内参（640x480）
        self.K_original = np.array([
            [320.0, 0, 320.0],
            [0, 320.0, 240.0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # 调整到目标图像大小的内参
        scale_x = target_image_size[0] / 640
        scale_y = target_image_size[1] / 480
        self.K = self.K_original.copy()
        self.K[0, 0] *= scale_x  # fx
        self.K[1, 1] *= scale_y  # fy
        self.K[0, 2] *= scale_x  # cx
        self.K[1, 2] *= scale_y  # cy
        
        # 加载所有序列信息
        self.sequences = self._discover_sequences()
        
        if max_sequences is not None and max_sequences > 0:
            if self.shuffle:
                random.shuffle(self.sequences)
            self.sequences = self.sequences[:max_sequences]
        
        # 构建片段索引
        self.segments = self._build_segments()
        
        print(f"多序列TartanAir数据集初始化完成:")
        print(f"  总序列数: {len(self.sequences)}")
        print(f"  总片段数: {len(self.segments)}")
        print(f"  片段长度: {n_view} 帧")
        print(f"  片段步长: {stride}")
        print(f"  裁剪尺寸: {crop_size} 体素")
        print(f"  体素大小: {voxel_size}米")
        print(f"  图像大小: {target_image_size}")
        
    def _discover_sequences(self) -> List[Dict]:
        """发现所有可用的序列"""
        sequences = []

        # 遍历data_root下的所有目录
        for item in os.listdir(self.data_root):
            item_path = os.path.join(self.data_root, item)
            if os.path.isdir(item_path):
                # 查找任何Pxxx格式的子目录（P001, P002, P003等）
                p_dirs = []
                for sub_item in os.listdir(item_path):
                    if sub_item.startswith('P') and os.path.isdir(os.path.join(item_path, sub_item)):
                        # 检查是否是数字（如P001, P002等）
                        if sub_item[1:].isdigit():
                            p_dirs.append(sub_item)

                if not p_dirs:
                    continue  # 没有找到Pxxx子目录，跳过

                # 使用第一个找到的Pxxx目录
                p_dir = p_dirs[0]
                p_path = os.path.join(item_path, p_dir)

                # 检查必要的子目录
                rgb_dir = os.path.join(p_path, "image_left")
                depth_dir = os.path.join(p_path, "depth_left")
                pose_file = os.path.join(p_path, "pose_left.txt")

                if (os.path.exists(rgb_dir) and
                    os.path.exists(depth_dir) and
                    os.path.exists(pose_file)):

                    # 获取RGB文件列表
                    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
                    if len(rgb_files) >= self.n_view:
                        sequences.append({
                            'name': item,
                            'path': item_path,
                            'p_path': p_path,  # 使用通用的p_path而不是p001_path
                            'p_dir': p_dir,     # 记录实际使用的Pxxx目录
                            'rgb_dir': rgb_dir,
                            'depth_dir': depth_dir,
                            'pose_file': pose_file,
                            'rgb_files': rgb_files,
                            'num_frames': len(rgb_files)
                        })
        
        if not sequences:
            raise FileNotFoundError(f"在 {self.data_root} 中未找到有效的TartanAir序列")
        
        print(f"发现 {len(sequences)} 个序列:")
        for seq in sequences[:5]:  # 只显示前5个
            print(f"  - {seq['name']}: {seq['num_frames']} 帧")
        if len(sequences) > 5:
            print(f"  ... 还有 {len(sequences)-5} 个序列")
        
        return sequences
    
    def _build_segments(self) -> List[Dict]:
        """为所有序列构建片段索引"""
        segments = []
        
        for seq_idx, seq in enumerate(self.sequences):
            num_frames = seq['num_frames']
            
            # 计算该序列可生成的片段数量
            num_segments = max(1, (num_frames - self.n_view) // self.stride + 1)
            
            for seg_idx in range(num_segments):
                start_frame = seg_idx * self.stride
                end_frame = start_frame + self.n_view
                
                # 确保不超过序列长度
                if end_frame > num_frames:
                    continue
                
                segments.append({
                    'seq_idx': seq_idx,
                    'seq_name': seq['name'],
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'frame_indices': list(range(start_frame, end_frame))
                })
        
        if self.shuffle:
            random.shuffle(segments)
        
        return segments
    
    def _load_pose_file(self, pose_path: str) -> np.ndarray:
        """加载位姿文件"""
        poses = []
        with open(pose_path, 'r') as f:
            for line in f:
                if line.strip():
                    values = list(map(float, line.strip().split()))
                    if len(values) == 12:  # 3x4矩阵
                        pose = np.eye(4, dtype=np.float32)
                        pose[:3, :4] = np.array(values).reshape(3, 4)
                        poses.append(pose)
        return np.array(poses)
    
    def _load_frame_data(self, seq_info: Dict, frame_idx: int) -> Dict:
        """加载单帧数据"""
        # 加载RGB图像
        rgb_path = seq_info['rgb_files'][frame_idx]
        rgb = imageio.imread(rgb_path)
        if rgb.shape[2] == 4:  # RGBA转RGB
            rgb = rgb[:, :, :3]
        
        # 调整图像大小
        if rgb.shape[:2] != self.target_image_size:
            rgb = np.array(Image.fromarray(rgb).resize(
                (self.target_image_size[1], self.target_image_size[0]),  # PIL使用(width, height)
                Image.BILINEAR
            ))
        
        # 加载深度图
        depth_path = os.path.join(
            seq_info['depth_dir'], 
            os.path.basename(rgb_path).replace('image', 'depth')
        )
        if os.path.exists(depth_path):
            depth = imageio.imread(depth_path).astype(np.float32) / 1000.0  # mm转米
            depth = np.array(Image.fromarray(depth).resize(
                (self.target_image_size[1], self.target_image_size[0]),
                Image.NEAREST
            ))
        else:
            # 如果没有深度图，创建虚拟深度
            depth = np.ones(self.target_image_size, dtype=np.float32) * 5.0
        
        # 加载位姿
        poses = self._load_pose_file(seq_info['pose_file'])
        if frame_idx < len(poses):
            pose = poses[frame_idx]
        else:
            pose = np.eye(4, dtype=np.float32)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'pose': pose
        }
    
    def _compute_tsdf_for_segment(self, frames_data: List[Dict]) -> Dict:
        """为片段计算TSDF"""
        # 这里简化实现，实际应该使用完整的TSDF融合
        # 暂时返回占位符数据
        
        # 提取所有位姿
        poses = np.stack([frame['pose'] for frame in frames_data])
        
        # 计算中心位姿
        center_pose = poses[len(poses) // 2]
        
        # 创建虚拟体素网格
        grid_shape = self.crop_size
        tsdf = np.zeros(grid_shape, dtype=np.float32)
        occupancy = np.zeros(grid_shape, dtype=np.float32)
        
        # 创建简单的测试数据
        center = grid_shape // 2
        radius = min(grid_shape) // 4
        
        # 创建球形TSDF
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                    tsdf_val = (dist - radius) / radius
                    tsdf[i, j, k] = np.clip(tsdf_val, -1.0, 1.0)
                    occupancy[i, j, k] = 1.0 if abs(tsdf_val) < 0.5 else 0.0
        
        # 创建体素坐标
        voxel_coords = np.zeros((*grid_shape, 3), dtype=np.float32)
        for i in range(grid_shape[0]):
            for j in range(grid_shape[1]):
                for k in range(grid_shape[2]):
                    voxel_coords[i, j, k] = [
                        (i - center[0]) * self.voxel_size,
                        (j - center[1]) * self.voxel_size,
                        (k - center[2]) * self.voxel_size
                    ]
        
        return {
            'tsdf': tsdf,
            'occupancy': occupancy,
            'voxel_coords': voxel_coords,
            'center_pose': center_pose
        }
    

    @staticmethod
    def collate_fn(batch):
        """
        自定义collate_fn，移除Dataset添加的额外维度1

        Args:
            batch: list of samples from dataset

        Returns:
            batched data with shape (batch, n_view, 3, H, W)
        """
        # 提取所有字段
        keys = batch[0].keys()

        result = {}

        # 对于字符串字段，直接列表
        for key in ['sequence_name']:
            result[key] = [item[key] for item in batch]

        # 对于标量字段，转为tensor
        for key in ['segment_idx', 'start_frame', 'end_frame']:
            result[key] = torch.tensor([item[key] for item in batch])

        # 对于tensor字段，stack并移除额外维度
        for key in ['rgb_images', 'poses', 'intrinsics']:
            # stack所有样本
            stacked = torch.stack([item[key] for item in batch], dim=0)  # (batch, 1, n_view, 3, H, W)
            # 移除Dataset添加的维度1（第1维）
            stacked = stacked.squeeze(1)  # (batch, n_view, 3, H, W)
            result[key] = stacked

        # 对于tsdf, occupancy，移除Dataset添加的维度1，保留一个维度1
        for key in ['tsdf', 'occupancy']:
            # stack所有样本
            stacked = torch.stack([item[key] for item in batch], dim=0)  # (batch, 1, 1, D, H, W)
            # 移除Dataset添加的维度1（第1维），保留原有的维度1（第2维）
            stacked = stacked.squeeze(1)  # (batch, 1, D, H, W)
            result[key] = stacked

        # 对于其他字段，保持原样
        for key in ['voxel_coords']:
            result[key] = [item[key] for item in batch]

        return result


    def __len__(self) -> int:
        """返回总片段数"""
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个片段的数据"""
        segment = self.segments[idx]
        seq_info = self.sequences[segment['seq_idx']]
        
        # 加载片段中的所有帧
        frames_data = []
        for frame_idx in segment['frame_indices']:
            frame_data = self._load_frame_data(seq_info, frame_idx)
            frames_data.append(frame_data)
        
        # 计算TSDF
        tsdf_data = self._compute_tsdf_for_segment(frames_data)
        
        # 提取RGB图像和位姿
        rgb_images = np.stack([frame['rgb'] for frame in frames_data])  # (n_view, H, W, 3)
        poses = np.stack([frame['pose'] for frame in frames_data])      # (n_view, 4, 4)
        
        # 转换为PyTorch张量并调整维度顺序
        rgb_images = torch.from_numpy(rgb_images).float() / 255.0  # 归一化到[0,1]
        rgb_images = rgb_images.permute(0, 3, 1, 2)  # (n_view, 3, H, W)
        rgb_images = rgb_images.unsqueeze(0)  # (1, n_view, 3, H, W) ← 添加维度
        
        poses = torch.from_numpy(poses).float()
        poses = poses.unsqueeze(0)  # (1, n_view, 4, 4) ← 添加维度

        tsdf = torch.from_numpy(tsdf_data['tsdf']).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W) ← 添加维度
        occupancy = torch.from_numpy(tsdf_data['occupancy']).float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W) ← 添加维度
        voxel_coords = torch.from_numpy(tsdf_data['voxel_coords']).float()
        
        # 内参（扩展到n_view维度，保持与其他字段一致的shape）
        intrinsics = torch.from_numpy(self.K).float()  # (3, 3)
        intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        intrinsics = intrinsics.expand(1, self.n_view, 3, 3)  # (1, n_view, 3, 3) ← 扩展到n_view维度
        
        return {
            'rgb_images': rgb_images,      # (1, n_view, 3, H, W)
            'poses': poses,                # (1, n_view, 4, 4)
            'intrinsics': intrinsics,      # (1, 3, 3)
            'tsdf': tsdf,                  # (1, 1, D, H, W)
            'occupancy': occupancy,        # (1, 1, D, H, W)
            'voxel_coords': voxel_coords,  # (D, H, W, 3)
            'sequence_name': seq_info['name'],
            'segment_idx': idx,
            'start_frame': segment['start_frame'],
            'end_frame': segment['end_frame']
        }


def test_dataset():
    """测试数据集"""
    import time
    
    # 使用实际数据目录
    data_root = "/home/cwh/Study/dataset/tartanair"
    
    print("测试多序列TartanAir数据集...")
    
    try:
        dataset = MultiSequenceTartanAirDataset(
            data_root=data_root,
            n_view=5,
            stride=2,
            crop_size=(48, 48, 32),
            voxel_size=0.04,
            target_image_size=(256, 256),
            max_sequences=3,  # 限制为3个序列用于测试
            shuffle=True
        )
        
        print(f"\n数据集大小: {len(dataset)} 个片段")
        
        # 测试第一个样本
        start_time = time.time()
        sample = dataset[0]
        load_time = time.time() - start_time
        
        print(f"\n样本加载时间: {load_time:.3f}秒")
        print(f"RGB图像形状: {sample['rgb_images'].shape}")
        print(f"位姿形状: {sample['poses'].shape}")
        print(f"TSDF形状: {sample['tsdf'].shape}")
        print(f"占用网格形状: {sample['occupancy'].shape}")
        print(f"序列名称: {sample['sequence_name']}")
        print(f"片段索引: {sample['segment_idx']}")
        print(f"帧范围: {sample['start_frame']}-{sample['end_frame']}")
        
        # 测试批量数据形状
        print(f"\n测试批量数据形状:")
        batch_size = 2
        indices = list(range(min(batch_size, len(dataset))))
        batch = [dataset[i] for i in indices]
        
        # 手动堆叠创建批量
        rgb_batch = torch.stack([item['rgb_images'] for item in batch])  # (batch_size, n_view, 3, H, W)
        poses_batch = torch.stack([item['poses'] for item in batch])     # (batch_size, n_view, 4, 4)
        tsdf_batch = torch.stack([item['tsdf'] for item in batch])       # (batch_size, 1, D, H, W)
        
        print(f"批量RGB形状: {rgb_batch.shape}")
        print(f"批量位姿形状: {poses_batch.shape}")
        print(f"批量TSDF形状: {tsdf_batch.shape}")
        
        # 验证数据范围
        print(f"\n数据范围验证:")
        print(f"RGB范围: [{rgb_batch.min():.3f}, {rgb_batch.max():.3f}]")
        print(f"TSDF范围: [{tsdf_batch.min():.3f}, {tsdf_batch.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dataset()
    if success:
        print("\n✅ 数据集测试通过!")
    else:
        print("\n❌ 数据集测试失败!")