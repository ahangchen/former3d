#!/usr/bin/env python3
"""
在线SDF生成的TartanAir数据集
基于real_tartanair_dataset.py，但简化并适配样本数据格式
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

class OnlineTartanAirDataset(Dataset):
    """
    在线生成SDF的TartanAir数据集
    从深度图实时计算TSDF，无需预先生成
    """
    
    def __init__(
        self,
        data_root: str,
        sequence_name: str = "abandonedfactory_sample_P001",
        n_frames: int = 5,
        crop_size: tuple = (48, 48, 32),
        voxel_size: float = 0.04,
        target_image_size: tuple = (256, 256),
        max_depth: float = 10.0,
        truncation_margin: float = 0.2,
        augment: bool = False
    ):
        """
        初始化数据集
        
        Args:
            data_root: TartanAir数据根目录
            sequence_name: 序列名称
            n_frames: 每场景使用的帧数
            crop_size: 裁剪尺寸（体素单位）
            voxel_size: 体素大小（米）
            target_image_size: 目标图像大小
            max_depth: 最大深度值（米）
            truncation_margin: TSDF截断边界
            augment: 是否使用数据增强
        """
        super().__init__()
        
        self.data_root = data_root
        self.sequence_name = sequence_name
        self.n_frames = n_frames
        self.crop_size = np.array(crop_size)
        self.voxel_size = voxel_size
        self.target_image_size = target_image_size
        self.max_depth = max_depth
        self.truncation_margin = truncation_margin
        self.augment = augment
        
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
        
        # 构建序列路径
        self.sequence_dir = os.path.join(data_root, sequence_name, "P001")
        if not os.path.exists(self.sequence_dir):
            raise FileNotFoundError(f"序列目录不存在: {self.sequence_dir}")
        
        # 加载数据路径
        self._load_data_paths()
        
        print(f"在线TartanAir数据集初始化完成:")
        print(f"  序列: {sequence_name}")
        print(f"  总帧数: {len(self.rgb_files)}")
        print(f"  使用帧数: {n_frames}")
        print(f"  裁剪尺寸: {crop_size} 体素")
        print(f"  体素大小: {voxel_size}米")
        print(f"  图像大小: {target_image_size}")
        
    def _load_data_paths(self):
        """加载数据文件路径"""
        # RGB图像
        rgb_dir = os.path.join(self.sequence_dir, "image_left")
        if not os.path.exists(rgb_dir):
            raise FileNotFoundError(f"RGB目录不存在: {rgb_dir}")
        
        self.rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        
        # 深度图
        depth_dir = os.path.join(self.sequence_dir, "depth_left")
        if not os.path.exists(depth_dir):
            raise FileNotFoundError(f"深度目录不存在: {depth_dir}")
        
        self.depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))
        
        # 位姿文件
        pose_file = os.path.join(self.sequence_dir, "pose_left.txt")
        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"位姿文件不存在: {pose_file}")
        
        # 加载位姿
        self.poses = self._load_poses(pose_file)
        
        # 确保数量匹配
        min_len = min(len(self.rgb_files), len(self.depth_files), len(self.poses))
        self.rgb_files = self.rgb_files[:min_len]
        self.depth_files = self.depth_files[:min_len]
        self.poses = self.poses[:min_len]
        
        # 如果帧数太多，均匀采样
        if len(self.rgb_files) > self.n_frames:
            indices = np.linspace(0, len(self.rgb_files) - 1, self.n_frames, dtype=int)
            self.rgb_files = [self.rgb_files[i] for i in indices]
            self.depth_files = [self.depth_files[i] for i in indices]
            self.poses = self.poses[indices]
        
    def _load_poses(self, pose_file):
        """加载位姿文件"""
        poses_data = np.loadtxt(pose_file)
        
        poses = []
        for i in range(len(poses_data)):
            pose_row = poses_data[i]
            
            # TartanAir格式: [x, y, z, qx, qy, qz, qw]
            if len(pose_row) == 7:
                x, y, z = pose_row[0:3]
                qx, qy, qz, qw = pose_row[3:7]
                
                # 转换为旋转矩阵
                rotation = Rotation.from_quat([qx, qy, qz, qw])
                rot_mat = rotation.as_matrix()
                
                # 创建4x4变换矩阵
                pose_mat = np.eye(4, dtype=np.float32)
                pose_mat[:3, :3] = rot_mat
                pose_mat[:3, 3] = [x, y, z]
                
            elif len(pose_row) == 16:
                # 已经是4x4矩阵格式
                pose_mat = pose_row.reshape(4, 4)
            else:
                raise ValueError(f"未知的位姿格式: {pose_row.shape}")
            
            poses.append(pose_mat)
        
        return np.array(poses)
    
    def _compute_tsdf_online(self, depth_images, poses, anchor_point):
        """
        在线计算TSDF
        
        Args:
            depth_images: 深度图像列表 (H, W)
            poses: 位姿列表 (4, 4)
            anchor_point: 锚点世界坐标 (3,)
            
        Returns:
            tsdf_grid: TSDF网格 (D, H, W)
            voxel_coords: 体素坐标网格
        """
        # 创建局部裁剪
        min_bound = anchor_point - self.crop_size[:3] * self.voxel_size / 2
        max_bound = anchor_point + self.crop_size[:3] * self.voxel_size / 2
        
        # 创建体素网格
        x = np.arange(min_bound[0], max_bound[0], self.voxel_size, dtype=np.float32)
        y = np.arange(min_bound[1], max_bound[1], self.voxel_size, dtype=np.float32)
        z = np.arange(min_bound[2], max_bound[2], self.voxel_size, dtype=np.float32)
        
        # 确保尺寸正确
        x = x[:self.crop_size[0]]
        y = y[:self.crop_size[1]]
        z = z[:self.crop_size[2]]
        
        yy, xx, zz = np.meshgrid(y, x, z, indexing='ij')
        voxel_coords = np.stack([xx, yy, zz], axis=-1)  # (D, H, W, 3)
        voxel_points = voxel_coords.reshape(-1, 3)  # (N, 3)
        
        # 初始化TSDF网格
        tsdf_grid = np.ones(self.crop_size, dtype=np.float32)
        
        # 多帧融合
        for frame_idx in range(len(depth_images)):
            depth_img = depth_images[frame_idx]
            pose = poses[frame_idx]
            
            # 将体素点转换到相机坐标系
            voxel_points_hom = np.concatenate([
                voxel_points, 
                np.ones((len(voxel_points), 1))
            ], axis=-1)
            
            cam_points = (np.linalg.inv(pose) @ voxel_points_hom.T).T[:, :3]
            
            # 投影到图像平面
            z = cam_points[:, 2]
            x_proj = cam_points[:, 0] / z * self.K[0, 0] + self.K[0, 2]
            y_proj = cam_points[:, 1] / z * self.K[1, 1] + self.K[1, 2]
            
            # 检查投影是否在图像范围内
            valid = (
                (x_proj >= 0) & (x_proj < self.target_image_size[0]) &
                (y_proj >= 0) & (y_proj < self.target_image_size[1]) &
                (z > 0)
            )
            
            if not np.any(valid):
                continue
            
            # 获取深度值（双线性插值）
            x_valid = x_proj[valid]
            y_valid = y_proj[valid]
            
            # 简单最近邻采样（为了速度）
            x_idx = np.clip(x_valid.astype(int), 0, depth_img.shape[1] - 1)
            y_idx = np.clip(y_valid.astype(int), 0, depth_img.shape[0] - 1)
            depth_values = depth_img[y_idx, x_idx]
            
            # 计算有符号距离
            signed_distance = z[valid] - depth_values
            
            # 截断和归一化
            signed_distance = np.clip(
                signed_distance, 
                -self.truncation_margin, 
                self.truncation_margin
            ) / self.truncation_margin
            
            # 更新TSDF（取最小绝对距离）
            valid_indices = np.where(valid.reshape(self.crop_size))
            for j in range(len(valid_indices[0])):
                idx = (valid_indices[0][j], valid_indices[1][j], valid_indices[2][j])
                if abs(signed_distance[j]) < abs(tsdf_grid[idx]):
                    tsdf_grid[idx] = signed_distance[j]
        
        return tsdf_grid, voxel_coords
    
    def __len__(self):
        """返回数据集大小（每个序列作为一个样本）"""
        return 1  # 暂时只处理一个序列
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Returns:
            dict: 包含图像、位姿、TSDF等数据
        """
        # 加载RGB图像
        rgb_images = []
        for rgb_file in self.rgb_files:
            img = Image.open(rgb_file).convert('RGB')
            img = img.resize(self.target_image_size, Image.Resampling.LANCZOS)
            rgb_images.append(np.array(img))
        
        rgb_images = np.array(rgb_images)  # (F, H, W, 3)
        
        # 加载深度图像
        depth_images = []
        for depth_file in self.depth_files:
            depth = np.load(depth_file).astype(np.float32)
            # 调整大小
            depth_img = Image.fromarray(depth)
            depth_img = depth_img.resize(self.target_image_size, Image.Resampling.NEAREST)
            depth = np.array(depth_img)
            depth = np.clip(depth, 0, self.max_depth)
            depth_images.append(depth)
        
        depth_images = np.array(depth_images)  # (F, H, W)
        
        # 选择锚点（使用第一帧的点云中心）
        # 从第一帧深度图创建点云
        depth0 = depth_images[0]
        pose0 = self.poses[0]
        
        # 创建像素网格
        h, w = depth0.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        u = u.reshape(-1)
        v = v.reshape(-1)
        z = depth0.reshape(-1)
        
        # 过滤无效深度
        valid = z > 0
        u = u[valid]
        v = v[valid]
        z = z[valid]
        
        # 转换到相机坐标系
        x = (u - self.K[0, 2]) * z / self.K[0, 0]
        y = (v - self.K[1, 2]) * z / self.K[1, 1]
        
        # 转换到世界坐标系
        points_cam = np.stack([x, y, z], axis=-1)
        points_cam_hom = np.concatenate([
            points_cam, 
            np.ones((len(points_cam), 1))
        ], axis=-1)
        
        points_world = (pose0 @ points_cam_hom.T).T[:, :3]
        
        # 选择锚点（点云中心）
        anchor_point = np.median(points_world, axis=0)
        
        # 在线计算TSDF
        tsdf_grid, voxel_coords = self._compute_tsdf_online(
            depth_images, self.poses, anchor_point
        )
        
        # 创建占用网格
        occupancy_grid = np.abs(tsdf_grid) < 0.999
        
        # 转换为张量
        rgb_tensor = torch.from_numpy(rgb_images).float() / 255.0
        rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)  # (F, 3, H, W)
        
        pose_tensor = torch.from_numpy(self.poses).float()
        intrinsics_tensor = torch.from_numpy(self.K).float()
        
        # TSDF和占用网格
        tsdf_tensor = torch.from_numpy(tsdf_grid).float()
        occ_tensor = torch.from_numpy(occupancy_grid).float()
        
        # 体素坐标
        voxel_coords_tensor = torch.from_numpy(voxel_coords).float()
        
        return {
            'rgb_images': rgb_tensor,          # (F, 3, H, W)
            'poses': pose_tensor,              # (F, 4, 4)
            'intrinsics': intrinsics_tensor,   # (3, 3)
            'tsdf': tsdf_tensor,               # (D, H, W)
            'occupancy': occ_tensor,           # (D, H, W)
            'voxel_coords': voxel_coords_tensor, # (D, H, W, 3)
            'anchor_point': torch.from_numpy(anchor_point).float(),
            'sequence_name': self.sequence_name,
            'crop_size': torch.tensor(self.crop_size),
            'voxel_size': torch.tensor(self.voxel_size)
        }


def test_dataset():
    """测试数据集"""
    data_root = "/home/cwh/Study/dataset/tartanair"
    sequence_name = "abandonedfactory_sample_P001"
    
    print("测试在线TartanAir数据集...")
    
    try:
        dataset = OnlineTartanAirDataset(
            data_root=data_root,
            sequence_name=sequence_name,
            n_frames=5,
            crop_size=(32, 32, 24),  # 小尺寸用于测试
            voxel_size=0.08,
            target_image_size=(128, 128),
            max_depth=10.0,
            truncation_margin=0.2,
            augment=False
        )
        
        print(f"数据集创建成功!")
        print(f"序列: {dataset.sequence_name}")
        print(f"RGB文件数: {len(dataset.rgb_files)}")
        print(f"深度文件数: {len(dataset.depth_files)}")
        print(f"位姿数: {len(dataset.poses)}")
        
        # 获取一个样本
        sample = dataset[0]
        print(f"\n样本信息:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} {value.dtype}")
            else:
                print(f"  {key}: {value}")
        
        # 检查TSDF值
        tsdf = sample['tsdf']
        print(f"\nTSDF统计:")
        print(f"  形状: {tsdf.shape}")
        print(f"  最小值: {tsdf.min():.4f}")
        print(f"  最大值: {tsdf.max():.4f}")
        print(f"  均值: {tsdf.mean():.4f}")
        print(f"  标准差: {tsdf.std():.4f}")
        
        # 检查占用率
        occ = sample['occupancy']
        occ_rate = occ.sum() / occ.numel()
        print(f"\n占用统计:")
        print(f"  占用体素: {occ.sum().item():.0f}")
        print(f"  总体素数: {occ.numel()}")
        print(f"  占用率: {occ_rate:.2%}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_dataset()