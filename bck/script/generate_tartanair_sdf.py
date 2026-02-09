#!/usr/bin/env python3
"""
TartanAir SDF Ground Truth Generation Script
从TartanAir深度图生成SDF和占用真值

功能：
1. 加载TartanAir深度图和位姿
2. 计算场景边界
3. 创建体素网格
4. 使用TSDF融合生成SDF
5. 生成占用网格
6. 保存为训练格式
"""

import os
import sys
import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation as R

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 强制使用CPU版本（GPU版本有问题）
try:
    from simple_tsdf_fusion import SimpleTSDFVolume
    USE_GPU_TSDF = False
    print("使用CPU版本的简化TSDF融合")
except ImportError as e:
    print(f"错误: 无法导入TSDF模块: {e}")
    sys.exit(1)


class TartanAirSDFGenerator:
    """TartanAir SDF真值生成器"""
    
    def __init__(self, data_root, sequence_name, voxel_size=0.08, trunc_margin=5):
        """
        初始化生成器
        
        Args:
            data_root: TartanAir数据根目录
            sequence_name: 序列名称 (如: abandonedfactory_sample_P001)
            voxel_size: 体素大小 (米)
            trunc_margin: TSDF截断边界 (体素倍数)
        """
        self.data_root = Path(data_root)
        self.sequence_name = sequence_name
        self.voxel_size = voxel_size
        self.trunc_margin = trunc_margin
        
        # 序列路径
        self.sequence_path = self.data_root / sequence_name / sequence_name.split('_')[-1]
        
        # 检查路径是否存在
        if not self.sequence_path.exists():
            raise FileNotFoundError(f"序列路径不存在: {self.sequence_path}")
        
        # 固定内参矩阵 (TartanAir 640x480)
        self.intrinsics = np.array([
            [320.0, 0.0, 320.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        print(f"初始化TartanAir SDF生成器:")
        print(f"  数据根目录: {self.data_root}")
        print(f"  序列名称: {self.sequence_name}")
        print(f"  序列路径: {self.sequence_path}")
        print(f"  体素大小: {voxel_size}米")
        print(f"  截断边界: {trunc_margin}个体素")
    
    def load_poses(self):
        """加载位姿文件"""
        pose_file = self.sequence_path / "pose_left.txt"
        if not pose_file.exists():
            raise FileNotFoundError(f"位姿文件不存在: {pose_file}")
        
        poses = np.loadtxt(pose_file)
        print(f"加载位姿: {len(poses)} 帧")
        
        # TartanAir位姿格式: [x, y, z, qx, qy, qz, qw]
        # 转换为4x4变换矩阵
        pose_matrices = []
        for i, pose in enumerate(poses):
            if len(pose) != 7:
                print(f"警告: 第{i}帧位姿格式错误: {pose}")
                continue
            
            # 提取位置和四元数
            position = pose[:3]
            quaternion = pose[3:]
            
            # 创建旋转矩阵
            rotation = R.from_quat(quaternion).as_matrix()
            
            # 创建4x4变换矩阵
            T = np.eye(4)
            T[:3, :3] = rotation
            T[:3, 3] = position
            
            pose_matrices.append(T)
        
        return np.array(pose_matrices)
    
    def load_depth_maps(self, max_frames=None):
        """加载深度图"""
        depth_dir = self.sequence_path / "depth_left"
        if not depth_dir.exists():
            raise FileNotFoundError(f"深度图目录不存在: {depth_dir}")
        
        # 获取所有.npy深度图文件
        depth_files = sorted([f for f in depth_dir.glob("*.npy")])
        if max_frames:
            depth_files = depth_files[:max_frames]
        
        print(f"加载深度图: {len(depth_files)} 帧")
        
        depth_maps = []
        for depth_file in tqdm(depth_files, desc="加载深度图"):
            depth = np.load(depth_file)
            
            # TartanAir深度图单位为毫米，转换为米
            depth = depth.astype(np.float32) / 1000.0
            
            # 检查深度图有效性
            if depth.max() > 50:  # 如果最大深度大于50米，可能有错误
                print(f"警告: {depth_file.name} 深度异常: {depth.max():.2f}米")
            
            depth_maps.append(depth)
        
        return depth_maps
    
    def compute_scene_bounds(self, poses, depth_maps, padding=1.0):
        """计算场景边界"""
        print("计算场景边界...")
        
        # 收集所有相机位置
        camera_positions = []
        for pose in poses:
            camera_positions.append(pose[:3, 3])
        
        camera_positions = np.array(camera_positions)
        
        # 计算相机位置的边界
        min_bounds = camera_positions.min(axis=0) - padding
        max_bounds = camera_positions.max(axis=0) + padding
        
        # 考虑深度范围
        max_depth = max([d.max() for d in depth_maps])
        print(f"最大深度: {max_depth:.2f}米")
        
        # 扩展边界以包含可见区域
        # 简单方法：根据最大深度扩展
        max_bounds = np.maximum(max_bounds, min_bounds + max_depth * 0.5)
        
        bounds = np.array([min_bounds, max_bounds]).T
        print(f"场景边界: {bounds}")
        
        return bounds
    
    def generate_sdf(self, poses, depth_maps, bounds):
        """生成SDF体素网格"""
        print("生成SDF体素网格...")
        
        if USE_GPU_TSDF:
            # 使用GPU加速的TSDF
            tsdf_volume = TSDFVolume(
                vol_bnds=bounds,
                voxel_size=self.voxel_size,
                margin=self.trunc_margin
            )
        else:
            # 使用CPU版本的简化TSDF
            tsdf_volume = SimpleTSDFVolume(
                vol_bnds=bounds,
                voxel_size=self.voxel_size,
                margin=self.trunc_margin
            )
        
        # 融合所有深度帧
        print("融合深度帧到TSDF...")
        for i, (pose, depth) in enumerate(tqdm(zip(poses, depth_maps), 
                                              total=len(poses), 
                                              desc="TSDF融合")):
            
            # 转换为相机到世界变换 (TSDF期望世界到相机)
            # TartanAir位姿是相机到世界，需要求逆
            cam_to_world = pose
            world_to_cam = np.linalg.inv(cam_to_world)
            
            if USE_GPU_TSDF:
                # GPU版本接口
                tsdf_volume.integrate(
                    color_image=None,  # 不需要颜色
                    depth_image=depth,
                    cam_intr=self.intrinsics[:3, :3],
                    cam_pose=world_to_cam,
                    obs_weight=1.0
                )
            else:
                # CPU版本接口
                tsdf_volume.integrate(
                    depth_im=depth,
                    cam_intr=self.intrinsics[:3, :3],
                    cam_pose=world_to_cam,
                    obs_weight=1.0
                )
        
        # 获取TSDF网格
        tsdf_grid, weight_grid = tsdf_volume.get_volume()
        
        print(f"TSDF网格形状: {tsdf_grid.shape}")
        print(f"TSDF值范围: [{tsdf_grid.min():.3f}, {tsdf_grid.max():.3f}]")
        
        return tsdf_grid, weight_grid
    
    def generate_occupancy(self, tsdf_grid, threshold=0.03):
        """从SDF生成占用网格"""
        print("生成占用网格...")
        
        # 使用SDF阈值生成占用
        # SDF < threshold 表示在表面内部
        occupancy_grid = (tsdf_grid < threshold).astype(np.float32)
        
        occupied_voxels = occupancy_grid.sum()
        total_voxels = occupancy_grid.size
        
        print(f"占用体素: {occupied_voxels} / {total_voxels} ({occupied_voxels/total_voxels*100:.2f}%)")
        
        return occupancy_grid
    
    def save_results(self, tsdf_grid, occupancy_grid, bounds, output_dir):
        """保存结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为.npz文件
        output_path = output_dir / f"{self.sequence_name}_sdf_occ.npz"
        
        np.savez_compressed(
            output_path,
            sdf=tsdf_grid,
            occupancy=occupancy_grid,
            voxel_size=self.voxel_size,
            bounds=bounds,
            intrinsics=self.intrinsics,
            sequence_name=self.sequence_name
        )
        
        print(f"结果保存到: {output_path}")
        
        # 保存元数据
        metadata = {
            "sequence_name": self.sequence_name,
            "voxel_size": float(self.voxel_size),
            "trunc_margin": self.trunc_margin,
            "bounds": bounds.tolist(),
            "grid_shape": tsdf_grid.shape,
            "intrinsics": self.intrinsics.tolist(),
            "num_frames": len(self.load_poses())
        }
        
        metadata_path = output_dir / f"{self.sequence_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"元数据保存到: {metadata_path}")
        
        return output_path
    
    def visualize_results(self, tsdf_grid, occupancy_grid, bounds):
        """可视化结果（可选）"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            print("生成可视化...")
            
            # 提取占用体素的坐标
            occ_coords = np.where(occupancy_grid > 0.5)
            
            if len(occ_coords[0]) > 0:
                # 随机采样一些点用于可视化
                sample_size = min(10000, len(occ_coords[0]))
                indices = np.random.choice(len(occ_coords[0]), sample_size, replace=False)
                
                x = occ_coords[0][indices]
                y = occ_coords[1][indices]
                z = occ_coords[2][indices]
                
                # 创建3D图
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # 绘制占用体素
                scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.6)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f'TartanAir SDF - {self.sequence_name}')
                
                plt.colorbar(scatter, ax=ax, label='Z coordinate')
                plt.tight_layout()
                
                # 保存图像
                vis_dir = Path("visualizations")
                vis_dir.mkdir(exist_ok=True)
                vis_path = vis_dir / f"{self.sequence_name}_sdf_visualization.png"
                plt.savefig(vis_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"可视化保存到: {vis_path}")
                return vis_path
            else:
                print("警告: 没有找到占用体素，跳过可视化")
                return None
                
        except ImportError:
            print("警告: matplotlib未安装，跳过可视化")
            return None
    
    def run(self, max_frames=None, output_dir="./tartanair_sdf_output", visualize=True):
        """运行完整的SDF生成流程"""
        print(f"\n{'='*60}")
        print(f"开始生成TartanAir SDF真值")
        print(f"{'='*60}")
        
        # 1. 加载数据
        poses = self.load_poses()
        depth_maps = self.load_depth_maps(max_frames)
        
        # 确保帧数匹配
        min_frames = min(len(poses), len(depth_maps))
        poses = poses[:min_frames]
        depth_maps = depth_maps[:min_frames]
        
        print(f"使用帧数: {min_frames}")
        
        # 2. 计算场景边界
        bounds = self.compute_scene_bounds(poses, depth_maps)
        
        # 3. 生成SDF
        tsdf_grid, weight_grid = self.generate_sdf(poses, depth_maps, bounds)
        
        # 4. 生成占用
        occupancy_grid = self.generate_occupancy(tsdf_grid)
        
        # 5. 保存结果
        output_path = self.save_results(tsdf_grid, occupancy_grid, bounds, output_dir)
        
        # 6. 可视化（可选）
        vis_path = None
        if visualize:
            vis_path = self.visualize_results(tsdf_grid, occupancy_grid, bounds)
        
        print(f"\n{'='*60}")
        print(f"SDF生成完成!")
        print(f"输出文件: {output_path}")
        if vis_path:
            print(f"可视化文件: {vis_path}")
        print(f"{'='*60}")
        
        return {
            "sdf_grid": tsdf_grid,
            "occupancy_grid": occupancy_grid,
            "weight_grid": weight_grid,
            "bounds": bounds,
            "output_path": output_path,
            "visualization_path": vis_path
        }


def main():
    parser = argparse.ArgumentParser(description="从TartanAir深度图生成SDF真值")
    parser.add_argument("--data_root", type=str, default="/home/cwh/Study/dataset/tartanair",
                       help="TartanAir数据根目录")
    parser.add_argument("--sequence", type=str, default="abandonedfactory_sample_P001",
                       help="序列名称")
    parser.add_argument("--voxel_size", type=float, default=0.08,
                       help="体素大小（米）")
    parser.add_argument("--max_frames", type=int, default=50,
                       help="最大处理帧数（用于测试）")
    parser.add_argument("--output_dir", type=str, default="./tartanair_sdf_output",
                       help="输出目录")
    parser.add_argument("--no_visualize", action="store_true",
                       help="不生成可视化")
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = TartanAirSDFGenerator(
        data_root=args.data_root,
        sequence_name=args.sequence,
        voxel_size=args.voxel_size
    )
    
    # 运行生成流程
    try:
        results = generator.run(
            max_frames=args.max_frames,
            output_dir=args.output_dir,
            visualize=not args.no_visualize
        )
        
        print("\n🎉 SDF生成成功!")
        print(f"可以在训练脚本中使用以下路径加载SDF真值:")
        print(f"  npz文件: {results['output_path']}")
        
    except Exception as e:
        print(f"\n❌ SDF生成失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()