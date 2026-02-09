#!/usr/bin/env python3
"""
优化TartanAir TSDF生成参数
测试不同参数组合，找到最佳配置
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
import json
import time
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入GPU版本，如果失败则使用CPU版本
try:
    from former3d.tsdf_fusion import TSDFVolume
    print("使用GPU版本的TSDF融合")
except ImportError as e:
    print(f"GPU版本导入失败: {e}")
    print("使用CPU版本的简化TSDF融合")
    # 使用简单的CPU TSDF实现
    import sys
    sys.path.append('/home/cwh/ubuntu18/home/ubuntu/coding/former3d')
    from simple_tsdf_fusion import TSDFVolume


class OptimizedTartanAirSDFGenerator:
    """优化的TartanAir SDF生成器"""
    
    def __init__(self, data_root, sequence_name):
        self.data_root = Path(data_root)
        self.sequence_name = sequence_name
        self.sequence_path = self.data_root / sequence_name / sequence_name.split('_')[-1]
        
        # 固定内参矩阵
        self.intrinsics = np.array([
            [320.0, 0.0, 320.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
    
    def load_data(self, max_frames=None):
        """加载位姿和深度图"""
        # 加载位姿
        pose_file = self.sequence_path / "pose_left.txt"
        poses = np.loadtxt(pose_file)
        
        # 转换为4x4变换矩阵
        from scipy.spatial.transform import Rotation as R
        pose_matrices = []
        for pose in poses:
            if len(pose) != 7:
                continue
            position = pose[:3]
            quaternion = pose[3:]
            rotation = R.from_quat(quaternion).as_matrix()
            T = np.eye(4)
            T[:3, :3] = rotation
            T[:3, 3] = position
            pose_matrices.append(T)
        
        # 加载深度图
        depth_dir = self.sequence_path / "depth_left"
        depth_files = sorted([f for f in depth_dir.glob("*.npy")])
        if max_frames:
            depth_files = depth_files[:max_frames]
        
        depth_maps = []
        for depth_file in tqdm(depth_files, desc="加载深度图"):
            depth = np.load(depth_file).astype(np.float32) / 1000.0  # 毫米转米
            depth_maps.append(depth)
        
        # 确保帧数匹配
        min_frames = min(len(pose_matrices), len(depth_maps))
        pose_matrices = pose_matrices[:min_frames]
        depth_maps = depth_maps[:min_frames]
        
        return np.array(pose_matrices), depth_maps
    
    def compute_optimal_bounds(self, poses, depth_maps, padding_factor=1.2):
        """计算优化的场景边界"""
        
        # 收集所有相机位置
        camera_positions = np.array([pose[:3, 3] for pose in poses])
        
        # 计算相机轨迹的边界
        min_bounds = camera_positions.min(axis=0)
        max_bounds = camera_positions.max(axis=0)
        
        # 计算相机轨迹的中心和范围
        center = (min_bounds + max_bounds) / 2
        size = max_bounds - min_bounds
        
        # 考虑深度范围
        max_depth = max([d.max() for d in depth_maps])
        
        # 优化边界：基于相机轨迹和最大深度
        # 扩展边界以包含可见区域
        expanded_size = np.maximum(size, max_depth * 0.6)
        
        # 应用填充因子
        expanded_size = expanded_size * padding_factor
        
        # 计算新边界
        new_min_bounds = center - expanded_size / 2
        new_max_bounds = center + expanded_size / 2
        
        bounds = np.array([new_min_bounds, new_max_bounds]).T
        
        print(f"相机轨迹边界: [{min_bounds}, {max_bounds}]")
        print(f"场景中心: {center}")
        print(f"场景大小: {expanded_size}")
        print(f"优化后边界: {bounds}")
        
        return bounds
    
    def generate_sdf_with_params(self, poses, depth_maps, voxel_size=0.04, trunc_margin=3, 
                                bounds=None, use_optimized_bounds=True):
        """使用指定参数生成SDF"""
        
        print(f"\n使用参数生成SDF:")
        print(f"  体素大小: {voxel_size}米")
        print(f"  截断边界: {trunc_margin}个体素 ({trunc_margin*voxel_size:.3f}米)")
        
        # 计算场景边界
        if bounds is None or use_optimized_bounds:
            bounds = self.compute_optimal_bounds(poses, depth_maps)
        
        # 创建TSDF体积
        try:
            tsdf_volume = TSDFVolume(
                vol_bnds=bounds,
                voxel_size=voxel_size,
                margin=trunc_margin
            )
        except Exception as e:
            print(f"GPU TSDF初始化失败: {e}")
            print("尝试使用CPU版本...")
            # 使用简单的CPU TSDF实现
            import sys
            sys.path.append('/home/cwh/ubuntu18/home/ubuntu/coding/former3d')
            from simple_tsdf_fusion import TSDFVolume as SimpleTSDFVolume
            tsdf_volume = SimpleTSDFVolume(
                vol_bnds=bounds,
                voxel_size=voxel_size,
                margin=trunc_margin
            )
        
        # 融合深度帧
        print("融合深度帧到TSDF...")
        start_time = time.time()
        
        for i, (pose, depth) in enumerate(tqdm(zip(poses, depth_maps), 
                                              total=len(poses), 
                                              desc="TSDF融合")):
            
            # 转换为世界到相机变换
            cam_to_world = pose
            world_to_cam = np.linalg.inv(cam_to_world)
            
            # 融合到TSDF
            tsdf_volume.integrate(
                color_image=None,
                depth_image=depth,
                cam_intr=self.intrinsics[:3, :3],
                cam_pose=world_to_cam,
                obs_weight=1.0
            )
        
        fusion_time = time.time() - start_time
        
        # 获取TSDF网格
        tsdf_grid, weight_grid = tsdf_volume.get_volume()
        
        # 生成占用网格
        occupancy_grid = (tsdf_grid < 0.03).astype(np.float32)
        
        # 计算统计信息
        stats = {
            "voxel_size": voxel_size,
            "trunc_margin": trunc_margin,
            "grid_shape": tsdf_grid.shape,
            "total_voxels": tsdf_grid.size,
            "occupied_voxels": occupancy_grid.sum(),
            "occupancy_rate": occupancy_grid.sum() / occupancy_grid.size * 100,
            "sdf_min": tsdf_grid.min(),
            "sdf_max": tsdf_grid.max(),
            "sdf_mean": tsdf_grid.mean(),
            "sdf_std": tsdf_grid.std(),
            "fusion_time": fusion_time,
            "bounds": bounds.tolist(),
            "num_frames": len(poses)
        }
        
        return tsdf_grid, occupancy_grid, stats
    
    def run_parameter_sweep(self, max_frames=50, output_dir="./tsdf_optimization"):
        """运行参数扫描，找到最佳配置"""
        
        print("=" * 60)
        print("开始TSDF生成参数优化")
        print("=" * 60)
        
        # 加载数据
        poses, depth_maps = self.load_data(max_frames)
        print(f"加载数据: {len(poses)}帧")
        
        # 参数组合
        voxel_sizes = [0.08, 0.06, 0.04, 0.02]
        trunc_margins = [3, 5, 7]
        
        results = []
        
        # 运行参数扫描
        for voxel_size in voxel_sizes:
            for trunc_margin in trunc_margins:
                print(f"\n{'='*40}")
                print(f"测试参数: 体素大小={voxel_size}米, 截断边界={trunc_margin}")
                print(f"{'='*40}")
                
                try:
                    tsdf_grid, occupancy_grid, stats = self.generate_sdf_with_params(
                        poses, depth_maps, voxel_size, trunc_margin
                    )
                    
                    results.append({
                        "voxel_size": voxel_size,
                        "trunc_margin": trunc_margin,
                        "stats": stats,
                        "success": True
                    })
                    
                    print(f"结果:")
                    print(f"  网格形状: {stats['grid_shape']}")
                    print(f"  占用率: {stats['occupancy_rate']:.2f}%")
                    print(f"  SDF范围: [{stats['sdf_min']:.3f}, {stats['sdf_max']:.3f}]")
                    print(f"  融合时间: {stats['fusion_time']:.1f}秒")
                    
                except Exception as e:
                    print(f"错误: {e}")
                    results.append({
                        "voxel_size": voxel_size,
                        "trunc_margin": trunc_margin,
                        "error": str(e),
                        "success": False
                    })
        
        # 分析结果
        print(f"\n{'='*60}")
        print("参数优化结果分析")
        print(f"{'='*60}")
        
        successful_results = [r for r in results if r["success"]]
        
        if not successful_results:
            print("所有参数组合都失败了！")
            return None
        
        # 按占用率排序（越高越好）
        successful_results.sort(key=lambda x: x["stats"]["occupancy_rate"], reverse=True)
        
        print("\n最佳参数组合（按占用率排序）:")
        for i, result in enumerate(successful_results[:5]):
            stats = result["stats"]
            print(f"{i+1}. 体素大小={result['voxel_size']}米, 截断边界={result['trunc_margin']}")
            print(f"   占用率: {stats['occupancy_rate']:.2f}%")
            print(f"   网格形状: {stats['grid_shape']}")
            print(f"   融合时间: {stats['fusion_time']:.1f}秒")
            print()
        
        # 保存结果
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        results_path = output_dir / "parameter_sweep_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                "sequence": self.sequence_name,
                "num_frames": len(poses),
                "results": results
            }, f, indent=2, default=str)
        
        print(f"详细结果保存到: {results_path}")
        
        # 选择最佳参数
        best_result = successful_results[0]
        best_params = {
            "voxel_size": best_result["voxel_size"],
            "trunc_margin": best_result["trunc_margin"],
            "stats": best_result["stats"]
        }
        
        # 使用最佳参数生成最终SDF
        print(f"\n{'='*60}")
        print("使用最佳参数生成最终SDF")
        print(f"{'='*60}")
        
        tsdf_grid, occupancy_grid, stats = self.generate_sdf_with_params(
            poses, depth_maps,
            voxel_size=best_params["voxel_size"],
            trunc_margin=best_params["trunc_margin"]
        )
        
        # 保存最终SDF
        final_output_dir = output_dir / "best_result"
        final_output_dir.mkdir(exist_ok=True)
        
        output_path = final_output_dir / f"{self.sequence_name}_sdf_occ.npz"
        np.savez_compressed(
            output_path,
            sdf=tsdf_grid,
            occupancy=occupancy_grid,
            voxel_size=best_params["voxel_size"],
            bounds=stats["bounds"],
            intrinsics=self.intrinsics,
            sequence_name=self.sequence_name
        )
        
        # 保存元数据
        metadata = {
            "sequence_name": self.sequence_name,
            "voxel_size": best_params["voxel_size"],
            "trunc_margin": best_params["trunc_margin"],
            "bounds": stats["bounds"],
            "grid_shape": stats["grid_shape"],
            "intrinsics": self.intrinsics.tolist(),
            "num_frames": stats["num_frames"],
            "occupancy_rate": stats["occupancy_rate"],
            "fusion_time": stats["fusion_time"],
            "optimization_results": results_path.name
        }
        
        metadata_path = final_output_dir / f"{self.sequence_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n最终SDF保存到: {output_path}")
        print(f"元数据保存到: {metadata_path}")
        
        return {
            "best_params": best_params,
            "output_path": output_path,
            "metadata_path": metadata_path,
            "all_results": results
        }


def main():
    parser = argparse.ArgumentParser(description="优化TartanAir TSDF生成参数")
    parser.add_argument("--data_root", type=str, default="/home/cwh/Study/dataset/tartanair",
                       help="TartanAir数据根目录")
    parser.add_argument("--sequence", type=str, default="abandonedfactory_sample_P001",
                       help="序列名称")
    parser.add_argument("--max_frames", type=int, default=50,
                       help="最大处理帧数")
    parser.add_argument("--output_dir", type=str, default="./tsdf_optimization",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 创建优化器
    optimizer = OptimizedTartanAirSDFGenerator(
        data_root=args.data_root,
        sequence_name=args.sequence
    )
    
    # 运行参数优化
    try:
        results = optimizer.run_parameter_sweep(
            max_frames=args.max_frames,
            output_dir=args.output_dir
        )
        
        if results:
            print(f"\n{'='*60}")
            print("🎉 TSDF参数优化完成!")
            print(f"{'='*60}")
            
            best_params = results["best_params"]
            print(f"最佳参数:")
            print(f"  体素大小: {best_params['voxel_size']}米")
            print(f"  截断边界: {best_params['trunc_margin']}个体素")
            print(f"  占用率: {best_params['stats']['occupancy_rate']:.2f}%")
            print(f"  网格形状: {best_params['stats']['grid_shape']}")
            print(f"\n输出文件:")
            print(f"  SDF数据: {results['output_path']}")
            print(f"  元数据: {results['metadata_path']}")
            
        else:
            print("\n❌ TSDF参数优化失败!")
            
    except Exception as e:
        print(f"\n❌ TSDF参数优化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()