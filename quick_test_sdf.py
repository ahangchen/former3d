#!/usr/bin/env python3
"""
快速测试SDF生成（使用少量帧）
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def quick_test():
    """快速测试"""
    print("快速测试SDF生成...")
    
    # 测试数据路径
    data_root = "/home/cwh/Study/dataset/tartanair"
    sequence_name = "abandonedfactory_sample_P001"
    sequence_path = Path(data_root) / sequence_name / "P001"
    
    print(f"测试序列: {sequence_name}")
    print(f"序列路径: {sequence_path}")
    
    # 1. 检查位姿文件
    pose_file = sequence_path / "pose_left.txt"
    if not pose_file.exists():
        print(f"❌ 位姿文件不存在")
        return False
    
    poses = np.loadtxt(pose_file)
    print(f"✅ 位姿文件: {len(poses)} 帧")
    
    # 2. 检查深度图
    depth_dir = sequence_path / "depth_left"
    if not depth_dir.exists():
        print(f"❌ 深度图目录不存在")
        return False
    
    depth_files = sorted([f for f in depth_dir.glob("*.npy")])
    print(f"✅ 深度图文件: {len(depth_files)} 个")
    
    # 3. 加载少量数据用于测试
    test_frames = 5
    test_poses = poses[:test_frames]
    
    test_depths = []
    for i in range(test_frames):
        depth_path = depth_files[i]
        depth = np.load(depth_path).astype(np.float32) / 1000.0  # 毫米转米
        test_depths.append(depth)
    
    print(f"✅ 加载测试数据: {test_frames} 帧")
    
    # 4. 计算场景边界
    camera_positions = test_poses[:, :3]
    min_bounds = camera_positions.min(axis=0) - 1.0
    max_bounds = camera_positions.max(axis=0) + 1.0
    bounds = np.array([min_bounds, max_bounds]).T
    
    print(f"✅ 场景边界: {bounds}")
    
    # 5. 测试TSDF模块
    try:
        from simple_tsdf_fusion import SimpleTSDFVolume
        
        voxel_size = 0.2  # 使用较大的体素加速测试
        tsdf = SimpleTSDFVolume(bounds, voxel_size=voxel_size, margin=3)
        
        print(f"✅ TSDF体积创建成功")
        print(f"   体素网格: {tsdf._vol_dim}")
        
        # 6. 测试融合一帧
        from scipy.spatial.transform import Rotation as R
        
        # 使用第一帧位姿
        pose = test_poses[0]
        position = pose[:3]
        quaternion = pose[3:]
        rotation = R.from_quat(quaternion).as_matrix()
        
        # 创建变换矩阵
        cam_to_world = np.eye(4)
        cam_to_world[:3, :3] = rotation
        cam_to_world[:3, 3] = position
        
        # TSDF需要世界到相机的变换
        world_to_cam = np.linalg.inv(cam_to_world)
        
        # 固定内参
        intrinsics = np.array([
            [320.0, 0.0, 320.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        
        # 融合
        depth_im = test_depths[0]
        tsdf.integrate(depth_im, intrinsics, world_to_cam)
        
        print(f"✅ TSDF融合成功")
        
        # 7. 获取结果
        tsdf_grid, weight_grid = tsdf.get_volume()
        print(f"✅ 获取TSDF网格")
        print(f"   TSDF形状: {tsdf_grid.shape}")
        print(f"   TSDF范围: [{tsdf_grid.min():.3f}, {tsdf_grid.max():.3f}]")
        print(f"   有效体素: {weight_grid.sum()}")
        
        # 8. 生成占用网格
        occupancy_grid = (tsdf_grid < 0.03).astype(np.float32)
        occupied = occupancy_grid.sum()
        total = occupancy_grid.size
        print(f"✅ 生成占用网格")
        print(f"   占用体素: {occupied}/{total} ({occupied/total*100:.2f}%)")
        
        # 9. 保存测试结果
        output_dir = Path("./test_output")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "test_sdf.npz"
        np.savez_compressed(
            output_path,
            sdf=tsdf_grid,
            occupancy=occupancy_grid,
            voxel_size=voxel_size,
            bounds=bounds,
            intrinsics=intrinsics
        )
        
        print(f"✅ 测试结果保存到: {output_path}")
        
        # 10. 验证文件可加载
        loaded = np.load(output_path)
        print(f"✅ 文件验证:")
        print(f"   加载的SDF形状: {loaded['sdf'].shape}")
        print(f"   加载的占用形状: {loaded['occupancy'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("SDF生成快速测试")
    print("=" * 60)
    
    success = quick_test()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 快速测试成功!")
        print("\n可以运行完整生成:")
        print("  python generate_tartanair_sdf.py --max_frames 20 --voxel_size 0.1")
    else:
        print("❌ 快速测试失败")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)