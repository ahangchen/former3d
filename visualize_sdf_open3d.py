#!/usr/bin/env python3
"""
使用Open3D进行TartanAir SDF结果可视化
提供交互式3D查看和高质量渲染
"""

import numpy as np
import open3d as o3d
import argparse
import os
import json
from pathlib import Path

def load_sdf_data(input_path):
    """加载SDF数据"""
    print(f"加载SDF数据: {input_path}")
    data = np.load(input_path)
    
    sdf_grid = data['sdf']
    occupancy_grid = data['occupancy']
    voxel_size = float(data['voxel_size'])
    bounds = data['bounds']
    
    print(f"SDF网格形状: {sdf_grid.shape}")
    print(f"体素大小: {voxel_size}米")
    print(f"占用体素: {occupancy_grid.sum():.0f}/{occupancy_grid.size} ({occupancy_grid.sum()/occupancy_grid.size*100:.2f}%)")
    
    return sdf_grid, occupancy_grid, voxel_size, bounds

def create_mesh_from_sdf(sdf_grid, voxel_size, level=0.0):
    """从SDF网格创建网格"""
    print("从SDF创建网格...")
    
    # 使用Marching Cubes算法从SDF提取网格
    try:
        # 将SDF转换为Open3D格式
        sdf_array = sdf_grid.astype(np.float32)
        
        # 创建体素网格
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            o3d.geometry.PointCloud(), 
            voxel_size=voxel_size
        )
        
        # 直接使用Marching Cubes
        vertices, triangles = o3d.geometry.TriangleMesh.create_from_volume(
            sdf_array, 
            voxel_size=voxel_size,
            level=level
        )
        
        if vertices is None or len(vertices) == 0:
            print("警告: 无法从SDF提取网格，可能没有表面")
            return None
            
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        
        # 计算法线用于渲染
        mesh.compute_vertex_normals()
        
        print(f"网格创建完成: {len(vertices)}个顶点, {len(triangles)}个三角形")
        return mesh
        
    except Exception as e:
        print(f"网格创建失败: {e}")
        return None

def create_point_cloud_from_occupancy(occupancy_grid, voxel_size, bounds, sample_rate=0.1):
    """从占用网格创建点云"""
    print("从占用网格创建点云...")
    
    # 获取占用体素的坐标
    occ_coords = np.where(occupancy_grid > 0.5)
    
    if len(occ_coords[0]) == 0:
        print("警告: 没有找到占用体素")
        return None
    
    # 采样以减少点数
    if sample_rate < 1.0:
        num_points = len(occ_coords[0])
        sample_size = int(num_points * sample_rate)
        indices = np.random.choice(num_points, sample_size, replace=False)
        x = occ_coords[0][indices]
        y = occ_coords[1][indices]
        z = occ_coords[2][indices]
    else:
        x, y, z = occ_coords
    
    # 将体素坐标转换为世界坐标
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    points = points * voxel_size + bounds[:, 0].reshape(1, 3)
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 根据Z坐标添加颜色
    z_values = points[:, 2]
    z_min, z_max = z_values.min(), z_values.max()
    if z_max > z_min:
        colors = (z_values - z_min) / (z_max - z_min)
        colors = np.stack([colors, 1-colors, np.zeros_like(colors)], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"点云创建完成: {len(points)}个点")
    return pcd

def create_camera_trajectory(metadata_path):
    """从元数据创建相机轨迹"""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 检查是否有相机位姿信息
        if 'camera_poses' in metadata:
            poses = np.array(metadata['camera_poses'])
            print(f"加载相机轨迹: {len(poses)}个位姿")
            
            # 创建相机位置点云
            positions = poses[:, :3, 3]  # 提取位置
            
            trajectory = o3d.geometry.LineSet()
            trajectory.points = o3d.utility.Vector3dVector(positions)
            
            # 创建连接线
            lines = []
            for i in range(len(positions) - 1):
                lines.append([i, i + 1])
            trajectory.lines = o3d.utility.Vector2iVector(lines)
            trajectory.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(lines))
            
            return trajectory
        else:
            print("元数据中没有相机位姿信息")
            return None
            
    except Exception as e:
        print(f"创建相机轨迹失败: {e}")
        return None

def create_coordinate_frame(size=1.0):
    """创建坐标轴"""
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    return coordinate_frame

def create_bounding_box(bounds):
    """创建场景边界框"""
    min_bound = bounds[:, 0]
    max_bound = bounds[:, 1]
    
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    bbox.color = [0, 1, 0]  # 绿色
    
    # 创建线框
    lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
    lines.paint_uniform_color([0, 1, 0])
    
    return lines

def visualize_interactive(sdf_grid, occupancy_grid, voxel_size, bounds, metadata_path=None):
    """交互式可视化"""
    print("准备交互式可视化...")
    
    # 创建可视化元素
    geometries = []
    
    # 1. 从SDF创建网格
    mesh = create_mesh_from_sdf(sdf_grid, voxel_size)
    if mesh is not None:
        mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 灰色
        geometries.append(mesh)
    
    # 2. 从占用创建点云
    pcd = create_point_cloud_from_occupancy(occupancy_grid, voxel_size, bounds, sample_rate=0.05)
    if pcd is not None:
        geometries.append(pcd)
    
    # 3. 创建相机轨迹
    if metadata_path and os.path.exists(metadata_path):
        trajectory = create_camera_trajectory(metadata_path)
        if trajectory is not None:
            geometries.append(trajectory)
    
    # 4. 添加坐标轴
    coordinate_frame = create_coordinate_frame(size=0.5)
    geometries.append(coordinate_frame)
    
    # 5. 添加边界框
    bbox = create_bounding_box(bounds)
    geometries.append(bbox)
    
    # 设置可视化选项
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="TartanAir SDF可视化", width=1200, height=800)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
    render_option.point_size = 2.0
    render_option.mesh_show_wireframe = True
    render_option.mesh_show_back_face = False
    
    # 设置相机视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    print("可视化窗口已打开，使用以下控制:")
    print("  鼠标左键: 旋转")
    print("  鼠标右键: 平移")
    print("  鼠标滚轮: 缩放")
    print("  F: 切换全屏")
    print("  H: 显示帮助")
    print("  Q/Esc: 退出")
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

def save_visualization(sdf_grid, occupancy_grid, voxel_size, bounds, output_dir):
    """保存可视化结果"""
    print("保存可视化结果...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 保存网格为PLY文件
    mesh = create_mesh_from_sdf(sdf_grid, voxel_size)
    if mesh is not None:
        mesh_path = os.path.join(output_dir, "sdf_mesh.ply")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        print(f"  网格保存到: {mesh_path}")
    
    # 2. 保存点云为PLY文件
    pcd = create_point_cloud_from_occupancy(occupancy_grid, voxel_size, bounds, sample_rate=0.1)
    if pcd is not None:
        pcd_path = os.path.join(output_dir, "occupancy_pointcloud.ply")
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"  点云保存到: {pcd_path}")
    
    # 3. 保存截图
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Screenshot", width=1600, height=900, visible=False)
        
        if mesh is not None:
            vis.add_geometry(mesh)
        if pcd is not None:
            vis.add_geometry(pcd)
        
        # 添加坐标轴和边界框
        vis.add_geometry(create_coordinate_frame(size=0.5))
        vis.add_geometry(create_bounding_box(bounds))
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)
        
        # 渲染并保存截图
        vis.poll_events()
        vis.update_renderer()
        
        screenshot_path = os.path.join(output_dir, "sdf_screenshot.png")
        vis.capture_screen_image(screenshot_path, do_render=True)
        vis.destroy_window()
        
        print(f"  截图保存到: {screenshot_path}")
        
    except Exception as e:
        print(f"  截图保存失败: {e}")
    
    # 4. 保存统计信息
    stats_path = os.path.join(output_dir, "sdf_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TartanAir SDF统计信息\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"SDF网格形状: {sdf_grid.shape}\n")
        f.write(f"体素大小: {voxel_size}米\n")
        f.write(f"总体素数: {sdf_grid.size}\n\n")
        
        f.write(f"场景边界 (米):\n")
        f.write(f"  X: [{bounds[0,0]:.2f}, {bounds[0,1]:.2f}]\n")
        f.write(f"  Y: [{bounds[1,0]:.2f}, {bounds[1,1]:.2f}]\n")
        f.write(f"  Z: [{bounds[2,0]:.2f}, {bounds[2,1]:.2f}]\n\n")
        
        f.write(f"SDF统计:\n")
        f.write(f"  最小值: {sdf_grid.min():.6f}\n")
        f.write(f"  最大值: {sdf_grid.max():.6f}\n")
        f.write(f"  均值: {sdf_grid.mean():.6f}\n")
        f.write(f"  标准差: {sdf_grid.std():.6f}\n\n")
        
        f.write(f"占用统计:\n")
        f.write(f"  占用体素: {occupancy_grid.sum():.0f}\n")
        f.write(f"  空闲体素: {occupancy_grid.size - occupancy_grid.sum():.0f}\n")
        f.write(f"  占用率: {occupancy_grid.sum()/occupancy_grid.size*100:.2f}%\n")
    
    print(f"  统计信息保存到: {stats_path}")

def main():
    parser = argparse.ArgumentParser(description="使用Open3D可视化TartanAir SDF结果")
    parser.add_argument("--input", type=str, default="tartanair_sdf_output/abandonedfactory_sample_P001_sdf_occ.npz",
                       help="输入NPZ文件路径")
    parser.add_argument("--metadata", type=str, default="tartanair_sdf_output/abandonedfactory_sample_P001_metadata.json",
                       help="元数据文件路径")
    parser.add_argument("--output_dir", type=str, default="open3d_visualizations",
                       help="输出目录")
    parser.add_argument("--no_interactive", action="store_true",
                       help="不显示交互式窗口，只保存文件")
    parser.add_argument("--level", type=float, default=0.0,
                       help="Marching Cubes等值面水平")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 加载数据
    sdf_grid, occupancy_grid, voxel_size, bounds = load_sdf_data(args.input)
    
    print(f"\n{'='*60}")
    print("Open3D可视化准备就绪")
    print(f"{'='*60}")
    
    # 保存可视化结果
    save_visualization(sdf_grid, occupancy_grid, voxel_size, bounds, args.output_dir)
    
    # 交互式可视化
    if not args.no_interactive:
        print("\n启动交互式可视化...")
        visualize_interactive(sdf_grid, occupancy_grid, voxel_size, bounds, args.metadata)
    
    print(f"\n{'='*60}")
    print("可视化完成!")
    print(f"所有文件保存到: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()