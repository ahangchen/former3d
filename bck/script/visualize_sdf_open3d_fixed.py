#!/usr/bin/env python3
"""
使用Open3D进行TartanAir SDF结果可视化（修复版）
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

def extract_surface_points(sdf_grid, voxel_size, bounds, threshold=0.03):
    """从SDF提取表面点"""
    print("从SDF提取表面点...")
    
    # 找到接近表面的体素
    surface_mask = np.abs(sdf_grid) < threshold
    
    if not surface_mask.any():
        print("警告: 没有找到表面点")
        return None
    
    # 获取表面体素的坐标
    surface_coords = np.where(surface_mask)
    
    # 将体素坐标转换为世界坐标
    points = np.stack(surface_coords, axis=1).astype(np.float32)
    points = points * voxel_size + bounds[:, 0].reshape(1, 3)
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 根据SDF值添加颜色（负值表示内部，正值表示外部）
    sdf_values = sdf_grid[surface_mask]
    
    # 归一化SDF值用于颜色映射
    sdf_min, sdf_max = sdf_values.min(), sdf_values.max()
    if sdf_max > sdf_min:
        normalized_sdf = (sdf_values - sdf_min) / (sdf_max - sdf_min)
        # 使用彩虹色映射：内部（负值）为蓝色，表面为绿色，外部（正值）为红色
        colors = np.zeros((len(normalized_sdf), 3))
        
        # 负值（内部）到蓝色
        internal_mask = sdf_values < 0
        if internal_mask.any():
            internal_norm = (sdf_values[internal_mask] - sdf_min) / (-sdf_min)
            colors[internal_mask, 0] = 0.2  # 少量红色
            colors[internal_mask, 1] = 0.2  # 少量绿色
            colors[internal_mask, 2] = 0.8 + 0.2 * internal_norm  # 蓝色渐变
        
        # 正值（外部）到红色
        external_mask = sdf_values > 0
        if external_mask.any():
            external_norm = sdf_values[external_mask] / sdf_max
            colors[external_mask, 0] = 0.8 + 0.2 * external_norm  # 红色渐变
            colors[external_mask, 1] = 0.2  # 少量绿色
            colors[external_mask, 2] = 0.2  # 少量蓝色
        
        # 接近表面的点（绝对值小）为绿色
        surface_mask = np.abs(sdf_values) < threshold * 0.5
        if surface_mask.any():
            colors[surface_mask] = [0.2, 0.8, 0.2]  # 绿色
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"表面点云创建完成: {len(points)}个点")
    return pcd

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
        # 使用热图颜色：低处为蓝色，高处为红色
        colors = np.stack([colors, np.zeros_like(colors), 1-colors], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"占用点云创建完成: {len(points)}个点")
    return pcd

def create_coordinate_frame(size=1.0):
    """创建坐标轴"""
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    return coordinate_frame

def create_bounding_box(bounds):
    """创建场景边界框"""
    min_bound = bounds[:, 0]
    max_bound = bounds[:, 1]
    
    # 创建线框
    lines = o3d.geometry.LineSet()
    
    # 定义立方体的8个顶点
    vertices = np.array([
        [min_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]],
    ])
    
    # 定义12条边
    lines_indices = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
    ])
    
    lines.points = o3d.utility.Vector3dVector(vertices)
    lines.lines = o3d.utility.Vector2iVector(lines_indices)
    lines.paint_uniform_color([0, 1, 0])  # 绿色
    
    return lines

def visualize_interactive(geometries, window_name="TartanAir SDF可视化"):
    """交互式可视化"""
    print("启动交互式可视化...")
    
    # 设置可视化选项
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1200, height=800)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
    render_option.point_size = 2.0
    render_option.line_width = 2.0
    
    # 设置相机视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    
    print("\n可视化窗口已打开，使用以下控制:")
    print("  鼠标左键: 旋转")
    print("  鼠标右键: 平移")
    print("  鼠标滚轮: 缩放")
    print("  F: 切换全屏")
    print("  H: 显示帮助")
    print("  Q/Esc: 退出")
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

def save_visualization_results(geometries, output_dir):
    """保存可视化结果"""
    print("保存可视化结果...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存每个几何体
    for i, geom in enumerate(geometries):
        if isinstance(geom, o3d.geometry.PointCloud):
            file_path = os.path.join(output_dir, f"geometry_{i}_pointcloud.ply")
            o3d.io.write_point_cloud(file_path, geom)
            print(f"  点云保存到: {file_path}")
        elif isinstance(geom, o3d.geometry.LineSet):
            file_path = os.path.join(output_dir, f"geometry_{i}_lineset.ply")
            o3d.io.write_line_set(file_path, geom)
            print(f"  线集保存到: {file_path}")
        elif isinstance(geom, o3d.geometry.TriangleMesh):
            file_path = os.path.join(output_dir, f"geometry_{i}_mesh.ply")
            o3d.io.write_triangle_mesh(file_path, geom)
            print(f"  网格保存到: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="使用Open3D可视化TartanAir SDF结果（修复版）")
    parser.add_argument("--input", type=str, default="tartanair_sdf_output/abandonedfactory_sample_P001_sdf_occ.npz",
                       help="输入NPZ文件路径")
    parser.add_argument("--output_dir", type=str, default="open3d_visualizations_fixed",
                       help="输出目录")
    parser.add_argument("--no_interactive", action="store_true",
                       help="不显示交互式窗口，只保存文件")
    parser.add_argument("--sdf_threshold", type=float, default=0.03,
                       help="SDF表面提取阈值")
    parser.add_argument("--sample_rate", type=float, default=0.05,
                       help="点云采样率")
    
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
    
    # 创建可视化几何体
    geometries = []
    
    # 1. 从SDF提取表面点云
    surface_pcd = extract_surface_points(sdf_grid, voxel_size, bounds, args.sdf_threshold)
    if surface_pcd is not None:
        geometries.append(surface_pcd)
    
    # 2. 从占用创建点云
    occupancy_pcd = create_point_cloud_from_occupancy(occupancy_grid, voxel_size, bounds, args.sample_rate)
    if occupancy_pcd is not None:
        geometries.append(occupancy_pcd)
    
    # 3. 添加坐标轴
    coordinate_frame = create_coordinate_frame(size=0.5)
    geometries.append(coordinate_frame)
    
    # 4. 添加边界框
    bbox = create_bounding_box(bounds)
    geometries.append(bbox)
    
    # 保存可视化结果
    save_visualization_results(geometries, args.output_dir)
    
    # 保存统计信息
    stats_path = os.path.join(args.output_dir, "sdf_statistics.txt")
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
    
    # 交互式可视化
    if not args.no_interactive and len(geometries) > 0:
        visualize_interactive(geometries, "TartanAir SDF可视化")
    
    print(f"\n{'='*60}")
    print("可视化完成!")
    print(f"所有文件保存到: {args.output_dir}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()