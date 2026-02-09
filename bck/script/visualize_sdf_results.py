#!/usr/bin/env python3
"""
可视化TartanAir SDF生成结果
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import json
import os

def visualize_sdf_3d(sdf_grid, occupancy_grid, bounds, title="TartanAir SDF"):
    """3D可视化SDF和占用网格"""
    
    # 提取占用体素的坐标
    occ_coords = np.where(occupancy_grid > 0.5)
    
    if len(occ_coords[0]) == 0:
        print("警告: 没有找到占用体素")
        return None
    
    # 随机采样一些点用于可视化（避免太多点）
    max_points = 5000
    if len(occ_coords[0]) > max_points:
        indices = np.random.choice(len(occ_coords[0]), max_points, replace=False)
        x = occ_coords[0][indices]
        y = occ_coords[1][indices]
        z = occ_coords[2][indices]
    else:
        x, y, z = occ_coords
    
    # 创建3D图
    fig = plt.figure(figsize=(15, 10))
    
    # 子图1: 占用体素
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(x, y, z, c=z, cmap='viridis', s=2, alpha=0.6)
    ax1.set_xlabel('X (体素索引)')
    ax1.set_ylabel('Y (体素索引)')
    ax1.set_zlabel('Z (体素索引)')
    ax1.set_title(f'{title} - 占用体素 ({len(x)}个点)')
    plt.colorbar(scatter1, ax=ax1, label='Z坐标')
    
    # 子图2: SDF切片
    ax2 = fig.add_subplot(122)
    
    # 显示中间Z切片的SDF
    z_mid = sdf_grid.shape[2] // 2
    sdf_slice = sdf_grid[:, :, z_mid]
    
    im = ax2.imshow(sdf_slice.T, cmap='coolwarm', origin='lower',
                   vmin=-1.0, vmax=1.0, aspect='auto')
    ax2.set_xlabel('X (体素索引)')
    ax2.set_ylabel('Y (体素索引)')
    ax2.set_title(f'{title} - SDF切片 (Z={z_mid})')
    plt.colorbar(im, ax=ax2, label='SDF值')
    
    plt.tight_layout()
    return fig

def visualize_sdf_slices(sdf_grid, occupancy_grid, title="TartanAir SDF切片"):
    """显示多个SDF切片"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 选择6个不同的Z切片
    z_slices = np.linspace(0, sdf_grid.shape[2]-1, 6, dtype=int)
    
    for i, z in enumerate(z_slices):
        ax = axes[i]
        sdf_slice = sdf_grid[:, :, z]
        occ_slice = occupancy_grid[:, :, z]
        
        # 创建组合图像：SDF为颜色，占用为轮廓
        im = ax.imshow(sdf_slice.T, cmap='coolwarm', origin='lower',
                      vmin=-1.0, vmax=1.0, aspect='auto', alpha=0.7)
        
        # 添加占用轮廓
        if occ_slice.sum() > 0:
            ax.contour(occ_slice.T, levels=[0.5], colors='black', linewidths=0.5)
        
        ax.set_title(f'Z切片 {z}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig

def visualize_statistics(sdf_grid, occupancy_grid, voxel_size, bounds):
    """显示统计信息"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. SDF值分布直方图
    ax1 = axes[0, 0]
    sdf_values = sdf_grid.flatten()
    ax1.hist(sdf_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('SDF值')
    ax1.set_ylabel('频率')
    ax1.set_title('SDF值分布')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='表面 (SDF=0)')
    ax1.legend()
    
    # 2. 占用统计
    ax2 = axes[0, 1]
    occ_values = occupancy_grid.flatten()
    labels = ['空闲', '占用']
    sizes = [len(occ_values) - occ_values.sum(), occ_values.sum()]
    colors = ['lightblue', 'orange']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'占用统计 ({occ_values.sum():.0f}/{len(occ_values)}体素)')
    
    # 3. SDF沿坐标轴的变化
    ax3 = axes[1, 0]
    x_mid = sdf_grid.shape[0] // 2
    y_mid = sdf_grid.shape[1] // 2
    z_mid = sdf_grid.shape[2] // 2
    
    # X轴
    x_profile = sdf_grid[:, y_mid, z_mid]
    ax3.plot(x_profile, label='X轴', alpha=0.7)
    
    # Y轴
    y_profile = sdf_grid[x_mid, :, z_mid]
    ax3.plot(y_profile, label='Y轴', alpha=0.7)
    
    # Z轴
    z_profile = sdf_grid[x_mid, y_mid, :]
    ax3.plot(z_profile, label='Z轴', alpha=0.7)
    
    ax3.set_xlabel('体素索引')
    ax3.set_ylabel('SDF值')
    ax3.set_title('SDF沿坐标轴变化')
    ax3.legend()
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax3.grid(True, alpha=0.3)
    
    # 4. 边界信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    info_text = f"""场景边界:
X: [{bounds[0,0]:.2f}, {bounds[0,1]:.2f}]米
Y: [{bounds[1,0]:.2f}, {bounds[1,1]:.2f}]米
Z: [{bounds[2,0]:.2f}, {bounds[2,1]:.2f}]米

体素网格:
形状: {sdf_grid.shape}
体素大小: {voxel_size}米
总体素数: {sdf_grid.size}

SDF统计:
最小值: {sdf_grid.min():.3f}
最大值: {sdf_grid.max():.3f}
均值: {sdf_grid.mean():.3f}
标准差: {sdf_grid.std():.3f}

占用统计:
占用体素: {occupancy_grid.sum():.0f}
占用率: {occupancy_grid.sum()/occupancy_grid.size*100:.2f}%
"""
    
    ax4.text(0.1, 0.5, info_text, fontfamily='monospace',
            verticalalignment='center', fontsize=10)
    
    plt.suptitle('TartanAir SDF统计信息', fontsize=16)
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="可视化TartanAir SDF结果")
    parser.add_argument("--input", type=str, default="tartanair_sdf_output/abandonedfactory_sample_P001_sdf_occ.npz",
                       help="输入NPZ文件路径")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print(f"加载数据: {args.input}")
    data = np.load(args.input)
    
    sdf_grid = data['sdf']
    occupancy_grid = data['occupancy']
    voxel_size = float(data['voxel_size'])
    bounds = data['bounds']
    
    print(f"SDF网格形状: {sdf_grid.shape}")
    print(f"占用网格形状: {occupancy_grid.shape}")
    print(f"体素大小: {voxel_size}米")
    print(f"占用体素: {occupancy_grid.sum():.0f}/{occupancy_grid.size} ({occupancy_grid.sum()/occupancy_grid.size*100:.2f}%)")
    
    # 生成可视化
    print("\n生成可视化...")
    
    # 1. 3D可视化
    print("  生成3D可视化...")
    fig_3d = visualize_sdf_3d(sdf_grid, occupancy_grid, bounds, 
                             title="TartanAir SDF - abandonedfactory_sample_P001")
    if fig_3d:
        fig_3d_path = os.path.join(args.output_dir, "sdf_3d_visualization.png")
        fig_3d.savefig(fig_3d_path, dpi=150, bbox_inches='tight')
        plt.close(fig_3d)
        print(f"    保存到: {fig_3d_path}")
    
    # 2. 切片可视化
    print("  生成切片可视化...")
    fig_slices = visualize_sdf_slices(sdf_grid, occupancy_grid,
                                     title="TartanAir SDF切片 - abandonedfactory_sample_P001")
    fig_slices_path = os.path.join(args.output_dir, "sdf_slices.png")
    fig_slices.savefig(fig_slices_path, dpi=150, bbox_inches='tight')
    plt.close(fig_slices)
    print(f"    保存到: {fig_slices_path}")
    
    # 3. 统计可视化
    print("  生成统计可视化...")
    fig_stats = visualize_statistics(sdf_grid, occupancy_grid, voxel_size, bounds)
    fig_stats_path = os.path.join(args.output_dir, "sdf_statistics.png")
    fig_stats.savefig(fig_stats_path, dpi=150, bbox_inches='tight')
    plt.close(fig_stats)
    print(f"    保存到: {fig_stats_path}")
    
    # 4. 保存汇总报告
    print("  生成汇总报告...")
    report_path = os.path.join(args.output_dir, "sdf_generation_report.txt")
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TartanAir SDF生成结果报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("输入文件:\n")
        f.write(f"  {args.input}\n\n")
        
        f.write("网格信息:\n")
        f.write(f"  SDF形状: {sdf_grid.shape}\n")
        f.write(f"  占用形状: {occupancy_grid.shape}\n")
        f.write(f"  体素大小: {voxel_size}米\n")
        f.write(f"  总体素数: {sdf_grid.size}\n\n")
        
        f.write("场景边界 (米):\n")
        f.write(f"  X: [{bounds[0,0]:.2f}, {bounds[0,1]:.2f}]\n")
        f.write(f"  Y: [{bounds[1,0]:.2f}, {bounds[1,1]:.2f}]\n")
        f.write(f"  Z: [{bounds[2,0]:.2f}, {bounds[2,1]:.2f}]\n\n")
        
        f.write("SDF统计:\n")
        f.write(f"  最小值: {sdf_grid.min():.6f}\n")
        f.write(f"  最大值: {sdf_grid.max():.6f}\n")
        f.write(f"  均值: {sdf_grid.mean():.6f}\n")
        f.write(f"  标准差: {sdf_grid.std():.6f}\n\n")
        
        f.write("占用统计:\n")
        f.write(f"  占用体素: {occupancy_grid.sum():.0f}\n")
        f.write(f"  空闲体素: {occupancy_grid.size - occupancy_grid.sum():.0f}\n")
        f.write(f"  占用率: {occupancy_grid.sum()/occupancy_grid.size*100:.2f}%\n\n")
        
        f.write("生成的可视化文件:\n")
        f.write(f"  1. 3D可视化: {fig_3d_path if fig_3d else 'N/A'}\n")
        f.write(f"  2. 切片可视化: {fig_slices_path}\n")
        f.write(f"  3. 统计可视化: {fig_stats_path}\n")
        f.write(f"  4. 本报告: {report_path}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 60 + "\n")
    
    print(f"    保存到: {report_path}")
    
    print(f"\n{'='*60}")
    print("可视化完成!")
    print(f"所有文件保存到: {args.output_dir}")
    print(f"{'='*60}")
    
    # 显示其中一个图像（如果有显示环境）
    try:
        plt.show()
    except:
        print("注意: 无法显示图形界面，请检查生成的文件")

if __name__ == "__main__":
    main()