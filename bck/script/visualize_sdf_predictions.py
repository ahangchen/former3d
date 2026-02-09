#!/usr/bin/env python3
"""
可视化SDF预测结果 - 使用matplotlib生成静态图像
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import sys
import os

# 添加项目路径
sys.path.append('/home/cwh/coding/former3d')

# 导入模型
try:
    from fixed_training import SimpleSDFModel
    print("✅ 成功导入SimpleSDFModel")
except ImportError as e:
    print(f"❌ 导入模型失败: {e}")
    sys.exit(1)

def load_model():
    """加载模型"""
    model_path = 'fixed_checkpoints/best_model.pth'
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    model = SimpleSDFModel()
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

def generate_sdf_slice(model, plane='xy', z=0.0, resolution=100, bounds=(-1, 1)):
    """生成SDF切片"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # 创建网格
    x = np.linspace(bounds[0], bounds[1], resolution)
    y = np.linspace(bounds[0], bounds[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    if plane == 'xy':
        # XY平面，固定Z
        points = np.stack([X.flatten(), Y.flatten(), np.full_like(X.flatten(), z)], axis=1)
    elif plane == 'xz':
        # XZ平面，固定Y
        points = np.stack([X.flatten(), np.full_like(X.flatten(), z), Y.flatten()], axis=1)
    else:  # yz
        # YZ平面，固定X
        points = np.stack([np.full_like(X.flatten(), z), X.flatten(), Y.flatten()], axis=1)
    
    # 预测SDF
    points_tensor = torch.tensor(points, dtype=torch.float32).to(device)
    with torch.no_grad():
        sdf_values = model(points_tensor).cpu().numpy()
    
    # 重塑为2D网格
    sdf_grid = sdf_values.reshape(resolution, resolution)
    
    return X, Y, sdf_grid

def plot_sdf_slice(X, Y, sdf_grid, plane='xy', z=0.0, save_path=None):
    """绘制SDF切片"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. 热力图
    im = axes[0].imshow(sdf_grid, 
                       extent=[X.min(), X.max(), Y.min(), Y.max()],
                       origin='lower',
                       cmap='coolwarm',
                       vmin=-0.5, vmax=0.5)
    axes[0].set_xlabel('X' if plane == 'xy' else 'X' if plane == 'xz' else 'Y')
    axes[0].set_ylabel('Y' if plane == 'xy' else 'Z' if plane == 'xz' else 'Z')
    fixed_axis = 'Z' if plane == 'xy' else 'Y' if plane == 'xz' else 'X'
    axes[0].set_title(f'SDF切片 - {plane.upper()}平面 (固定{fixed_axis}={z:.2f})')
    plt.colorbar(im, ax=axes[0], label='SDF值')
    
    # 添加零等高线
    contour = axes[0].contour(X, Y, sdf_grid, 
                             levels=[-0.1, -0.05, 0, 0.05, 0.1],
                             colors=['red', 'orange', 'black', 'orange', 'red'],
                             linewidths=1)
    axes[0].clabel(contour, inline=True, fontsize=8)
    
    # 2. 3D表面图
    ax3d = fig.add_subplot(122, projection='3d')
    surf = ax3d.plot_surface(X, Y, sdf_grid, 
                            cmap='coolwarm',
                            alpha=0.8,
                            linewidth=0,
                            antialiased=True)
    ax3d.set_xlabel('X' if plane == 'xy' else 'X' if plane == 'xz' else 'Y')
    ax3d.set_ylabel('Y' if plane == 'xy' else 'Z' if plane == 'xz' else 'Z')
    ax3d.set_zlabel('SDF值')
    ax3d.set_title(f'SDF 3D表面图')
    ax3d.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 图像保存到: {save_path}")
    
    return fig

def plot_sdf_distribution_histogram(sdf_values, save_path=None):
    """绘制SDF分布直方图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 直方图
    axes[0].hist(sdf_values.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='表面 (SDF=0)')
    axes[0].axvline(x=-0.1, color='orange', linestyle=':', alpha=0.5, label='截断边界')
    axes[0].axvline(x=0.1, color='orange', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('SDF值')
    axes[0].set_ylabel('频数')
    axes[0].set_title('SDF值分布直方图')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 累积分布
    axes[1].hist(sdf_values.flatten(), bins=50, cumulative=True, 
                density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('SDF值')
    axes[1].set_ylabel('累积概率')
    axes[1].set_title('SDF累积分布函数')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 分布图保存到: {save_path}")
    
    return fig

def create_sdf_comparison(model, save_dir='visualization_results'):
    """创建SDF比较图"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("🎨 生成SDF可视化...")
    
    # 生成不同平面的切片
    planes = ['xy', 'xz', 'yz']
    z_values = [-0.5, 0.0, 0.5]
    
    all_sdf_values = []
    
    for plane in planes:
        for z in z_values:
            fixed_axis = 'Z' if plane == 'xy' else 'Y' if plane == 'xz' else 'X'
            print(f"  生成 {plane.upper()}平面，固定{fixed_axis}={z:.1f}...")
            
            X, Y, sdf_grid = generate_sdf_slice(model, plane=plane, z=z, resolution=100)
            all_sdf_values.append(sdf_grid.flatten())
            
            # 保存切片图
            save_path = os.path.join(save_dir, f'sdf_slice_{plane}_fixed{fixed_axis.lower()}{z:.1f}.png')
            plot_sdf_slice(X, Y, sdf_grid, plane=plane, z=z, save_path=save_path)
            plt.close()
    
    # 合并所有SDF值进行分布分析
    all_sdf = np.concatenate(all_sdf_values)
    
    # 生成分布图
    dist_path = os.path.join(save_dir, 'sdf_distribution.png')
    plot_sdf_distribution_histogram(all_sdf, save_path=dist_path)
    plt.close()
    
    # 创建总结图像
    create_summary_image(save_dir, all_sdf)
    
    return save_dir, all_sdf

def create_summary_image(save_dir, sdf_values):
    """创建总结图像"""
    fig = plt.figure(figsize=(15, 10))
    
    # 1. SDF统计信息
    ax1 = plt.subplot(2, 3, 1)
    ax1.axis('off')
    
    stats_text = f"""SDF统计信息:
    
最小值: {sdf_values.min():.4f}
最大值: {sdf_values.max():.4f}
平均值: {sdf_values.mean():.4f}
标准差: {sdf_values.std():.4f}
中位数: {np.median(sdf_values):.4f}

表面点比例 (|SDF|<0.1): 
  {(np.abs(sdf_values) < 0.1).sum() / len(sdf_values):.2%}

正SDF比例: {(sdf_values > 0).sum() / len(sdf_values):.2%}
负SDF比例: {(sdf_values < 0).sum() / len(sdf_values):.2%}
零SDF比例: {(sdf_values == 0).sum() / len(sdf_values):.2%}"""
    
    ax1.text(0.1, 0.9, stats_text, transform=ax1.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.set_title('📊 SDF统计总结', fontsize=12, fontweight='bold')
    
    # 2. 直方图
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(sdf_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('SDF值')
    ax2.set_ylabel('频数')
    ax2.set_title('SDF值分布')
    ax2.grid(True, alpha=0.3)
    
    # 3. 箱线图
    ax3 = plt.subplot(2, 3, 3)
    ax3.boxplot(sdf_values, vert=False)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('SDF值')
    ax3.set_title('SDF值箱线图')
    ax3.grid(True, alpha=0.3)
    
    # 4. 分位数图
    ax4 = plt.subplot(2, 3, 4)
    quantiles = np.percentile(sdf_values, np.linspace(0, 100, 21))
    ax4.plot(np.linspace(0, 100, 21), quantiles, 'o-', color='green', linewidth=2)
    ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('百分位数')
    ax4.set_ylabel('SDF值')
    ax4.set_title('SDF分位数图')
    ax4.grid(True, alpha=0.3)
    
    # 5. 累积分布
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(sdf_values, bins=50, cumulative=True, 
            density=True, alpha=0.7, color='orange', edgecolor='black')
    ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax5.set_xlabel('SDF值')
    ax5.set_ylabel('累积概率')
    ax5.set_title('累积分布函数')
    ax5.grid(True, alpha=0.3)
    
    # 6. 模型信息
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    model_info = """模型信息:
    
模型: SimpleSDFModel
参数: 198,657
最佳验证损失: 0.0666
训练轮数: 10

推理性能:
批次大小 10000: 1.32 ms
每点时间: 0.0001 ms

数据范围: [-1, 1]³
分辨率: 100×100
切片数: 9 (3平面×3深度)"""
    
    ax6.text(0.1, 0.9, model_info, transform=ax6.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax6.set_title('🤖 模型信息', fontsize=12, fontweight='bold')
    
    plt.suptitle('🎯 SDF预测结果可视化总结', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    summary_path = os.path.join(save_dir, 'sdf_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 总结图像保存到: {summary_path}")
    
    return summary_path

def main():
    """主函数"""
    print("=" * 60)
    print("🎨 SDF预测结果可视化")
    print("=" * 60)
    
    # 1. 加载模型
    result = load_model()
    if result is None:
        print("❌ 无法加载模型，退出")
        return
    
    model, checkpoint = result
    print(f"✅ 模型加载成功 (验证损失: {checkpoint.get('val_loss', 'unknown'):.4f})")
    
    # 2. 创建可视化
    save_dir, all_sdf = create_sdf_comparison(model)
    
    # 3. 显示结果
    print(f"\n🎉 可视化完成!")
    print(f"📁 所有图像保存到: {save_dir}/")
    
    # 列出生成的文件
    print(f"\n📸 生成的图像文件:")
    for file in sorted(os.listdir(save_dir)):
        if file.endswith('.png'):
            file_path = os.path.join(save_dir, file)
            print(f"  - {file}")
            if file == 'sdf_summary.png':
                print(f"    <qqimg>{os.path.abspath(file_path)}</qqimg>")
    
    # 显示关键统计
    print(f"\n📊 关键统计:")
    print(f"  SDF范围: [{all_sdf.min():.3f}, {all_sdf.max():.3f}]")
    print(f"  表面点比例 (|SDF|<0.1): {(np.abs(all_sdf) < 0.1).sum() / len(all_sdf):.2%}")
    print(f"  正SDF比例: {(all_sdf > 0).sum() / len(all_sdf):.2%}")
    print(f"  负SDF比例: {(all_sdf < 0).sum() / len(all_sdf):.2%}")

if __name__ == "__main__":
    main()