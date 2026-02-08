#!/usr/bin/env python3
"""
测试最佳模型推理脚本
加载训练好的最佳模型，进行推理测试并可视化结果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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

def load_best_model(model_path='fixed_checkpoints/best_model.pth'):
    """加载最佳模型"""
    print(f"📂 加载模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None
    
    try:
        # 创建模型实例
        model = SimpleSDFModel()
        
        # 加载检查点
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 获取训练信息
        best_val_loss = checkpoint.get('val_loss', '未知')
        epoch = checkpoint.get('epoch', '未知')
        
        print(f"✅ 模型加载成功")
        print(f"   - 最佳验证损失: {best_val_loss}")
        print(f"   - 训练轮数: {epoch}")
        print(f"   - 模型参数: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, checkpoint
    
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None

def test_model_inference(model, device='cuda'):
    """测试模型推理"""
    print("\n🧪 测试模型推理...")
    
    # 将模型移到指定设备
    model = model.to(device)
    model.eval()
    
    # 生成测试点
    n_points = 1000
    print(f"生成 {n_points} 个测试点...")
    
    # 在单位立方体内生成随机点
    test_points = torch.randn(n_points, 3).to(device)
    
    with torch.no_grad():
        # 推理
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        sdf_predictions = model(test_points)
        end_time.record()
        
        torch.cuda.synchronize()
        inference_time = start_time.elapsed_time(end_time)
    
    # 分析结果
    sdf_values = sdf_predictions.cpu().numpy().flatten()
    
    print(f"✅ 推理完成")
    print(f"   - 推理时间: {inference_time:.2f} ms")
    print(f"   - 平均时间/点: {inference_time/n_points:.4f} ms")
    print(f"   - SDF统计:")
    print(f"     * 最小值: {sdf_values.min():.4f}")
    print(f"     * 最大值: {sdf_values.max():.4f}")
    print(f"     * 平均值: {sdf_values.mean():.4f}")
    print(f"     * 标准差: {sdf_values.std():.4f}")
    
    # 统计表面点比例 (|SDF| < 0.1)
    surface_mask = np.abs(sdf_values) < 0.1
    surface_ratio = surface_mask.sum() / len(sdf_values)
    print(f"   - 表面点比例 (|SDF|<0.1): {surface_ratio:.2%}")
    
    return test_points.cpu().numpy(), sdf_values

def visualize_sdf_distribution(sdf_values):
    """可视化SDF值分布"""
    print("\n📊 可视化SDF分布...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 直方图
    axes[0].hist(sdf_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='表面 (SDF=0)')
    axes[0].axvline(x=-0.1, color='orange', linestyle=':', alpha=0.5, label='截断边界')
    axes[0].axvline(x=0.1, color='orange', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('SDF值')
    axes[0].set_ylabel('频数')
    axes[0].set_title('SDF值分布直方图')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 箱线图
    axes[1].boxplot(sdf_values, vert=False)
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('SDF值')
    axes[1].set_title('SDF值箱线图')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    output_path = 'inference_results/sdf_distribution.png'
    os.makedirs('inference_results', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 可视化保存到: {output_path}")
    
    return output_path

def test_specific_points(model, device='cuda'):
    """测试特定点（原点、表面附近、远处）"""
    print("\n🎯 测试特定点...")
    
    model = model.to(device)
    model.eval()
    
    # 定义测试点
    test_cases = [
        ("原点", [0.0, 0.0, 0.0]),
        ("表面附近(+0.05)", [0.05, 0.05, 0.05]),
        ("表面附近(-0.05)", [-0.05, -0.05, -0.05]),
        ("远处(+1.0)", [1.0, 1.0, 1.0]),
        ("远处(-1.0)", [-1.0, -1.0, -1.0]),
        ("随机点", np.random.randn(3).tolist()),
    ]
    
    results = []
    with torch.no_grad():
        for name, point in test_cases:
            point_tensor = torch.tensor([point], dtype=torch.float32).to(device)
            sdf = model(point_tensor).item()
            results.append((name, point, sdf))
    
    # 打印结果
    print("测试点结果:")
    print("-" * 60)
    print(f"{'名称':<20} {'坐标':<30} {'SDF预测值':<10}")
    print("-" * 60)
    for name, point, sdf in results:
        coord_str = f"[{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]"
        print(f"{name:<20} {coord_str:<30} {sdf:>10.6f}")
    
    return results

def create_3d_scatter(points, sdf_values):
    """创建3D散点图"""
    print("\n🎨 创建3D散点图...")
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 根据SDF值着色
    scatter = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=sdf_values, cmap='coolwarm', alpha=0.6,
        s=10, vmin=-0.5, vmax=0.5
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D点云SDF预测值')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
    cbar.set_label('SDF值')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = 'inference_results/3d_sdf_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 3D散点图保存到: {output_path}")
    
    return output_path

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 最佳模型推理测试")
    print("=" * 60)
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 使用设备: {device}")
    if device == 'cuda':
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - CUDA版本: {torch.version.cuda}")
    
    # 1. 加载最佳模型
    result = load_best_model()
    if result is None:
        print("❌ 无法加载模型，退出")
        return
    
    model, checkpoint = result
    
    # 2. 测试推理性能
    points, sdf_values = test_model_inference(model, device)
    
    # 3. 测试特定点
    specific_results = test_specific_points(model, device)
    
    # 4. 可视化SDF分布
    dist_path = visualize_sdf_distribution(sdf_values)
    
    # 5. 创建3D散点图
    if len(points) > 0:
        scatter_path = create_3d_scatter(points, sdf_values)
    
    # 6. 保存推理结果
    output_dir = 'inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存结果到文件
    results_file = os.path.join(output_dir, 'inference_summary.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("最佳模型推理测试结果\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"模型文件: fixed_checkpoints/best_model.pth\n")
        f.write(f"设备: {device}\n")
        f.write(f"最佳验证损失: {checkpoint.get('val_loss', '未知')}\n")
        f.write(f"训练轮数: {checkpoint.get('epoch', '未知')}\n\n")
        
        f.write("SDF统计:\n")
        f.write(f"  最小值: {sdf_values.min():.6f}\n")
        f.write(f"  最大值: {sdf_values.max():.6f}\n")
        f.write(f"  平均值: {sdf_values.mean():.6f}\n")
        f.write(f"  标准差: {sdf_values.std():.6f}\n\n")
        
        f.write("特定点测试结果:\n")
        f.write("-" * 60 + "\n")
        for name, point, sdf in specific_results:
            coord_str = f"[{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]"
            f.write(f"{name:<20} {coord_str:<30} {sdf:>10.6f}\n")
    
    print(f"\n✅ 推理测试完成!")
    print(f"📁 结果保存到: {output_dir}/")
    print(f"📄 总结文件: {results_file}")
    print(f"📊 分布图: {dist_path}")
    
    # 显示图像路径（用于QQ Bot发送）
    if os.path.exists(dist_path):
        print(f"\n📸 图像文件路径:")
        print(f"<qqimg>{os.path.abspath(dist_path)}</qqimg>")

if __name__ == "__main__":
    main()