#!/usr/bin/env python3
"""
直接模型测试 - 使用检查点中的模型定义
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_model_from_checkpoint(checkpoint):
    """从检查点创建模型"""
    print("从检查点创建模型...")
    
    # 检查点中的模型定义
    class CheckpointModel(torch.nn.Module):
        def __init__(self, layer_dims):
            super().__init__()
            # 创建网络层
            layers = []
            for i in range(0, len(layer_dims)-1):
                layer = torch.nn.Linear(layer_dims[i], layer_dims[i+1])
                layers.append(layer)
                # 使用正确的命名
                setattr(self, f'network.{i*2}', layer)
            
            self.layers = layers
            self.num_layers = len(layers)
        
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < self.num_layers - 1:  # 除了最后一层都加激活函数
                    x = torch.nn.functional.relu(x)
            return x
    
    # 从参数推断维度
    param_names = list(checkpoint['model_state_dict'].keys())
    
    # 提取层维度
    layer_dims = []
    for i in range(0, 10, 2):  # 检查点中有5层：0,2,4,6,8
        weight_key = f'network.{i}.weight'
        if weight_key in checkpoint['model_state_dict']:
            weight = checkpoint['model_state_dict'][weight_key]
            if i == 0:
                layer_dims.append(weight.shape[1])  # 输入维度
            layer_dims.append(weight.shape[0])      # 输出维度
    
    print(f"层维度: {layer_dims}")
    
    # 创建模型
    model = CheckpointModel(layer_dims)
    
    # 加载状态字典
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型创建成功:")
    print(f"  - 输入维度: {layer_dims[0]}")
    print(f"  - 输出维度: {layer_dims[-1]}")
    print(f"  - 隐藏层: {layer_dims[1:-1]}")
    print(f"  - 总参数: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def test_model_inference(model):
    """测试模型推理"""
    print("\n测试模型推理...")
    
    # 创建测试数据
    batch_size = 4
    num_points = 1000
    input_dim = 3  # 3D点
    
    # 生成测试点
    test_points = torch.randn(batch_size * num_points, input_dim)
    
    # 推理
    with torch.no_grad():
        predictions = model(test_points)
    
    print(f"测试数据:")
    print(f"  - 点数: {batch_size * num_points:,}")
    print(f"  - 输入形状: {test_points.shape}")
    print(f"  - 输出形状: {predictions.shape}")
    
    # 统计信息
    print(f"\n预测统计:")
    print(f"  - 最小值: {predictions.min().item():.6f}")
    print(f"  - 最大值: {predictions.max().item():.6f}")
    print(f"  - 平均值: {predictions.mean().item():.6f}")
    print(f"  - 标准差: {predictions.std().item():.6f}")
    
    # 检查异常值
    nan_count = torch.isnan(predictions).sum().item()
    inf_count = torch.isinf(predictions).sum().item()
    print(f"  - NaN数量: {nan_count}")
    print(f"  - Inf数量: {inf_count}")
    
    # 测试不同区域的预测
    print(f"\n不同区域测试:")
    
    test_regions = [
        ("原点", torch.zeros(100, input_dim)),
        ("近点", torch.randn(100, input_dim) * 0.1),
        ("中距离", torch.randn(100, input_dim) * 1.0),
        ("远点", torch.randn(100, input_dim) * 5.0),
    ]
    
    results = []
    for name, points in test_regions:
        with torch.no_grad():
            pred = model(points)
        mean_val = pred.mean().item()
        std_val = pred.std().item()
        results.append((name, mean_val, std_val))
        print(f"  {name}: 平均值={mean_val:.6f}, 标准差={std_val:.6f}")
    
    return predictions, test_points

def visualize_predictions(predictions, points):
    """可视化预测结果"""
    print("\n可视化预测结果...")
    
    try:
        import matplotlib.pyplot as plt
        
        # 创建可视化目录
        vis_dir = project_root / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # 1. 预测值分布直方图
        plt.figure(figsize=(12, 4))
        
        # 预测值直方图
        plt.subplot(1, 3, 1)
        pred_np = predictions.numpy().flatten()
        plt.hist(pred_np, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title('SDF预测值分布')
        plt.xlabel('SDF值')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        
        # 2. 点云距离与SDF关系
        plt.subplot(1, 3, 2)
        distances = torch.norm(points, dim=1).numpy()
        plt.scatter(distances[:1000], pred_np[:1000], alpha=0.5, s=1)
        plt.title('距离 vs SDF')
        plt.xlabel('到原点的距离')
        plt.ylabel('SDF预测值')
        plt.grid(True, alpha=0.3)
        
        # 3. 3D点云颜色编码
        plt.subplot(1, 3, 3)
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 采样一些点
        sample_idx = np.random.choice(len(points), min(1000, len(points)), replace=False)
        sample_points = points[sample_idx].numpy()
        sample_pred = pred_np[sample_idx]
        
        # 颜色映射
        scatter = ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
                           c=sample_pred, cmap='coolwarm', alpha=0.6, s=10)
        plt.colorbar(scatter, ax=ax, label='SDF值')
        ax.set_title('3D点云SDF预测')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 保存图像
        plt.tight_layout()
        vis_path = vis_dir / "sdf_predictions.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 可视化保存到: {vis_path}")
        
        # 保存数据供后续分析
        data_path = vis_dir / "prediction_data.npz"
        np.savez(data_path, 
                points=points.numpy(), 
                predictions=pred_np,
                distances=distances)
        print(f"✓ 数据保存到: {data_path}")
        
    except ImportError:
        print("警告: matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"可视化失败: {e}")

def main():
    print("=" * 60)
    print("直接模型测试")
    print("=" * 60)
    
    # 1. 加载检查点
    checkpoint_path = project_root / "fixed_checkpoints" / "best_model.pth"
    
    if not checkpoint_path.exists():
        print(f"错误：检查点文件不存在: {checkpoint_path}")
        return
    
    print(f"加载检查点: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"检查点信息:")
        print(f"  - 训练epoch: {checkpoint.get('epoch', '未知')}")
        print(f"  - 验证损失: {checkpoint.get('val_loss', '未知'):.6f}")
        
        # 2. 创建模型
        model = create_model_from_checkpoint(checkpoint)
        
        # 3. 测试推理
        predictions, points = test_model_inference(model)
        
        # 4. 可视化
        visualize_predictions(predictions, points)
        
        # 5. 模型性能评估
        print("\n" + "=" * 60)
        print("模型性能评估:")
        print(f"  - 验证损失: {checkpoint.get('val_loss', 0.0):.6f}")
        print(f"  - 训练epoch: {checkpoint.get('epoch', 0)}")
        
        # 计算一些指标
        pred_np = predictions.numpy().flatten()
        zero_crossings = np.sum((pred_np[:-1] * pred_np[1:]) < 0)
        positive_ratio = np.mean(pred_np > 0)
        negative_ratio = np.mean(pred_np < 0)
        
        print(f"  - 零交叉点数量: {zero_crossings}")
        print(f"  - 正值比例: {positive_ratio:.2%}")
        print(f"  - 负值比例: {negative_ratio:.2%}")
        print(f"  - 绝对值小于0.1的比例: {np.mean(np.abs(pred_np) < 0.1):.2%}")
        
        print("\n" + "=" * 60)
        print("✅ 模型测试完成！")
        print("\n下一步:")
        print("  1. 创建端到端训练脚本")
        print("  2. 集成OnlineTartanAirDataset")
        print("  3. 最大化GPU内存利用率训练")
        print("  4. 添加流式融合功能")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()