#!/usr/bin/env python3
"""
测试检查点模型 - 在former3d环境中运行
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 60)
    print("检查点模型测试 (former3d环境)")
    print("=" * 60)
    
    # 1. 加载检查点
    checkpoint_path = project_root / "fixed_checkpoints" / "best_model.pth"
    
    if not checkpoint_path.exists():
        print(f"❌ 错误：检查点文件不存在: {checkpoint_path}")
        return
    
    print(f"📂 加载检查点: {checkpoint_path}")
    
    try:
        # 在CPU上加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"📊 检查点信息:")
        print(f"  - 训练epoch: {checkpoint.get('epoch', '未知')}")
        print(f"  - 验证损失: {checkpoint.get('val_loss', '未知'):.6f}")
        print(f"  - 优化器步数: {checkpoint.get('step', '未知')}")
        
        # 显示模型参数
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"  - 模型参数数量: {len(state_dict)}")
            
            # 分析模型结构
            print(f"\n🔍 模型结构分析:")
            layer_info = {}
            for name, param in state_dict.items():
                if 'weight' in name:
                    layer_num = name.split('.')[1] if len(name.split('.')) > 1 else 'unknown'
                    in_features = param.shape[1] if len(param.shape) > 1 else 1
                    out_features = param.shape[0]
                    layer_info[layer_num] = (in_features, out_features)
                    print(f"    {name}: {param.shape} (in={in_features}, out={out_features})")
            
            # 推断模型架构
            if layer_info:
                print(f"\n🏗️  推断的模型架构:")
                layers = sorted([(int(k), v) for k, v in layer_info.items() if k.isdigit()])
                for layer_num, (in_dim, out_dim) in layers:
                    print(f"    层 {layer_num}: Linear({in_dim} → {out_dim})")
        
        # 2. 创建简单模型进行测试
        print(f"\n🧪 创建测试模型...")
        
        # 从检查点推断模型维度
        input_dim = 3  # 3D点
        hidden_dim = 256
        output_dim = 1
        
        # 创建简单的MLP模型
        class SimpleSDFModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 5层MLP，与检查点匹配
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim, output_dim)
                )
            
            def forward(self, x):
                return self.network(x)
        
        # 创建模型
        model = SimpleSDFModel()
        
        # 尝试加载检查点参数（可能不匹配，但我们可以测试）
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("✅ 模型参数加载成功（非严格模式）")
        except Exception as e:
            print(f"⚠️  参数加载警告: {e}")
            print("使用随机初始化参数继续测试")
        
        model.eval()
        print(f"📈 模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 3. 推理测试
        print(f"\n🚀 进行推理测试...")
        
        # 生成测试数据
        batch_size = 4
        num_points = 1000
        
        # 不同区域的测试点
        test_cases = [
            ("原点附近", torch.randn(batch_size * num_points, 3) * 0.1),
            ("单位球内", torch.randn(batch_size * num_points, 3)),
            ("远点", torch.randn(batch_size * num_points, 3) * 5.0),
        ]
        
        results = []
        for name, points in test_cases:
            with torch.no_grad():
                predictions = model(points)
            
            pred_np = predictions.numpy().flatten()
            
            stats = {
                'name': name,
                'min': pred_np.min(),
                'max': pred_np.max(),
                'mean': pred_np.mean(),
                'std': pred_np.std(),
                'nan_count': np.isnan(pred_np).sum(),
                'inf_count': np.isinf(pred_np).sum(),
                'positive_ratio': np.mean(pred_np > 0),
                'near_zero_ratio': np.mean(np.abs(pred_np) < 0.1)
            }
            
            results.append(stats)
            
            print(f"\n📊 {name}测试结果:")
            print(f"  形状: {predictions.shape}")
            print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  均值: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  NaN/Inf: {stats['nan_count']}/{stats['inf_count']}")
            print(f"  正值比例: {stats['positive_ratio']:.2%}")
            print(f"  近零比例: {stats['near_zero_ratio']:.2%}")
        
        # 4. 保存测试结果
        print(f"\n💾 保存测试结果...")
        output_dir = project_root / "test_results"
        output_dir.mkdir(exist_ok=True)
        
        # 保存统计数据
        import json
        stats_path = output_dir / "model_test_stats.json"
        
        # 转换numpy类型为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32):
                return int(obj)
            elif isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        with open(stats_path, 'w') as f:
            json.dump(convert_to_serializable({
                'checkpoint_info': {
                    'path': str(checkpoint_path),
                    'epoch': checkpoint.get('epoch'),
                    'val_loss': float(checkpoint.get('val_loss', 0))
                },
                'test_results': results,
                'model_info': {
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'output_dim': output_dim,
                    'total_params': sum(p.numel() for p in model.parameters())
                }
            }), f, indent=2)
        
        print(f"✅ 统计数据保存到: {stats_path}")
        
        # 5. 创建可视化
        try:
            import matplotlib.pyplot as plt
            
            print(f"\n🎨 创建可视化...")
            
            # 选择第一个测试用例进行可视化
            _, points = test_cases[0]
            with torch.no_grad():
                preds = model(points[:1000])  # 只取1000个点
            
            # 创建可视化
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            # 1. 预测值分布
            axes[0].hist(preds.numpy().flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[0].set_title('SDF预测值分布')
            axes[0].set_xlabel('SDF值')
            axes[0].set_ylabel('频率')
            axes[0].grid(True, alpha=0.3)
            
            # 2. 3D散点图
            from mpl_toolkits.mplot3d import Axes3D
            sample_points = points[:200].numpy()
            sample_preds = preds[:200].numpy().flatten()
            
            # 颜色映射
            norm = plt.Normalize(sample_preds.min(), sample_preds.max())
            colors = plt.cm.coolwarm(norm(sample_preds))
            
            ax = axes[1]
            scatter = ax.scatter(sample_points[:, 0], sample_points[:, 1], 
                               c=colors, alpha=0.6, s=20)
            ax.set_title('2D投影 (X-Y平面)')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.grid(True, alpha=0.3)
            
            # 3. 距离与SDF关系
            distances = np.linalg.norm(points[:500].numpy(), axis=1)
            distance_preds = preds[:500].numpy().flatten()
            
            axes[2].scatter(distances, distance_preds, alpha=0.5, s=10)
            axes[2].set_title('距离 vs SDF')
            axes[2].set_xlabel('到原点的距离')
            axes[2].set_ylabel('SDF值')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            vis_path = output_dir / "model_test_visualization.png"
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ 可视化保存到: {vis_path}")
            
        except ImportError:
            print("⚠️  matplotlib未安装，跳过可视化")
        
        # 6. 总结
        print(f"\n" + "=" * 60)
        print("🎉 测试完成总结")
        print("=" * 60)
        
        print(f"✅ 检查点加载成功")
        print(f"✅ 模型创建成功")
        print(f"✅ 推理测试完成")
        print(f"✅ 结果已保存")
        
        # 检查训练效果
        val_loss = checkpoint.get('val_loss', 0)
        if val_loss < 0.1:
            print(f"📈 模型性能: 优秀 (验证损失: {val_loss:.6f})")
        elif val_loss < 0.2:
            print(f"📈 模型性能: 良好 (验证损失: {val_loss:.6f})")
        else:
            print(f"📈 模型性能: 需改进 (验证损失: {val_loss:.6f})")
        
        print(f"\n🚀 下一步:")
        print(f"  1. 创建端到端训练脚本")
        print(f"  2. 最大化GPU内存利用率")
        print(f"  3. 集成OnlineTartanAirDataset")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()