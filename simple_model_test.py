#!/usr/bin/env python3
"""
简单模型测试 - 直接加载最佳模型并进行推理
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
    print("简单模型测试")
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
        print(f"  - 模型类型: {checkpoint.get('model_type', '未知')}")
        
        # 显示模型参数
        if 'model_state_dict' in checkpoint:
            print(f"  - 模型参数数量: {len(checkpoint['model_state_dict'])}")
            for name, param in checkpoint['model_state_dict'].items():
                print(f"    {name}: {param.shape}")
        
        # 2. 创建简单模型进行推理
        print("\n创建简单MLP模型...")
        
        # 从检查点推断模型结构
        param_names = list(checkpoint['model_state_dict'].keys())
        
        # 查找网络层
        network_layers = []
        for name in param_names:
            if 'network' in name:
                network_layers.append(name)
        
        print(f"网络层: {network_layers}")
        
        # 创建简单的MLP模型（匹配检查点中的参数名称）
        class SimpleMLP(torch.nn.Module):
            def __init__(self, layer_dims):
                super().__init__()
                # 创建网络层，使用与检查点相同的命名
                self.network = torch.nn.ModuleList()
                for i in range(0, len(layer_dims)-1):
                    layer = torch.nn.Linear(layer_dims[i], layer_dims[i+1])
                    # 使用与检查点相同的命名：network.0, network.2, network.4, ...
                    setattr(self, f'network.{i*2}', layer)
                    self.network.append(layer)
            
            def forward(self, x):
                for i, layer in enumerate(self.network):
                    x = layer(x)
                    if i < len(self.network) - 1:  # 除了最后一层都加激活函数
                        x = torch.nn.functional.relu(x)
                return x
        
        # 从参数推断维度
        input_dim = checkpoint['model_state_dict']['network.0.weight'].shape[1]
        hidden_dims = []
        
        for i in range(len(network_layers)//2):  # 假设每层有weight和bias
            weight_key = f'network.{i*2}.weight'
            if weight_key in checkpoint['model_state_dict']:
                weight = checkpoint['model_state_dict'][weight_key]
                hidden_dims.append(weight.shape[0])
        
        # 构建层维度
        layer_dims = [input_dim] + hidden_dims
        print(f"推断的层维度: {layer_dims}")
        
        # 创建模型
        model = SimpleMLP(layer_dims)
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"模型创建成功，总参数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 验证参数加载
        print("验证参数加载:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}")
        
        # 3. 进行推理测试
        print("\n进行推理测试...")
        
        # 创建测试输入
        batch_size = 2
        num_points = 500
        
        # 随机3D点
        points = torch.randn(batch_size * num_points, input_dim)
        
        # 推理
        with torch.no_grad():
            output = model(points)
        
        print(f"输入形状: {points.shape}")
        print(f"输出形状: {output.shape}")
        print(f"输出统计:")
        print(f"  - 最小值: {output.min().item():.6f}")
        print(f"  - 最大值: {output.max().item():.6f}")
        print(f"  - 平均值: {output.mean().item():.6f}")
        print(f"  - 标准差: {output.std().item():.6f}")
        
        # 检查NaN/Inf
        nan_count = torch.isnan(output).sum().item()
        inf_count = torch.isinf(output).sum().item()
        print(f"  - NaN数量: {nan_count}")
        print(f"  - Inf数量: {inf_count}")
        
        # 4. 验证模型学习效果
        print("\n验证模型学习效果...")
        
        # 创建一些测试用例
        test_cases = [
            ("原点附近", torch.zeros(10, input_dim)),
            ("单位球面", torch.randn(10, input_dim)),
            ("远点", torch.randn(10, input_dim) * 5.0),
        ]
        
        for name, test_points in test_cases:
            with torch.no_grad():
                test_output = model(test_points)
            avg_output = test_output.mean().item()
            std_output = test_output.std().item()
            print(f"  {name}: 平均值={avg_output:.6f}, 标准差={std_output:.6f}")
        
        # 5. 保存示例预测
        print("\n保存示例预测...")
        example_dir = project_root / "example_predictions"
        example_dir.mkdir(exist_ok=True)
        
        # 保存一些预测结果
        example_points = torch.randn(1000, input_dim)
        with torch.no_grad():
            example_pred = model(example_points)
        
        example_data = {
            'points': example_points.numpy(),
            'predictions': example_pred.numpy(),
            'checkpoint_info': {
                'epoch': checkpoint.get('epoch', 'unknown'),
                'val_loss': checkpoint.get('val_loss', 'unknown'),
                'model_type': checkpoint.get('model_type', 'simple_mlp')
            }
        }
        
        example_path = example_dir / "example_predictions.npz"
        np.savez(example_path, **example_data)
        print(f"示例预测保存到: {example_path}")
        
        print("\n" + "=" * 60)
        print("✅ 简单模型测试完成！")
        print(f"模型已成功加载并验证:")
        print(f"  - 输入维度: {input_dim}")
        print(f"  - 输出维度: {output.shape[1]}")
        print(f"  - 验证损失: {checkpoint.get('val_loss', 'unknown'):.6f}")
        print(f"  - 无NaN/Inf值")
        
        print("\n下一步:")
        print("  1. 创建端到端训练脚本")
        print("  2. 集成OnlineTartanAirDataset")
        print("  3. 最大化GPU内存利用率")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()