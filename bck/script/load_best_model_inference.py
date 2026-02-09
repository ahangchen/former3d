#!/usr/bin/env python3
"""
加载最佳模型进行推理测试
加载 fixed_checkpoints/best_model.pth 并进行SDF预测
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_best_model():
    """加载最佳模型检查点"""
    checkpoint_path = project_root / "fixed_checkpoints" / "best_model.pth"
    
    if not checkpoint_path.exists():
        print(f"错误：检查点文件不存在: {checkpoint_path}")
        return None
    
    print(f"加载检查点: {checkpoint_path}")
    
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"检查点信息:")
        print(f"  - 训练epoch: {checkpoint.get('epoch', '未知')}")
        print(f"  - 验证损失: {checkpoint.get('val_loss', '未知'):.6f}")
        print(f"  - 模型类型: {checkpoint.get('model_type', '未知')}")
        
        # 检查模型状态
        if 'model_state_dict' in checkpoint:
            print(f"  - 模型参数数量: {len(checkpoint['model_state_dict'])}")
            # 打印前几个参数名称
            for i, (name, param) in enumerate(checkpoint['model_state_dict'].items()):
                if i < 5:  # 只显示前5个
                    print(f"    {name}: {param.shape}")
        
        return checkpoint
    
    except Exception as e:
        print(f"加载检查点时出错: {e}")
        return None

def create_test_input():
    """创建测试输入数据"""
    print("\n创建测试输入数据...")
    
    # 创建模拟输入
    batch_size = 2
    num_points = 1000
    
    # 创建随机3D点（在[-1, 1]范围内）
    points = torch.randn(batch_size, num_points, 3) * 0.5
    
    # 创建随机相机位姿
    poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 添加一些随机旋转和平移
    for i in range(batch_size):
        # 随机平移
        poses[i, :3, 3] = torch.randn(3) * 0.1
        
        # 随机小旋转
        angle = torch.randn(1) * 0.1
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        poses[i, 0, 0] = cos_a
        poses[i, 0, 1] = -sin_a
        poses[i, 1, 0] = sin_a
        poses[i, 1, 1] = cos_a
    
    # 创建随机图像特征
    img_size = 128
    img_channels = 256
    img_feats = torch.randn(batch_size, img_channels, img_size, img_size)
    
    # 创建相机内参
    intrinsics = torch.tensor([
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 240.0],
        [0.0, 0.0, 1.0]
    ]).unsqueeze(0).repeat(batch_size, 1, 1)
    
    print(f"输入数据形状:")
    print(f"  - 3D点: {points.shape}")
    print(f"  - 相机位姿: {poses.shape}")
    print(f"  - 图像特征: {img_feats.shape}")
    print(f"  - 相机内参: {intrinsics.shape}")
    
    return {
        'points': points,
        'poses': poses,
        'img_feats': img_feats,
        'intrinsics': intrinsics
    }

def test_simple_model_inference(checkpoint):
    """使用简化模型进行推理测试"""
    print("\n=== 使用简化模型进行推理测试 ===")
    
    try:
        # 导入简化模型
        from former3d.simplified_stream_sdfformer import SimpleSDFModel
        
        # 创建模型
        model = SimpleSDFModel(
            hidden_dim=256,
            num_layers=8,
            dropout=0.1
        )
        
        # 加载模型参数
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ 模型参数加载成功")
        
        # 设置为评估模式
        model.eval()
        
        # 创建测试输入
        test_input = create_test_input()
        
        # 进行推理
        with torch.no_grad():
            # 准备输入
            points = test_input['points']
            poses = test_input['poses']
            img_feats = test_input['img_feats']
            intrinsics = test_input['intrinsics']
            
            # 重塑点云数据
            batch_size, num_points, _ = points.shape
            points_flat = points.view(-1, 3)
            
            # 为每个点添加批次索引
            batch_inds = torch.arange(batch_size, device=points.device)
            batch_inds = batch_inds.unsqueeze(1).repeat(1, num_points).view(-1)
            
            # 调用模型
            output = model(
                points=points_flat,
                batch_inds=batch_inds,
                img_feats=img_feats,
                poses=poses,
                intrinsics=intrinsics
            )
            
            # 解析输出
            if isinstance(output, dict):
                sdf_pred = output.get('sdf', output.get('features'))
            else:
                sdf_pred = output
            
            print(f"\n推理结果:")
            print(f"  - 输出类型: {type(output)}")
            print(f"  - SDF预测形状: {sdf_pred.shape if sdf_pred is not None else 'None'}")
            
            if sdf_pred is not None:
                print(f"  - SDF统计:")
                print(f"    * 最小值: {sdf_pred.min().item():.4f}")
                print(f"    * 最大值: {sdf_pred.max().item():.4f}")
                print(f"    * 平均值: {sdf_pred.mean().item():.4f}")
                print(f"    * 标准差: {sdf_pred.std().item():.4f}")
                
                # 检查NaN值
                nan_count = torch.isnan(sdf_pred).sum().item()
                inf_count = torch.isinf(sdf_pred).sum().item()
                print(f"    * NaN数量: {nan_count}")
                print(f"    * Inf数量: {inf_count}")
                
                # 重塑回批次格式
                sdf_batch = sdf_pred.view(batch_size, num_points)
                print(f"  - 批次SDF形状: {sdf_batch.shape}")
                
                # 计算每个批次的统计
                for i in range(batch_size):
                    batch_sdf = sdf_batch[i]
                    print(f"    批次 {i}: min={batch_sdf.min().item():.4f}, "
                          f"max={batch_sdf.max().item():.4f}, "
                          f"mean={batch_sdf.mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"简化模型推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_stream_sdfformer_inference():
    """测试StreamSDFFormerIntegrated模型推理"""
    print("\n=== 测试StreamSDFFormerIntegrated模型推理 ===")
    
    try:
        # 尝试导入StreamSDFFormerIntegrated
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        # 创建模型配置
        config = {
            'voxel_size': 0.08,
            'crop_size': (24, 48, 48),
            'resolutions': {
                'coarse': 16,
                'fine': 1
            },
            'feature_dim': 256,
            'use_stream_fusion': False,  # 先禁用流式融合
            'memory_optimized': True
        }
        
        # 创建模型
        model = StreamSDFFormerIntegrated(**config)
        
        print(f"模型创建成功:")
        print(f"  - 总参数: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # 设置为评估模式
        model.eval()
        
        # 创建测试输入
        test_input = create_test_input()
        
        # 进行单帧推理
        with torch.no_grad():
            output = model.forward_single_frame(
                img_feats=test_input['img_feats'],
                poses=test_input['poses'],
                intrinsics=test_input['intrinsics']
            )
            
            print(f"\n单帧推理结果:")
            print(f"  - 输出类型: {type(output)}")
            
            if isinstance(output, dict):
                print(f"  - 输出键: {list(output.keys())}")
                
                # 检查SDF输出
                if 'sdf' in output and output['sdf'] is not None:
                    sdf = output['sdf']
                    print(f"  - SDF形状: {sdf.shape}")
                    print(f"  - SDF统计:")
                    print(f"    * 最小值: {sdf.min().item():.4f}")
                    print(f"    * 最大值: {sdf.max().item():.4f}")
                    print(f"    * 平均值: {sdf.mean().item():.4f}")
                
                # 检查占用输出
                if 'occupancy' in output and output['occupancy'] is not None:
                    occ = output['occupancy']
                    print(f"  - 占用形状: {occ.shape}")
                    print(f"    * 最小值: {occ.min().item():.4f}")
                    print(f"    * 最大值: {occ.max().item():.4f}")
                    print(f"    * 平均值: {occ.mean().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"StreamSDFFormerIntegrated推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def visualize_sdf_results():
    """可视化SDF结果（如果可能）"""
    print("\n=== 可视化SDF结果 ===")
    
    try:
        # 检查是否有可视化脚本
        vis_script = project_root / "visualize_sdf_open3d_fixed.py"
        
        if vis_script.exists():
            print(f"找到可视化脚本: {vis_script}")
            
            # 尝试运行可视化
            import subprocess
            result = subprocess.run(
                [sys.executable, str(vis_script)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("✓ 可视化脚本运行成功")
                # 提取关键信息
                lines = result.stdout.split('\n')
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['sdf', 'point', 'cloud', 'visual']):
                        print(f"  {line}")
            else:
                print(f"可视化脚本运行失败:")
                print(f"  错误: {result.stderr}")
        else:
            print("未找到可视化脚本，跳过可视化")
            
    except Exception as e:
        print(f"可视化失败: {e}")

def main():
    """主函数"""
    print("=" * 60)
    print("最佳模型加载和推理测试")
    print("=" * 60)
    
    # 1. 加载最佳模型
    checkpoint = load_best_model()
    if checkpoint is None:
        print("无法加载最佳模型，退出")
        return
    
    # 2. 测试简化模型推理
    simple_success = test_simple_model_inference(checkpoint)
    
    # 3. 测试StreamSDFFormerIntegrated推理
    stream_success = test_stream_sdfformer_inference()
    
    # 4. 可视化结果
    visualize_sdf_results()
    
    # 5. 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print(f"  - 简化模型推理: {'✓ 成功' if simple_success else '✗ 失败'}")
    print(f"  - StreamSDFFormer推理: {'✓ 成功' if stream_success else '✗ 失败'}")
    
    if simple_success or stream_success:
        print("\n✅ 模型推理测试完成！")
        print("下一步:")
        print("  1. 创建端到端训练脚本")
        print("  2. 最大化GPU内存利用率")
        print("  3. 使用OnlineTartanAirDataset进行训练")
    else:
        print("\n❌ 模型推理测试失败，需要进一步调试")

if __name__ == "__main__":
    main()