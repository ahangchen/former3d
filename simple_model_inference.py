#!/usr/bin/env python3
"""
简单模型推理测试 - 无GUI版本
"""

import torch
import numpy as np
import sys
import os
import json

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

def test_inference_performance(model, device='cuda'):
    """测试推理性能"""
    print("\n🧪 测试推理性能...")
    
    # 将模型移到指定设备
    model = model.to(device)
    model.eval()
    
    # 测试不同批次的推理时间
    batch_sizes = [1, 10, 100, 1000, 10000]
    results = []
    
    for batch_size in batch_sizes:
        # 生成测试点
        test_points = torch.randn(batch_size, 3).to(device)
        
        # 预热
        with torch.no_grad():
            _ = model(test_points[:min(10, batch_size)])
        
        # 正式测试
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        with torch.no_grad():
            sdf_predictions = model(test_points)
        end_time.record()
        
        torch.cuda.synchronize()
        inference_time = start_time.elapsed_time(end_time)
        
        # 获取结果统计
        sdf_values = sdf_predictions.cpu().numpy().flatten()
        
        results.append({
            'batch_size': batch_size,
            'inference_time_ms': inference_time,
            'time_per_point_ms': inference_time / batch_size,
            'sdf_min': float(sdf_values.min()),
            'sdf_max': float(sdf_values.max()),
            'sdf_mean': float(sdf_values.mean()),
            'sdf_std': float(sdf_values.std()),
        })
        
        print(f"  批次大小 {batch_size:6d}: {inference_time:6.2f} ms ({inference_time/batch_size:.4f} ms/点)")
    
    return results

def test_specific_points(model, device='cuda'):
    """测试特定点"""
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
        ("随机点1", np.random.randn(3).tolist()),
        ("随机点2", np.random.randn(3).tolist()),
        ("随机点3", np.random.randn(3).tolist()),
    ]
    
    results = []
    with torch.no_grad():
        for name, point in test_cases:
            point_tensor = torch.tensor([point], dtype=torch.float32).to(device)
            sdf = model(point_tensor).item()
            results.append({
                'name': name,
                'point': point,
                'sdf': sdf
            })
    
    # 打印结果
    print("测试点结果:")
    print("-" * 70)
    print(f"{'名称':<15} {'X':>8} {'Y':>8} {'Z':>8} {'SDF预测值':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<15} {r['point'][0]:>8.3f} {r['point'][1]:>8.3f} {r['point'][2]:>8.3f} {r['sdf']:>12.6f}")
    
    return results

def analyze_sdf_consistency(model, device='cuda'):
    """分析SDF一致性（对称性、连续性）"""
    print("\n🔍 分析SDF一致性...")
    
    model = model.to(device)
    model.eval()
    
    # 测试对称性：f(x) ≈ -f(-x) 对于SDF
    test_points = torch.randn(100, 3).to(device)
    
    with torch.no_grad():
        sdf_pos = model(test_points)
        sdf_neg = model(-test_points)
    
    # 计算对称误差
    symmetry_error = torch.mean(torch.abs(sdf_pos + sdf_neg)).item()
    
    # 测试连续性：相邻点的SDF值应该接近
    base_points = torch.randn(50, 3).to(device)
    offsets = torch.randn(50, 3).to(device) * 0.01  # 小偏移
    
    with torch.no_grad():
        sdf_base = model(base_points)
        sdf_offset = model(base_points + offsets)
    
    continuity_error = torch.mean(torch.abs(sdf_base - sdf_offset)).item()
    
    print(f"  对称性误差 (f(x)+f(-x)): {symmetry_error:.6f}")
    print(f"  连续性误差 (相邻点): {continuity_error:.6f}")
    
    return {
        'symmetry_error': symmetry_error,
        'continuity_error': continuity_error
    }

def save_results(inference_results, point_results, consistency_results, checkpoint):
    """保存结果到文件"""
    output_dir = 'inference_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存JSON结果
    results_dict = {
        'model_info': {
            'best_val_loss': checkpoint.get('val_loss', 'unknown'),
            'epoch': checkpoint.get('epoch', 'unknown'),
            'total_params': sum(p.numel() for p in SimpleSDFModel().parameters()),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        },
        'inference_performance': inference_results,
        'specific_points': point_results,
        'sdf_consistency': consistency_results,
        'timestamp': np.datetime64('now').astype(str)
    }
    
    json_path = os.path.join(output_dir, 'inference_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    # 保存文本总结
    txt_path = os.path.join(output_dir, 'inference_summary.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("最佳模型推理测试结果\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("📊 模型信息:\n")
        f.write(f"   最佳验证损失: {checkpoint.get('val_loss', 'unknown')}\n")
        f.write(f"   训练轮数: {checkpoint.get('epoch', 'unknown')}\n")
        f.write(f"   总参数: {results_dict['model_info']['total_params']:,}\n")
        f.write(f"   推理设备: {results_dict['model_info']['device']}\n\n")
        
        f.write("⚡ 推理性能:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'批次大小':<10} {'推理时间(ms)':<15} {'每点时间(ms)':<15} {'SDF均值':<12}\n")
        f.write("-" * 70 + "\n")
        for r in inference_results:
            f.write(f"{r['batch_size']:<10} {r['inference_time_ms']:<15.2f} {r['time_per_point_ms']:<15.4f} {r['sdf_mean']:<12.6f}\n")
        
        f.write("\n🎯 特定点测试:\n")
        f.write("-" * 70 + "\n")
        for r in point_results:
            f.write(f"{r['name']:<15} [{r['point'][0]:.3f}, {r['point'][1]:.3f}, {r['point'][2]:.3f}] -> SDF = {r['sdf']:.6f}\n")
        
        f.write("\n🔍 SDF一致性分析:\n")
        f.write(f"   对称性误差: {consistency_results['symmetry_error']:.6f}\n")
        f.write(f"   连续性误差: {consistency_results['continuity_error']:.6f}\n")
    
    print(f"\n✅ 结果保存到:")
    print(f"   JSON文件: {json_path}")
    print(f"   文本总结: {txt_path}")
    
    return json_path, txt_path

def main():
    """主函数"""
    print("=" * 60)
    print("🎯 最佳模型推理测试 (无GUI版本)")
    print("=" * 60)
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"📱 使用设备: {device}")
    if device == 'cuda':
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 1. 加载最佳模型
    result = load_best_model()
    if result is None:
        print("❌ 无法加载模型，退出")
        return
    
    model, checkpoint = result
    
    # 2. 测试推理性能
    inference_results = test_inference_performance(model, device)
    
    # 3. 测试特定点
    point_results = test_specific_points(model, device)
    
    # 4. 分析SDF一致性
    consistency_results = analyze_sdf_consistency(model, device)
    
    # 5. 保存结果
    json_path, txt_path = save_results(inference_results, point_results, consistency_results, checkpoint)
    
    print(f"\n🎉 推理测试完成!")
    print(f"📊 关键发现:")
    print(f"   - 模型推理速度: {inference_results[-1]['time_per_point_ms']:.4f} ms/点 (10000点批次)")
    print(f"   - SDF范围: [{inference_results[0]['sdf_min']:.3f}, {inference_results[0]['sdf_max']:.3f}]")
    print(f"   - 对称性良好: 误差 {consistency_results['symmetry_error']:.6f}")
    
    # 显示总结文件内容
    print(f"\n📄 推理总结:")
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f.readlines()[:20]:  # 显示前20行
            print(line.rstrip())

if __name__ == "__main__":
    main()