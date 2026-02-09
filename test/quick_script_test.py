#!/usr/bin/env python3
"""
快速脚本测试 - 验证修复后的训练脚本可以运行
"""

import sys
import os
import argparse

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_script_import():
    """测试脚本导入"""
    print("测试脚本导入...")
    try:
        from final_multi_sequence_training_fixed_fixed import main
        print("✅ 脚本导入成功")
        return True
    except Exception as e:
        print(f"❌ 脚本导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_argument_parsing():
    """测试参数解析"""
    print("\n测试参数解析...")
    
    # 模拟命令行参数
    test_args = [
        '--test-only',
        '--epochs', '1',
        '--batch-size', '2',
        '--device', 'cpu'  # 使用CPU避免GPU问题
    ]
    
    try:
        import argparse
        
        # 创建解析器
        parser = argparse.ArgumentParser(description='测试参数解析')
        
        # 添加参数（从原脚本复制）
        parser.add_argument('--test-only', action='store_true', help='仅测试模式')
        parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
        parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
        parser.add_argument('--device', type=str, default='cuda', help='设备 (cpu/cuda)')
        parser.add_argument('--data-dir', type=str, default='data', help='数据目录')
        parser.add_argument('--num-workers', type=int, default=4, help='数据加载工作线程数')
        parser.add_argument('--learning-rate', type=float, default=0.001, help='学习率')
        parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
        parser.add_argument('--save-interval', type=int, default=5, help='保存间隔')
        parser.add_argument('--validation-split', type=float, default=0.2, help='验证集比例')
        parser.add_argument('--seed', type=int, default=42, help='随机种子')
        
        # 解析参数
        args = parser.parse_args(test_args)
        
        print(f"✅ 参数解析成功")
        print(f"   test-only: {args.test_only}")
        print(f"   epochs: {args.epochs}")
        print(f"   batch-size: {args.batch_size}")
        print(f"   device: {args.device}")
        
        return True
        
    except Exception as e:
        print(f"❌ 参数解析失败: {e}")
        return False

def test_minimal_execution():
    """测试最小化执行"""
    print("\n测试最小化执行...")
    
    # 创建一个简化的测试环境
    import torch
    from final_multi_sequence_training_fixed_fixed import (
        SDF3DModel, 
        prepare_input_data,
        correct_tsdf_dimensions
    )
    
    try:
        # 使用CPU设备
        device = torch.device('cpu')
        print(f"使用设备: {device}")
        
        # 创建简化模型
        model = SDF3DModel().to(device)
        print(f"模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 创建测试数据
        batch_size = 2
        rgb_images = torch.randn(batch_size, 3, 3, 32, 32)
        tsdf_raw = torch.randn(batch_size, 1, 32, 32, 24)
        
        # 测试数据处理管道
        tsdf_corrected = correct_tsdf_dimensions(tsdf_raw)
        input_3d = prepare_input_data(rgb_images, tsdf_corrected, frame_idx=0, device=device)
        
        # 前向传播
        with torch.no_grad():
            output = model(input_3d)
        
        print(f"✅ 最小化执行测试通过")
        print(f"  输入形状: {input_3d.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  输出范围: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 最小化执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_device_consistency():
    """测试设备一致性"""
    print("\n测试设备一致性...")
    
    from device_consistency_utils import DeviceConsistency
    import torch
    
    # 测试数据
    test_data = {
        'tensor_cpu': torch.randn(2, 3),
        'list_data': [torch.randn(1, 2), torch.randn(2, 3)],
        'nested': {
            'tensor': torch.randn(3, 4),
            'scalar': 42
        }
    }
    
    # 测试CPU设备一致性
    device_cpu = torch.device('cpu')
    fixed_cpu = DeviceConsistency.ensure_consistency(test_data, device_cpu)
    
    if DeviceConsistency.check_consistency(fixed_cpu, device_cpu):
        print("✅ CPU设备一致性测试通过")
    else:
        print("❌ CPU设备一致性测试失败")
        return False
    
    # 如果CUDA可用，测试CUDA设备一致性
    if torch.cuda.is_available():
        device_cuda = torch.device('cuda:0')
        fixed_cuda = DeviceConsistency.ensure_consistency(test_data, device_cuda)
        
        if DeviceConsistency.check_consistency(fixed_cuda, device_cuda):
            print("✅ CUDA设备一致性测试通过")
        else:
            print("❌ CUDA设备一致性测试失败")
            return False
    
    return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("快速脚本测试套件")
    print("=" * 60)
    
    tests = [
        test_script_import,
        test_argument_parsing,
        test_device_consistency,
        test_minimal_execution
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            print(f"❌ {test_func.__name__} 异常: {e}")
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("✅ 所有快速测试通过!")
        print("\n建议下一步: 运行完整测试命令:")
        print("  conda activate former3d")
        print("  python final_multi_sequence_training_fixed_fixed.py --test-only --epochs 1 --batch-size 2")
        return True
    else:
        print("❌ 有测试失败，需要进一步调试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)