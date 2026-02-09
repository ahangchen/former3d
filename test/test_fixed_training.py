#!/usr/bin/env python3
"""
修复后训练脚本测试
"""

import sys
import os
import torch
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """测试所有必要的导入"""
    print("测试1: 导入测试")
    
    try:
        # 测试设备一致性工具导入
        from device_consistency_utils import DeviceConsistency, move_to_device
        print("  ✅ 设备一致性工具导入成功")
    except ImportError as e:
        print(f"  ❌ 设备一致性工具导入失败: {e}")
        return False
    
    try:
        # 测试数据集导入
        from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
        print("  ✅ 多序列数据集导入成功")
    except ImportError as e:
        print(f"  ❌ 多序列数据集导入失败: {e}")
        return False
    
    try:
        # 测试训练脚本导入
        from final_multi_sequence_training_fixed_fixed import (
            SDF3DModel, 
            prepare_input_data,
            correct_tsdf_dimensions
        )
        print("  ✅ 训练脚本组件导入成功")
    except ImportError as e:
        print(f"  ❌ 训练脚本组件导入失败: {e}")
        return False
    
    print("  ✅ 所有导入测试通过")
    return True

def test_device_consistency_integration():
    """测试设备一致性集成"""
    print("\n测试2: 设备一致性集成测试")
    
    from device_consistency_utils import DeviceConsistency
    
    # 创建测试设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  使用设备: {device}")
    
    # 创建测试数据
    test_batch = {
        'rgb_images': torch.randn(2, 3, 3, 32, 32),  # [batch, views, channels, H, W]
        'tsdf': torch.randn(2, 1, 32, 32, 24),       # [batch, 1, H, W, D]
        'camera_params': torch.randn(2, 3, 4),       # 相机参数
        'metadata': {
            'sequence_id': [1, 2],
            'frame_idx': [0, 1]
        }
    }
    
    # 测试设备一致性
    fixed_batch = DeviceConsistency.ensure_consistency(test_batch, device)
    
    # 检查所有张量都在正确设备上
    is_consistent = DeviceConsistency.check_consistency(fixed_batch, device)
    
    if is_consistent:
        print("  ✅ 设备一致性集成测试通过")
        return True
    else:
        print("  ❌ 设备一致性集成测试失败")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n测试3: 模型创建测试")
    
    from final_multi_sequence_training_fixed_fixed import SDF3DModel
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 创建模型
        model = SDF3DModel()
        model = model.to(device)
        
        # 检查模型参数设备
        for name, param in model.named_parameters():
            if param.device != device:
                print(f"  ❌ 模型参数 {name} 在错误设备上: {param.device}")
                return False
        
        print(f"  ✅ 模型创建成功，参数在 {device} 上")
        return True
        
    except Exception as e:
        print(f"  ❌ 模型创建失败: {e}")
        return False

def test_data_preparation():
    """测试数据准备函数"""
    print("\n测试4: 数据准备函数测试")
    
    from final_multi_sequence_training_fixed_fixed import prepare_input_data, correct_tsdf_dimensions
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 2
    rgb_images = torch.randn(batch_size, 3, 3, 32, 32)  # [batch, views, channels, H, W]
    tsdf_raw = torch.randn(batch_size, 1, 32, 32, 24)   # [batch, 1, H, W, D]
    
    # 修正TSDF维度
    tsdf_corrected = correct_tsdf_dimensions(tsdf_raw)  # [batch, 1, D, H, W]
    
    # 测试不带设备参数
    try:
        input_3d_no_device = prepare_input_data(rgb_images, tsdf_corrected, frame_idx=0)
        print("  ✅ 不带设备参数的数据准备测试通过")
    except Exception as e:
        print(f"  ❌ 不带设备参数的数据准备失败: {e}")
        return False
    
    # 测试带设备参数
    try:
        # 移动数据到设备
        rgb_images_device = rgb_images.to(device)
        tsdf_corrected_device = tsdf_corrected.to(device)
        
        input_3d_with_device = prepare_input_data(
            rgb_images_device, 
            tsdf_corrected_device, 
            frame_idx=0, 
            device=device
        )
        
        # 检查输出设备
        if input_3d_with_device.device == device:
            print(f"  ✅ 带设备参数的数据准备测试通过，输出在 {device} 上")
            return True
        else:
            print(f"  ❌ 输出在错误设备上: {input_3d_with_device.device}")
            return False
            
    except Exception as e:
        print(f"  ❌ 带设备参数的数据准备失败: {e}")
        return False

def test_single_iteration():
    """测试单次训练迭代"""
    print("\n测试5: 单次训练迭代测试")
    
    from final_multi_sequence_training_fixed_fixed import SDF3DModel, prepare_input_data, correct_tsdf_dimensions
    from device_consistency_utils import move_to_device
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 创建模型
        model = SDF3DModel().to(device)
        model.train()
        
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 创建损失函数
        criterion = torch.nn.MSELoss()
        
        # 创建模拟批次数据
        batch = {
            'rgb_images': torch.randn(2, 3, 3, 32, 32),
            'tsdf': torch.randn(2, 1, 32, 32, 24)
        }
        
        # 应用设备一致性
        batch = move_to_device(batch, device)
        
        # 获取数据
        rgb_images = batch['rgb_images']
        tsdf_raw = batch['tsdf']
        
        # 修正TSDF维度
        tsdf_corrected = correct_tsdf_dimensions(tsdf_raw)
        
        # 准备输入数据
        input_3d = prepare_input_data(rgb_images, tsdf_corrected, frame_idx=0, device=device)
        
        # 前向传播
        sdf_pred = model(input_3d)
        
        # 检查预测形状
        expected_shape = (2, 1, 24, 32, 32)  # [batch, channels, D, H, W]
        if sdf_pred.shape != expected_shape:
            print(f"  ❌ 预测形状错误: {sdf_pred.shape} != {expected_shape}")
            return False
        
        # 计算损失
        loss = criterion(sdf_pred, tsdf_corrected)
        
        # 检查损失值
        if torch.isnan(loss):
            print("  ❌ 损失值为NaN")
            return False
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 检查梯度
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        if not has_gradients:
            print("  ❌ 没有梯度")
            return False
        
        # 优化器步骤
        optimizer.step()
        
        print(f"  ✅ 单次迭代测试通过，损失: {loss.item():.6f}")
        return True
        
    except Exception as e:
        print(f"  ❌ 单次迭代测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """测试内存使用"""
    print("\n测试6: 内存使用测试")
    
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA不可用，跳过内存测试")
        return True
    
    device = torch.device('cuda:0')
    
    try:
        # 记录初始内存
        initial_memory = torch.cuda.memory_allocated(0)
        
        # 创建模型和数据
        from final_multi_sequence_training_fixed_fixed import SDF3DModel
        model = SDF3DModel().to(device)
        
        # 创建一些测试数据 - 需要5D输入 [batch, channels, D, H, W]
        test_data = torch.randn(4, 3, 24, 64, 64).to(device)  # [batch, channels, D, H, W]
        
        # 执行一些操作
        output = model(test_data)  # 直接使用5D输入
        
        # 记录最终内存
        final_memory = torch.cuda.memory_allocated(0)
        memory_used = (final_memory - initial_memory) / 1e6  # MB
        
        print(f"  ✅ 内存使用测试通过，使用了 {memory_used:.2f} MB")
        
        # 清理
        del model, test_data, output
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"  ❌ 内存测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("修复后训练脚本测试套件")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_device_consistency_integration,
        test_model_creation,
        test_data_preparation,
        test_single_iteration,
        test_memory_usage
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
            print(f"  ❌ {test_func.__name__} 异常: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("✅ 所有测试通过!")
        return True
    else:
        print("❌ 有测试失败")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)