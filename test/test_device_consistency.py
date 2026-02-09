#!/usr/bin/env python3
"""
设备一致性工具测试
"""

import sys
import os
import torch
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from device_consistency_utils import DeviceConsistency, move_to_device, batch_device_check

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_basic_tensor():
    """测试基本张量设备移动"""
    print("测试1: 基本张量设备移动")
    
    # 创建测试张量
    tensor_cpu = torch.randn(2, 3, 4)
    device_cuda = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 测试CPU到CPU
    result = DeviceConsistency.ensure_consistency(tensor_cpu, torch.device('cpu'))
    assert result.device == torch.device('cpu'), "CPU到CPU设备移动失败"
    print("  ✅ CPU到CPU测试通过")
    
    # 测试CPU到CUDA（如果可用）
    if torch.cuda.is_available():
        result = DeviceConsistency.ensure_consistency(tensor_cpu, device_cuda)
        assert result.device == device_cuda, "CPU到CUDA设备移动失败"
        print("  ✅ CPU到CUDA测试通过")
    
    print("  ✅ 基本张量测试完成")

def test_nested_structure():
    """测试嵌套结构设备移动"""
    print("\n测试2: 嵌套结构设备移动")
    
    device_target = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 创建嵌套测试数据
    test_data = {
        'tensor1': torch.randn(2, 3),
        'tensor2': torch.randn(3, 4),
        'list_data': [
            torch.randn(1, 2),
            torch.randn(2, 3)
        ],
        'tuple_data': (
            torch.randn(1, 1),
            {'nested_tensor': torch.randn(2, 2)}
        ),
        'scalar': 42,
        'string': 'test',
        'none_value': None
    }
    
    # 确保设备一致性
    fixed_data = DeviceConsistency.ensure_consistency(test_data, device_target)
    
    # 检查所有张量都在正确设备上
    def check_device(data, expected_device):
        if isinstance(data, torch.Tensor):
            assert data.device == expected_device, f"张量设备不匹配: {data.device} != {expected_device}"
        elif isinstance(data, dict):
            for value in data.values():
                check_device(value, expected_device)
        elif isinstance(data, (list, tuple)):
            for item in data:
                check_device(item, expected_device)
    
    check_device(fixed_data, device_target)
    print("  ✅ 嵌套结构测试通过")
    
    # 测试设备一致性检查
    is_consistent = DeviceConsistency.check_consistency(fixed_data, device_target)
    assert is_consistent, "设备一致性检查失败"
    print("  ✅ 设备一致性检查测试通过")

def test_device_info():
    """测试设备信息收集"""
    print("\n测试3: 设备信息收集")
    
    # 创建混合设备数据（如果CUDA可用）
    if torch.cuda.is_available():
        mixed_data = {
            'cpu_tensor': torch.randn(2, 3),
            'cuda_tensor': torch.randn(3, 4).cuda(),
            'list_mixed': [torch.randn(1, 2), torch.randn(2, 3).cuda()]
        }
        
        info = DeviceConsistency.get_device_info(mixed_data)
        assert info['tensor_count'] == 4, f"张量数量错误: {info['tensor_count']}"
        assert info['device_mismatch'] == True, "应检测到设备不匹配"
        assert len(info['devices']) >= 2, f"设备数量不足: {info['devices']}"
        print("  ✅ 混合设备信息收集测试通过")
    
    # 创建一致设备数据
    consistent_data = {
        'tensor1': torch.randn(2, 3),
        'tensor2': torch.randn(3, 4),
        'nested': {'tensor3': torch.randn(4, 5)}
    }
    
    info = DeviceConsistency.get_device_info(consistent_data)
    assert info['tensor_count'] == 3, f"张量数量错误: {info['tensor_count']}"
    assert info['device_mismatch'] == False, "不应检测到设备不匹配"
    print("  ✅ 一致设备信息收集测试通过")

def test_batch_device_check():
    """测试批量设备检查"""
    print("\n测试4: 批量设备检查")
    
    device_target = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试批次数据
    batch_data = {
        'images': torch.randn(4, 3, 32, 32),
        'labels': torch.randint(0, 10, (4,)),
        'metadata': {
            'ids': [1, 2, 3, 4],
            'scores': torch.randn(4)
        }
    }
    
    # 移动到目标设备
    batch_data = move_to_device(batch_data, device_target)
    
    # 执行批量检查
    info = batch_device_check(batch_data, stage="测试阶段")
    
    assert info['device_mismatch'] == False, "批量检查应显示设备一致"
    assert info['tensor_count'] == 3, f"张量数量错误: {info['tensor_count']}"
    print("  ✅ 批量设备检查测试通过")

def test_move_to_device_alias():
    """测试move_to_device别名函数"""
    print("\n测试5: move_to_device别名函数测试")
    
    device_target = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    test_tensor = torch.randn(2, 3)
    result = move_to_device(test_tensor, device_target)
    
    assert result.device == device_target, "move_to_device函数失败"
    print("  ✅ move_to_device函数测试通过")

def test_performance():
    """测试性能（简单验证）"""
    print("\n测试6: 性能测试")
    
    device_target = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # 创建大量数据
    large_data = {
        f'tensor_{i}': torch.randn(10, 10) for i in range(100)
    }
    
    import time
    start_time = time.time()
    fixed_data = DeviceConsistency.ensure_consistency(large_data, device_target)
    end_time = time.time()
    
    processing_time = end_time - start_time
    print(f"  处理100个张量耗时: {processing_time:.4f}秒")
    print(f"  平均每个张量: {processing_time/100:.6f}秒")
    
    # 验证所有张量都在正确设备上
    is_consistent = DeviceConsistency.check_consistency(fixed_data, device_target)
    assert is_consistent, "性能测试后设备一致性失败"
    print("  ✅ 性能测试通过")

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("设备一致性工具测试套件")
    print("=" * 60)
    
    tests = [
        test_basic_tensor,
        test_nested_structure,
        test_device_info,
        test_batch_device_check,
        test_move_to_device_alias,
        test_performance
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ {test_func.__name__} 失败: {e}")
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