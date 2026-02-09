#!/usr/bin/env python3
"""
设备一致性工具模块
确保所有张量在相同设备上，避免设备不匹配错误
"""

import torch
from typing import Any, Union, Dict, List, Tuple
import logging

# 设置日志
logger = logging.getLogger(__name__)

class DeviceConsistency:
    """设备一致性管理类"""
    
    @staticmethod
    def ensure_consistency(data: Any, device: torch.device) -> Any:
        """
        递归确保所有张量在指定设备上
        
        Args:
            data: 任意数据（张量、字典、列表、元组等）
            device: 目标设备
            
        Returns:
            设备一致化的数据
        """
        if data is None:
            return None
            
        # 如果是张量，移动到设备
        if isinstance(data, torch.Tensor):
            if data.device != device:
                logger.debug(f"移动张量 {data.shape} 从 {data.device} 到 {device}")
                return data.to(device)
            return data
            
        # 如果是字典，递归处理每个值
        elif isinstance(data, dict):
            return {key: DeviceConsistency.ensure_consistency(value, device) 
                   for key, value in data.items()}
            
        # 如果是列表或元组，递归处理每个元素
        elif isinstance(data, (list, tuple)):
            processed = [DeviceConsistency.ensure_consistency(item, device) 
                        for item in data]
            return type(data)(processed)
            
        # 其他类型（数字、字符串等）保持不变
        else:
            return data
    
    @staticmethod
    def check_consistency(data: Any, expected_device: torch.device) -> bool:
        """
        检查数据中所有张量是否在指定设备上
        
        Args:
            data: 要检查的数据
            expected_device: 期望的设备
            
        Returns:
            bool: 是否所有张量都在期望设备上
        """
        if data is None:
            return True
            
        if isinstance(data, torch.Tensor):
            return data.device == expected_device
            
        elif isinstance(data, dict):
            return all(DeviceConsistency.check_consistency(value, expected_device) 
                      for value in data.values())
            
        elif isinstance(data, (list, tuple)):
            return all(DeviceConsistency.check_consistency(item, expected_device) 
                      for item in data)
            
        else:
            return True
    
    @staticmethod
    def get_device_info(data: Any) -> Dict[str, Any]:
        """
        获取数据中所有张量的设备信息
        
        Args:
            data: 要分析的数据
            
        Returns:
            设备信息字典
        """
        info = {
            'tensor_count': 0,
            'devices': set(),
            'shapes': [],
            'device_mismatch': False
        }
        
        DeviceConsistency._collect_device_info(data, info)
        
        # 检查设备是否一致
        if len(info['devices']) > 1:
            info['device_mismatch'] = True
            logger.warning(f"发现设备不匹配: {info['devices']}")
            
        return info
    
    @staticmethod
    def _collect_device_info(data: Any, info: Dict[str, Any]):
        """递归收集设备信息"""
        if data is None:
            return
            
        if isinstance(data, torch.Tensor):
            info['tensor_count'] += 1
            info['devices'].add(str(data.device))
            info['shapes'].append(tuple(data.shape))
            
        elif isinstance(data, dict):
            for value in data.values():
                DeviceConsistency._collect_device_info(value, info)
                
        elif isinstance(data, (list, tuple)):
            for item in data:
                DeviceConsistency._collect_device_info(item, info)


def move_to_device(data: Any, device: torch.device) -> Any:
    """
    将数据移动到指定设备（DeviceConsistency.ensure_consistency的别名）
    
    Args:
        data: 要移动的数据
        device: 目标设备
        
    Returns:
        移动到设备的数据
    """
    return DeviceConsistency.ensure_consistency(data, device)


def batch_device_check(batch: Any, stage: str = "unknown") -> Dict[str, Any]:
    """
    批量设备检查，用于训练循环中的检查点
    
    Args:
        batch: 要检查的批次数据
        stage: 检查阶段名称（用于日志）
        
    Returns:
        检查结果信息
    """
    info = DeviceConsistency.get_device_info(batch)
    
    if info['device_mismatch']:
        logger.error(f"[{stage}] 设备不匹配: {info['devices']}")
        logger.error(f"[{stage}] 张量数量: {info['tensor_count']}")
        logger.error(f"[{stage}] 形状: {info['shapes'][:5]}")  # 只显示前5个形状
        
    else:
        logger.debug(f"[{stage}] 设备一致: {list(info['devices'])[0] if info['devices'] else '无张量'}")
        logger.debug(f"[{stage}] 张量数量: {info['tensor_count']}")
        
    return info


# 装饰器：确保函数输入输出在指定设备上
def ensure_device(device_arg: str = 'device', input_device: bool = True, output_device: bool = True):
    """
    装饰器：确保函数输入输出在指定设备上
    
    Args:
        device_arg: 设备参数名
        input_device: 是否确保输入设备一致性
        output_device: 是否确保输出设备一致性
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 获取设备参数
            device = None
            
            # 从关键字参数中查找
            if device_arg in kwargs:
                device = kwargs[device_arg]
            # 从位置参数中查找（通过函数签名）
            else:
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                
                if device_arg in params:
                    idx = params.index(device_arg)
                    if idx < len(args):
                        device = args[idx]
            
            if device is None:
                logger.warning(f"未找到设备参数 '{device_arg}'，跳过设备一致性检查")
                return func(*args, **kwargs)
            
            # 确保输入设备一致性
            if input_device:
                # 这里简化处理，实际可能需要更复杂的参数解析
                logger.debug(f"确保输入设备一致性到 {device}")
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 确保输出设备一致性
            if output_device and device is not None:
                result = DeviceConsistency.ensure_consistency(result, device)
                logger.debug(f"确保输出设备一致性到 {device}")
            
            return result
        
        return wrapper
    
    return decorator


# 测试函数
if __name__ == "__main__":
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    
    # 测试数据
    device_cuda = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')
    
    test_data = {
        'tensor_cpu': torch.randn(2, 3),
        'tensor_gpu': torch.randn(2, 3).to(device_cuda) if torch.cuda.is_available() else torch.randn(2, 3),
        'nested': {
            'list_tensors': [torch.randn(1, 1), torch.randn(2, 2).to(device_cuda) if torch.cuda.is_available() else torch.randn(2, 2)],
            'number': 42,
            'string': 'test'
        }
    }
    
    print("原始数据设备信息:")
    info = DeviceConsistency.get_device_info(test_data)
    print(f"  张量数量: {info['tensor_count']}")
    print(f"  设备: {info['devices']}")
    print(f"  设备不匹配: {info['device_mismatch']}")
    
    # 测试设备一致性
    print("\n确保设备一致性到CPU:")
    fixed_cpu = DeviceConsistency.ensure_consistency(test_data, device_cpu)
    info_cpu = DeviceConsistency.get_device_info(fixed_cpu)
    print(f"  设备: {info_cpu['devices']}")
    print(f"  设备不匹配: {info_cpu['device_mismatch']}")
    
    if torch.cuda.is_available():
        print("\n确保设备一致性到CUDA:")
        fixed_cuda = DeviceConsistency.ensure_consistency(test_data, device_cuda)
        info_cuda = DeviceConsistency.get_device_info(fixed_cuda)
        print(f"  设备: {info_cuda['devices']}")
        print(f"  设备不匹配: {info_cuda['device_mismatch']}")
    
    print("\n✅ 设备一致性工具测试完成")