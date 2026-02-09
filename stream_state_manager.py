#!/usr/bin/env python3
"""
流式状态管理器 - 管理StreamSDFFormerIntegrated的训练状态
"""

import torch
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class StreamStateManager:
    """
    管理流式训练状态
    
    在流式训练中，模型需要维护序列间的状态信息。
    这个类负责：
    1. 存储和检索序列状态
    2. 在序列边界重置状态
    3. 管理状态的生命周期
    """
    
    def __init__(self, device: torch.device = None):
        """
        初始化状态管理器
        
        Args:
            device: 状态存储的设备（默认与模型相同）
        """
        self.device = device
        self.state_cache: Dict[str, Dict[str, Any]] = {}
        self.current_sequence: Optional[str] = None
        self.current_state: Optional[Dict[str, Any]] = None
        
        logger.info(f"初始化StreamStateManager，设备: {device}")
    
    def reset(self, sequence_id: str = None) -> None:
        """
        重置状态
        
        Args:
            sequence_id: 序列ID，如果提供则只重置该序列的状态
        """
        if sequence_id is None:
            # 重置所有状态
            self.state_cache.clear()
            self.current_sequence = None
            self.current_state = None
            logger.debug("重置所有序列状态")
        else:
            # 重置特定序列的状态
            if sequence_id in self.state_cache:
                del self.state_cache[sequence_id]
                logger.debug(f"重置序列状态: {sequence_id}")
            
            # 如果当前序列被重置，清除当前状态
            if self.current_sequence == sequence_id:
                self.current_sequence = None
                self.current_state = None
    
    def get_state(self, sequence_id: str = None) -> Optional[Dict[str, Any]]:
        """
        获取序列状态
        
        Args:
            sequence_id: 序列ID，如果为None则返回当前状态
            
        Returns:
            序列状态字典，如果不存在则返回None
        """
        if sequence_id is None:
            # 返回当前状态
            return self.current_state
        else:
            # 返回指定序列的状态
            return self.state_cache.get(sequence_id)
    
    def update_state(self, 
                    new_state: Dict[str, Any], 
                    sequence_id: str, 
                    frame_idx: int = 0,
                    reset_state: bool = False) -> Dict[str, Any]:
        """
        更新序列状态
        
        Args:
            new_state: 新的状态字典
            sequence_id: 序列ID
            frame_idx: 帧索引（用于日志）
            reset_state: 是否重置状态（序列开始）
            
        Returns:
            更新后的状态
        """
        # 如果需要重置状态或序列改变
        if reset_state or self.current_sequence != sequence_id:
            self.reset(sequence_id)
            self.current_sequence = sequence_id
            logger.debug(f"序列 {sequence_id} 开始，帧 {frame_idx}")
        
        # 确保状态在正确设备上
        if self.device is not None:
            new_state = self._move_to_device(new_state, self.device)
        
        # 更新状态
        self.current_state = new_state
        self.state_cache[sequence_id] = new_state
        
        return self.current_state
    
    def _move_to_device(self, state: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        将状态字典中的所有张量移动到指定设备
        
        Args:
            state: 状态字典
            device: 目标设备
            
        Returns:
            移动后的状态字典
        """
        if not isinstance(state, dict):
            return state
        
        moved_state = {}
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                moved_state[key] = value.to(device)
            elif isinstance(value, dict):
                moved_state[key] = self._move_to_device(value, device)
            elif isinstance(value, (list, tuple)):
                # 递归处理列表/元组中的张量
                if isinstance(value, list):
                    moved_state[key] = [self._move_to_device(v, device) if isinstance(v, dict) else v.to(device) if isinstance(v, torch.Tensor) else v for v in value]
                else:  # tuple
                    moved_state[key] = tuple(self._move_to_device(v, device) if isinstance(v, dict) else v.to(device) if isinstance(v, torch.Tensor) else v for v in value)
            else:
                moved_state[key] = value
        
        return moved_state
    
    def save_state(self, filepath: str) -> None:
        """
        保存状态到文件
        
        Args:
            filepath: 文件路径
        """
        try:
            torch.save({
                'state_cache': self.state_cache,
                'current_sequence': self.current_sequence,
                'current_state': self.current_state
            }, filepath)
            logger.info(f"状态保存到: {filepath}")
        except Exception as e:
            logger.error(f"保存状态失败: {e}")
    
    def load_state(self, filepath: str) -> None:
        """
        从文件加载状态
        
        Args:
            filepath: 文件路径
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.state_cache = checkpoint['state_cache']
            self.current_sequence = checkpoint['current_sequence']
            self.current_state = checkpoint['current_state']
            logger.info(f"从 {filepath} 加载状态")
        except Exception as e:
            logger.error(f"加载状态失败: {e}")
    
    def clear_cache(self) -> None:
        """清除所有缓存的状态"""
        self.state_cache.clear()
        self.current_sequence = None
        self.current_state = None
        logger.info("清除所有缓存状态")
    
    def get_cache_size(self) -> int:
        """获取缓存中的序列数量"""
        return len(self.state_cache)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        获取内存使用情况
        
        Returns:
            包含内存使用信息的字典
        """
        memory_info = {
            'num_sequences': len(self.state_cache),
            'total_tensors': 0,
            'total_elements': 0,
            'estimated_memory_mb': 0.0
        }
        
        def count_tensors(obj):
            if isinstance(obj, torch.Tensor):
                memory_info['total_tensors'] += 1
                memory_info['total_elements'] += obj.numel()
                # 估计内存使用（假设float32）
                memory_info['estimated_memory_mb'] += obj.numel() * 4 / (1024 ** 2)
            elif isinstance(obj, dict):
                for v in obj.values():
                    count_tensors(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    count_tensors(v)
        
        for state in self.state_cache.values():
            count_tensors(state)
        
        if self.current_state is not None:
            count_tensors(self.current_state)
        
        return memory_info


# 测试函数
def test_stream_state_manager():
    """测试流式状态管理器"""
    print("="*80)
    print("测试StreamStateManager")
    print("="*80)
    
    # 创建管理器
    device = torch.device("cpu")
    manager = StreamStateManager(device=device)
    
    # 测试1: 基本功能
    print("\n1. 测试基本功能...")
    
    # 创建模拟状态
    state1 = {
        'features': torch.randn(1, 32, 16, 16, 16),
        'mask': torch.ones(1, 1, 16, 16, 16, dtype=torch.bool),
        'metadata': {'frame_idx': 0, 'processed': True}
    }
    
    # 更新状态
    updated_state = manager.update_state(state1, "seq_001", frame_idx=0, reset_state=True)
    print(f"   更新状态成功: {updated_state is not None}")
    
    # 获取状态
    retrieved_state = manager.get_state("seq_001")
    print(f"   获取状态成功: {retrieved_state is not None}")
    
    # 测试2: 序列切换
    print("\n2. 测试序列切换...")
    
    state2 = {
        'features': torch.randn(1, 32, 16, 16, 16),
        'mask': torch.ones(1, 1, 16, 16, 16, dtype=torch.bool),
        'metadata': {'frame_idx': 0, 'processed': True}
    }
    
    # 切换到新序列
    updated_state2 = manager.update_state(state2, "seq_002", frame_idx=0, reset_state=True)
    print(f"   切换到新序列成功: {updated_state2 is not None}")
    print(f"   当前序列: {manager.current_sequence}")
    
    # 测试3: 内存使用
    print("\n3. 测试内存使用...")
    memory_info = manager.get_memory_usage()
    print(f"   序列数量: {memory_info['num_sequences']}")
    print(f"   张量总数: {memory_info['total_tensors']}")
    print(f"   元素总数: {memory_info['total_elements']:,}")
    print(f"   估计内存: {memory_info['estimated_memory_mb']:.2f} MB")
    
    # 测试4: 重置状态
    print("\n4. 测试重置状态...")
    manager.reset("seq_001")
    print(f"   重置seq_001后缓存大小: {manager.get_cache_size()}")
    
    manager.reset()
    print(f"   重置所有后缓存大小: {manager.get_cache_size()}")
    
    print("\n" + "="*80)
    print("StreamStateManager测试完成!")
    print("="*80)


if __name__ == "__main__":
    test_stream_state_manager()