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
    管理流式训练状态（优化版本，防止内存泄漏）

    在流式训练中，模型需要维护序列间的状态信息。
    这个类负责：
    1. 存储和检索序列状态
    2. 在序列边界重置状态
    3. 管理状态的生命周期（防止内存泄漏）
    4. 使用LRU策略自动清理旧状态
    """

    def __init__(self, device: torch.device = None, max_cached_states: int = 5):
        """
        初始化状态管理器

        Args:
            device: 状态存储的设备（默认与模型相同）
            max_cached_states: 最大缓存的序列状态数量（LRU策略）
        """
        self.device = device
        self.state_cache: Dict[str, Dict[str, Any]] = {}
        self.current_sequence: Optional[str] = None
        self.current_state: Optional[Dict[str, Any]] = None
        self.max_cached_states = max_cached_states
        self.sequence_access_order = []  # 记录序列访问顺序（用于LRU）

        logger.info(f"初始化StreamStateManager，设备: {device}，最大缓存状态: {max_cached_states}")
    
    def reset(self, sequence_id: str = None) -> None:
        """
        重置状态（优化版本，防止内存泄漏）

        Args:
            sequence_id: 序列ID，如果提供则只重置该序列的状态
        """
        if sequence_id is None:
            # 重置所有状态 - 先释放所有张量
            for seq_id, state in self.state_cache.items():
                logger.debug(f"释放序列 {seq_id} 的状态")
                self._release_state_tensors(state)
            self.state_cache.clear()
            self.current_sequence = None
            self.current_state = None
            logger.debug("重置所有序列状态")
        else:
            # 重置特定序列的状态 - 先释放张量
            if sequence_id in self.state_cache:
                old_state = self.state_cache[sequence_id]
                logger.debug(f"释放序列 {sequence_id} 的状态")
                self._release_state_tensors(old_state)
                del self.state_cache[sequence_id]

            # 如果当前序列被重置，清除当前状态
            if self.current_sequence == sequence_id:
                self._release_state_tensors(self.current_state) if self.current_state is not None else None
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
        更新序列状态（优化版本，防止内存泄漏）

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
        else:
            # 如果序列ID已存在，先释放旧状态
            if sequence_id in self.state_cache:
                old_state = self.state_cache[sequence_id]
                logger.debug(f"释放序列 {sequence_id} 的旧状态")
                # 显式释放旧状态中的张量
                self._release_state_tensors(old_state)
                del old_state

        # 确保状态在正确设备上
        if self.device is not None:
            new_state = self._move_to_device(new_state, self.device)

        # 更新状态
        self.current_state = new_state
        self.state_cache[sequence_id] = new_state

        # 更新访问顺序（LRU机制）
        if sequence_id in self.sequence_access_order:
            self.sequence_access_order.remove(sequence_id)
        self.sequence_access_order.append(sequence_id)

        # 检查是否需要清理旧状态（LRU）
        if len(self.state_cache) > self.max_cached_states:
            # 移除最久未访问的状态
            excess_count = len(self.state_cache) - self.max_cached_states
            for _ in range(excess_count):
                if self.sequence_access_order:
                    oldest_seq_id = self.sequence_access_order.pop(0)
                    if oldest_seq_id in self.state_cache:
                        logger.debug(f"LRU清理: 移除最久未访问的状态 {oldest_seq_id}")
                        old_state = self.state_cache[oldest_seq_id]
                        self._release_state_tensors(old_state)
                        del self.state_cache[oldest_seq_id]

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

    def _release_state_tensors(self, state: Dict[str, Any]) -> None:
        """
        递归释放状态字典中的张量（防止内存泄漏）

        将张量移动到CPU并删除引用，让Python垃圾回收器回收内存

        Args:
            state: 状态字典
        """
        if not isinstance(state, dict):
            return

        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                # 将张量移动到CPU以释放GPU显存
                value = value.cpu()
                # 显式删除引用
                del value
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                self._release_state_tensors(value)
            elif isinstance(value, (list, tuple)):
                # 处理列表/元组中的张量
                for v in value:
                    if isinstance(v, torch.Tensor):
                        v = v.cpu()
                        del v
                    elif isinstance(v, dict):
                        self._release_state_tensors(v)

    def clear_old_states(self, keep_last_n: int = 3) -> None:
        """
        清理旧的状态，只保留最近的n个序列状态（防止内存泄漏）

        Args:
            keep_last_n: 要保留的序列数量
        """
        if len(self.state_cache) <= keep_last_n:
            logger.debug(f"状态数量({len(self.state_cache)}) <= 保留数量({keep_last_n})，无需清理")
            return

        # 根据访问顺序确定要保留哪些状态（LRU策略）
        sequences_to_keep = set(self.sequence_access_order[-keep_last_n:]) if self.sequence_access_order else set()

        sequences_to_remove = []
        for seq_id in list(self.state_cache.keys()):
            if seq_id not in sequences_to_keep and seq_id != self.current_sequence:
                sequences_to_remove.append(seq_id)

        removed_count = 0
        for seq_id in sequences_to_remove:
            # 释放旧状态中的张量
            old_state = self.state_cache[seq_id]
            self._release_state_tensors(old_state)
            # 从缓存中删除
            del self.state_cache[seq_id]
            # 从访问顺序中删除
            if seq_id in self.sequence_access_order:
                self.sequence_access_order.remove(seq_id)
            removed_count += 1

        logger.info(f"清理了 {removed_count} 个旧状态，保留 {len(self.state_cache)} 个状态")

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