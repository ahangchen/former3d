#!/usr/bin/env python3
"""
内存管理器
定期清理CUDA缓存和垃圾回收，防止显存泄漏
"""

import torch
import gc
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """内存管理器

    负责定期清理CUDA缓存和垃圾回收，防止显存泄漏
    """

    def __init__(self, cleanup_frequency: int = 10, memory_threshold_gb: float = 8.0):
        """
        初始化内存管理器

        Args:
            cleanup_frequency: 清理频率（每N步清理一次）
            memory_threshold_gb: 显存阈值（超过此值自动清理）
        """
        self.cleanup_frequency = cleanup_frequency
        self.memory_threshold_gb = memory_threshold_gb
        self.step_counter = 0

        logger.info(
            f"初始化内存管理器 - 清理频率: {cleanup_frequency}步, "
            f"显存阈值: {memory_threshold_gb}GB"
        )

    def step(self) -> None:
        """
        每步调用，根据需要清理内存

        根据cleanup_frequency定期执行清理
        """
        self.step_counter += 1

        if self.step_counter % self.cleanup_frequency == 0:
            self.cleanup()

    def cleanup(self, verbose: bool = False) -> None:
        """
        执行内存清理

        清理CUDA缓存和垃圾回收

        Args:
            verbose: 是否打印详细信息
        """
        # 1. 清理CUDA缓存
        if torch.cuda.is_available():
            before_reserved = torch.cuda.memory_reserved() / 1024**3
            before_allocated = torch.cuda.memory_allocated() / 1024**3

            torch.cuda.empty_cache()

            after_reserved = torch.cuda.memory_reserved() / 1024**3
            after_allocated = torch.cuda.memory_allocated() / 1024**3

            reserved_freed = before_reserved - after_reserved
            allocated_freed = before_allocated - after_allocated

            if verbose:
                logger.info(
                    f"CUDA缓存清理 - "
                    f"已保留: {before_reserved:.2f}GB → {after_reserved:.2f}GB (释放{reserved_freed:.2f}GB), "
                    f"已分配: {before_allocated:.2f}GB → {after_allocated:.2f}GB (释放{allocated_freed:.2f}GB)"
                )
            elif reserved_freed > 0.1 or allocated_freed > 0.1:
                # 只有释放了较多内存时才记录
                logger.info(
                    f"CUDA缓存清理 - "
                    f"释放保留: {reserved_freed:.2f}GB, "
                    f"释放分配: {allocated_freed:.2f}GB"
                )
        else:
            if verbose:
                logger.info("CUDA不可用，跳过CUDA缓存清理")

        # 2. 强制垃圾回收
        collected = gc.collect()

        if verbose and collected > 0:
            logger.info(f"垃圾回收 - 回收了{collected}个对象")
        elif collected > 100:
            logger.info(f"垃圾回收 - 回收了{collected}个对象")

        # 3. 记录当前显存使用情况
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3

            if verbose:
                logger.info(
                    f"当前显存 - 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB"
                )

    def cleanup_if_needed(self, threshold_gb: Optional[float] = None) -> bool:
        """
        如果显存使用超过阈值则清理

        Args:
            threshold_gb: 覆盖默认的阈值

        Returns:
            是否执行了清理
        """
        threshold = threshold_gb if threshold_gb is not None else self.memory_threshold_gb

        if not torch.cuda.is_available():
            return False

        allocated = torch.cuda.memory_allocated() / 1024**3

        if allocated > threshold:
            logger.warning(
                f"显存使用过高 ({allocated:.2f}GB > {threshold:.2f}GB)，执行清理..."
            )
            self.cleanup(verbose=True)
            return True

        return False

    def force_cleanup(self, verbose: bool = True) -> None:
        """
        强制执行内存清理（用于显存碎片化严重的情况）

        执行更激进的清理，包括重置显存分配器状态

        Args:
            verbose: 是否打印详细信息
        """
        logger.info("执行强制显存清理...")

        # 1. 清理CUDA缓存
        if torch.cuda.is_available():
            before_reserved = torch.cuda.memory_reserved() / 1024**3
            before_allocated = torch.cuda.memory_allocated() / 1024**3

            # 多次调用确保完全清理
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

            after_reserved = torch.cuda.memory_reserved() / 1024**3
            after_allocated = torch.cuda.memory_allocated() / 1024**3

            reserved_freed = before_reserved - after_reserved
            allocated_freed = before_allocated - after_allocated

            logger.info(
                f"强制CUDA缓存清理 - "
                f"已保留: {before_reserved:.2f}GB → {after_reserved:.2f}GB (释放{reserved_freed:.2f}GB), "
                f"已分配: {before_allocated:.2f}GB → {after_allocated:.2f}GB (释放{allocated_freed:.2f}GB)"
            )
        else:
            if verbose:
                logger.info("CUDA不可用，跳过CUDA缓存清理")

        # 2. 强制垃圾回收（多次执行）
        for _ in range(3):
            collected = gc.collect()
            if collected > 0 and verbose:
                logger.info(f"强制垃圾回收 - 回收了{collected}个对象")

        # 3. 重置峰值显存统计
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        logger.info("强制显存清理完成")

    def get_memory_info(self) -> dict:
        """
        获取当前显存信息

        Returns:
            显存信息字典
        """
        if not torch.cuda.is_available():
            return {"cuda_available": False}

        return {
            "cuda_available": True,
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        }

    def reset_peak_stats(self) -> None:
        """重置峰值显存统计"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            logger.debug("峰值显存统计已重置")

    def __repr__(self) -> str:
        return (
            f"MemoryManager(cleanup_frequency={self.cleanup_frequency}, "
            f"memory_threshold_gb={self.memory_threshold_gb})"
        )
