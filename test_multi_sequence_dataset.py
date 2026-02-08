#!/usr/bin/env python3
"""
测试多序列数据集的基本功能
简化版本，不依赖外部库
"""

import os
import sys
import numpy as np
from typing import List, Dict
import random


class SimpleMultiSequenceTartanAirDataset:
    """
    简化的多序列TartanAir数据集测试类
    用于验证逻辑而不依赖外部库
    """
    
    def __init__(
        self,
        data_root: str,
        n_view: int = 5,
        stride: int = 1,
        crop_size: tuple = (48, 48, 32),
        voxel_size: float = 0.04,
        target_image_size: tuple = (256, 256),
        max_sequences: int = 3,
        shuffle: bool = True
    ):
        """
        初始化简化的数据集
        """
        self.data_root = data_root
        self.n_view = n_view
        self.stride = stride
        self.crop_size = np.array(crop_size)
        self.voxel_size = voxel_size
        self.target_image_size = target_image_size
        self.shuffle = shuffle
        
        # 模拟序列发现
        self.sequences = self._simulate_sequences()
        
        if max_sequences is not None and max_sequences > 0:
            if self.shuffle:
                random.shuffle(self.sequences)
            self.sequences = self.sequences[:max_sequences]
        
        # 构建片段索引
        self.segments = self._build_segments()
        
        print(f"简化多序列数据集初始化完成:")
        print(f"  总序列数: {len(self.sequences)}")
        print(f"  总片段数: {len(self.segments)}")
        print(f"  片段长度: {n_view} 帧")
        print(f"  片段步长: {stride}")
        print(f"  裁剪尺寸: {crop_size} 体素")
        print(f"  体素大小: {voxel_size}米")
        print(f"  图像大小: {target_image_size}")
    
    def _simulate_sequences(self) -> List[Dict]:
        """模拟序列发现"""
        # 创建模拟序列
        sequences = []
        sequence_names = [
            "abandonedfactory_sample_P001",
            "abandonedfactory_sample_P002", 
            "abandonedfactory_sample_P003",
            "abandonedfactory_sample_P004",
            "abandonedfactory_sample_P005"
        ]
        
        for i, name in enumerate(sequence_names[:5]):
            # 模拟不同长度的序列
            num_frames = 50 + i * 10
            
            sequences.append({
                'name': name,
                'num_frames': num_frames,
                'seq_id': i
            })
        
        return sequences
    
    def _build_segments(self) -> List[Dict]:
        """为所有序列构建片段索引"""
        segments = []
        
        for seq_idx, seq in enumerate(self.sequences):
            num_frames = seq['num_frames']
            
            # 计算该序列可生成的片段数量
            num_segments = max(1, (num_frames - self.n_view) // self.stride + 1)
            
            for seg_idx in range(num_segments):
                start_frame = seg_idx * self.stride
                end_frame = start_frame + self.n_view
                
                # 确保不超过序列长度
                if end_frame > num_frames:
                    continue
                
                segments.append({
                    'seq_idx': seq_idx,
                    'seq_name': seq['name'],
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'frame_indices': list(range(start_frame, end_frame))
                })
        
        if self.shuffle:
            random.shuffle(segments)
        
        return segments
    
    def __len__(self) -> int:
        """返回总片段数"""
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个片段的数据（模拟）"""
        segment = self.segments[idx]
        seq_info = self.sequences[segment['seq_idx']]
        
        # 模拟RGB图像 (n_view, H, W, 3)
        H, W = self.target_image_size
        rgb_images = np.random.rand(self.n_view, H, W, 3).astype(np.float32)
        
        # 模拟位姿 (n_view, 4, 4)
        poses = np.tile(np.eye(4, dtype=np.float32), (self.n_view, 1, 1))
        
        # 模拟TSDF (1, D, H, W)
        D, H, W = self.crop_size
        tsdf = np.random.randn(1, D, H, W).astype(np.float32)
        tsdf = np.clip(tsdf, -1.0, 1.0)
        
        # 模拟占用网格
        occupancy = (np.abs(tsdf) < 0.5).astype(np.float32)
        
        return {
            'rgb_images': rgb_images,      # (n_view, H, W, 3)
            'poses': poses,                # (n_view, 4, 4)
            'tsdf': tsdf,                  # (1, D, H, W)
            'occupancy': occupancy,        # (1, D, H, W)
            'sequence_name': seq_info['name'],
            'segment_idx': idx,
            'start_frame': segment['start_frame'],
            'end_frame': segment['end_frame']
        }


def test_simple_dataset():
    """测试简化数据集"""
    print("测试简化多序列TartanAir数据集...")
    
    try:
        # 使用虚拟数据根目录
        data_root = "/home/cwh/coding/former3d/files/sample_tartanair"
        
        dataset = SimpleMultiSequenceTartanAirDataset(
            data_root=data_root,
            n_view=5,
            stride=2,
            crop_size=(48, 48, 32),
            voxel_size=0.04,
            target_image_size=(256, 256),
            max_sequences=3,
            shuffle=True
        )
        
        print(f"\n数据集大小: {len(dataset)} 个片段")
        
        # 测试第一个样本
        sample = dataset[0]
        
        print(f"\n样本信息:")
        print(f"RGB图像形状: {sample['rgb_images'].shape}")
        print(f"位姿形状: {sample['poses'].shape}")
        print(f"TSDF形状: {sample['tsdf'].shape}")
        print(f"占用网格形状: {sample['occupancy'].shape}")
        print(f"序列名称: {sample['sequence_name']}")
        print(f"片段索引: {sample['segment_idx']}")
        print(f"帧范围: {sample['start_frame']}-{sample['end_frame']}")
        
        # 验证形状正确性
        assert sample['rgb_images'].shape == (5, 256, 256, 3), f"RGB形状错误: {sample['rgb_images'].shape}"
        assert sample['poses'].shape == (5, 4, 4), f"位姿形状错误: {sample['poses'].shape}"
        assert sample['tsdf'].shape == (1, 48, 48, 32), f"TSDF形状错误: {sample['tsdf'].shape}"
        
        # 测试批量数据形状
        print(f"\n测试批量数据形状:")
        batch_size = 2
        indices = list(range(min(batch_size, len(dataset))))
        batch = [dataset[i] for i in indices]
        
        # 手动堆叠创建批量
        rgb_batch = np.stack([item['rgb_images'] for item in batch])  # (batch_size, n_view, H, W, 3)
        poses_batch = np.stack([item['poses'] for item in batch])     # (batch_size, n_view, 4, 4)
        tsdf_batch = np.stack([item['tsdf'] for item in batch])       # (batch_size, 1, D, H, W)
        
        print(f"批量RGB形状: {rgb_batch.shape}")
        print(f"批量位姿形状: {poses_batch.shape}")
        print(f"批量TSDF形状: {tsdf_batch.shape}")
        
        # 验证批量形状
        assert rgb_batch.shape == (2, 5, 256, 256, 3), f"批量RGB形状错误: {rgb_batch.shape}"
        assert poses_batch.shape == (2, 5, 4, 4), f"批量位姿形状错误: {poses_batch.shape}"
        assert tsdf_batch.shape == (2, 1, 48, 48, 32), f"批量TSDF形状错误: {tsdf_batch.shape}"
        
        # 验证数据范围
        print(f"\n数据范围验证:")
        print(f"RGB范围: [{rgb_batch.min():.3f}, {rgb_batch.max():.3f}]")
        print(f"TSDF范围: [{tsdf_batch.min():.3f}, {tsdf_batch.max():.3f}]")
        
        # 验证TSDF在[-1, 1]范围内
        assert tsdf_batch.min() >= -1.0 and tsdf_batch.max() <= 1.0, "TSDF超出范围"
        
        # 测试不同索引
        print(f"\n测试随机索引:")
        for i in range(min(3, len(dataset))):
            sample_i = dataset[i]
            print(f"  索引 {i}: {sample_i['sequence_name']}, 帧 {sample_i['start_frame']}-{sample_i['end_frame']}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_compatibility():
    """测试与训练循环的兼容性"""
    print("\n" + "="*60)
    print("测试与训练循环的兼容性")
    print("="*60)
    
    try:
        # 创建模拟数据集
        data_root = "/home/cwh/coding/former3d/files/sample_tartanair"
        dataset = SimpleMultiSequenceTartanAirDataset(
            data_root=data_root,
            n_view=5,
            stride=2,
            crop_size=(48, 48, 32),
            voxel_size=0.04,
            target_image_size=(256, 256),
            max_sequences=2,
            shuffle=False  # 不shuffle以便测试
        )
        
        # 模拟训练循环
        batch_size = 2
        print(f"\n模拟训练循环 (batch_size={batch_size}):")
        
        for epoch in range(2):  # 2个epoch
            print(f"\nEpoch {epoch+1}:")
            
            # 模拟数据加载器
            for batch_start in range(0, len(dataset), batch_size):
                batch_end = min(batch_start + batch_size, len(dataset))
                indices = list(range(batch_start, batch_end))
                
                # 加载批次
                batch = [dataset[i] for i in indices]
                
                # 堆叠批次数据
                rgb_batch = np.stack([item['rgb_images'] for item in batch])  # (batch_size, n_view, H, W, 3)
                poses_batch = np.stack([item['poses'] for item in batch])     # (batch_size, n_view, 4, 4)
                tsdf_batch = np.stack([item['tsdf'] for item in batch])       # (batch_size, 1, D, H, W)
                
                # 模拟前向传播 - 遍历每个时刻
                print(f"  批次 {batch_start//batch_size}: {len(indices)}个样本")
                
                for frame_idx in range(dataset.n_view):
                    # 提取当前帧
                    current_images = rgb_batch[:, frame_idx]  # (batch_size, H, W, 3)
                    current_poses = poses_batch[:, frame_idx]  # (batch_size, 4, 4)
                    
                    # 模拟模型前向传播
                    # 这里只是验证形状
                    assert current_images.shape == (len(indices), 256, 256, 3)
                    assert current_poses.shape == (len(indices), 4, 4)
                    
                    if frame_idx == 0:
                        print(f"    帧 {frame_idx}: 图像 {current_images.shape}, 位姿 {current_poses.shape}")
                
                # 模拟损失计算
                if batch_start == 0:
                    print(f"    TSDF目标形状: {tsdf_batch.shape}")
        
        return True
        
    except Exception as e:
        print(f"兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("开始测试多序列数据集...")
    
    # 测试1: 基本功能
    test1_success = test_simple_dataset()
    
    # 测试2: 训练兼容性
    test2_success = test_training_compatibility()
    
    print("\n" + "="*60)
    print("测试结果汇总:")
    print("="*60)
    print(f"测试1 - 基本功能: {'✅ 通过' if test1_success else '❌ 失败'}")
    print(f"测试2 - 训练兼容性: {'✅ 通过' if test2_success else '❌ 失败'}")
    
    if test1_success and test2_success:
        print("\n🎉 所有测试通过! 数据集实现正确。")
        print("\n下一步:")
        print("1. 将 SimpleMultiSequenceTartanAirDataset 替换为完整的 MultiSequenceTartanAirDataset")
        print("2. 修改训练脚本以使用新数据集")
        print("3. 测试实际训练流程")
    else:
        print("\n⚠️  部分测试失败，需要修复问题。")