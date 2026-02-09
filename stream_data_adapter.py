#!/usr/bin/env python3
"""
流式数据适配器
将多帧数据集适配为单帧流式输入
"""

import torch
from typing import Dict, Any, Optional

class StreamDataAdapter:
    """适配多帧数据为流式单帧输入"""
    
    def __init__(self, frame_selection: str = 'first'):
        """
        初始化适配器
        
        Args:
            frame_selection: 帧选择策略
                - 'first': 选择第一帧
                - 'middle': 选择中间帧
                - 'random': 随机选择一帧
                - 'all': 处理所有帧（返回迭代器）
        """
        self.frame_selection = frame_selection
        
    def adapt_batch(self, batch: Dict[str, Any], frame_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        适配批次数据为单帧输入
        
        Args:
            batch: 原始批次数据，包含多帧
            frame_idx: 指定帧索引（如果为None则根据策略选择）
            
        Returns:
            适配后的单帧批次数据
        """
        adapted_batch = {}
        
        # 获取批次大小和帧数
        if 'rgb_images' in batch:
            batch_size, n_frames, channels, height, width = batch['rgb_images'].shape
        else:
            # 如果没有rgb_images，尝试其他方式获取形状
            batch_size = 1
            n_frames = 1
            
        # 选择帧索引
        if frame_idx is None:
            if self.frame_selection == 'first':
                frame_idx = 0
            elif self.frame_selection == 'middle':
                frame_idx = n_frames // 2
            elif self.frame_selection == 'random':
                frame_idx = torch.randint(0, n_frames, (1,)).item()
            elif self.frame_selection == 'all':
                # 返回迭代器
                return self._create_frame_iterator(batch)
            else:
                raise ValueError(f"未知的帧选择策略: {self.frame_selection}")
        
        # 适配每个键
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                # 处理张量
                if key in ['rgb_images', 'poses', 'cam_poses']:
                    # 这些键有帧维度
                    if len(value.shape) >= 4:  # 至少有 [batch, frames, ...]
                        if value.shape[1] == n_frames:  # 确认有帧维度
                            # 移除帧维度，只保留单帧
                            adapted_batch[key] = value[:, frame_idx]
                        else:
                            # 没有帧维度，直接使用
                            adapted_batch[key] = value
                    else:
                        adapted_batch[key] = value
                elif key == 'intrinsics' or key == 'cam_intrinsics':
                    # 内参通常没有帧维度，但可能有批次维度
                    if len(value.shape) == 3:  # [batch, 3, 3]
                        adapted_batch[key] = value
                    elif len(value.shape) == 4:  # [batch, frames, 3, 3]
                        adapted_batch[key] = value[:, frame_idx]
                    else:
                        adapted_batch[key] = value
                elif key in ['tsdf', 'occupancy', 'voxel_coords']:
                    # 这些是3D数据，没有帧维度
                    adapted_batch[key] = value
                else:
                    # 其他张量
                    adapted_batch[key] = value
            else:
                # 非张量数据
                adapted_batch[key] = value
        
        # 重命名键以匹配模型期望
        adapted_batch = self._rename_keys(adapted_batch)
        
        # 添加帧索引信息
        adapted_batch['frame_idx'] = frame_idx
        adapted_batch['total_frames'] = n_frames
        
        return adapted_batch
    
    def _create_frame_iterator(self, batch: Dict[str, Any]):
        """创建帧迭代器"""
        if 'rgb_images' in batch:
            _, n_frames, _, _, _ = batch['rgb_images'].shape
        else:
            n_frames = 1
            
        for frame_idx in range(n_frames):
            yield self.adapt_batch(batch, frame_idx), frame_idx
    
    def _rename_keys(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """重命名键以匹配模型期望"""
        renamed = {}
        
        key_mapping = {
            'rgb_images': 'images',  # StreamSDFFormerIntegrated期望'images'
            'poses': 'poses',        # 保持不变
            'cam_poses': 'poses',    # 重命名
            'intrinsics': 'intrinsics',  # 保持不变
            'cam_intrinsics': 'intrinsics',  # 重命名
        }
        
        for key, value in batch.items():
            if key in key_mapping:
                renamed[key_mapping[key]] = value
            else:
                renamed[key] = value
        
        return renamed
    
    def adapt_for_stream_training(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        为流式训练适配数据
        
        Args:
            batch: 原始批次数据
            
        Returns:
            适配后的数据，包含：
            - sequence_data: 序列信息
            - frames: 帧列表
            - ground_truth: 真值数据
        """
        adapted = {
            'sequence_data': {},
            'frames': [],
            'ground_truth': {}
        }
        
        # 提取序列信息
        if 'sequence_name' in batch:
            adapted['sequence_data']['name'] = batch['sequence_name']
        if 'segment_idx' in batch:
            adapted['sequence_data']['segment_idx'] = batch['segment_idx']
        if 'start_frame' in batch:
            adapted['sequence_data']['start_frame'] = batch['start_frame']
        if 'end_frame' in batch:
            adapted['sequence_data']['end_frame'] = batch['end_frame']
        
        # 提取真值数据
        if 'tsdf' in batch:
            adapted['ground_truth']['tsdf'] = batch['tsdf']
        if 'occupancy' in batch:
            adapted['ground_truth']['occupancy'] = batch['occupancy']
        if 'voxel_coords' in batch:
            adapted['ground_truth']['voxel_coords'] = batch['voxel_coords']
        
        # 提取所有帧
        if 'rgb_images' in batch:
            batch_size, n_frames, channels, height, width = batch['rgb_images'].shape
            
            for frame_idx in range(n_frames):
                frame_data = {}
                
                # 提取当前帧图像
                if 'rgb_images' in batch:
                    frame_data['images'] = batch['rgb_images'][:, frame_idx]
                
                # 提取当前帧位姿
                if 'poses' in batch:
                    if len(batch['poses'].shape) == 4:  # [batch, frames, 4, 4]
                        frame_data['poses'] = batch['poses'][:, frame_idx]
                    else:  # [batch, 4, 4]
                        frame_data['poses'] = batch['poses']
                
                # 提取内参
                if 'intrinsics' in batch:
                    if len(batch['intrinsics'].shape) == 4:  # [batch, frames, 3, 3]
                        frame_data['intrinsics'] = batch['intrinsics'][:, frame_idx]
                    else:  # [batch, 3, 3]
                        frame_data['intrinsics'] = batch['intrinsics']
                
                # 添加帧索引
                frame_data['frame_idx'] = frame_idx
                
                adapted['frames'].append(frame_data)
        
        return adapted


# 测试适配器
if __name__ == "__main__":
    # 创建模拟数据
    batch_size = 2
    n_frames = 5
    height, width = 256, 256
    
    batch = {
        'rgb_images': torch.randn(batch_size, n_frames, 3, height, width),
        'poses': torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_frames, 1, 1),
        'intrinsics': torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_frames, 1, 1),
        'tsdf': torch.randn(batch_size, 1, 48, 48, 32),
        'sequence_name': ['seq1', 'seq2'],
        'segment_idx': [0, 1],
        'start_frame': [0, 0],
        'end_frame': [4, 4]
    }
    
    print("测试流式数据适配器...")
    print(f"原始批次形状:")
    print(f"  rgb_images: {batch['rgb_images'].shape}")
    print(f"  poses: {batch['poses'].shape}")
    print(f"  intrinsics: {batch['intrinsics'].shape}")
    
    # 测试不同策略
    adapter = StreamDataAdapter(frame_selection='first')
    
    print(f"\n使用'first'策略适配:")
    adapted = adapter.adapt_batch(batch)
    print(f"  适配后形状:")
    print(f"    images: {adapted['images'].shape}")
    print(f"    poses: {adapted['poses'].shape}")
    print(f"    intrinsics: {adapted['intrinsics'].shape}")
    print(f"    frame_idx: {adapted['frame_idx']}")
    
    print(f"\n使用'all'策略迭代:")
    adapter_all = StreamDataAdapter(frame_selection='all')
    for frame_data, frame_idx in adapter_all._create_frame_iterator(batch):
        print(f"  帧 {frame_idx}: images形状={frame_data['images'].shape}")
    
    print(f"\n为流式训练适配:")
    stream_data = adapter.adapt_for_stream_training(batch)
    print(f"  序列数: {len(stream_data['frames'])}")
    print(f"  序列名: {stream_data['sequence_data'].get('name', 'N/A')}")
    print(f"  真值键: {list(stream_data['ground_truth'].keys())}")
    
    print("\n✅ 适配器测试完成!")