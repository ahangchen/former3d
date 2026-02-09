#!/usr/bin/env python
"""
调试grid_sample维度问题
"""

import torch
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_grid_sample_dimensions():
    """测试grid_sample的维度要求"""
    print("=" * 60)
    print("调试grid_sample维度问题")
    print("=" * 60)
    
    # 模拟测试数据
    batch_size = 2
    depth, height, width = 48, 96, 96
    num_voxels = depth * height * width  # 442,368
    channels = 128
    
    print(f"模拟参数:")
    print(f"  batch_size: {batch_size}")
    print(f"  体素网格: {depth}x{height}x{width} = {num_voxels}个体素")
    print(f"  通道数: {channels}")
    
    # 1. 创建2D特征（如测试中使用的）
    print("\n1. 创建2D特征...")
    features_2d = torch.randn(batch_size * num_voxels, channels)
    print(f"  2D特征形状: {features_2d.shape}")
    print(f"  期望: [batch*num_voxels, channels] = [{batch_size*num_voxels}, {channels}]")
    
    # 2. 创建坐标映射
    print("\n2. 创建坐标映射...")
    coordinate_mapping = torch.randn(batch_size, depth, height, width, 3)
    print(f"  坐标映射形状: {coordinate_mapping.shape}")
    print(f"  期望: [batch, depth, height, width, 3]")
    
    # 3. 尝试重塑为5D
    print("\n3. 尝试重塑为5D...")
    try:
        # 按照pose_projection.py中的逻辑
        historical_features_5d = features_2d.reshape(
            batch_size, num_voxels, -1
        ).permute(0, 2, 1).reshape(
            batch_size, -1, depth, height, width
        )
        print(f"  重塑后的5D特征形状: {historical_features_5d.shape}")
        print(f"  期望: [batch, channels, depth, height, width] = [{batch_size}, {channels}, {depth}, {height}, {width}]")
        
        # 4. 尝试grid_sample
        print("\n4. 尝试grid_sample...")
        projected = F.grid_sample(
            historical_features_5d,
            coordinate_mapping,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        print(f"  ✅ grid_sample成功!")
        print(f"  投影后形状: {projected.shape}")
        
        # 5. 重塑回2D
        print("\n5. 重塑回2D...")
        batch_size_out, channels_out, depth_out, height_out, width_out = projected.shape
        projected_2d = projected.reshape(batch_size_out, channels_out, -1).permute(0, 2, 1)
        projected_2d = projected_2d.reshape(-1, channels_out)
        print(f"  最终2D特征形状: {projected_2d.shape}")
        print(f"  期望: [batch*num_voxels, channels] = [{batch_size*num_voxels}, {channels}]")
        
    except Exception as e:
        print(f"  ❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("调试完成")
    print("=" * 60)

def test_actual_problem():
    """测试实际的问题场景"""
    print("\n" + "=" * 60)
    print("测试实际的问题场景")
    print("=" * 60)
    
    try:
        from former3d.pose_projection import PoseProjection
        
        # 创建投影器
        projector = PoseProjection()
        
        # 创建测试数据（模拟测试中的情况）
        batch_size = 2
        depth, height, width = 48, 96, 96
        num_voxels = depth * height * width
        channels = 128
        
        print(f"测试参数:")
        print(f"  batch_size: {batch_size}")
        print(f"  体素网格: {depth}x{height}x{width}")
        print(f"  通道数: {channels}")
        
        # 创建历史状态（2D特征）
        historical_state = {
            'features': torch.randn(batch_size * num_voxels, channels),
            'sdf': torch.randn(batch_size * num_voxels, 1),
            'occupancy': torch.randn(batch_size * num_voxels, 1),
            'coords': torch.randint(0, 10, (batch_size * num_voxels, 4))
        }
        
        # 创建位姿
        historical_pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        current_pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        current_pose[:, :3, 3] += 0.1  # 添加平移
        
        print("\n调用pose_projection...")
        projected_state = projector(historical_state, historical_pose, current_pose)
        
        print(f"✅ 投影成功!")
        print(f"投影状态keys: {list(projected_state.keys())}")
        for key, value in projected_state.items():
            if torch.is_tensor(value):
                print(f"  {key}: {value.shape}")
        
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_grid_sample_dimensions()
    test_actual_problem()