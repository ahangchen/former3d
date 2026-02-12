#!/usr/bin/env python3
"""
简化的Pose-Aware投影测试
"""

import torch
from former3d.pose_aware_projection import PoseAwareFeatureProjector

def test_basic_functionality():
    """测试基本功能"""
    print("=== 测试PoseAwareFeatureProjector基本功能 ===")
    
    # 创建投影器
    projector = PoseAwareFeatureProjector(voxel_size=0.16)
    
    # 创建简单的测试数据
    B, N_hist = 1, 10  # 简单情况：1个batch，10个历史点
    C_fine = 128
    D, H, W = 16, 16, 16  # 小尺寸用于测试
    
    # 创建密集网格
    dense_grids = {
        'fine': torch.randn(B, C_fine, D, H, W),
    }
    
    # 创建稀疏索引
    sparse_indices = {
        'fine': torch.randint(0, min(D, H, W), (N_hist, 4)),  # [N, 4]
    }
    
    # 创建SDF网格
    sdf_grid = torch.randn(B, 1, D, H, W)
    
    # 创建poses
    historical_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)  # [B, 4, 4]
    current_pose = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)  # [B, 4, 4]
    
    # 添加一些小的旋转和平移变化
    current_pose[0, 0, 3] = 0.1  # x方向平移0.1米
    current_pose[0, 1, 3] = 0.2  # y方向平移0.2米
    
    # 历史特征字典
    historical_features = {
        'dense_grids': dense_grids,
        'sparse_indices': sparse_indices,
        'sdf_grid': sdf_grid,
        'sdf_indices': torch.randint(0, min(D, H, W), (N_hist, 4)),
        'sdf_spatial_shape': [D, H, W],
        'sdf_resolution': 0.16
    }
    
    # 创建当前体素索引
    current_voxel_indices = torch.randint(0, min(D, H, W), (N_hist, 4))
    
    try:
        # 执行投影
        projected = projector.project(
            historical_features,
            historical_pose,
            current_pose,
            current_voxel_indices
        )
        
        print(f"✅ 投影成功!")
        print(f"  投影结果: {list(projected.keys())}")
        for key in projected:
            print(f"    {key}: {projected[key].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 投影失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_main_model():
    """测试与主模型的集成"""
    print("\n=== 测试与StreamSDFFormerIntegrated的集成 ===")
    
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        # 创建模型实例
        model = StreamSDFFormerIntegrated(
            attn_heads=1,
            attn_layers=1,
            use_proj_occ=False,
            voxel_size=0.16,
            fusion_local_radius=2.0,
            crop_size=(12, 12, 8)
        )
        
        print("✅ 模型创建成功!")
        print(f"  - Pose-Aware投影器: {'存在' if hasattr(model, 'pose_aware_projector') else '缺失'}")
        print(f"  - 3D卷积融合: {'存在' if hasattr(model, 'fusion_3d') else '缺失'}")
        print(f"  - 3D卷积融合启用: {model.fusion_3d_enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_basic_functionality()
    success2 = test_integration_with_main_model()
    
    if success1 and success2:
        print("\n🎉 所有测试通过!")
    else:
        print("\n💥 测试失败!")
        exit(1)