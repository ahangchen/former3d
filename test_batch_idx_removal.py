#!/usr/bin/env python3
"""
测试batch_idx遍历移除的脚本
检查是否还有遍历batch_idx的操作
"""

import torch
import sys
import os
sys.path.append('.')

def check_batch_idx_traversal(file_path):
    """检查文件中是否有遍历batch_idx的操作"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 检查常见的遍历batch_idx的模式
    patterns = [
        r'for.*batch_idx.*in.*range',  # for batch_idx in range
        r'for.*i.*in.*range.*batch',   # for i in range(batch)
        r'for.*b.*in.*range.*batch',   # for b in range(batch)
        r'for.*batch.*in.*range',      # for batch in range
    ]
    
    issues = []
    for pattern in patterns:
        import re
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            issues.append(f"  发现模式: {pattern} -> {matches}")
    
    return issues

def main():
    print("检查batch_idx遍历移除情况...")
    print("=" * 60)
    
    # 检查关键文件
    files_to_check = [
        'former3d/stream_sdfformer_integrated.py',
        'former3d/pose_projection.py',
        'train_stream_integrated.py'
    ]
    
    all_issues = []
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"\n检查文件: {file_path}")
            issues = check_batch_idx_traversal(file_path)
            if issues:
                print(f"  ❌ 发现{len(issues)}个潜在问题:")
                for issue in issues:
                    print(issue)
                all_issues.extend(issues)
            else:
                print(f"  ✅ 未发现遍历batch_idx的操作")
    
    print("\n" + "=" * 60)
    if all_issues:
        print(f"总计发现 {len(all_issues)} 个潜在问题需要修复")
        return False
    else:
        print("✅ 所有文件都已移除遍历batch_idx的操作")
        return True

def test_model_batch_processing():
    """测试模型的批量处理能力"""
    print("\n测试模型的批量处理能力...")
    print("=" * 60)
    
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        # 创建模型
        model = StreamSDFFormerIntegrated(
            attn_heads=1,
            attn_layers=1,
            use_proj_occ=False,
            voxel_size=0.25,
            fusion_local_radius=2.0,
            crop_size=(8,8,6)
        ).to('cuda')
        
        print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 测试不同batch_size（跳过batch_size=1，因为BatchNorm在训练模式下需要batch_size>1）
        batch_sizes = [2, 4]  # 跳过batch_size=1
        n_view = 5
        
        for batch_size in batch_sizes:
            print(f"\n测试 batch_size={batch_size}...")
            
            # 创建输入
            images = torch.randn(batch_size, n_view, 3, 256, 256).to('cuda')
            poses = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(batch_size, n_view, 4, 4).to('cuda')
            intrinsics = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(batch_size, n_view, 3, 3).to('cuda')
            
            print(f"  输入形状: images={images.shape}, poses={poses.shape}, intrinsics={intrinsics.shape}")
            
            try:
                # 测试forward_sequence
                outputs, states = model.forward_sequence(images, poses, intrinsics)
                print(f"  ✅ forward_sequence成功")
                print(f"    输出类型: {type(outputs)}")
                if isinstance(outputs, dict):
                    print(f"    输出键: {list(outputs.keys())}")
                print(f"    状态数量: {len(states)}")
                
                # 重置模型状态
                model.reset_state()
                
            except Exception as e:
                print(f"  ❌ forward_sequence失败: {e}")
                return False
        
        print("\n✅ 所有batch_size测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pose_projection():
    """测试姿态投影模块的批量处理"""
    print("\n测试姿态投影模块的批量处理...")
    print("=" * 60)
    
    try:
        from former3d.pose_projection import PoseProjection
        
        # 创建投影模块
        projector = PoseProjection(voxel_size=0.0625, crop_size=(48, 96, 96))
        
        # 测试不同batch_size
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            print(f"\n测试 batch_size={batch_size}...")
            
            # 创建模拟稀疏数据
            num_voxels = 1000
            channels = 64
            
            # 随机生成体素坐标
            voxel_coords = torch.randn(num_voxels, 3) * 0.5
            voxel_batch_inds = torch.randint(0, batch_size, (num_voxels,))
            
            # 随机生成特征
            historical_features = torch.randn(num_voxels, channels)
            
            # 恒等变换
            identity_pose = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            
            # 创建历史状态
            historical_state = {
                'features': historical_features,
                'coords': voxel_coords,
                'batch_inds': voxel_batch_inds
            }
            
            # 投影
            projected_state = projector(
                historical_state,
                identity_pose,
                identity_pose
            )
            
            print(f"  ✅ 投影成功")
            print(f"    输入特征形状: {historical_features.shape}")
            print(f"    输出特征形状: {projected_state['features'].shape}")
            print(f"    有效体素数量: {projected_state['mask'].sum().item()}")
        
        print("\n✅ 姿态投影模块批量处理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 姿态投影模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("批量处理优化测试套件")
    print("=" * 60)
    
    # 检查代码
    code_check_passed = main()
    
    # 测试模型
    model_test_passed = test_model_batch_processing()
    
    # 测试姿态投影
    projection_test_passed = test_pose_projection()
    
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    print(f"  代码检查: {'✅ 通过' if code_check_passed else '❌ 失败'}")
    print(f"  模型测试: {'✅ 通过' if model_test_passed else '❌ 失败'}")
    print(f"  姿态投影测试: {'✅ 通过' if projection_test_passed else '❌ 失败'}")
    
    if code_check_passed and model_test_passed and projection_test_passed:
        print("\n🎉 所有测试通过！批量处理优化完成。")
        sys.exit(0)
    else:
        print("\n⚠️ 部分测试失败，需要进一步优化。")
        sys.exit(1)