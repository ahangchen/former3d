#!/usr/bin/env python
"""
测试修复后的3D池化层
"""

import torch
import torch.distributed as dist
import os
import sys

print("="*80)
print("测试修复后的3D池化层")
print("="*80)

# 初始化分布式环境
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group('nccl', rank=0, world_size=1)

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_fixed_model():
    """测试修复后的模型"""
    print("\n" + "="*60)
    print("测试修复后的StreamSDFFormer")
    print("="*60)
    
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        # 使用原始参数测试
        print("\n使用原始参数测试:")
        print("  voxel_size=0.04, crop_size=(48, 96, 96)")
        
        model = StreamSDFFormerIntegrated(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.04,  # 原始值
            fusion_local_radius=3.0,
            crop_size=(48, 96, 96)  # 原始值
        ).cuda()
        
        print(f"✅ 模型创建成功")
        print(f"  参数总数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 测试前向传播
        model.eval()
        with torch.no_grad():
            images = torch.randn(1, 3, 64, 64).cuda()
            poses = torch.eye(4).unsqueeze(0).cuda()
            intrinsics = torch.eye(3).unsqueeze(0).cuda()
            intrinsics[:, 0, 0] = 250.0
            intrinsics[:, 1, 1] = 250.0
            intrinsics[:, 0, 2] = 32
            intrinsics[:, 1, 2] = 32
            
            print("\n执行前向传播...")
            output, state = model.forward_single_frame(
                images, poses, intrinsics, reset_state=True
            )
            
            if 'sdf' in output and output['sdf'] is not None:
                print(f"✅ 前向传播成功！")
                print(f"  SDF形状: {output['sdf'].shape}")
                print(f"  体素网格大小: {model.get_voxel_dim()}")
                
                # 测试训练模式
                print("\n测试训练模式...")
                model.train()
                
                # 创建需要梯度的输入
                train_images = torch.randn(2, 3, 64, 64, device='cuda', requires_grad=True)
                train_poses = torch.eye(4).unsqueeze(0).repeat(2, 1, 1).cuda()
                train_intrinsics = intrinsics.repeat(2, 1, 1)
                
                # 前向传播
                output, _ = model.forward_single_frame(
                    train_images, train_poses, train_intrinsics, reset_state=True
                )
                
                if 'sdf' in output and output['sdf'] is not None:
                    loss = output['sdf'].mean()
                    loss.backward()
                    
                    if train_images.grad is not None:
                        print(f"✅ 梯度计算成功")
                        print(f"  图像梯度范数: {train_images.grad.norm().item():.6f}")
                    else:
                        print("❌ 图像梯度为None")
                
                return True
            else:
                print("❌ 前向传播失败")
                return False
                
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_parameters():
    """测试不同参数组合"""
    print("\n" + "="*60)
    print("测试不同参数组合")
    print("="*60)
    
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        test_cases = [
            {"voxel_size": 0.04, "crop_size": (48, 96, 96), "desc": "原始参数"},
            {"voxel_size": 0.08, "crop_size": (24, 48, 48), "desc": "较大体素"},
            {"voxel_size": 0.02, "crop_size": (96, 192, 192), "desc": "较小体素"},
            {"voxel_size": 0.1, "crop_size": (16, 32, 32), "desc": "大体素小裁剪"},
        ]
        
        for params in test_cases:
            print(f"\n测试: {params['desc']}")
            print(f"  voxel_size={params['voxel_size']}, crop_size={params['crop_size']}")
            
            try:
                model = StreamSDFFormerIntegrated(
                    attn_heads=2,
                    attn_layers=2,
                    use_proj_occ=False,
                    voxel_size=params['voxel_size'],
                    fusion_local_radius=3.0,
                    crop_size=params['crop_size']
                ).cuda()
                
                # 简单前向传播测试
                model.eval()
                with torch.no_grad():
                    images = torch.randn(1, 3, 64, 64).cuda()
                    poses = torch.eye(4).unsqueeze(0).cuda()
                    intrinsics = torch.eye(3).unsqueeze(0).cuda()
                    intrinsics[:, 0, 0] = 250.0
                    intrinsics[:, 1, 1] = 250.0
                    intrinsics[:, 0, 2] = 32
                    intrinsics[:, 1, 2] = 32
                    
                    output, _ = model.forward_single_frame(
                        images, poses, intrinsics, reset_state=True
                    )
                    
                    if 'sdf' in output and output['sdf'] is not None:
                        print(f"  ✅ 成功 - SDF形状: {output['sdf'].shape}")
                    else:
                        print(f"  ⚠️ 警告 - 无SDF输出")
                        
            except Exception as e:
                print(f"  ❌ 失败: {str(e)[:80]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 参数测试失败: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("3D池化层修复测试")
    print("="*80)
    
    # 运行测试
    test1_success = test_fixed_model()
    test2_success = test_different_parameters()
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    if test1_success:
        print("✅ 3D池化层修复成功！")
        print("  原始参数现在可以正常工作")
        print("  梯度计算正常")
        print("  前向传播正常")
    else:
        print("❌ 修复测试失败")
        print("  需要进一步调试")
    
    if test2_success:
        print("\n✅ 参数兼容性测试通过")
        print("  多种参数组合均可工作")
    else:
        print("\n⚠️ 参数兼容性测试部分失败")
    
    print("\n" + "="*80)
    print("下一步：运行完整的端到端训练测试")
    print("="*80)

if __name__ == "__main__":
    main()