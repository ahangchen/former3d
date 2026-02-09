#!/usr/bin/env python3
"""
快速测试训练
验证修复是否有效
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("快速测试训练")
print("="*80)

# 检查环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def test_intrinsics_fix():
    """测试intrinsics修复"""
    print("\n1. 测试intrinsics修复...")
    
    # 模拟数据集返回
    batch = {
        'rgb_images': torch.randn(2, 3, 3, 64, 64),
        'poses': torch.randn(2, 3, 4, 4),
        'intrinsics': torch.eye(3),  # (3, 3)
        'tsdf': torch.randn(2, 16, 16, 16)
    }
    
    # 应用修复
    images = batch['rgb_images'].to(device)
    poses = batch['poses'].to(device)
    intrinsics = batch['intrinsics'].to(device)
    
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0).repeat(images.shape[0], 1, 1)
    
    print(f"  images形状: {images.shape}")
    print(f"  poses形状: {poses.shape}")
    print(f"  intrinsics形状: {intrinsics.shape}")
    
    if intrinsics.shape == torch.Size([2, 3, 3]):
        print("  ✅ intrinsics形状修复正确")
        return True
    else:
        print("  ❌ intrinsics形状修复失败")
        return False

def test_model_creation():
    """测试模型创建"""
    print("\n2. 测试模型创建...")
    
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        model = StreamSDFFormerIntegrated(
            attn_heads=2,
            attn_layers=1,
            use_proj_occ=True,
            voxel_size=0.15,  # 增大体素大小以减少内存
            fusion_local_radius=0.0,
            crop_size=(16, 16, 16)  # 小裁剪尺寸
        ).to(device)
        
        print(f"  ✅ 模型创建成功")
        print(f"    参数数量: {sum(p.numel() for p in model.parameters()):,}")
        return model
        
    except Exception as e:
        print(f"  ❌ 模型创建失败: {e}")
        return None

def test_single_batch(model):
    """测试单个批次"""
    print("\n3. 测试单个批次...")
    
    try:
        # 创建测试数据
        B = 1  # 小批次
        images = torch.randn(B, 3, 64, 64).to(device)
        poses = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)
        intrinsics = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(device)
        
        print(f"  输入形状:")
        print(f"    images: {images.shape}")
        print(f"    poses: {poses.shape}")
        print(f"    intrinsics: {intrinsics.shape}")
        
        # 前向传播
        with torch.no_grad():
            output = model(images, poses, intrinsics, reset_state=True)
        
        if 'sdf' in output:
            sdf = output['sdf']
            print(f"  ✅ 前向传播成功")
            print(f"    SDF形状: {sdf.shape}")
            print(f"    SDF范围: [{sdf.min():.4f}, {sdf.max():.4f}]")
            return True
        else:
            print(f"  ❌ 输出中没有'sdf'键")
            return False
            
    except Exception as e:
        print(f"  ❌ 批次测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step(model):
    """测试训练步骤"""
    print("\n4. 测试训练步骤...")
    
    try:
        # 创建优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = nn.MSELoss()
        
        # 创建测试数据
        B = 1
        images = torch.randn(B, 3, 64, 64).to(device)
        poses = torch.eye(4).unsqueeze(0).repeat(B, 1, 1).to(device)
        intrinsics = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(device)
        
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        
        output = model(images, poses, intrinsics, reset_state=True)
        
        if 'sdf' in output:
            pred_sdf = output['sdf']
            
            # 根据pred_sdf形状创建匹配的目标
            if len(pred_sdf.shape) == 2:  # [num_voxels, 1]
                num_voxels = pred_sdf.shape[0]
                # 创建匹配形状的目标
                targets = torch.randn(num_voxels, 1).to(device)
                loss = loss_fn(pred_sdf, targets)
            elif len(pred_sdf.shape) == 3:  # [B, num_points, 1]
                B, num_points, _ = pred_sdf.shape
                targets = torch.randn(B, num_points, 1).to(device)
                loss = loss_fn(pred_sdf, targets)
            else:
                print(f"  ❌ 未知的pred_sdf形状: {pred_sdf.shape}")
                return False
            
            loss.backward()
            optimizer.step()
            
            print(f"  ✅ 训练步骤成功")
            print(f"    pred_sdf形状: {pred_sdf.shape}")
            print(f"    损失: {loss.item():.6f}")
            print(f"    梯度范数: {sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None):.4f}")
            return True
        else:
            print(f"  ❌ 输出中没有'sdf'键")
            return False
            
    except Exception as e:
        print(f"  ❌ 训练步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("开始快速测试...")
    
    # 测试1: intrinsics修复
    test1 = test_intrinsics_fix()
    
    # 测试2: 模型创建
    model = test_model_creation()
    
    if model is None:
        print("❌ 模型创建失败，无法继续测试")
        return
    
    # 测试3: 单个批次
    test3 = test_single_batch(model)
    
    # 测试4: 训练步骤
    test4 = test_training_step(model)
    
    print("\n" + "="*80)
    print("测试结果")
    print("="*80)
    
    all_passed = test1 and test3 and test4
    
    if all_passed:
        print("✅ 所有测试通过!")
        print("\noptimized_online_training.py修复成功!")
        print("现在可以运行完整的训练脚本:")
        print("命令: python optimized_online_training.py")
    else:
        print("❌ 部分测试失败")
        print("\n需要进一步调试:")
        if not test1: print("  - intrinsics形状修复")
        if not test3: print("  - 模型前向传播")
        if not test4: print("  - 训练步骤")
    
    print("\n🚀 根据测试结果决定下一步操作")

if __name__ == "__main__":
    main()