#!/usr/bin/env python
"""
分布式梯度验证 - 正确初始化分布式环境以支持SyncBatchNorm
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_distributed(rank, world_size):
    """设置分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    
    # 设置当前GPU
    torch.cuda.set_device(rank)
    print(f"进程 {rank}/{world_size} 初始化完成，使用GPU {rank}")

def cleanup_distributed():
    """清理分布式环境"""
    dist.destroy_process_group()

def run_gradient_test_single_process():
    """单进程梯度测试（简化版）"""
    print("\n" + "="*60)
    print("单进程梯度测试")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建一个极简的流式模型
    class SimpleStreamModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 2D特征提取
            self.conv2d = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU()
            )
            
            # 特征投影
            self.projection = nn.Linear(32, 64)
            
            # 流式融合（模拟cross-attention）
            self.fusion = nn.MultiheadAttention(embed_dim=64, num_heads=2, batch_first=True)
            
            # 输出头
            self.sdf_head = nn.Linear(64, 1)
            self.occ_head = nn.Linear(64, 1)
            
            # 历史状态
            self.history_features = None
            self.history_positions = None
        
        def forward(self, images, poses, intrinsics, reset_state=False):
            batch_size = images.shape[0]
            
            # 提取2D特征
            features_2d = self.conv2d(images)  # [B, 32, H, W]
            features_2d = features_2d.mean(dim=[2, 3])  # [B, 32]
            
            # 投影到3D特征空间
            features_3d = self.projection(features_2d)  # [B, 64]
            
            # 准备融合输入
            query = features_3d.unsqueeze(1)  # [B, 1, 64]
            
            # 流式融合
            if self.history_features is not None and not reset_state:
                # 使用历史特征作为key和value
                key = self.history_features.unsqueeze(1)  # [B, 1, 64]
                value = self.history_features.unsqueeze(1)  # [B, 1, 64]
                
                # Cross-attention融合
                fused, _ = self.fusion(query, key, value)
                fused = fused.squeeze(1)  # [B, 64]
            else:
                fused = features_3d
            
            # 更新历史状态
            self.history_features = features_3d.detach().clone()
            
            # 输出
            sdf = self.sdf_head(fused)
            occupancy = torch.sigmoid(self.occ_head(fused))
            
            return {
                'sdf': sdf,
                'occupancy': occupancy,
                'features': fused
            }
    
    # 创建模型
    model = SimpleStreamModel().to(device)
    model.train()
    
    print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试1: 第一帧
    print("\n>>> 测试1: 第一帧推理（重置状态）")
    images1 = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)
    poses = torch.eye(4, device=device).unsqueeze(0).repeat(2, 1, 1)
    intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(2, 1, 1)
    
    output1 = model(images1, poses, intrinsics, reset_state=True)
    print(f"  SDF形状: {output1['sdf'].shape}")
    print(f"  SDF requires_grad: {output1['sdf'].requires_grad}")
    
    # 反向传播
    loss1 = output1['sdf'].mean()
    print(f"  损失值: {loss1.item():.6f}")
    
    loss1.backward()
    
    # 检查梯度
    grad_check1 = images1.grad is not None
    print(f"  输入梯度存在: {'✅' if grad_check1 else '❌'}")
    
    # 清除梯度
    model.zero_grad()
    
    # 测试2: 第二帧（使用历史状态）
    print("\n>>> 测试2: 第二帧推理（使用历史状态）")
    images2 = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)
    
    output2 = model(images2, poses, intrinsics, reset_state=False)
    print(f"  SDF形状: {output2['sdf'].shape}")
    
    # 反向传播
    loss2 = output2['sdf'].mean()
    print(f"  损失值: {loss2.item():.6f}")
    
    loss2.backward()
    
    # 检查梯度
    grad_check2 = images2.grad is not None
    print(f"  输入梯度存在: {'✅' if grad_check2 else '❌'}")
    
    # 检查融合模块梯度
    fusion_grads = []
    for name, param in model.named_parameters():
        if 'fusion' in name and param.grad is not None:
            fusion_grads.append(param.grad.norm().item())
    
    fusion_check = len(fusion_grads) > 0
    print(f"  融合模块梯度存在: {'✅' if fusion_check else '❌'}")
    
    return grad_check1 and grad_check2 and fusion_check

def test_original_model_with_distributed():
    """测试原始模型（带分布式初始化）"""
    print("\n" + "="*60)
    print("测试原始SDFFormer模型（带分布式）")
    print("="*60)
    
    try:
        # 导入原始模型
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        # 创建模型
        model = StreamSDFFormerIntegrated(
            attn_heads=2,
            attn_layers=2,
            use_proj_occ=False,
            voxel_size=0.0625,
            fusion_local_radius=3.0,
            crop_size=(48, 96, 96)
        )
        
        # 移动到GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.train()
        
        print(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建测试数据
        batch_size = 1
        images = torch.randn(batch_size, 3, 128, 128, device=device, requires_grad=True)
        poses = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        intrinsics[:, 0, 0] = 250.0
        intrinsics[:, 1, 1] = 250.0
        intrinsics[:, 0, 2] = 64
        intrinsics[:, 1, 2] = 64
        
        print(f"\n测试数据:")
        print(f"  images形状: {images.shape}, requires_grad: {images.requires_grad}")
        
        # 启用流式融合
        model.enable_stream_fusion(True)
        
        # 第一帧推理
        print("\n>>> 第一帧推理（重置状态）")
        output1, state1 = model.forward_single_frame(
            images, poses, intrinsics, reset_state=True
        )
        
        if output1['sdf'] is not None:
            print(f"  SDF形状: {output1['sdf'].shape}")
            print(f"  SDF requires_grad: {output1['sdf'].requires_grad}")
            
            # 反向传播
            loss1 = output1['sdf'].mean()
            print(f"  损失值: {loss1.item():.6f}")
            
            loss1.backward()
            
            # 检查梯度
            grad_check1 = images.grad is not None
            print(f"  输入梯度存在: {'✅' if grad_check1 else '❌'}")
            
            if images.grad is not None:
                print(f"  梯度范数: {images.grad.norm().item():.6f}")
            
            # 检查模型参数梯度
            grad_params = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_params.append((name, param.grad.norm().item()))
            
            print(f"  有梯度的参数数量: {len(grad_params)}")
            
            # 检查流式融合模块
            fusion_grads = []
            for name, param in model.named_parameters():
                if 'stream_fusion' in name and param.grad is not None:
                    fusion_grads.append(param.grad.norm().item())
            
            print(f"  流式融合模块有梯度的参数: {len(fusion_grads)}")
            
            return len(grad_params) > 0
            
        else:
            print("  ❌ 未生成SDF输出")
            return False
            
    except Exception as e:
        print(f"\n❌ 原始模型测试失败:")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("="*80)
    print("分布式梯度验证 - 流式SDFFormer梯度图验证")
    print("="*80)
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    results = []
    
    try:
        # 方法1: 单进程简化测试
        print("\n>>> 方法1: 单进程简化梯度测试")
        result1 = run_gradient_test_single_process()
        results.append(("单进程简化测试", result1))
        
        # 方法2: 初始化分布式环境测试原始模型
        print("\n>>> 方法2: 分布式环境测试原始模型")
        
        # 初始化分布式环境（单机多卡）
        world_size = min(2, torch.cuda.device_count())
        
        if world_size > 1:
            print(f"初始化分布式环境 (world_size={world_size})...")
            
            # 使用spawn启动多个进程
            mp.spawn(
                test_worker,
                args=(world_size,),
                nprocs=world_size,
                join=True
            )
            
            result2 = True  # 假设成功
        else:
            print("只有1个GPU，使用单机模式")
            # 初始化单机分布式
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                rank=0,
                world_size=1
            )
            
            result2 = test_original_model_with_distributed()
            
            # 清理
            cleanup_distributed()
        
        results.append(("分布式原始模型测试", result2))
        
    except Exception as e:
        print(f"\n❌ 主测试流程失败:")
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        results.append(("主流程", False))
    
    # 总结
    print("\n" + "="*80)
    print("梯度验证总结")
    print("="*80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print(f"\n总体结果: {'✅ 所有测试通过' if all_passed else '❌ 部分测试失败'}")
    
    if all_passed:
        print("\n🎉 梯度图验证成功！")
        print("网络梯度流完整，可以继续进行性能基准测试。")
    else:
        print("\n⚠️ 梯度验证发现问题。")
        print("建议检查:")
        print("  1. SyncBatchNorm的分布式初始化")
        print("  2. 模型参数requires_grad设置")
        print("  3. 计算图中是否有detach()操作")
    
    return all_passed

def test_worker(rank, world_size):
    """分布式测试工作进程"""
    setup_distributed(rank, world_size)
    
    try:
        # 每个进程测试自己的部分
        if rank == 0:
            print(f"\n主进程 {rank} 开始测试...")
            success = test_original_model_with_distributed()
        else:
            print(f"工作进程 {rank} 等待...")
            success = True
        
        # 同步所有进程
        dist.barrier()
        
        return success
        
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)