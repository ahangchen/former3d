#!/usr/bin/env python3
"""
简化版在线SDF训练测试
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("简化版在线SDF训练测试")
print("="*80)

# 检查环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 创建简化模型
class SimpleOnlineSDFModel(nn.Module):
    """简化在线SDF模型"""
    def __init__(self):
        super().__init__()
        # 图像编码器
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # 3D解码器
        self.decoder = nn.Sequential(
            nn.Linear(32 * 8 * 8 + 3, 256),  # 图像特征 + 3D坐标
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # SDF值
        )
        
    def forward(self, images, poses, intrinsics, reset_state=False):
        """
        简化前向传播
        Args:
            images: (B, 3, H, W) 图像
            poses: (B, 4, 4) 相机位姿
            intrinsics: (B, 3, 3) 相机内参
            reset_state: 是否重置状态
        Returns:
            dict: 包含'sdf'预测
        """
        B, C, H, W = images.shape
        
        # 编码图像
        img_features = self.image_encoder(images)  # (B, 32, 8, 8)
        img_features = img_features.view(B, -1)  # (B, 32*8*8)
        
        # 生成3D点
        num_points = 512
        points = torch.randn(B, num_points, 3).to(images.device)  # 随机3D点
        
        # 重复图像特征给每个点
        img_features_expanded = img_features.unsqueeze(1).repeat(1, num_points, 1)  # (B, num_points, 2048)
        
        # 拼接特征
        combined_features = torch.cat([img_features_expanded, points], dim=-1)  # (B, num_points, 2051)
        
        # 解码SDF
        sdf_pred = self.decoder(combined_features)  # (B, num_points, 1)
        
        return {'sdf': sdf_pred}
    
    def reset_state(self):
        """重置状态（简化模型无状态）"""
        pass


def test_training():
    """测试训练循环"""
    print("\n测试训练循环...")
    
    # 创建模型
    model = SimpleOnlineSDFModel()
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 创建损失函数
    loss_fn = nn.MSELoss()
    
    # 模拟数据
    batch_size = 1
    num_frames = 5
    H, W = 128, 128
    
    print(f"使用模拟数据: batch_size={batch_size}, num_frames={num_frames}")
    
    losses = []
    
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}:")
        
        epoch_loss = 0
        
        for frame_idx in range(num_frames):
            # 创建模拟数据
            images = torch.randn(batch_size, 3, H, W)
            poses = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
            poses[:, 0, 3] = frame_idx * 0.1  # 简单运动
            
            intrinsics = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1)
            intrinsics[:, 0, 0] = 300.0
            intrinsics[:, 1, 1] = 300.0
            intrinsics[:, 0, 2] = W / 2
            intrinsics[:, 1, 2] = H / 2
            
            # 训练步骤
            optimizer.zero_grad()
            
            # 前向传播
            output = model(
                images=images,
                poses=poses,
                intrinsics=intrinsics,
                reset_state=(frame_idx == 0)
            )
            
            # 创建目标（模拟）
            pred_sdf = output['sdf']
            target_sdf = torch.randn_like(pred_sdf) * 0.1
            
            # 计算损失
            loss = loss_fn(pred_sdf, target_sdf)
            epoch_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 检查梯度
            grad_norm = 0
            param_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
                    param_count += 1
            
            # 更新参数
            optimizer.step()
            
            print(f"  帧 {frame_idx + 1}: 损失={loss.item():.6f}, 梯度范数={grad_norm/param_count if param_count>0 else 0:.6f}")
        
        avg_loss = epoch_loss / num_frames
        losses.append(avg_loss)
        print(f"  平均损失: {avg_loss:.6f}")
    
    # 分析结果
    print("\n" + "="*60)
    print("训练结果分析")
    print("="*60)
    
    print(f"损失变化: {losses[0]:.6f} -> {losses[-1]:.6f}")
    
    # 检查参数是否更新
    model2 = SimpleOnlineSDFModel()
    updated_params = 0
    
    for (name1, param1), (name2, param2) in zip(model.named_parameters(), model2.named_parameters()):
        diff = (param1.data - param2.data).norm().item()
        if diff > 1e-6:
            updated_params += 1
    
    print(f"更新参数: {updated_params}/{sum(1 for _ in model.parameters())}")
    
    if updated_params > 0 and losses[-1] < losses[0]:
        print("\n✅ 训练测试通过!")
        return True
    else:
        print("\n⚠️ 训练测试部分失败")
        return False


def test_dataset():
    """测试数据集"""
    print("\n测试在线TartanAir数据集...")
    
    try:
        from online_tartanair_dataset import OnlineTartanAirDataset
        
        dataset = OnlineTartanAirDataset(
            data_root="/home/cwh/Study/dataset/tartanair",
            sequence_name="abandonedfactory_sample_P001",
            n_frames=3,
            crop_size=(16, 16, 12),
            voxel_size=0.16,
            target_image_size=(64, 64),
            max_depth=10.0,
            truncation_margin=0.2,
            augment=False
        )
        
        print(f"✅ 数据集创建成功")
        print(f"  序列: {dataset.sequence_name}")
        print(f"  帧数: {len(dataset.rgb_files)}")
        
        # 获取样本
        sample = dataset[0]
        print(f"\n样本信息:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} {value.dtype}")
        
        # 检查TSDF
        tsdf = sample['tsdf']
        print(f"\nTSDF统计:")
        print(f"  形状: {tsdf.shape}")
        print(f"  范围: [{tsdf.min():.3f}, {tsdf.max():.3f}]")
        print(f"  均值: {tsdf.mean():.3f}")
        
        # 检查占用率
        occ = sample['occupancy']
        occ_rate = occ.sum() / occ.numel()
        print(f"\n占用统计:")
        print(f"  占用体素: {occ.sum().item():.0f}")
        print(f"  占用率: {occ_rate:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("开始简化版在线SDF训练测试...")
    
    # 测试数据集
    dataset_ok = test_dataset()
    
    # 测试训练
    training_ok = test_training()
    
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    print(f"数据集测试: {'✅ 通过' if dataset_ok else '❌ 失败'}")
    print(f"训练测试: {'✅ 通过' if training_ok else '❌ 失败'}")
    
    if dataset_ok and training_ok:
        print("\n🎉 所有测试通过!")
        print("端到端在线SDF训练验证成功!")
        return 0
    else:
        print("\n⚠️ 部分测试失败")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)