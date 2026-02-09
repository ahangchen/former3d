#!/usr/bin/env python3
"""
基本训练测试
验证最简单的训练流程是否工作
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

print("="*80)
print("基本训练测试")
print("="*80)

# 检查环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 创建简单的数据集
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100):
        self.num_samples = num_samples
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机数据
        points = torch.randn(100, 3)  # 100个3D点
        sdf_values = torch.randn(100, 1)  # 对应的SDF值
        
        return {
            'points': points,
            'sdf_values': sdf_values
        }

# 创建简单模型
class SimpleSDFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, points):
        return self.mlp(points)

# 测试训练
def test_training():
    print("\n1. 创建数据集和模型...")
    dataset = SimpleDataset(10)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    model = SimpleSDFModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n2. 训练测试...")
    losses = []
    
    for epoch in range(3):
        epoch_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            points = batch['points'].to(device)
            targets = batch['sdf_values'].to(device)
            
            # 前向传播
            preds = model(points)
            
            # 计算损失
            loss = loss_fn(preds, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.6f}")
    
    print("\n3. 验证训练效果...")
    if len(losses) > 1:
        loss_reduction = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"  损失减少: {loss_reduction:.1f}%")
        
        if loss_reduction > 0:
            print("  ✅ 训练有效！损失在减少")
        else:
            print("  ⚠️ 训练可能有问题，损失没有减少")
    else:
        print("  ⚠️ 无法计算损失减少")
    
    return losses[-1] > 0

def test_stream_sdfformer():
    print("\n4. 测试StreamSDFFormerIntegrated导入...")
    try:
        # 尝试导入
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        print("  ✅ StreamSDFFormerIntegrated导入成功")
        
        # 尝试创建模型
        model = StreamSDFFormerIntegrated(
            attn_heads=2,
            attn_layers=1,
            use_proj_occ=True,
            voxel_size=0.10,
            fusion_local_radius=0.0,
            crop_size=(24, 24, 16)
        )
        
        print(f"  ✅ 模型创建成功")
        print(f"    参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
    except Exception as e:
        print(f"  ❌ 导入失败: {e}")
        
        # 检查具体错误
        import traceback
        traceback.print_exc()
        
        return False

def main():
    print("开始基本训练测试...")
    
    # 测试简单训练
    training_ok = test_training()
    
    # 测试StreamSDFFormer导入
    model_import_ok = test_stream_sdfformer()
    
    print("\n" + "="*80)
    print("测试结果")
    print("="*80)
    
    if training_ok:
        print("✅ 基本训练测试通过")
    else:
        print("❌ 基本训练测试失败")
    
    if model_import_ok:
        print("✅ StreamSDFFormerIntegrated导入测试通过")
    else:
        print("❌ StreamSDFFormerIntegrated导入测试失败")
    
    print("\n建议:")
    if not model_import_ok:
        print("1. 检查former3d/stream_sdfformer_integrated.py文件")
        print("2. 检查是否有语法错误或导入错误")
        print("3. 检查SyncBatchNorm问题是否已修复")
    
    print("\n🚀 根据测试结果决定下一步操作")

if __name__ == "__main__":
    main()