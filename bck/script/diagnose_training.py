#!/usr/bin/env python3
"""
诊断训练问题
"""

import os
import sys
import torch
import time
import traceback

print("=" * 60)
print("训练问题诊断")
print("=" * 60)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. 检查GPU内存
print("1. 检查GPU内存...")
if torch.cuda.is_available():
    device = torch.device('cuda')
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e6
    cached = torch.cuda.memory_reserved(0) / 1e6
    
    print(f"   GPU总内存: {total_memory:.2f} GB")
    print(f"   已分配: {allocated:.2f} MB")
    print(f"   缓存: {cached:.2f} MB")
else:
    print("   ⚠️  GPU不可用")
    device = torch.device('cpu')
print()

# 2. 尝试导入数据集
print("2. 尝试导入数据集...")
try:
    from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
    print("   ✅ 数据集导入成功")
except Exception as e:
    print(f"   ❌ 数据集导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)
print()

# 3. 尝试创建小数据集
print("3. 创建小数据集...")
try:
    start_time = time.time()
    
    dataset = MultiSequenceTartanAirDataset(
        data_root="/home/cwh/Study/dataset/tartanair",
        n_view=5,
        stride=2,
        crop_size=(48, 48, 32),
        voxel_size=0.04,
        target_image_size=(256, 256),
        max_sequences=1,  # 最小化
        shuffle=True
    )
    
    load_time = time.time() - start_time
    print(f"   ✅ 数据集创建成功")
    print(f"     大小: {len(dataset)} 个样本")
    print(f"     加载时间: {load_time:.2f}秒")
    
    # 限制为前5个样本
    from torch.utils.data import Subset
    small_dataset = Subset(dataset, indices=range(min(5, len(dataset))))
    print(f"     测试数据集: {len(small_dataset)} 个样本")
    
except Exception as e:
    print(f"   ❌ 数据集创建失败: {e}")
    traceback.print_exc()
    sys.exit(1)
print()

# 4. 测试数据加载
print("4. 测试数据加载...")
try:
    sample_times = []
    
    for i in range(3):  # 测试3个样本
        start_time = time.time()
        sample = dataset[i]
        sample_time = time.time() - start_time
        sample_times.append(sample_time)
        
        if i == 0:
            print(f"   样本 {i} 加载时间: {sample_time:.2f}秒")
            print(f"   样本形状:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"     {key}: {value.shape}")
    
    avg_time = sum(sample_times) / len(sample_times)
    print(f"   平均加载时间: {avg_time:.2f}秒")
    
except Exception as e:
    print(f"   ❌ 数据加载失败: {e}")
    traceback.print_exc()
print()

# 5. 测试模型创建
print("5. 测试模型创建...")
try:
    start_time = time.time()
    
    # 简单模型
    model = torch.nn.Sequential(
        torch.nn.Conv3d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv3d(8, 1, 3, padding=1)
    ).to(device)
    
    model_time = time.time() - start_time
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ 模型创建成功")
    print(f"     参数数量: {total_params:,}")
    print(f"     创建时间: {model_time:.2f}秒")
    print(f"     设备: {next(model.parameters()).device}")
    
except Exception as e:
    print(f"   ❌ 模型创建失败: {e}")
    traceback.print_exc()
print()

# 6. 测试训练循环（极简）
print("6. 测试训练循环...")
try:
    # 准备一个样本
    sample = dataset[0]
    rgb_images = sample['rgb_images'].unsqueeze(0).to(device)  # [1, 5, 3, 256, 256]
    tsdf_gt = sample['tsdf'].unsqueeze(0).to(device)           # [1, 1, 48, 48, 32]
    
    # 准备输入（使用第一帧）
    current_images = rgb_images[:, 0]  # [1, 3, 256, 256]
    current_images_resized = torch.nn.functional.interpolate(
        current_images,
        size=(48, 48),
        mode='bilinear',
        align_corners=False
    )
    input_3d = current_images_resized.unsqueeze(2).repeat(1, 1, 32, 1, 1)  # [1, 3, 32, 48, 48]
    
    print(f"   输入形状: {input_3d.shape}")
    print(f"   目标形状: {tsdf_gt.shape}")
    
    # 简单训练循环
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    start_time = time.time()
    
    for i in range(3):
        # 前向传播
        output = model(input_3d)
        loss = criterion(output, tsdf_gt)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   迭代 {i+1}: 损失 = {loss.item():.6f}")
    
    train_time = time.time() - start_time
    print(f"   训练时间: {train_time:.2f}秒")
    
    # 检查GPU内存
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(0) / 1e6
        cached = torch.cuda.memory_reserved(0) / 1e6
        max_allocated = torch.cuda.max_memory_allocated(0) / 1e6
        
        print(f"   GPU内存使用:")
        print(f"     当前已分配: {allocated:.2f} MB")
        print(f"     当前缓存: {cached:.2f} MB")
        print(f"     最大已分配: {max_allocated:.2f} MB")
    
    print("   ✅ 训练循环测试通过")
    
except Exception as e:
    print(f"   ❌ 训练循环失败: {e}")
    traceback.print_exc()
print()

print("=" * 60)
print("诊断完成")
print("=" * 60)

# 总结
print("\n📋 诊断总结:")
print("-" * 40)

if device.type == 'cuda':
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"1. GPU内存: {total_memory:.2f} GB - {'✅ 充足' if total_memory > 4 else '⚠️ 可能不足'}")
else:
    print("1. GPU: ❌ 不可用")

print(f"2. 数据集: ✅ 正常 ({len(dataset)} 样本)")
print(f"3. 样本加载: {avg_time:.2f}秒/样本 - {'✅ 正常' if avg_time < 2 else '⚠️ 较慢'}")
print(f"4. 模型训练: ✅ 基础训练循环通过")
print(f"5. 内存峰值: {max_allocated:.2f} MB - {'✅ 正常' if max_allocated < 2000 else '⚠️ 较高'}")

print("\n💡 建议:")
if avg_time > 1.5:
    print("  - 样本加载较慢，考虑优化数据预处理")
if max_allocated > 2000:
    print("  - GPU内存使用较高，减少批次大小")
print("  - 先运行小规模训练验证稳定性")