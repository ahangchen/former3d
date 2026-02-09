#!/usr/bin/env python3
"""
调试训练 - 找出问题所在
"""

import os
import sys
import torch

print("=" * 60)
print("调试训练")
print("=" * 60)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 检查GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"1. 设备检查: {device}")
if device.type == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
print()

# 尝试导入数据集
print("2. 尝试导入数据集...")
try:
    from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
    print("   ✅ 数据集导入成功")
    
    # 尝试创建数据集
    print("3. 尝试创建数据集实例...")
    try:
        dataset = MultiSequenceTartanAirDataset(
            data_root="/home/cwh/Study/dataset/tartanair",
            n_view=5,
            stride=2,
            crop_size=(48, 48, 32),
            voxel_size=0.04,
            target_image_size=(256, 256),
            max_sequences=1,  # 只用一个序列加快速度
            shuffle=True
        )
        print(f"   ✅ 数据集创建成功，大小: {len(dataset)}")
        
        # 尝试获取一个样本
        print("4. 尝试获取样本...")
        try:
            sample = dataset[0]
            print(f"   ✅ 样本获取成功")
            print(f"     样本键: {list(sample.keys())}")
            
            # 检查数据形状
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"     {key}: {value.shape} ({value.device})")
                elif isinstance(value, str):
                    print(f"     {key}: '{value}'")
                elif isinstance(value, int):
                    print(f"     {key}: {value}")
                    
        except Exception as e:
            print(f"   ❌ 样本获取失败: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"   ❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"   ❌ 数据集导入失败: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("调试完成")
print("=" * 60)