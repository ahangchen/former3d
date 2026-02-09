#!/usr/bin/env python3
"""
修复数据集问题 - 为流式训练准备数据
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_tartanair_dataset():
    """检查TartanAir数据集状态"""
    print("="*80)
    print("检查TartanAir数据集状态")
    print("="*80)
    
    data_root = "./tartanair_sdf_output"
    print(f"数据根目录: {data_root}")
    
    if not os.path.exists(data_root):
        print(f"❌ 目录不存在: {data_root}")
        return False
    
    # 查找所有文件
    files = list(Path(data_root).glob("*.json")) + list(Path(data_root).glob("*.npz"))
    print(f"找到 {len(files)} 个文件:")
    
    for f in files:
        print(f"  - {f.name}")
    
    # 检查文件配对
    json_files = list(Path(data_root).glob("*_metadata.json"))
    npz_files = list(Path(data_root).glob("*_sdf_occ.npz"))
    
    print(f"\nJSON文件: {len(json_files)} 个")
    print(f"NPZ文件: {len(npz_files)} 个")
    
    # 检查配对
    paired = []
    for json_file in json_files:
        base_name = json_file.name.replace("_metadata.json", "")
        npz_file = Path(data_root) / f"{base_name}_sdf_occ.npz"
        
        if npz_file.exists():
            paired.append((json_file, npz_file))
            print(f"✅ 配对成功: {base_name}")
        else:
            print(f"❌ 缺少NPZ文件: {base_name}")
    
    print(f"\n✅ 成功配对: {len(paired)} 对文件")
    
    # 检查文件内容
    if paired:
        json_file, npz_file = paired[0]
        print(f"\n检查第一个样本:")
        print(f"  JSON文件: {json_file}")
        print(f"  NPZ文件: {npz_file}")
        
        # 读取JSON
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            print(f"  ✅ JSON读取成功")
            print(f"    序列ID: {metadata.get('sequence_id', '未知')}")
            print(f"    帧数: {metadata.get('num_frames', '未知')}")
        except Exception as e:
            print(f"  ❌ JSON读取失败: {e}")
        
        # 读取NPZ
        try:
            data = np.load(npz_file)
            print(f"  ✅ NPZ读取成功")
            print(f"    包含的数组: {list(data.keys())}")
            
            for key in data.keys():
                arr = data[key]
                print(f"    {key}: {arr.shape}, {arr.dtype}")
        except Exception as e:
            print(f"  ❌ NPZ读取失败: {e}")
    
    return len(paired) > 0

def create_simulated_dataset(num_samples=10):
    """创建模拟数据集用于测试"""
    print("\n" + "="*80)
    print("创建模拟数据集")
    print("="*80)
    
    output_dir = "./simulated_tartanair"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"创建 {num_samples} 个模拟样本到: {output_dir}")
    
    for i in range(num_samples):
        # 创建元数据
        metadata = {
            "sequence_id": f"simulated_sequence_{i:03d}",
            "num_frames": 10,
            "scene_name": "simulated_scene",
            "resolution": [256, 256],
            "voxel_size": 0.04,
            "crop_size": [32, 32, 24],
            "is_simulated": True
        }
        
        # 创建数据
        data = {
            "sdf": np.random.randn(32, 32, 24).astype(np.float32),
            "occupancy": np.random.rand(32, 32, 24).astype(np.float32),
            "rgb_images": np.random.rand(10, 3, 128, 128).astype(np.float32),
            "poses": np.tile(np.eye(4), (10, 1, 1)).astype(np.float32),
            "intrinsics": np.tile(np.eye(3), (10, 1, 1)).astype(np.float32)
        }
        
        # 保存文件
        metadata_file = os.path.join(output_dir, f"simulated_sequence_{i:03d}_metadata.json")
        data_file = os.path.join(output_dir, f"simulated_sequence_{i:03d}_sdf_occ.npz")
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        np.savez_compressed(data_file, **data)
        
        print(f"  ✅ 创建样本 {i+1}/{num_samples}: {metadata['sequence_id']}")
    
    print(f"\n✅ 模拟数据集创建完成: {output_dir}")
    print(f"   包含 {num_samples} 个样本")
    return output_dir

def test_with_simulated_data():
    """使用模拟数据测试训练脚本"""
    print("\n" + "="*80)
    print("使用模拟数据测试训练脚本")
    print("="*80)
    
    # 创建模拟数据
    data_dir = create_simulated_dataset(5)
    
    # 测试训练脚本
    print(f"\n测试训练脚本使用数据目录: {data_dir}")
    
    try:
        # 导入必要的模块
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        from former3d.multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
        import torch
        
        print("✅ 模块导入成功")
        
        # 创建模型
        model = StreamSDFFormerIntegrated(
            attn_heads=2,
            attn_layers=1,
            voxel_size=0.04,
            crop_size=(32, 32, 24),
            use_proj_occ=True
        )
        print(f"✅ 模型创建成功，参数: {sum(p.numel() for p in model.parameters()):,}")
        
        # 创建数据集
        dataset = MultiSequenceTartanAirDataset(
            data_root=data_dir,
            sequence_length=5,
            max_sequences=5
        )
        print(f"✅ 数据集创建成功，样本数: {len(dataset)}")
        
        # 测试数据加载
        sample = dataset[0]
        print(f"✅ 数据加载成功")
        print(f"   样本键: {list(sample.keys())}")
        
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}, {value.dtype}")
            else:
                print(f"   {key}: {type(value)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("修复数据集问题 - 为流式训练准备数据")
    print("="*80)
    
    # 检查现有数据集
    has_data = check_tartanair_dataset()
    
    if not has_data:
        print("\n⚠️ 现有数据集不足，创建模拟数据集...")
        success = test_with_simulated_data()
        
        if success:
            print("\n" + "="*80)
            print("✅ 修复完成!")
            print("="*80)
            print("\n下一步:")
            print("1. 使用模拟数据测试训练:")
            print("   python train_stream_integrated.py --data-root ./simulated_tartanair --epochs 1 --batch-size 1")
            print("\n2. 收集更多真实数据到 tartanair_sdf_output 目录")
            print("\n3. 开始完整训练:")
            print("   python train_stream_integrated.py --epochs 10 --batch-size 2")
        else:
            print("\n❌ 修复失败，请检查错误信息")
    else:
        print("\n✅ 现有数据集可用")
        print("\n下一步:")
        print("1. 使用现有数据测试训练:")
        print("   python train_stream_integrated.py --epochs 1 --batch-size 1 --test-only")
        print("\n2. 开始完整训练:")
        print("   python train_stream_integrated.py --epochs 10 --batch-size 2")

if __name__ == "__main__":
    main()