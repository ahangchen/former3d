#!/usr/bin/env python3
"""
创建测试数据集 - 为流式训练准备兼容格式的数据
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
import shutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def convert_existing_data():
    """转换现有数据为训练脚本可用的格式"""
    print("="*80)
    print("转换现有数据为训练脚本可用的格式")
    print("="*80)
    
    source_dir = "./tartanair_sdf_output"
    target_dir = "./tartanair_training_data"
    
    if not os.path.exists(source_dir):
        print(f"❌ 源目录不存在: {source_dir}")
        return False
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 查找所有文件
    json_files = list(Path(source_dir).glob("*_metadata.json"))
    npz_files = list(Path(source_dir).glob("*_sdf_occ.npz"))
    
    print(f"找到 {len(json_files)} 个JSON文件和 {len(npz_files)} 个NPZ文件")
    
    converted_count = 0
    
    for json_file in json_files:
        base_name = json_file.name.replace("_metadata.json", "")
        npz_file = Path(source_dir) / f"{base_name}_sdf_occ.npz"
        
        if not npz_file.exists():
            print(f"❌ 缺少NPZ文件: {base_name}")
            continue
        
        try:
            # 读取原始数据
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            data = np.load(npz_file)
            
            # 创建新的目录结构
            seq_dir = os.path.join(target_dir, base_name)
            os.makedirs(seq_dir, exist_ok=True)
            
            # 创建新的元数据
            new_metadata = {
                "sequence_id": base_name,
                "scene_name": metadata.get("scene_name", "unknown"),
                "num_frames": metadata.get("num_frames", 10),
                "resolution": [256, 256],
                "voxel_size": float(data.get("voxel_size", 0.04)),
                "bounds": data.get("bounds", [[-1, 1], [-1, 1], [-1, 1]]).tolist(),
                "intrinsics": data.get("intrinsics", np.eye(3)).tolist(),
                "is_converted": True
            }
            
            # 保存新元数据
            with open(os.path.join(seq_dir, "metadata.json"), 'w') as f:
                json.dump(new_metadata, f, indent=2)
            
            # 创建模拟的RGB图像目录
            rgb_dir = os.path.join(seq_dir, "image_left")
            os.makedirs(rgb_dir, exist_ok=True)
            
            # 创建模拟的深度图像目录
            depth_dir = os.path.join(seq_dir, "depth_left")
            os.makedirs(depth_dir, exist_ok=True)
            
            # 创建模拟的位姿文件
            num_frames = new_metadata["num_frames"]
            poses = []
            
            # 生成简单的相机轨迹
            for i in range(num_frames):
                # 创建简单的平移运动
                pose = np.eye(4)
                pose[0, 3] = i * 0.1  # 沿X轴移动
                pose[1, 3] = 0.0
                pose[2, 3] = 0.0
                poses.append(pose)
            
            pose_file = os.path.join(seq_dir, "pose_left.txt")
            np.savetxt(pose_file, np.vstack(poses))
            
            # 创建模拟的RGB和深度文件（占位符）
            for i in range(min(num_frames, 10)):  # 最多创建10个文件
                # RGB文件（占位符）
                rgb_file = os.path.join(rgb_dir, f"{i:06d}_left.png")
                with open(rgb_file, 'w') as f:
                    f.write("placeholder")
                
                # 深度文件（占位符）
                depth_file = os.path.join(depth_dir, f"{i:06d}_left_depth.npy")
                np.save(depth_file, np.zeros((256, 256), dtype=np.float32))
            
            # 保存SDF数据
            sdf_data = {
                "sdf": data.get("sdf", np.zeros((32, 32, 24), dtype=np.float32)),
                "occupancy": data.get("occupancy", np.zeros((32, 32, 24), dtype=np.float32)),
                "voxel_size": float(data.get("voxel_size", 0.04)),
                "bounds": data.get("bounds", [[-1, 1], [-1, 1], [-1, 1]]),
                "intrinsics": data.get("intrinsics", np.eye(3))
            }
            
            sdf_file = os.path.join(seq_dir, "sdf_data.npz")
            np.savez_compressed(sdf_file, **sdf_data)
            
            print(f"✅ 转换序列: {base_name}")
            print(f"   帧数: {num_frames}")
            print(f"   体素大小: {new_metadata['voxel_size']}")
            print(f"   保存到: {seq_dir}")
            
            converted_count += 1
            
        except Exception as e:
            print(f"❌ 转换失败 {base_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✅ 转换完成: {converted_count} 个序列")
    print(f"   目标目录: {target_dir}")
    
    return converted_count > 0

def create_minimal_test_dataset():
    """创建最小测试数据集"""
    print("\n" + "="*80)
    print("创建最小测试数据集")
    print("="*80)
    
    target_dir = "./minimal_test_data"
    os.makedirs(target_dir, exist_ok=True)
    
    # 创建3个测试序列
    sequences = ["test_seq_001", "test_seq_002", "test_seq_003"]
    
    for seq_name in sequences:
        seq_dir = os.path.join(target_dir, seq_name)
        os.makedirs(seq_dir, exist_ok=True)
        
        # 创建元数据
        metadata = {
            "sequence_id": seq_name,
            "scene_name": "test_scene",
            "num_frames": 5,
            "resolution": [128, 128],
            "voxel_size": 0.04,
            "bounds": [[-0.64, 0.64], [-0.64, 0.64], [-0.48, 0.48]],
            "intrinsics": [[320, 0, 64], [0, 320, 64], [0, 0, 1]],
            "is_test": True
        }
        
        with open(os.path.join(seq_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 创建RGB目录和文件
        rgb_dir = os.path.join(seq_dir, "image_left")
        os.makedirs(rgb_dir, exist_ok=True)
        
        for i in range(5):
            rgb_file = os.path.join(rgb_dir, f"{i:06d}_left.png")
            with open(rgb_file, 'w') as f:
                f.write("placeholder")
        
        # 创建深度目录和文件
        depth_dir = os.path.join(seq_dir, "depth_left")
        os.makedirs(depth_dir, exist_ok=True)
        
        for i in range(5):
            depth_file = os.path.join(depth_dir, f"{i:06d}_left_depth.npy")
            np.save(depth_file, np.random.randn(128, 128).astype(np.float32))
        
        # 创建位姿文件
        poses = []
        for i in range(5):
            pose = np.eye(4)
            pose[0, 3] = i * 0.05  # 简单的平移
            poses.append(pose)
        
        pose_file = os.path.join(seq_dir, "pose_left.txt")
        np.savetxt(pose_file, np.vstack(poses))
        
        # 创建SDF数据
        sdf_data = {
            "sdf": np.random.randn(32, 32, 24).astype(np.float32),
            "occupancy": np.random.rand(32, 32, 24).astype(np.float32),
            "voxel_size": 0.04,
            "bounds": np.array([[-0.64, 0.64], [-0.64, 0.64], [-0.48, 0.48]]),
            "intrinsics": np.array([[320, 0, 64], [0, 320, 64], [0, 0, 1]])
        }
        
        sdf_file = os.path.join(seq_dir, "sdf_data.npz")
        np.savez_compressed(sdf_file, **sdf_data)
        
        print(f"✅ 创建测试序列: {seq_name}")
    
    print(f"\n✅ 最小测试数据集创建完成: {target_dir}")
    print(f"   包含 {len(sequences)} 个序列，每个5帧")
    
    return target_dir

def test_dataset_compatibility(data_dir):
    """测试数据集兼容性"""
    print("\n" + "="*80)
    print("测试数据集兼容性")
    print("="*80)
    
    try:
        from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
        
        print(f"测试数据目录: {data_dir}")
        
        dataset = MultiSequenceTartanAirDataset(
            data_root=data_dir,
            n_view=3,  # 相当于sequence_length
            max_sequences=3
        )
        
        print(f"✅ 数据集创建成功")
        print(f"   序列数: {len(dataset.sequences)}")
        print(f"   总样本数: {len(dataset)}")
        
        # 测试获取样本
        sample = dataset[0]
        print(f"✅ 样本加载成功")
        print(f"   样本键: {list(sample.keys())}")
        
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}, {value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"   {key}: {type(value)}, 长度: {len(value)}")
            else:
                print(f"   {key}: {type(value)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据集兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("创建测试数据集 - 为流式训练准备兼容格式的数据")
    print("="*80)
    
    # 尝试转换现有数据
    print("\n1. 尝试转换现有数据...")
    converted = convert_existing_data()
    
    if converted:
        data_dir = "./tartanair_training_data"
        print(f"\n✅ 现有数据转换成功，使用目录: {data_dir}")
    else:
        print("\n⚠️ 现有数据转换失败，创建最小测试数据集...")
        data_dir = create_minimal_test_dataset()
    
    # 测试数据集兼容性
    print(f"\n2. 测试数据集兼容性...")
    compatible = test_dataset_compatibility(data_dir)
    
    if compatible:
        print("\n" + "="*80)
        print("✅ 数据集准备完成!")
        print("="*80)
        
        print(f"\n数据集目录: {data_dir}")
        print("\n下一步:")
        print("1. 使用测试数据运行训练:")
        print(f"   python train_stream_integrated.py --data-root {data_dir} --epochs 1 --batch-size 1 --test-only")
        print("\n2. 开始完整训练:")
        print(f"   python train_stream_integrated.py --data-root {data_dir} --epochs 3 --batch-size 1")
        print("\n3. 收集更多真实数据并更新数据集")
    else:
        print("\n❌ 数据集兼容性测试失败")
        print("\n建议:")
        print("1. 检查数据集目录结构")
        print("2. 确保有足够的序列数据")
        print("3. 检查MultiSequenceTartanAirDataset的实现")

if __name__ == "__main__":
    main()