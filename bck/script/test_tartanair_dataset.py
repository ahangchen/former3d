#!/usr/bin/env python
"""
测试TartanAir数据集加载功能
验证是否能正常加载磁盘中的TartanAir数据
"""

import torch
import torch.distributed as dist
import numpy as np
import os
import sys
import time
from pathlib import Path

print("="*80)
print("TartanAir数据集加载测试")
print("="*80)

# 初始化分布式环境
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group('nccl', rank=0, world_size=1)

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dataset_loading():
    """测试数据集加载功能"""
    print("\n" + "="*60)
    print("测试1: 数据集加载")
    print("="*60)
    
    try:
        # 导入数据集类
        from former3d.datasets.tartanair_dataset import TartanAirStreamingDataset
        
        # TartanAir数据根目录
        data_root = "/home/cwh/Study/dataset/tartanair"
        
        print(f"数据根目录: {data_root}")
        print(f"目录存在: {os.path.exists(data_root)}")
        
        # 检查样本目录
        sample_dirs = []
        for item in os.listdir(data_root):
            if "_sample_" in item:
                sample_dirs.append(item)
        
        print(f"找到的样本目录: {len(sample_dirs)} 个")
        for i, dir_name in enumerate(sample_dirs[:5]):  # 显示前5个
            print(f"  {i+1}. {dir_name}")
        
        if len(sample_dirs) == 0:
            print("❌ 没有找到样本目录")
            return False
        
        # 使用第一个样本目录创建数据集
        sample_dir = sample_dirs[0]
        print(f"\n使用样本目录: {sample_dir}")
        
        # 检查目录结构
        sample_path = Path(data_root) / sample_dir
        print(f"样本路径: {sample_path}")
        
        # 查找Pxxx子目录
        p_dirs = []
        for item in sample_path.iterdir():
            if item.is_dir() and item.name.startswith('P'):
                p_dirs.append(item.name)
        
        print(f"找到的轨迹目录: {p_dirs}")
        
        if len(p_dirs) == 0:
            print("❌ 没有找到轨迹目录")
            return False
        
        # 使用第一个轨迹目录
        traj_dir = p_dirs[0]
        print(f"使用轨迹目录: {traj_dir}")
        
        # 检查轨迹目录内容
        traj_path = sample_path / traj_dir
        print(f"轨迹路径: {traj_path}")
        
        # 列出轨迹目录内容
        print(f"轨迹目录内容:")
        for item in traj_path.iterdir():
            if item.is_dir():
                print(f"  📁 {item.name}")
            else:
                print(f"  📄 {item.name} ({item.stat().st_size} bytes)")
        
        # 检查必要文件
        image_dir = traj_path / "image_left"
        pose_file = traj_path / "pose_left.txt"
        
        print(f"\n检查必要文件:")
        print(f"  图像目录: {image_dir} - 存在: {image_dir.exists()}")
        print(f"  位姿文件: {pose_file} - 存在: {pose_file.exists()}")
        
        if not (image_dir.exists() and pose_file.exists()):
            print("❌ 缺少必要文件")
            return False
        
        # 检查图像文件
        image_files = list(image_dir.glob("*.png"))
        print(f"  图像文件数量: {len(image_files)}")
        
        if len(image_files) == 0:
            print("❌ 没有图像文件")
            return False
        
        # 检查位姿文件
        try:
            poses = np.loadtxt(pose_file)
            print(f"  位姿数量: {poses.shape[0] if poses.ndim > 1 else 1}")
            print(f"  位姿形状: {poses.shape}")
        except Exception as e:
            print(f"❌ 位姿文件加载失败: {e}")
            return False
        
        print("\n✅ 数据集结构检查通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据集加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_instantiation():
    """测试数据集实例化"""
    print("\n" + "="*60)
    print("测试2: 数据集实例化")
    print("="*60)
    
    try:
        from former3d.datasets.tartanair_dataset import TartanAirStreamingDataset
        
        data_root = "/home/cwh/Study/dataset/tartanair"
        
        # 查找所有样本目录
        sample_dirs = []
        for item in os.listdir(data_root):
            if "_sample_" in item:
                sample_dir = Path(data_root) / item
                # 查找Pxxx子目录
                for subitem in sample_dir.iterdir():
                    if subitem.is_dir() and subitem.name.startswith('P'):
                        # 序列ID格式: env/difficulty/trajectory
                        # 对于样本目录，我们使用简化的格式
                        seq_id = f"{item}/Easy/{subitem.name}"
                        sample_dirs.append(seq_id)
        
        if len(sample_dirs) == 0:
            print("❌ 没有找到有效的序列")
            return False
        
        print(f"找到的序列: {len(sample_dirs)} 个")
        for i, seq_id in enumerate(sample_dirs[:3]):  # 显示前3个
            print(f"  {i+1}. {seq_id}")
        
        # 使用前2个序列创建数据集
        selected_sequences = sample_dirs[:2]
        
        print(f"\n创建数据集，使用序列: {selected_sequences}")
        
        dataset = TartanAirStreamingDataset(
            data_root=data_root,
            split='train',
            sequence_ids=selected_sequences,
            load_depth=False,
            load_sdf=False,
            max_sequence_length=10,  # 限制序列长度
            image_size=(256, 256),
            normalize_images=True,
            cache_data=False,
            use_left_camera=True,
            frame_interval=5  # 每5帧采样一帧
        )
        
        print(f"✅ 数据集创建成功")
        print(f"  总帧数: {len(dataset)}")
        print(f"  序列数: {len(dataset.sequence_info)}")
        
        # 显示序列信息
        for seq_id, seq_info in dataset.sequence_info.items():
            print(f"\n  序列 {seq_id}:")
            print(f"    路径: {seq_info['path']}")
            print(f"    帧索引: {seq_info['start_idx']} - {seq_info['end_idx']}")
            print(f"    采样帧数: {seq_info['length']}")
            print(f"    总帧数: {seq_info['total_frames']}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ 数据集实例化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_data_iteration(dataset):
    """测试数据迭代"""
    print("\n" + "="*60)
    print("测试3: 数据迭代")
    print("="*60)
    
    if dataset is None:
        print("❌ 数据集为空")
        return False
    
    try:
        # 测试前5个样本
        num_samples = min(5, len(dataset))
        print(f"测试前 {num_samples} 个样本:")
        
        for i in range(num_samples):
            print(f"\n样本 {i+1}/{num_samples}:")
            
            # 获取数据
            data = dataset[i]
            
            # 检查数据键
            print(f"  数据键: {list(data.keys())}")
            
            # 检查图像
            if 'image' in data:
                image = data['image']
                print(f"  图像形状: {image.shape}")
                print(f"  图像类型: {image.dtype}")
                print(f"  图像范围: [{image.min():.3f}, {image.max():.3f}]")
            
            # 检查位姿
            if 'pose' in data:
                pose = data['pose']
                print(f"  位姿形状: {pose.shape}")
                print(f"  位姿类型: {pose.dtype}")
                # 显示旋转和平移部分
                if pose.shape == (4, 4):
                    print(f"  旋转矩阵:\n{pose[:3, :3]}")
                    print(f"  平移向量: {pose[:3, 3]}")
            
            # 检查内参
            if 'intrinsic' in data:
                intrinsic = data['intrinsic']
                print(f"  内参形状: {intrinsic.shape}")
                print(f"  内参:\n{intrinsic}")
            
            # 检查序列信息
            if 'sequence_id' in data:
                print(f"  序列ID: {data['sequence_id']}")
            
            if 'frame_idx' in data:
                print(f"  帧索引: {data['frame_idx']}")
            
            # 检查深度图（如果加载）
            if 'depth' in data and data['depth'] is not None:
                depth = data['depth']
                print(f"  深度图形状: {depth.shape}")
                print(f"  深度图范围: [{depth.min():.3f}, {depth.max():.3f}]")
            
            # 检查SDF真值（如果加载）
            if 'sdf' in data and data['sdf'] is not None:
                sdf = data['sdf']
                print(f"  SDF形状: {sdf.shape}")
                print(f"  SDF范围: [{sdf.min():.3f}, {sdf.max():.3f}]")
            
            # 检查占用真值（如果加载）
            if 'occ' in data and data['occ'] is not None:
                occ = data['occ']
                print(f"  占用形状: {occ.shape}")
                print(f"  占用范围: [{occ.min():.3f}, {occ.max():.3f}]")
        
        print(f"\n✅ 数据迭代测试完成")
        return True
        
    except Exception as e:
        print(f"❌ 数据迭代失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader():
    """测试DataLoader"""
    print("\n" + "="*60)
    print("测试4: DataLoader")
    print("="*60)
    
    try:
        from former3d.datasets.tartanair_dataset import TartanAirStreamingDataset
        from torch.utils.data import DataLoader
        
        data_root = "/home/cwh/Study/dataset/tartanair"
        
        # 创建数据集
        dataset = TartanAirStreamingDataset(
            data_root=data_root,
            split='train',
            sequence_ids=None,  # 使用所有找到的序列
            load_depth=False,
            load_sdf=False,
            max_sequence_length=5,
            image_size=(256, 256),
            normalize_images=True,
            cache_data=False,
            use_left_camera=True,
            frame_interval=10
        )
        
        if len(dataset) == 0:
            print("❌ 数据集为空")
            return False
        
        print(f"数据集大小: {len(dataset)}")
        
        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,  # 在测试中使用0避免多进程问题
            pin_memory=True
        )
        
        print(f"DataLoader创建成功")
        print(f"  批次大小: {2}")
        print(f"  是否打乱: {True}")
        
        # 测试一个批次
        print(f"\n测试一个批次:")
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            print(f"\n批次 {batch_idx + 1}:")
            
            # 检查批次数据
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
                elif isinstance(value, list):
                    print(f"  {key}: 列表长度={len(value)}")
                else:
                    print(f"  {key}: {type(value)}")
            
            # 只测试第一个批次
            if batch_idx == 0:
                break
        
        elapsed_time = time.time() - start_time
        print(f"\n批次加载时间: {elapsed_time:.3f}秒")
        
        print(f"\n✅ DataLoader测试完成")
        return True
        
    except Exception as e:
        print(f"❌ DataLoader测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("TartanAir数据集完整测试")
    print("="*80)
    
    # 运行所有测试
    test_results = []
    
    # 测试1: 数据集加载
    test1_success = test_dataset_loading()
    test_results.append(("数据集加载", test1_success))
    
    # 测试2: 数据集实例化
    dataset = test_dataset_instantiation()
    test2_success = dataset is not None
    test_results.append(("数据集实例化", test2_success))
    
    # 测试3: 数据迭代
    if dataset is not None:
        test3_success = test_data_iteration(dataset)
        test_results.append(("数据迭代", test3_success))
    else:
        test_results.append(("数据迭代", False))
    
    # 测试4: DataLoader
    test4_success = test_dataloader()
    test_results.append(("DataLoader", test4_success))
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    all_passed = True
    for test_name, success in test_results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("🎉 所有测试通过！TartanAir数据集加载功能正常。")
        print("下一步：基于这些场景组织训练集，执行端到端训练循环验证。")
    else:
        print("⚠️ 部分测试失败，需要进一步调试。")
    
    print("="*80)

if __name__ == "__main__":
    main()