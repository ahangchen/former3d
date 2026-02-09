#!/usr/bin/env python
"""
测试TartanAir数据集加载功能 - 修复版本
直接使用样本目录作为数据根目录
"""

import torch
import torch.distributed as dist
import numpy as np
import os
import sys
import time
from pathlib import Path

print("="*80)
print("TartanAir数据集加载测试 - 修复版本")
print("="*80)

# 初始化分布式环境
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
dist.init_process_group('nccl', rank=0, world_size=1)

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_direct_loading():
    """直接测试数据集加载"""
    print("\n" + "="*60)
    print("直接测试数据集加载")
    print("="*60)
    
    try:
        from former3d.datasets.tartanair_dataset import TartanAirStreamingDataset
        
        # 使用样本目录作为数据根目录
        data_root = "/home/cwh/Study/dataset/tartanair"
        
        print(f"数据根目录: {data_root}")
        
        # 直接创建一个简单的数据集，不指定序列ID
        print("\n创建数据集（不指定序列ID）...")
        dataset = TartanAirStreamingDataset(
            data_root=data_root,
            split='train',
            sequence_ids=None,  # 让数据集自己发现序列
            load_depth=False,
            load_sdf=False,
            max_sequence_length=5,
            image_size=(256, 256),
            normalize_images=True,
            cache_data=False,
            use_left_camera=True,
            frame_interval=5
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) == 0:
            print("❌ 数据集为空，检查目录结构")
            
            # 调试：检查数据集内部状态
            print(f"\n调试信息:")
            print(f"  data_root: {dataset.data_root}")
            print(f"  sequence_info: {dataset.sequence_info}")
            print(f"  frame_indices: {len(dataset.frame_indices)}")
            
            # 尝试手动查找序列
            print(f"\n手动查找序列:")
            env_dirs = sorted([d for d in Path(data_root).iterdir() if d.is_dir()])
            print(f"  环境目录数量: {len(env_dirs)}")
            
            for env_dir in env_dirs[:3]:  # 显示前3个
                print(f"    环境: {env_dir.name}")
                
                # 检查是否是样本目录
                if "_sample_" in env_dir.name:
                    print(f"      -> 样本目录")
                    # 查找Pxxx子目录
                    for subitem in env_dir.iterdir():
                        if subitem.is_dir() and subitem.name.startswith('P'):
                            print(f"        轨迹: {subitem.name}")
                            # 检查必要文件
                            image_dir = subitem / "image_left"
                            pose_file = subitem / "pose_left.txt"
                            print(f"          图像目录: {image_dir.exists()}")
                            print(f"          位姿文件: {pose_file.exists()}")
            
            return None
        
        print(f"✅ 数据集创建成功，包含 {len(dataset)} 帧")
        
        # 测试加载第一帧
        print(f"\n测试加载第一帧:")
        try:
            data = dataset[0]
            print(f"✅ 数据加载成功")
            print(f"  数据键: {list(data.keys())}")
            
            if 'image' in data:
                print(f"  图像形状: {data['image'].shape}")
            
            if 'pose' in data:
                print(f"  位姿形状: {data['pose'].shape}")
            
            return dataset
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_simple_dataset():
    """创建简单的自定义数据集"""
    print("\n" + "="*60)
    print("创建简单的自定义数据集")
    print("="*60)
    
    try:
        import torch
        from torch.utils.data import Dataset
        from PIL import Image
        import numpy as np
        
        class SimpleTartanAirDataset(Dataset):
            """简单的TartanAir数据集，直接加载样本目录"""
            
            def __init__(self, sample_dir, max_frames=10):
                self.sample_dir = Path(sample_dir)
                self.max_frames = max_frames
                
                # 查找Pxxx子目录
                self.traj_dirs = []
                for item in self.sample_dir.iterdir():
                    if item.is_dir() and item.name.startswith('P'):
                        self.traj_dirs.append(item)
                
                print(f"找到轨迹目录: {len(self.traj_dirs)} 个")
                
                # 收集所有帧
                self.frames = []
                for traj_dir in self.traj_dirs:
                    image_dir = traj_dir / "image_left"
                    pose_file = traj_dir / "pose_left.txt"
                    
                    if not (image_dir.exists() and pose_file.exists()):
                        continue
                    
                    # 加载位姿
                    poses = np.loadtxt(pose_file)
                    if poses.ndim == 1:
                        poses = poses.reshape(1, -1)
                    
                    # 获取图像文件
                    image_files = sorted(image_dir.glob("*.png"))
                    
                    # 限制帧数
                    num_frames = min(len(image_files), self.max_frames, poses.shape[0])
                    
                    for i in range(num_frames):
                        self.frames.append({
                            'traj_dir': traj_dir,
                            'image_path': image_files[i],
                            'pose_idx': i,
                            'poses': poses
                        })
                
                print(f"总帧数: {len(self.frames)}")
            
            def __len__(self):
                return len(self.frames)
            
            def __getitem__(self, idx):
                frame_info = self.frames[idx]
                
                # 加载图像
                image = Image.open(frame_info['image_path'])
                image = image.resize((256, 256))
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # HWC -> CHW
                
                # 加载位姿
                pose_row = frame_info['poses'][frame_info['pose_idx']]
                
                # TartanAir位姿格式: [x, y, z, qx, qy, qz, qw]
                if pose_row.shape[0] == 7:
                    # 四元数+平移向量格式
                    x, y, z, qx, qy, qz, qw = pose_row
                    
                    # 将四元数转换为旋转矩阵
                    from scipy.spatial.transform import Rotation
                    rot = Rotation.from_quat([qx, qy, qz, qw])
                    rot_matrix = rot.as_matrix()
                    
                    # 创建4x4位姿矩阵
                    pose_matrix = np.eye(4, dtype=np.float32)
                    pose_matrix[:3, :3] = rot_matrix
                    pose_matrix[:3, 3] = [x, y, z]
                else:
                    # 假设已经是矩阵格式
                    pose_matrix = pose_row.reshape(4, 4)
                
                pose_tensor = torch.from_numpy(pose_matrix)
                
                # 固定内参（TartanAir 640x480）
                intrinsic = np.array([
                    [320.0, 0.0, 320.0],
                    [0.0, 320.0, 240.0],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float32)
                
                # 调整内参到256x256
                scale_h = 256 / 480
                scale_w = 256 / 640
                intrinsic_scaled = intrinsic.copy()
                intrinsic_scaled[0, 0] *= scale_w
                intrinsic_scaled[1, 1] *= scale_h
                intrinsic_scaled[0, 2] *= scale_w
                intrinsic_scaled[1, 2] *= scale_h
                
                intrinsic_tensor = torch.from_numpy(intrinsic_scaled)
                
                return {
                    'image': image_tensor,
                    'pose': pose_tensor,
                    'intrinsic': intrinsic_tensor,
                    'sequence_id': frame_info['traj_dir'].name,
                    'frame_idx': frame_info['pose_idx']
                }
        
        # 使用第一个样本目录
        sample_dirs = []
        data_root = Path("/home/cwh/Study/dataset/tartanair")
        for item in data_root.iterdir():
            if item.is_dir() and "_sample_" in item.name:
                sample_dirs.append(item)
        
        if len(sample_dirs) == 0:
            print("❌ 没有找到样本目录")
            return None
        
        sample_dir = sample_dirs[0]
        print(f"使用样本目录: {sample_dir}")
        
        dataset = SimpleTartanAirDataset(sample_dir, max_frames=5)
        
        if len(dataset) == 0:
            print("❌ 数据集为空")
            return None
        
        print(f"✅ 自定义数据集创建成功，包含 {len(dataset)} 帧")
        
        # 测试加载
        print(f"\n测试数据加载:")
        for i in range(min(3, len(dataset))):
            data = dataset[i]
            print(f"\n帧 {i}:")
            print(f"  图像形状: {data['image'].shape}")
            print(f"  位姿形状: {data['pose'].shape}")
            print(f"  内参形状: {data['intrinsic'].shape}")
            print(f"  序列ID: {data['sequence_id']}")
            print(f"  帧索引: {data['frame_idx']}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ 自定义数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_with_original_structure():
    """使用原始TartanAir目录结构测试"""
    print("\n" + "="*60)
    print("使用原始TartanAir目录结构测试")
    print("="*60)
    
    try:
        # 检查是否有解压的完整数据集
        data_root = Path("/home/cwh/Study/dataset/tartanair")
        
        # 查找非样本目录
        env_dirs = []
        for item in data_root.iterdir():
            if item.is_dir() and "_sample_" not in item.name and not item.name.endswith('.zip'):
                env_dirs.append(item)
        
        print(f"找到的环境目录: {len(env_dirs)} 个")
        for env_dir in env_dirs[:5]:
            print(f"  环境: {env_dir.name}")
            
            # 检查是否有Easy目录
            easy_dir = env_dir / "Easy"
            if easy_dir.exists():
                print(f"    Easy目录存在")
                # 查找Pxxx子目录
                p_dirs = []
                for subitem in easy_dir.iterdir():
                    if subitem.is_dir() and subitem.name.startswith('P'):
                        p_dirs.append(subitem.name)
                
                print(f"    轨迹目录: {len(p_dirs)} 个")
                if p_dirs:
                    print(f"    示例: {p_dirs[0]}")
        
        # 如果没有解压的数据，建议解压一个样本
        if len(env_dirs) == 0:
            print("\n⚠️ 没有找到解压的完整数据集")
            print("建议解压一个样本目录的zip文件:")
            print("  例如: unzip abandonedfactory_Easy_image_left.zip -d /tmp/tartanair_test")
            print("  然后使用解压后的目录作为数据根目录")
        
        return len(env_dirs) > 0
        
    except Exception as e:
        print(f"❌ 目录结构检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("\n" + "="*80)
    print("TartanAir数据集加载测试 - 多种方法")
    print("="*80)
    
    # 方法1: 使用原始数据集类
    print("\n方法1: 使用原始TartanAirStreamingDataset类")
    dataset1 = test_direct_loading()
    
    # 方法2: 使用自定义简单数据集
    print("\n方法2: 使用自定义简单数据集")
    dataset2 = test_simple_dataset()
    
    # 方法3: 检查原始目录结构
    print("\n方法3: 检查原始目录结构")
    has_original_structure = test_with_original_structure()
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    
    print(f"方法1 (原始类): {'✅ 成功' if dataset1 is not None else '❌ 失败'}")
    print(f"方法2 (自定义): {'✅ 成功' if dataset2 is not None else '❌ 失败'}")
    print(f"方法3 (目录结构): {'✅ 有完整结构' if has_original_structure else '⚠️ 只有样本'}")
    
    print("\n" + "="*80)
    if dataset2 is not None:
        print("🎉 成功创建自定义数据集！")
        print("下一步：基于这个数据集创建训练循环。")
        
        # 建议下一步
        print("\n建议:")
        print("1. 使用自定义数据集进行训练循环测试")
        print("2. 解压完整的TartanAir数据集以获得更多训练数据")
        print("3. 调整数据集类以匹配实际目录结构")
    else:
        print("⚠️ 需要进一步调试数据集加载。")
    
    print("="*80)

if __name__ == "__main__":
    main()