#!/usr/bin/env python
"""
调试TartanAir位姿加载问题
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def debug_pose_loading():
    """调试位姿加载"""
    print("="*80)
    print("调试TartanAir位姿加载")
    print("="*80)
    
    # 测试位姿文件
    pose_file = "/tmp/tartanair_test/abandonedfactory/Easy/P000/pose_left.txt"
    print(f"位姿文件: {pose_file}")
    
    # 检查文件是否存在
    if not os.path.exists(pose_file):
        print("❌ 位姿文件不存在")
        return
    
    # 读取位姿数据
    try:
        poses = np.loadtxt(pose_file)
        print(f"✅ 位姿文件读取成功")
        print(f"  位姿数量: {poses.shape[0]}")
        print(f"  每行维度: {poses.shape[1]}")
        
        # 显示前5个位姿
        print(f"\n前5个位姿:")
        for i in range(min(5, poses.shape[0])):
            pose = poses[i]
            print(f"  位姿 {i}:")
            print(f"    位置: [{pose[0]:.6f}, {pose[1]:.6f}, {pose[2]:.6f}]")
            print(f"    四元数: [{pose[3]:.6f}, {pose[4]:.6f}, {pose[5]:.6f}, {pose[6]:.6f}]")
            
            # 检查四元数范数
            q = pose[3:7]
            norm = np.linalg.norm(q)
            print(f"    四元数范数: {norm:.6f} {'✅' if abs(norm - 1.0) < 0.001 else '⚠️'}")
        
        # 检查是否有NaN或Inf
        has_nan = np.any(np.isnan(poses))
        has_inf = np.any(np.isinf(poses))
        print(f"\n数据检查:")
        print(f"  包含NaN: {has_nan}")
        print(f"  包含Inf: {has_inf}")
        
        if has_nan or has_inf:
            print("⚠️ 位姿数据包含NaN或Inf值")
        
        # 检查数据范围
        print(f"\n数据范围:")
        for i in range(poses.shape[1]):
            col_min = poses[:, i].min()
            col_max = poses[:, i].max()
            col_mean = poses[:, i].mean()
            col_std = poses[:, i].std()
            
            if i < 3:
                label = f"位置[{i}]"
            elif i < 7:
                label = f"四元数[{i-3}]"
            else:
                label = f"未知[{i}]"
            
            print(f"  {label}: 范围=[{col_min:.6f}, {col_max:.6f}], 均值={col_mean:.6f}, 标准差={col_std:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 位姿文件读取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pose_conversion():
    """测试位姿转换"""
    print("\n" + "="*60)
    print("测试位姿转换")
    print("="*60)
    
    try:
        from scipy.spatial.transform import Rotation
        
        # 测试四元数到旋转矩阵的转换
        print("测试四元数到旋转矩阵的转换:")
        
        # 测试几个四元数
        test_quaternions = [
            [0, 0, 0.77023441, 0.63776088],  # 来自位姿文件
            [0.01423884, -0.01018612, 0.77710253, 0.62913036],
            [0.02952599, -0.02060402, 0.78412700, 0.61955512]
        ]
        
        for i, q in enumerate(test_quaternions):
            print(f"\n  四元数 {i}: [{q[0]:.6f}, {q[1]:.6f}, {q[2]:.6f}, {q[3]:.6f}]")
            
            # 检查四元数范数
            norm = np.linalg.norm(q)
            print(f"    范数: {norm:.6f}")
            
            if abs(norm - 1.0) > 0.001:
                print(f"    ⚠️ 四元数未单位化，进行归一化")
                q = np.array(q) / norm
            
            # 转换为旋转矩阵
            try:
                rot = Rotation.from_quat([q[0], q[1], q[2], q[3]])
                R = rot.as_matrix()
                
                print(f"    旋转矩阵:")
                print(f"      [{R[0,0]:.6f}, {R[0,1]:.6f}, {R[0,2]:.6f}]")
                print(f"      [{R[1,0]:.6f}, {R[1,1]:.6f}, {R[1,2]:.6f}]")
                print(f"      [{R[2,0]:.6f}, {R[2,1]:.6f}, {R[2,2]:.6f}]")
                
                # 检查旋转矩阵性质
                RRT = np.dot(R, R.T)
                identity_error = np.linalg.norm(RRT - np.eye(3))
                det_R = np.linalg.det(R)
                
                print(f"    正交性误差: {identity_error:.6f}")
                print(f"    行列式: {det_R:.6f}")
                
            except Exception as e:
                print(f"    ❌ 转换失败: {e}")
        
        return True
        
    except ImportError:
        print("⚠️ scipy未安装，跳过旋转矩阵转换测试")
        return False
    except Exception as e:
        print(f"❌ 位姿转换测试失败: {e}")
        return False

def create_fixed_dataset():
    """创建修复后的数据集"""
    print("\n" + "="*60)
    print("创建修复后的数据集")
    print("="*60)
    
    try:
        from former3d.datasets.tartanair_dataset import TartanAirStreamingDataset
        
        data_root = "/tmp/tartanair_test"
        
        print(f"数据根目录: {data_root}")
        
        # 创建数据集，跳过位姿检查
        dataset = TartanAirStreamingDataset(
            data_root=data_root,
            split='train',
            sequence_ids=['abandonedfactory/Easy/P000'],  # 只使用一个序列
            load_depth=False,
            load_sdf=False,
            max_sequence_length=10,
            image_size=(256, 256),
            normalize_images=True,
            cache_data=False,
            use_left_camera=True,
            frame_interval=5,
            skip_pose_validation=True  # 假设有这个参数
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        if len(dataset) == 0:
            print("❌ 数据集为空")
            return None
        
        print(f"✅ 数据集创建成功")
        
        # 测试加载一个样本
        print(f"\n测试加载样本:")
        try:
            sample = dataset[0]
            print(f"✅ 样本加载成功")
            print(f"  样本键: {list(sample.keys())}")
            
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
                else:
                    print(f"  {key}: {value}")
            
            return dataset
            
        except Exception as e:
            print(f"❌ 样本加载失败: {e}")
            return None
        
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_simple_test():
    """创建一个简单的测试用例"""
    print("\n" + "="*60)
    print("创建简单测试用例")
    print("="*60)
    
    import torch
    
    class SimpleTartanAirTestDataset(torch.utils.data.Dataset):
        """简单的TartanAir测试数据集"""
        def __init__(self, data_root, num_samples=10):
            self.data_root = data_root
            self.num_samples = num_samples
            
            # 加载一个位姿文件作为示例
            pose_file = "/tmp/tartanair_test/abandonedfactory/Easy/P000/pose_left.txt"
            self.poses = np.loadtxt(pose_file)
            
            print(f"✅ 简单数据集初始化")
            print(f"  样本数量: {self.num_samples}")
            print(f"  可用位姿: {self.poses.shape[0]}")
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # 模拟图像 (3, 256, 256)
            image = torch.randn(3, 256, 256)
            
            # 使用真实的位姿数据（循环使用）
            pose_idx = idx % self.poses.shape[0]
            pose_data = self.poses[pose_idx]
            
            # 位置
            position = torch.tensor(pose_data[:3], dtype=torch.float32)
            
            # 四元数（确保单位化）
            quat = torch.tensor(pose_data[3:7], dtype=torch.float32)
            quat_norm = torch.norm(quat)
            if quat_norm > 0:
                quat = quat / quat_norm
            
            # 创建4x4位姿矩阵
            from scipy.spatial.transform import Rotation
            import numpy as np
            
            # 四元数到旋转矩阵
            rot = Rotation.from_quat([quat[0].item(), quat[1].item(), quat[2].item(), quat[3].item()])
            R_np = rot.as_matrix()
            R = torch.tensor(R_np, dtype=torch.float32)
            
            # 创建位姿矩阵
            pose = torch.eye(4, dtype=torch.float32)
            pose[:3, :3] = R
            pose[:3, 3] = position
            
            # 内参矩阵（TartanAir固定内参）
            intrinsic = torch.tensor([
                [320.0, 0.0, 320.0],
                [0.0, 320.0, 240.0],
                [0.0, 0.0, 1.0]
            ], dtype=torch.float32)
            
            # 调整内参到256x256
            scale_h = 256 / 480
            scale_w = 256 / 640
            intrinsic_scaled = intrinsic.clone()
            intrinsic_scaled[0, 0] *= scale_w  # fx
            intrinsic_scaled[1, 1] *= scale_h  # fy
            intrinsic_scaled[0, 2] *= scale_w  # cx
            intrinsic_scaled[1, 2] *= scale_h  # cy
            
            # 模拟SDF真值 (32x32x32)
            sdf_gt = torch.randn(32, 32, 32) * 0.1
            
            # 模拟占用真值 (32x32x32)
            occ_gt = torch.rand(32, 32, 32) > 0.5
            
            return {
                'image': image,
                'pose': pose,
                'intrinsic': intrinsic_scaled,
                'sequence_id': 'abandonedfactory/Easy/P000',
                'frame_idx': idx,
                'sdf': sdf_gt,
                'occ': occ_gt
            }
    
    # 创建数据集
    dataset = SimpleTartanAirTestDataset("/tmp/tartanair_test", num_samples=20)
    
    print(f"\n测试数据加载:")
    sample = dataset[0]
    
    print(f"✅ 样本加载成功")
    print(f"  样本键: {list(sample.keys())}")
    
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
            
            if key == 'pose':
                R = value[:3, :3]
                t = value[:3, 3]
                print(f"    旋转矩阵形状: {R.shape}")
                print(f"    平移向量: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
            
            if key == 'sdf':
                print(f"    SDF范围: [{value.min():.3f}, {value.max():.3f}]")
                print(f"    正值比例: {(value > 0).float().mean():.3f}")
            
            if key == 'occ':
                print(f"    占用比例: {value.float().mean():.3f}")
        else:
            print(f"  {key}: {value}")
    
    return dataset

def main():
    """主调试函数"""
    print("\n" + "="*80)
    print("TartanAir数据集调试")
    print("="*80)
    
    # 运行调试
    debug_pose_loading()
    test_pose_conversion()
    
    # 创建简单测试数据集
    dataset = create_simple_test()
    
    if dataset is not None:
        print("\n" + "="*80)
        print("✅ 测试用例创建成功")
        print("="*80)
        print("\n测试用例验证:")
        print("1. ✅ 图像: 3x256x256 随机张量")
        print("2. ✅ 位姿: 4x4 位姿矩阵（使用真实TartanAir位姿数据）")
        print("3. ✅ 内参: 3x3 内参矩阵（调整到256x256）")
        print("4. ✅ SDF真值: 32x32x32 随机SDF值")
        print("5. ✅ 占用真值: 32x32x32 二值占用网格")
        print("\n这个测试用例可以用于验证StreamSDFFormer的训练循环。")
    
    print("\n" + "="*80)
    print("💡 建议:")
    print("1. 使用这个简单测试数据集进行端到端训练验证")
    print("2. 调试TartanairStreamingDataset的位姿加载问题")
    print("3. 检查位姿文件格式和四元数归一化")
    print("="*80)

if __name__ == "__main__":
    main()