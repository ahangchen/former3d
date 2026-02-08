"""
TartanAir数据集测试
测试TartanAirStreamingDataset的功能
"""

import os
import sys
import tempfile
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from former3d.datasets.tartanair_dataset import TartanAirStreamingDataset


def create_mock_tartanair_data(temp_dir: Path, num_frames: int = 5):
    """
    创建模拟TartanAir数据
    
    TartanAir目录结构:
    abandonedfactory/
    └── Easy/
        └── P000/
            ├── image_left/      # *.png
            ├── depth_left/      # *.npy (可选)
            └── pose_left.txt    # 所有帧的位姿
    """
    # 创建环境目录
    env_dir = temp_dir / "abandonedfactory"
    env_dir.mkdir(exist_ok=True)
    
    # 创建难度目录
    difficulty_dir = env_dir / "Easy"
    difficulty_dir.mkdir(exist_ok=True)
    
    # 创建轨迹目录
    traj_dir = difficulty_dir / "P000"
    traj_dir.mkdir(exist_ok=True)
    
    # 创建子目录
    image_dir = traj_dir / "image_left"
    depth_dir = traj_dir / "depth_left"
    
    image_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    
    # TartanAir原始图像尺寸
    original_size = (480, 640)
    
    # 创建位姿文件（所有帧的位姿在一个文件中）
    poses = []
    for i in range(num_frames):
        # 创建位姿矩阵（4x4）
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [i*0.2, 0.0, 0.0]  # 沿x轴移动
        poses.append(pose.flatten())
    
    # 保存位姿文件（每行16个值）
    pose_file = traj_dir / "pose_left.txt"
    np.savetxt(pose_file, np.array(poses))
    
    # 创建数据帧
    for i in range(num_frames):
        # 创建图像（480x640 RGB）
        img = Image.new('RGB', original_size[::-1], color=(i*40, i*40, i*40))
        img.save(image_dir / f"{i:06d}_left.png")
        
        # 创建深度图（.npy格式，浮点数）
        depth = np.ones((original_size[0], original_size[1]), dtype=np.float32) * (i * 0.5 + 1.0)
        np.save(depth_dir / f"{i:06d}_left_depth.npy", depth)
    
    return str(temp_dir)


def test_tartanair_initialization():
    """测试TartanAir数据集初始化"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        data_root = create_mock_tartanair_data(temp_dir, num_frames=6)
        
        dataset = TartanAirStreamingDataset(
            data_root=data_root,
            split="train",
            image_size=(240, 320),  # 缩小一半
            load_depth=True,
            use_left_camera=True,
            frame_interval=2
        )
        
        # 检查基本属性
        assert len(dataset) == 3, f"数据集长度应为3（6帧，间隔2），实际为{len(dataset)}"
        assert len(dataset.sequence_info) == 1, f"应有一个序列，实际有{len(dataset.sequence_info)}"
        
        # 检查序列信息
        seq_id = "abandonedfactory/Easy/P000"
        assert seq_id in dataset.sequence_info, f"序列{seq_id}不存在"
        
        seq_info = dataset.sequence_info[seq_id]
        assert seq_info['length'] == 3, f"序列长度应为3，实际为{seq_info['length']}"
        assert seq_info['total_frames'] == 6, f"序列总帧数应为6，实际为{seq_info['total_frames']}"
        assert seq_info['num_poses'] == 6, f"位姿数量应为6，实际为{seq_info['num_poses']}"
        
        print("✅ test_tartanair_initialization 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_tartanair_data_loading():
    """测试TartanAir数据加载"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        data_root = create_mock_tartanair_data(temp_dir, num_frames=3)
        
        dataset = TartanAirStreamingDataset(
            data_root=data_root,
            split="train",
            image_size=(120, 160),  # 缩小到1/4
            load_depth=True,
            use_left_camera=True
        )
        
        # 获取第一帧数据
        frame_data = dataset[0]
        
        # 检查数据格式
        assert 'image' in frame_data, "缺少image字段"
        assert 'pose' in frame_data, "缺少pose字段"
        assert 'intrinsics' in frame_data, "缺少intrinsics字段"
        assert 'depth' in frame_data, "缺少depth字段"
        
        # 检查形状
        assert frame_data['image'].shape == (3, 120, 160), f"image形状应为(3, 120, 160)，实际为{frame_data['image'].shape}"
        assert frame_data['pose'].shape == (4, 4), f"pose形状应为(4, 4)，实际为{frame_data['pose'].shape}"
        assert frame_data['intrinsics'].shape == (3, 3), f"intrinsics形状应为(3, 3)，实际为{frame_data['intrinsics'].shape}"
        assert frame_data['depth'].shape == (120, 160), f"depth形状应为(120, 160)，实际为{frame_data['depth'].shape}"
        
        # 检查深度值（浮点数，单位米）
        depth_value = frame_data['depth'][0, 0].item()
        expected_depth = 1.0  # 第一帧深度1.0米
        assert abs(depth_value - expected_depth) < 0.1, f"深度值应为{expected_depth}，实际为{depth_value}"
        
        print("✅ test_tartanair_data_loading 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_tartanair_intrinsic_fixed():
    """测试TartanAir固定内参"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # 直接创建测试数据，避免使用create_mock_tartanair_data
        env_dir = temp_dir / "abandonedfactory"
        env_dir.mkdir(exist_ok=True)
        
        difficulty_dir = env_dir / "Easy"
        difficulty_dir.mkdir(exist_ok=True)
        
        traj_dir = difficulty_dir / "P000"
        traj_dir.mkdir(exist_ok=True)
        
        # 创建图像目录
        image_dir = traj_dir / "image_left"
        image_dir.mkdir(exist_ok=True)
        
        # 创建一张测试图像
        img = Image.new('RGB', (640, 480), color=(128, 128, 128))
        img.save(image_dir / "000000_left.png")
        
        # 创建位姿文件（一行，16个值）
        pose = np.eye(4, dtype=np.float32).flatten()
        pose_file = traj_dir / "pose_left.txt"
        np.savetxt(pose_file, [pose])  # 注意：要包装在列表中
        
        # TartanAir固定内参（针对640x480）
        fixed_intrinsic = np.array([
            [320.0, 0.0, 320.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # 测试不同尺寸
        test_sizes = [(240, 320), (120, 160), (480, 640)]
        
        for h, w in test_sizes:
            dataset = TartanAirStreamingDataset(
                data_root=str(temp_dir),
                split="train",
                image_size=(h, w),
                load_depth=False,
                use_left_camera=True
            )
            
            # 确保数据集有数据
            assert len(dataset) > 0, f"数据集应为空，尺寸{h}x{w}"
            
            frame_data = dataset[0]
            adjusted_intrinsic = frame_data['intrinsics'].numpy()
            
            # 计算缩放因子
            scale_h = h / 480.0
            scale_w = w / 640.0
            
            # 检查内参调整是否正确
            expected_fx = fixed_intrinsic[0, 0] * scale_w
            expected_fy = fixed_intrinsic[1, 1] * scale_h
            expected_cx = fixed_intrinsic[0, 2] * scale_w
            expected_cy = fixed_intrinsic[1, 2] * scale_h
            
            assert abs(adjusted_intrinsic[0, 0] - expected_fx) < 0.1, f"fx调整错误: 期望{expected_fx}, 实际{adjusted_intrinsic[0, 0]}"
            assert abs(adjusted_intrinsic[1, 1] - expected_fy) < 0.1, f"fy调整错误: 期望{expected_fy}, 实际{adjusted_intrinsic[1, 1]}"
            assert abs(adjusted_intrinsic[0, 2] - expected_cx) < 0.1, f"cx调整错误: 期望{expected_cx}, 实际{adjusted_intrinsic[0, 2]}"
            assert abs(adjusted_intrinsic[1, 2] - expected_cy) < 0.1, f"cy调整错误: 期望{expected_cy}, 实际{adjusted_intrinsic[1, 2]}"
            
            print(f"  ✅ 尺寸{h}x{w}内参调整正确")
        
        print("✅ test_tartanair_intrinsic_fixed 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_tartanair_pose_loading():
    """测试TartanAir位姿加载"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        data_root = create_mock_tartanair_data(temp_dir, num_frames=4)
        
        dataset = TartanAirStreamingDataset(
            data_root=data_root,
            split="train",
            image_size=(240, 320),
            load_depth=False,
            use_left_camera=True
        )
        
        # 检查每帧的位姿
        for i in range(len(dataset)):
            frame_data = dataset[i]
            pose = frame_data['pose'].numpy()
            
            # 检查是否是4x4矩阵
            assert pose.shape == (4, 4), f"位姿形状应为(4, 4)，实际为{pose.shape}"
            
            # 检查最后一行为[0, 0, 0, 1]
            assert np.allclose(pose[3, :], [0, 0, 0, 1]), f"位姿最后一行应为[0,0,0,1]，实际为{pose[3, :]}"
            
            # 检查平移部分（模拟数据中沿x轴移动）
            # 注意：TartanAir位姿文件包含所有帧，但数据集可能采样
            frame_idx = dataset.frame_indices[i][1]  # 获取原始帧索引
            expected_translation = [frame_idx * 0.2, 0.0, 0.0]
            actual_translation = pose[:3, 3]
            
            assert np.allclose(actual_translation, expected_translation, atol=0.01), \
                f"第{i}帧（原始帧{frame_idx}）平移应为{expected_translation}，实际为{actual_translation}"
        
        print("✅ test_tartanair_pose_loading 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_tartanair_right_camera():
    """测试使用右相机"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # 创建左右相机数据
        env_dir = temp_dir / "abandonedfactory"
        env_dir.mkdir(exist_ok=True)
        
        difficulty_dir = env_dir / "Easy"
        difficulty_dir.mkdir(exist_ok=True)
        
        traj_dir = difficulty_dir / "P000"
        traj_dir.mkdir(exist_ok=True)
        
        # 创建左相机数据
        image_left_dir = traj_dir / "image_left"
        image_left_dir.mkdir(exist_ok=True)
        
        # 创建右相机数据
        image_right_dir = traj_dir / "image_right"
        image_right_dir.mkdir(exist_ok=True)
        
        # 创建位姿文件
        pose_left_file = traj_dir / "pose_left.txt"
        pose_right_file = traj_dir / "pose_right.txt"
        
        # 创建一些测试数据
        for i in range(3):
            # 左相机图像（红色）
            img_left = Image.new('RGB', (640, 480), color=(255, 0, 0))
            img_left.save(image_left_dir / f"{i:06d}_left.png")
            
            # 右相机图像（蓝色）
            img_right = Image.new('RGB', (640, 480), color=(0, 0, 255))
            img_right.save(image_right_dir / f"{i:06d}_right.png")
        
        # 创建位姿文件
        poses = [np.eye(4).flatten() for _ in range(3)]
        np.savetxt(pose_left_file, poses)
        np.savetxt(pose_right_file, poses)
        
        # 测试左相机
        dataset_left = TartanAirStreamingDataset(
            data_root=str(temp_dir),
            split="train",
            image_size=(64, 64),
            load_depth=False,
            use_left_camera=True
        )
        
        # 测试右相机
        dataset_right = TartanAirStreamingDataset(
            data_root=str(temp_dir),
            split="train",
            image_size=(64, 64),
            load_depth=False,
            use_left_camera=False  # 使用右相机
        )
        
        # 检查两个数据集都能正确初始化
        assert len(dataset_left) == 3, f"左相机数据集应有3帧，实际有{len(dataset_left)}帧"
        assert len(dataset_right) == 3, f"右相机数据集应有3帧，实际有{len(dataset_right)}帧"
        
        print("✅ test_tartanair_right_camera 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_tartanair_without_depth():
    """测试不加载深度图"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        data_root = create_mock_tartanair_data(temp_dir, num_frames=2)
        
        dataset = TartanAirStreamingDataset(
            data_root=data_root,
            split="train",
            image_size=(240, 320),
            load_depth=False,  # 不加载深度
            use_left_camera=True
        )
        
        frame_data = dataset[0]
        
        # 检查深度图不存在
        assert 'depth' not in frame_data, "不加载深度时应没有depth字段"
        
        print("✅ test_tartanair_without_depth 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_tartanair_sequence_utilities():
    """测试TartanAir序列工具函数"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # 创建多个环境的数据
        envs = ["abandonedfactory", "carwelding", "hospital"]
        
        for env in envs:
            env_dir = temp_dir / env
            env_dir.mkdir(exist_ok=True)
            
            for difficulty in ["Easy", "Hard"]:
                difficulty_dir = env_dir / difficulty
                difficulty_dir.mkdir(exist_ok=True)
                
                for traj in ["P000", "P001"]:
                    traj_dir = difficulty_dir / traj
                    traj_dir.mkdir(exist_ok=True)
        
        # 测试工具函数
        envs = TartanAirStreamingDataset.get_available_environments(str(temp_dir))
        assert len(envs) == 3, f"应有3个环境，实际有{len(envs)}个"
        assert "abandonedfactory" in envs, "环境列表应包含'abandonedfactory'"
        
        # 测试获取序列
        all_seqs = TartanAirStreamingDataset.get_available_sequences(str(temp_dir))
        expected_seqs = 3 * 2 * 2  # 3环境 * 2难度 * 2轨迹
        assert len(all_seqs) == expected_seqs, f"应有{expected_seqs}个序列，实际有{len(all_seqs)}个"
        
        # 测试按环境获取序列
        factory_seqs = TartanAirStreamingDataset.get_available_sequences(
            str(temp_dir), environment="abandonedfactory")
        assert len(factory_seqs) == 4, f"abandonedfactory应有4个序列，实际有{len(factory_seqs)}个"
        
        print("✅ test_tartanair_sequence_utilities 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def run_all_tests():
    """运行所有TartanAir测试"""
    tests = [
        test_tartanair_initialization,
        test_tartanair_data_loading,
        test_tartanair_intrinsic_fixed,
        test_tartanair_pose_loading,
        test_tartanair_right_camera,
        test_tartanair_without_depth,
        test_tartanair_sequence_utilities
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n📊 TartanAir测试结果: {passed} 通过, {failed} 失败")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)