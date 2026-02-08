"""
ScanNet数据集测试
测试ScanNetStreamingDataset的功能
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

from former3d.datasets.scannet_dataset import ScanNetStreamingDataset


def create_mock_scannet_data(temp_dir: Path, num_frames: int = 5):
    """
    创建模拟ScanNet数据
    
    ScanNet目录结构:
    scans/
    └── scene_test/
        ├── color/          # *.jpg
        ├── pose/           # *.txt
        ├── intrinsic/      # intrinsic_color.txt
        └── depth/          # *.png (可选)
    """
    # 创建序列目录
    scans_dir = temp_dir / "scans"
    scans_dir.mkdir(exist_ok=True)
    
    seq_dir = scans_dir / "scene_test"
    seq_dir.mkdir(exist_ok=True)
    
    # 创建子目录
    color_dir = seq_dir / "color"
    pose_dir = seq_dir / "pose"
    intrinsic_dir = seq_dir / "intrinsic"
    depth_dir = seq_dir / "depth"
    
    color_dir.mkdir(exist_ok=True)
    pose_dir.mkdir(exist_ok=True)
    intrinsic_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    
    # ScanNet原始图像尺寸
    original_size = (480, 640)
    
    # 创建内参文件（针对480x640图像）
    intrinsic = np.array([
        [577.870605, 0.0, 319.5],
        [0.0, 577.870605, 239.5],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    np.savetxt(intrinsic_dir / "intrinsic_color.txt", intrinsic)
    
    # 创建数据帧
    for i in range(num_frames):
        # 创建图像（480x640 RGB）
        img = Image.new('RGB', original_size[::-1], color=(i*50, i*50, i*50))
        img.save(color_dir / f"{i:06d}.jpg")
        
        # 创建位姿文件（4x4矩阵，稍微变化）
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [i*0.1, 0.0, 0.0]  # 沿x轴移动
        np.savetxt(pose_dir / f"{i:06d}.txt", pose)
        
        # 创建深度图（16位PNG）
        depth = np.ones((original_size[0], original_size[1]), dtype=np.uint16) * (i * 1000 + 1000)
        depth_img = Image.fromarray(depth)
        depth_img.save(depth_dir / f"{i:06d}.png")
    
    return str(temp_dir)


def test_scannet_initialization():
    """测试ScanNet数据集初始化"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        data_root = create_mock_scannet_data(temp_dir, num_frames=5)
        
        dataset = ScanNetStreamingDataset(
            data_root=data_root,
            split="train",
            image_size=(240, 320),  # 缩小一半
            load_depth=True,
            frame_interval=2
        )
        
        # 检查基本属性
        assert len(dataset) == 3, f"数据集长度应为3（5帧，间隔2），实际为{len(dataset)}"
        assert len(dataset.sequence_info) == 1, f"应有一个序列，实际有{len(dataset.sequence_info)}"
        
        # 检查序列信息
        seq_id = "scene_test"
        assert seq_id in dataset.sequence_info, f"序列{seq_id}不存在"
        
        seq_info = dataset.sequence_info[seq_id]
        assert seq_info['length'] == 3, f"序列长度应为3，实际为{seq_info['length']}"
        assert seq_info['total_frames'] == 5, f"序列总帧数应为5，实际为{seq_info['total_frames']}"
        
        print("✅ test_scannet_initialization 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_scannet_data_loading():
    """测试ScanNet数据加载"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        data_root = create_mock_scannet_data(temp_dir, num_frames=3)
        
        dataset = ScanNetStreamingDataset(
            data_root=data_root,
            split="train",
            image_size=(120, 160),  # 缩小到1/4
            load_depth=True
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
        
        # 检查深度值（毫米转米）
        depth_value = frame_data['depth'][0, 0].item()
        expected_depth = 1000.0 / 1000.0  # 第一帧深度1000毫米 = 1.0米
        assert abs(depth_value - expected_depth) < 0.1, f"深度值应为{expected_depth}，实际为{depth_value}"
        
        print("✅ test_scannet_data_loading 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_scannet_intrinsic_adjustment():
    """测试ScanNet内参矩阵调整"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        data_root = create_mock_scannet_data(temp_dir, num_frames=1)
        
        # 原始内参（针对480x640）
        original_intrinsic = np.array([
            [577.870605, 0.0, 319.5],
            [0.0, 577.870605, 239.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # 测试不同尺寸
        test_sizes = [(240, 320), (120, 160), (480, 640)]
        
        for h, w in test_sizes:
            dataset = ScanNetStreamingDataset(
                data_root=data_root,
                split="train",
                image_size=(h, w),
                load_depth=False
            )
            
            frame_data = dataset[0]
            adjusted_intrinsic = frame_data['intrinsics'].numpy()
            
            # 计算缩放因子
            scale_h = h / 480.0
            scale_w = w / 640.0
            
            # 检查内参调整是否正确
            expected_fx = original_intrinsic[0, 0] * scale_w
            expected_fy = original_intrinsic[1, 1] * scale_h
            expected_cx = original_intrinsic[0, 2] * scale_w
            expected_cy = original_intrinsic[1, 2] * scale_h
            
            assert abs(adjusted_intrinsic[0, 0] - expected_fx) < 0.1, f"fx调整错误: 期望{expected_fx}, 实际{adjusted_intrinsic[0, 0]}"
            assert abs(adjusted_intrinsic[1, 1] - expected_fy) < 0.1, f"fy调整错误: 期望{expected_fy}, 实际{adjusted_intrinsic[1, 1]}"
            assert abs(adjusted_intrinsic[0, 2] - expected_cx) < 0.1, f"cx调整错误: 期望{expected_cx}, 实际{adjusted_intrinsic[0, 2]}"
            assert abs(adjusted_intrinsic[1, 2] - expected_cy) < 0.1, f"cy调整错误: 期望{expected_cy}, 实际{adjusted_intrinsic[1, 2]}"
            
            print(f"  ✅ 尺寸{h}x{w}内参调整正确")
        
        print("✅ test_scannet_intrinsic_adjustment 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_scannet_pose_loading():
    """测试ScanNet位姿加载"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        data_root = create_mock_scannet_data(temp_dir, num_frames=3)
        
        dataset = ScanNetStreamingDataset(
            data_root=data_root,
            split="train",
            image_size=(240, 320),
            load_depth=False
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
            expected_translation = [i * 0.1, 0.0, 0.0]
            actual_translation = pose[:3, 3]
            
            assert np.allclose(actual_translation, expected_translation, atol=0.01), \
                f"第{i}帧平移应为{expected_translation}，实际为{actual_translation}"
        
        print("✅ test_scannet_pose_loading 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_scannet_frame_sampling():
    """测试ScanNet帧采样"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        data_root = create_mock_scannet_data(temp_dir, num_frames=10)
        
        # 测试不同采样间隔
        test_intervals = [1, 2, 3, 5]
        
        for interval in test_intervals:
            dataset = ScanNetStreamingDataset(
                data_root=data_root,
                split="train",
                image_size=(240, 320),
                load_depth=False,
                frame_interval=interval,
                max_sequence_length=100  # 不限制长度
            )
            
            expected_frames = (10 + interval - 1) // interval  # 向上取整
            actual_frames = len(dataset)
            
            assert actual_frames == expected_frames, \
                f"采样间隔{interval}：期望{expected_frames}帧，实际{actual_frames}帧"
            
            # 检查采样的帧索引
            seq_info = dataset.sequence_info["scene_test"]
            sampled_frames = seq_info['sampled_frames']
            
            # 检查采样是否正确
            for j, frame_idx in enumerate(sampled_frames):
                expected_idx = j * interval
                assert frame_idx == expected_idx, \
                    f"第{j}个采样帧应为{expected_idx}，实际为{frame_idx}"
            
            print(f"  ✅ 采样间隔{interval}正确: {actual_frames}帧")
        
        print("✅ test_scannet_frame_sampling 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_scannet_max_sequence_length():
    """测试ScanNet最大序列长度限制"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        data_root = create_mock_scannet_data(temp_dir, num_frames=20)
        
        # 测试不同最大长度
        test_lengths = [5, 10, 15]
        
        for max_len in test_lengths:
            dataset = ScanNetStreamingDataset(
                data_root=data_root,
                split="train",
                image_size=(240, 320),
                load_depth=False,
                frame_interval=1,
                max_sequence_length=max_len
            )
            
            actual_length = len(dataset)
            expected_length = min(max_len, 20)  # 总共20帧
            
            assert actual_length == expected_length, \
                f"最大长度{max_len}：期望{expected_length}帧，实际{actual_length}帧"
            
            print(f"  ✅ 最大长度{max_len}正确: {actual_length}帧")
        
        print("✅ test_scannet_max_sequence_length 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_scannet_without_depth():
    """测试不加载深度图"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        data_root = create_mock_scannet_data(temp_dir, num_frames=3)
        
        dataset = ScanNetStreamingDataset(
            data_root=data_root,
            split="train",
            image_size=(240, 320),
            load_depth=False  # 不加载深度
        )
        
        frame_data = dataset[0]
        
        # 检查深度图不存在
        assert 'depth' not in frame_data, "不加载深度时应没有depth字段"
        
        print("✅ test_scannet_without_depth 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_scannet_train_val_split():
    """测试ScanNet训练验证集划分"""
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # 创建多个序列
        scans_dir = temp_dir / "scans"
        scans_dir.mkdir(exist_ok=True)
        
        # 创建5个序列
        for i in range(5):
            seq_dir = scans_dir / f"scene_{i:04d}_00"
            seq_dir.mkdir(exist_ok=True)
            
            # 创建必要的空目录
            (seq_dir / "color").mkdir(exist_ok=True)
            (seq_dir / "pose").mkdir(exist_ok=True)
            (seq_dir / "intrinsic").mkdir(exist_ok=True)
        
        # 测试划分函数
        split = ScanNetStreamingDataset.create_train_val_split(
            data_root=str(temp_dir),
            val_ratio=0.2,  # 20%验证集
            random_seed=42
        )
        
        # 检查划分结果
        assert 'train' in split, "划分结果应包含'train'键"
        assert 'val' in split, "划分结果应包含'val'键"
        
        train_seqs = split['train']
        val_seqs = split['val']
        
        # 检查数量
        total_seqs = len(train_seqs) + len(val_seqs)
        assert total_seqs == 5, f"总序列数应为5，实际为{total_seqs}"
        
        # 验证集应为1个序列（5 * 0.2 = 1）
        assert len(val_seqs) == 1, f"验证集应有1个序列，实际有{len(val_seqs)}个"
        assert len(train_seqs) == 4, f"训练集应有4个序列，实际有{len(train_seqs)}个"
        
        # 检查没有重叠
        assert len(set(train_seqs) & set(val_seqs)) == 0, "训练集和验证集不应有重叠"
        
        print("✅ test_scannet_train_val_split 通过")
        return True
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def run_all_tests():
    """运行所有ScanNet测试"""
    tests = [
        test_scannet_initialization,
        test_scannet_data_loading,
        test_scannet_intrinsic_adjustment,
        test_scannet_pose_loading,
        test_scannet_frame_sampling,
        test_scannet_max_sequence_length,
        test_scannet_without_depth,
        test_scannet_train_val_split
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
    
    print(f"\n📊 ScanNet测试结果: {passed} 通过, {failed} 失败")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)