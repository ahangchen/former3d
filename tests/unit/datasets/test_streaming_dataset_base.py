"""
流式数据集基础测试
测试StreamingDataset基类的功能
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

from former3d.datasets.streaming_dataset import StreamingDataset


class MockStreamingDataset(StreamingDataset):
    """
    用于测试的模拟数据集
    继承StreamingDataset并实现抽象方法
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 sequence_ids = None,
                 transform = None,
                 load_depth: bool = False,
                 load_sdf: bool = False,
                 max_sequence_length = None,
                 image_size = (64, 64),
                 normalize_images: bool = True,
                 cache_data: bool = False):
        
        # 创建模拟数据目录
        self.temp_dir = Path(tempfile.mkdtemp())
        self._create_mock_data()
        
        super().__init__(
            data_root=str(self.temp_dir),
            split=split,
            sequence_ids=sequence_ids,
            transform=transform,
            load_depth=load_depth,
            load_sdf=load_sdf,
            max_sequence_length=max_sequence_length,
            image_size=image_size,
            normalize_images=normalize_images,
            cache_data=cache_data
        )
    
    def _create_mock_data(self):
        """创建模拟数据"""
        # 创建序列目录
        seq_dir = self.temp_dir / "scene_test"
        seq_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        color_dir = seq_dir / "color"
        pose_dir = seq_dir / "pose"
        intrinsic_dir = seq_dir / "intrinsic"
        
        color_dir.mkdir(exist_ok=True)
        pose_dir.mkdir(exist_ok=True)
        intrinsic_dir.mkdir(exist_ok=True)
        
        # 创建10帧数据
        for i in range(10):
            # 创建图像
            img = Image.new('RGB', (128, 96), color=(i*25, i*25, i*25))
            img.save(color_dir / f"{i:06d}.jpg")
            
            # 创建位姿文件（4x4单位矩阵）
            pose = np.eye(4, dtype=np.float32)
            np.savetxt(pose_dir / f"{i:06d}.txt", pose)
        
        # 创建内参文件
        intrinsic = np.array([
            [320.0, 0.0, 64.0],
            [0.0, 320.0, 48.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        np.savetxt(intrinsic_dir / "intrinsic_color.txt", intrinsic)
    
    def _load_dataset(self, sequence_ids):
        """加载模拟数据集"""
        seq_id = "scene_test"
        seq_path = self.temp_dir / seq_id
        
        # 获取图像文件
        color_dir = seq_path / "color"
        image_files = sorted(color_dir.glob("*.jpg"))
        
        # 添加帧索引
        start_idx = len(self.frame_indices)
        for frame_idx in range(len(image_files)):
            self.frame_indices.append((seq_id, frame_idx))
        
        end_idx = len(self.frame_indices) - 1
        
        # 保存序列信息
        self.sequence_info[seq_id] = {
            'path': str(seq_path),
            'start_idx': start_idx,
            'end_idx': end_idx,
            'length': len(image_files),
            'total_frames': len(image_files),
            'color_dir': str(color_dir),
            'pose_dir': str(seq_path / "pose"),
            'intrinsic_dir': str(seq_path / "intrinsic")
        }
    
    def _get_image_path(self, sequence_id, frame_idx):
        seq_info = self.sequence_info[sequence_id]
        color_dir = Path(seq_info['color_dir'])
        return str(color_dir / f"{frame_idx:06d}.jpg")
    
    def _get_pose_path(self, sequence_id, frame_idx):
        seq_info = self.sequence_info[sequence_id]
        pose_dir = Path(seq_info['pose_dir'])
        return str(pose_dir / f"{frame_idx:06d}.txt")
    
    def _get_intrinsic_path(self, sequence_id, frame_idx):
        seq_info = self.sequence_info[sequence_id]
        intrinsic_dir = Path(seq_info['intrinsic_dir'])
        return str(intrinsic_dir / "intrinsic_color.txt")
    
    def _get_depth_path(self, sequence_id, frame_idx):
        return ""
    
    def _get_sdf_path(self, sequence_id, frame_idx):
        return ""
    
    def cleanup(self):
        """清理临时文件"""
        import shutil
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


def test_dataset_initialization():
    """测试数据集初始化"""
    dataset = MockStreamingDataset(
        data_root="/tmp/mock",
        split="train",
        image_size=(64, 64),
        normalize_images=True
    )
    
    try:
        # 检查基本属性
        assert len(dataset) == 10, f"数据集长度应为10，实际为{len(dataset)}"
        assert len(dataset.sequence_info) == 1, f"应有一个序列，实际有{len(dataset.sequence_info)}"
        
        # 检查序列信息
        seq_id = "scene_test"
        assert seq_id in dataset.sequence_info, f"序列{seq_id}不存在"
        
        seq_info = dataset.sequence_info[seq_id]
        assert seq_info['length'] == 10, f"序列长度应为10，实际为{seq_info['length']}"
        
        print("✅ test_dataset_initialization 通过")
        return True
        
    finally:
        dataset.cleanup()


def test_get_item():
    """测试获取单帧数据"""
    dataset = MockStreamingDataset(
        data_root="/tmp/mock",
        split="train",
        image_size=(64, 64)
    )
    
    try:
        # 获取第一帧
        frame_data = dataset[0]
        
        # 检查数据格式
        assert 'image' in frame_data, "缺少image字段"
        assert 'pose' in frame_data, "缺少pose字段"
        assert 'intrinsics' in frame_data, "缺少intrinsics字段"
        assert 'frame_id' in frame_data, "缺少frame_id字段"
        assert 'sequence_id' in frame_data, "缺少sequence_id字段"
        
        # 检查数据类型和形状
        assert isinstance(frame_data['image'], torch.Tensor), "image应为torch.Tensor"
        assert frame_data['image'].shape == (3, 64, 64), f"image形状应为(3, 64, 64)，实际为{frame_data['image'].shape}"
        
        assert isinstance(frame_data['pose'], torch.Tensor), "pose应为torch.Tensor"
        assert frame_data['pose'].shape == (4, 4), f"pose形状应为(4, 4)，实际为{frame_data['pose'].shape}"
        
        assert isinstance(frame_data['intrinsics'], torch.Tensor), "intrinsics应为torch.Tensor"
        assert frame_data['intrinsics'].shape == (3, 3), f"intrinsics形状应为(3, 3)，实际为{frame_data['intrinsics'].shape}"
        
        assert frame_data['frame_id'] == 0, f"frame_id应为0，实际为{frame_data['frame_id']}"
        assert frame_data['sequence_id'] == "scene_test", f"sequence_id应为'scene_test'，实际为{frame_data['sequence_id']}"
        
        print("✅ test_get_item 通过")
        return True
        
    finally:
        dataset.cleanup()


def test_sequence_continuity():
    """测试序列连续性"""
    dataset = MockStreamingDataset(
        data_root="/tmp/mock",
        split="train",
        image_size=(64, 64)
    )
    
    try:
        # 检查帧索引连续性
        for i in range(len(dataset)):
            seq_id, frame_idx = dataset.frame_indices[i]
            assert seq_id == "scene_test", f"第{i}帧的序列ID应为'scene_test'，实际为{seq_id}"
            assert frame_idx == i, f"第{i}帧的帧索引应为{i}，实际为{frame_idx}"
        
        # 检查序列信息
        seq_info = dataset.get_sequence_info("scene_test")
        assert seq_info['start_idx'] == 0, f"序列起始索引应为0，实际为{seq_info['start_idx']}"
        assert seq_info['end_idx'] == 9, f"序列结束索引应为9，实际为{seq_info['end_idx']}"
        assert seq_info['length'] == 10, f"序列长度应为10，实际为{seq_info['length']}"
        
        print("✅ test_sequence_continuity 通过")
        return True
        
    finally:
        dataset.cleanup()


def test_collate_fn():
    """测试批处理函数"""
    dataset = MockStreamingDataset(
        data_root="/tmp/mock",
        split="train",
        image_size=(64, 64)
    )
    
    try:
        # 创建批次数据
        batch = [dataset[i] for i in range(3)]
        
        # 应用collate_fn
        collated = StreamingDataset.collate_fn(batch)
        
        # 检查批处理结果
        assert 'image' in collated, "批处理结果缺少image字段"
        assert collated['image'].shape == (3, 3, 64, 64), f"批处理image形状应为(3, 3, 64, 64)，实际为{collated['image'].shape}"
        
        assert 'pose' in collated, "批处理结果缺少pose字段"
        assert collated['pose'].shape == (3, 4, 4), f"批处理pose形状应为(3, 4, 4)，实际为{collated['pose'].shape}"
        
        assert 'intrinsics' in collated, "批处理结果缺少intrinsics字段"
        assert collated['intrinsics'].shape == (3, 3, 3), f"批处理intrinsics形状应为(3, 3, 3)，实际为{collated['intrinsics'].shape}"
        
        assert 'frame_id' in collated, "批处理结果缺少frame_id字段"
        assert collated['frame_id'].shape == (3,), f"批处理frame_id形状应为(3,)，实际为{collated['frame_id'].shape}"
        
        assert 'sequence_id' in collated, "批处理结果缺少sequence_id字段"
        assert len(collated['sequence_id']) == 3, f"批处理sequence_id长度应为3，实际为{len(collated['sequence_id'])}"
        
        print("✅ test_collate_fn 通过")
        return True
        
    finally:
        dataset.cleanup()


def test_image_normalization():
    """测试图像归一化"""
    dataset = MockStreamingDataset(
        data_root="/tmp/mock",
        split="train",
        image_size=(64, 64),
        normalize_images=True
    )
    
    try:
        frame_data = dataset[0]
        image = frame_data['image']
        
        # 检查归一化范围
        # ImageNet归一化：mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        # 对于灰度图像（值在0-1之间），归一化后的值会在[-2.12, 2.64]范围内
        
        # 检查图像值在合理范围内
        # 归一化后的值通常不会超过±3
        assert torch.all(image > -5.0) and torch.all(image < 5.0), \
            f"归一化后图像值应在[-5, 5]范围内，实际范围为[{image.min():.3f}, {image.max():.3f}]"
        
        # 检查图像不是全零或全一
        assert not torch.allclose(image, torch.zeros_like(image)), "图像不应全为零"
        assert not torch.allclose(image, torch.ones_like(image)), "图像不应全为一"
        
        print("✅ test_image_normalization 通过")
        return True
        
    finally:
        dataset.cleanup()


def test_without_normalization():
    """测试不进行图像归一化"""
    dataset = MockStreamingDataset(
        data_root="/tmp/mock",
        split="train",
        image_size=(64, 64),
        normalize_images=False
    )
    
    try:
        frame_data = dataset[0]
        image = frame_data['image']
        
        # 不归一化的图像值应该在[0, 1]范围内
        assert torch.all(image >= 0) and torch.all(image <= 1.0), f"不归一化的图像值应在[0, 1]范围内，实际范围为[{image.min():.3f}, {image.max():.3f}]"
        
        print("✅ test_without_normalization 通过")
        return True
        
    finally:
        dataset.cleanup()


def test_cache_functionality():
    """测试缓存功能"""
    dataset = MockStreamingDataset(
        data_root="/tmp/mock",
        split="train",
        image_size=(64, 64),
        cache_data=True
    )
    
    try:
        # 第一次访问应该填充缓存
        frame_data1 = dataset[0]
        
        # 检查缓存是否被填充
        cache_key = "scene_test_0"
        assert cache_key in dataset.image_cache, "图像缓存应被填充"
        assert cache_key in dataset.pose_cache, "位姿缓存应被填充"
        assert cache_key in dataset.intrinsic_cache, "内参缓存应被填充"
        
        # 清空缓存并检查
        dataset.clear_cache()
        assert len(dataset.image_cache) == 0, "图像缓存应被清空"
        assert len(dataset.pose_cache) == 0, "位姿缓存应被清空"
        assert len(dataset.intrinsic_cache) == 0, "内参缓存应被清空"
        
        print("✅ test_cache_functionality 通过")
        return True
        
    finally:
        dataset.cleanup()


def run_all_tests():
    """运行所有测试"""
    tests = [
        test_dataset_initialization,
        test_get_item,
        test_sequence_continuity,
        test_collate_fn,
        test_image_normalization,
        test_without_normalization,
        test_cache_functionality
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} 失败: {e}")
            failed += 1
    
    print(f"\n📊 测试结果: {passed} 通过, {failed} 失败")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)