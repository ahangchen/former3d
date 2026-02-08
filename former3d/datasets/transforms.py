"""
流式数据集的数据增强变换
"""

import torch
import numpy as np
import random
from typing import Dict, Tuple, Optional
import torchvision.transforms as T
import torchvision.transforms.functional as F


class RandomCrop:
    """随机裁剪变换（保持序列一致性）"""
    
    def __init__(self, size: Tuple[int, int], padding: int = 0):
        """
        Args:
            size: 裁剪尺寸 (H, W)
            padding: 填充大小
        """
        self.size = size
        self.padding = padding
    
    def __call__(self, data_dict: Dict) -> Dict:
        """
        应用随机裁剪
        
        注意：对于同一序列的帧，应该使用相同的裁剪位置
        这里我们使用帧ID作为随机种子来确保一致性
        """
        if 'image' not in data_dict:
            return data_dict
        
        image = data_dict['image']
        h, w = image.shape[1], image.shape[2]
        
        # 使用帧ID作为随机种子，确保同一帧的裁剪位置一致
        frame_id = data_dict.get('frame_id', 0)
        random.seed(frame_id)
        
        # 生成裁剪位置
        top = random.randint(0, h - self.size[0])
        left = random.randint(0, w - self.size[1])
        
        # 裁剪图像
        data_dict['image'] = F.crop(image, top, left, self.size[0], self.size[1])
        
        # 裁剪深度图（如果存在）
        if 'depth' in data_dict and data_dict['depth'] is not None:
            depth = data_dict['depth']
            if depth.shape[0] == h and depth.shape[1] == w:
                data_dict['depth'] = F.crop(depth.unsqueeze(0), top, left, 
                                          self.size[0], self.size[1]).squeeze(0)
        
        # 调整内参矩阵
        if 'intrinsics' in data_dict:
            K = data_dict['intrinsics'].clone()
            K[0, 2] -= left  # 调整cx
            K[1, 2] -= top   # 调整cy
            data_dict['intrinsics'] = K
        
        return data_dict


class RandomHorizontalFlip:
    """随机水平翻转（保持序列一致性）"""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: 翻转概率
        """
        self.p = p
    
    def __call__(self, data_dict: Dict) -> Dict:
        """
        应用随机水平翻转
        
        使用帧ID作为随机种子确保一致性
        """
        if 'image' not in data_dict:
            return data_dict
        
        # 使用帧ID决定是否翻转
        frame_id = data_dict.get('frame_id', 0)
        random.seed(frame_id)
        should_flip = random.random() < self.p
        
        if not should_flip:
            return data_dict
        
        # 翻转图像
        data_dict['image'] = F.hflip(data_dict['image'])
        
        # 翻转深度图（如果存在）
        if 'depth' in data_dict and data_dict['depth'] is not None:
            data_dict['depth'] = F.hflip(data_dict['depth'])
        
        # 调整内参矩阵
        if 'intrinsics' in data_dict:
            K = data_dict['intrinsics'].clone()
            w = data_dict['image'].shape[2]
            K[0, 2] = w - K[0, 2]  # 调整cx
            data_dict['intrinsics'] = K
        
        # 调整位姿（水平翻转相当于绕Y轴旋转180度）
        if 'pose' in data_dict:
            # 创建绕Y轴旋转180度的变换矩阵
            flip_y = torch.eye(4, dtype=data_dict['pose'].dtype)
            flip_y[0, 0] = -1  # 反转X轴
            flip_y[2, 2] = -1  # 反转Z轴
            
            # 应用变换
            data_dict['pose'] = flip_y @ data_dict['pose']
        
        return data_dict


class ColorJitter:
    """颜色抖动（保持序列一致性）"""
    
    def __init__(self, 
                 brightness: float = 0.2,
                 contrast: float = 0.2,
                 saturation: float = 0.2,
                 hue: float = 0.1):
        """
        Args:
            brightness: 亮度抖动范围
            contrast: 对比度抖动范围
            saturation: 饱和度抖动范围
            hue: 色调抖动范围
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, data_dict: Dict) -> Dict:
        """
        应用颜色抖动
        
        使用序列ID作为随机种子，确保同一序列的所有帧使用相同的颜色变换
        """
        if 'image' not in data_dict:
            return data_dict
        
        # 使用序列ID作为随机种子
        seq_id = data_dict.get('sequence_id', 'default')
        seed = hash(seq_id) % 10000
        random.seed(seed)
        
        # 生成随机变换参数
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)
        
        # 应用变换
        image = data_dict['image']
        
        # 亮度
        if brightness_factor != 1:
            image = F.adjust_brightness(image, brightness_factor)
        
        # 对比度
        if contrast_factor != 1:
            image = F.adjust_contrast(image, contrast_factor)
        
        # 饱和度
        if saturation_factor != 1:
            image = F.adjust_saturation(image, saturation_factor)
        
        # 色调
        if hue_factor != 0:
            image = F.adjust_hue(image, hue_factor)
        
        data_dict['image'] = image
        return data_dict


class RandomRotation:
    """随机旋转（保持序列一致性）"""
    
    def __init__(self, degrees: float = 10.0):
        """
        Args:
            degrees: 旋转角度范围（-degrees到+degrees）
        """
        self.degrees = degrees
    
    def __call__(self, data_dict: Dict) -> Dict:
        """
        应用随机旋转
        
        使用帧ID作为随机种子确保一致性
        """
        if 'image' not in data_dict:
            return data_dict
        
        # 使用帧ID决定旋转角度
        frame_id = data_dict.get('frame_id', 0)
        random.seed(frame_id)
        angle = random.uniform(-self.degrees, self.degrees)
        
        if abs(angle) < 1e-3:
            return data_dict
        
        # 旋转图像
        data_dict['image'] = F.rotate(data_dict['image'], angle)
        
        # 旋转深度图（如果存在）
        if 'depth' in data_dict and data_dict['depth'] is not None:
            data_dict['depth'] = F.rotate(data_dict['depth'].unsqueeze(0), 
                                        angle).squeeze(0)
        
        # 注意：旋转会改变内参矩阵，这里简化处理
        # 实际应用中可能需要更复杂的处理
        
        return data_dict


class Normalize:
    """标准化图像（使用ImageNet统计量）"""
    
    def __init__(self, mean=None, std=None):
        """
        Args:
            mean: 均值，默认为ImageNet均值
            std: 标准差，默认为ImageNet标准差
        """
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
    
    def __call__(self, data_dict: Dict) -> Dict:
        """应用标准化"""
        if 'image' in data_dict:
            image = data_dict['image']
            # 确保图像值在[0, 1]范围内
            if image.max() > 1.0:
                image = image / 255.0
            
            # 应用标准化
            data_dict['image'] = (image - self.mean) / self.std
        
        return data_dict


class Compose:
    """组合多个变换"""
    
    def __init__(self, transforms):
        """
        Args:
            transforms: 变换列表
        """
        self.transforms = transforms
    
    def __call__(self, data_dict: Dict) -> Dict:
        """按顺序应用所有变换"""
        for transform in self.transforms:
            data_dict = transform(data_dict)
        return data_dict
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


def create_train_transforms(image_size: Tuple[int, int] = (128, 128)):
    """
    创建训练数据增强变换
    
    Args:
        image_size: 输出图像尺寸 (H, W)
        
    Returns:
        组合变换
    """
    transforms = [
        RandomCrop(size=image_size),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        RandomRotation(degrees=10),
        Normalize()
    ]
    
    return Compose(transforms)


def create_val_transforms(image_size: Tuple[int, int] = (128, 128)):
    """
    创建验证数据变换
    
    Args:
        image_size: 输出图像尺寸 (H, W)
        
    Returns:
        组合变换
    """
    transforms = [
        RandomCrop(size=image_size),  # 验证集也使用裁剪确保尺寸一致
        Normalize()
    ]
    
    return Compose(transforms)


def create_test_transforms(image_size: Tuple[int, int] = (128, 128)):
    """
    创建测试数据变换
    
    Args:
        image_size: 输出图像尺寸 (H, W)
        
    Returns:
        组合变换
    """
    transforms = [
        # 测试集通常使用中心裁剪
        T.CenterCrop(size=image_size),
        Normalize()
    ]
    
    return Compose(transforms)


class SequenceAwareTransform:
    """
    序列感知变换
    
    确保同一序列的所有帧应用相同的随机变换
    """
    
    def __init__(self, base_transform):
        """
        Args:
            base_transform: 基础变换
        """
        self.base_transform = base_transform
        self.sequence_states = {}  # 缓存序列的随机状态
    
    def __call__(self, data_dict: Dict) -> Dict:
        """
        应用序列感知变换
        
        对于同一序列的所有帧，使用相同的随机状态
        """
        seq_id = data_dict.get('sequence_id', 'default')
        frame_id = data_dict.get('frame_id', 0)
        
        # 为每个序列保存随机状态
        if seq_id not in self.sequence_states:
            # 为序列生成随机种子
            seed = hash(seq_id) % 10000
            self.sequence_states[seq_id] = {
                'seed': seed,
                'applied_frames': set()
            }
        
        seq_state = self.sequence_states[seq_id]
        
        # 如果这是序列的第一帧，生成随机参数
        if frame_id == 0 or frame_id not in seq_state['applied_frames']:
            # 设置随机种子
            random.seed(seq_state['seed'] + frame_id)
            
            # 记录已处理的帧
            seq_state['applied_frames'].add(frame_id)
        
        # 应用基础变换
        return self.base_transform(data_dict)
    
    def reset(self):
        """重置所有序列状态"""
        self.sequence_states.clear()