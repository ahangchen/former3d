"""
Task 3.2: 端到端小循环训练验证
串联TartanAirStreamingDataset进行训练验证
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import tempfile
import shutil

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*80)
print("端到端小循环训练验证（串联TartanAir数据集）")
print("="*80)

# 检查环境
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")


def create_mock_tartanair_dataset():
    """创建模拟TartanAir数据集"""
    print("\n▶️ 创建模拟TartanAir数据集")
    
    # 创建临时目录结构
    temp_dir = tempfile.mkdtemp(prefix="tartanair_mock_")
    print(f"  临时目录: {temp_dir}")
    
    # 创建TartanAir目录结构
    env_name = "abandonedfactory"
    difficulty = "Easy"
    trajectory = "P000"
    
    dataset_root = Path(temp_dir) / "tartanair"
    sequence_dir = dataset_root / env_name / difficulty / trajectory
    
    # 创建目录
    sequence_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建图像目录
    image_dir = sequence_dir / "image_left"
    depth_dir = sequence_dir / "depth_left"
    image_dir.mkdir(exist_ok=True)
    depth_dir.mkdir(exist_ok=True)
    
    frame_count = 10
    image_size = (64, 64)
    
    for i in range(frame_count):
        # 创建模拟图像文件
        img_path = image_dir / f"{i:06d}.png"
        
        # 创建模拟numpy数组并保存为PNG（通过PIL）
        import PIL.Image as Image
        img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(str(img_path))
        
        # 创建模拟深度图
        depth_path = depth_dir / f"{i:06d}.npy"
        depth_array = np.random.rand(*image_size).astype(np.float32) * 10.0  # 0-10米
        np.save(str(depth_path), depth_array)
    
    # 创建位姿文件
    pose_path = sequence_dir / "pose_left.txt"
    with open(pose_path, 'w') as f:
        for i in range(frame_count):
            # 创建单位矩阵加上小扰动
            pose = np.eye(4)
            pose[:3, 3] = [i*0.1, 0.0, 0.0]  # 沿x轴移动
            pose_str = ' '.join(map(str, pose.flatten()))
            f.write(pose_str + '\n')
    
    print(f"  创建了 {frame_count} 帧模拟数据")
    print(f"  环境: {env_name}")
    print(f"  难度: {difficulty}")
    print(f"  轨迹: {trajectory}")
    
    return str(dataset_root), temp_dir


def test_dataset_loading(dataset_root):
    """测试数据集加载"""
    print("\n▶️ 测试数据集加载")
    
    try:
        from former3d.datasets.tartanair_dataset import TartanAirStreamingDataset
        
        # 创建数据集
        dataset = TartanAirStreamingDataset(
            data_root=dataset_root,
            split='train',
            sequence_ids=None,
            transform=None,
            load_depth=True,
            load_sdf=False,
            max_sequence_length=3,
            image_size=(64, 64),
            normalize_images=True,
            cache_data=False,
            use_left_camera=True,
            frame_interval=1
        )
        
        print(f"✅ 数据集创建成功")
        print(f"  总帧数: {len(dataset)}")
        
        # 测试数据加载
        try:
            sample = dataset[0]
            print(f"✅ 数据加载成功")
            print(f"  图像形状: {sample['image'].shape}")
            print(f"  位姿形状: {sample['pose'].shape}")
            print(f"  内参形状: {sample['intrinsics'].shape}")
            print(f"  帧ID: {sample['frame_id']}")
            print(f"  序列ID: {sample['sequence_id']}")
            
            if 'depth' in sample:
                print(f"  深度图形状: {sample['depth'].shape}")
            
            return dataset, True
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None, False
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def create_simplified_model():
    """创建简化模型用于训练测试"""
    print("\n▶️ 创建简化模型")
    
    try:
        # 创建一个非常简化的模型用于测试
        class SimplifiedStreamModel(nn.Module):
            def __init__(self):
                super().__init__()
                # 简化的2D特征提取
                self.net2d = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten()
                )
                
                # 简化的3D处理
                self.net3d = nn.Sequential(
                    nn.Linear(32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)  # SDF输出
                )
                
                # 历史状态
                self.historical_feature = None
            
            def forward(self, image, pose=None, intrinsics=None, reset_state=False):
                # 提取2D特征
                features = self.net2d(image)  # [B, 32]
                
                # 如果有历史特征，进行简单融合
                if self.historical_feature is not None and not reset_state:
                    # 简单平均融合
                    features = (features + self.historical_feature) / 2
                
                # 更新历史特征
                self.historical_feature = features.detach()
                
                # 3D处理
                sdf = self.net3d(features)
                
                return {'sdf': sdf}
            
            def reset_state(self):
                self.historical_feature = None
        
        model = SimplifiedStreamModel()
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        print(f"✅ 简化模型创建成功")
        print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  设备: {next(model.parameters()).device}")
        
        return model
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return None


def test_training_with_dataset(dataset, model):
    """使用数据集进行训练测试"""
    print("\n▶️ 使用数据集进行训练测试")
    
    try:
        # 创建数据加载器
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        # 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练循环
        epochs = 3
        losses = []
        
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= 5:  # 只测试前5个批次
                    break
                
                # 移动到GPU
                if torch.cuda.is_available():
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].cuda()
                
                # 获取数据
                images = batch['image']
                poses = batch['pose']
                intrinsics = batch['intrinsics']
                
                # 重置优化器
                optimizer.zero_grad()
                
                # 重置模型状态（每批次开始）
                model.reset_state()
                
                # 序列处理（模拟流式）
                seq_len = images.shape[1] if len(images.shape) > 4 else 1
                total_loss = 0
                
                if seq_len > 1:
                    # 多帧序列
                    for t in range(seq_len):
                        frame_image = images[:, t]
                        frame_pose = poses[:, t] if poses.dim() > 3 else poses
                        frame_intrinsics = intrinsics[:, t] if intrinsics.dim() > 3 else intrinsics
                        
                        # 第一帧重置状态
                        reset = (t == 0)
                        
                        # 前向传播
                        output = model(
                            image=frame_image,
                            pose=frame_pose,
                            intrinsics=frame_intrinsics,
                            reset_state=reset
                        )
                        
                        # 计算损失（模拟SDF损失）
                        if 'sdf' in output:
                            sdf_pred = output['sdf']
                            sdf_target = torch.randn_like(sdf_pred) * 0.1
                            loss = nn.functional.mse_loss(sdf_pred, sdf_target)
                            total_loss += loss
                else:
                    # 单帧
                    output = model(
                        image=images,
                        pose=poses,
                        intrinsics=intrinsics,
                        reset_state=True
                    )
                    
                    if 'sdf' in output:
                        sdf_pred = output['sdf']
                        sdf_target = torch.randn_like(sdf_pred) * 0.1
                        total_loss = nn.functional.mse_loss(sdf_pred, sdf_target)
                
                # 平均损失
                avg_loss = total_loss / max(seq_len, 1)
                epoch_loss += avg_loss.item()
                batch_count += 1
                
                # 反向传播
                avg_loss.backward()
                
                # 检查梯度
                has_gradients = False
                for param in model.parameters():
                    if param.grad is not None:
                        has_gradients = True
                        break
                
                if not has_gradients:
                    print(f"  ⚠️ 批次 {batch_idx}: 无梯度")
                    continue
                
                # 更新参数
                optimizer.step()
                
                if batch_idx % 2 == 0:
                    print(f"  批次 {batch_idx}: 损失={avg_loss.item():.6f}, 有梯度={has_gradients}")
            
            # 计算epoch平均损失
            if batch_count > 0:
                avg_epoch_loss = epoch_loss / batch_count
                losses.append(avg_epoch_loss)
                print(f"  Epoch {epoch+1}: 平均损失={avg_epoch_loss:.6f}")
        
        # 检查训练效果
        if len(losses) >= 2:
            print(f"\n  训练总结:")
            print(f"    初始损失: {losses[0]:.6f}")
            print(f"    最终损失: {losses[-1]:.6f}")
            
            if losses[-1] < losses[0]:
                improvement = losses[0] - losses[-1]
                print(f"  ✅ 训练成功: 损失下降 {improvement:.6f}")
                return True
            else:
                print(f"  ❌ 训练失败: 损失未下降")
                return False
        else:
            print(f"  ❌ 训练失败: 损失记录不足")
            return False
            
    except Exception as e:
        print(f"❌ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_integration():
    """测试完整集成"""
    print("\n▶️ 测试完整集成（模型 + 数据集）")
    
    try:
        # 1. 创建模拟数据集
        dataset_root, temp_dir = create_mock_tartanair_dataset()
        
        try:
            # 2. 测试数据集加载
            dataset, dataset_ok = test_dataset_loading(dataset_root)
            if not dataset_ok:
                return False
            
            # 3. 创建简化模型
            model = create_simplified_model()
            if model is None:
                return False
            
            # 4. 测试训练
            training_ok = test_training_with_dataset(dataset, model)
            
            if training_ok:
                print("✅ 完整集成测试通过")
                return True
            else:
                print("❌ 完整集成测试失败")
                return False
                
        finally:
            # 清理临时目录
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"  清理临时目录: {temp_dir}")
                
    except Exception as e:
        print(f"❌ 完整集成测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    tests = [
        ("完整集成测试", test_full_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"开始测试: {test_name}")
            print('='*60)
            
            success = test_func()
            results.append((test_name, success))
            
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 总结结果
    print("\n" + "="*80)
    print("端到端训练验证结果")
    print("="*80)
    
    passed = 0
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\n总体结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 端到端训练验证通过！")
        return True
    else:
        print("⚠️ 端到端训练验证失败")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)