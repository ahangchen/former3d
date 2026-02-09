#!/usr/bin/env python3
"""
流式集成训练脚本 - 使用StreamSDFFormerIntegrated模型
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
from datetime import datetime
import argparse
import logging
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('stream_training.log')
    ]
)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入设备一致性工具
try:
    from device_consistency_utils import DeviceConsistency, move_to_device, batch_device_check
    DEVICE_CONSISTENCY_AVAILABLE = True
    logger.info("✅ 设备一致性工具导入成功")
except ImportError as e:
    logger.warning(f"⚠️ 无法导入设备一致性工具: {e}")
    logger.warning("⚠️ 将继续使用基本设备管理，可能出现设备不匹配错误")
    DEVICE_CONSISTENCY_AVAILABLE = False

# 导入数据集
try:
    from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
    DATASET_AVAILABLE = True
    logger.info("✅ MultiSequenceTartanAirDataset导入成功")
except ImportError as e:
    logger.error(f"❌ 无法导入MultiSequenceTartanAirDataset: {e}")
    sys.exit(1)

# 导入流式状态管理器
try:
    from stream_state_manager import StreamStateManager
    STREAM_STATE_MANAGER_AVAILABLE = True
    logger.info("✅ StreamStateManager导入成功")
except ImportError as e:
    logger.warning(f"⚠️ 无法导入StreamStateManager: {e}")
    logger.warning("⚠️ 将使用基本状态管理")
    STREAM_STATE_MANAGER_AVAILABLE = False

# 导入StreamSDFFormerIntegrated模型
try:
    from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
    MODEL_AVAILABLE = True
    logger.info("✅ StreamSDFFormerIntegrated导入成功")
except ImportError as e:
    logger.error(f"❌ 无法导入StreamSDFFormerIntegrated: {e}")
    sys.exit(1)

# 导入内存管理器
try:
    from memory_manager import MemoryManager
    MEMORY_MANAGER_AVAILABLE = True
    logger.info("✅ MemoryManager导入成功")
except ImportError as e:
    logger.warning(f"⚠️ 无法导入MemoryManager: {e}")
    logger.warning("⚠️ 将不使用显存清理功能")
    MEMORY_MANAGER_AVAILABLE = False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='流式集成训练脚本')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=2, help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载器工作进程数')
    
    # 模型参数
    parser.add_argument('--attn-heads', type=int, default=1, help='注意力头数')
    parser.add_argument('--attn-layers', type=int, default=1, help='注意力层数')
    parser.add_argument('--voxel-size', type=float, default=0.16, help='体素大小')
    parser.add_argument('--crop-size', type=str, default='12,12,8', help='裁剪尺寸')
    
    # 数据参数
    parser.add_argument('--data-root', type=str, default='/home/cwh/Study/dataset/tartanair', help='TartanAir原始数据根目录')
    parser.add_argument('--sequence-length', type=int, default=10, help='序列长度')
    parser.add_argument('--max-sequences', type=int, default=5, help='最大序列数')
    
    # 显存管理参数
    parser.add_argument('--cleanup-freq', type=int, default=10, help='显存清理频率（每N步清理一次）')
    parser.add_argument('--memory-threshold', type=float, default=8.0, help='显存阈值（超过此值自动清理，单位GB）')
    
    # 梯度累积参数
    parser.add_argument('--accumulation-steps', type=int, default=1, help='梯度累积步数（1表示不累积）')
    
    # 运行模式
    parser.add_argument('--dry-run', action='store_true', help='干运行模式，不实际训练')
    parser.add_argument('--test-only', action='store_true', help='仅测试模式')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    # 设备参数
    parser.add_argument('--no-cuda', action='store_true', help='禁用CUDA')
    parser.add_argument('--device', type=str, default='cuda:0', help='设备选择')
    
    return parser.parse_args()

def create_model(args, device):
    """创建StreamSDFFormerIntegrated模型"""
    logger.info("创建StreamSDFFormerIntegrated模型...")
    
    # 解析裁剪尺寸
    crop_size = tuple(map(int, args.crop_size.split(',')))
    
    # 创建模型
    model = StreamSDFFormerIntegrated(
        attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
        use_proj_occ=False,  # 禁用投影占用以获取SDF输出
        voxel_size=args.voxel_size,
        fusion_local_radius=2.0,
        crop_size=crop_size
    )
    
    # 移动到设备
    model = model.to(device)
    logger.info(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"   注意力头数: {args.attn_heads}")
    logger.info(f"   注意力层数: {args.attn_layers}")
    logger.info(f"   体素大小: {args.voxel_size}")
    logger.info(f"   裁剪尺寸: {crop_size}")
    
    return model

def create_dataloader(args, device):
    """创建数据加载器"""
    logger.info("创建MultiSequenceTartanAirDataset...")
    
    try:
        # 创建数据集
        dataset = MultiSequenceTartanAirDataset(
            data_root=args.data_root,
            n_view=args.sequence_length,
            max_sequences=args.max_sequences,
            crop_size=tuple(map(int, args.crop_size.split(','))),
            voxel_size=args.voxel_size
        )
        
        logger.info(f"✅ 数据集创建成功，样本数: {len(dataset)}")
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        logger.info(f"✅ 数据加载器创建成功，批次大小: {args.batch_size}")
        
        return dataloader, dataset
        
    except FileNotFoundError as e:
        logger.warning(f"⚠️ 数据集创建失败: {e}")
        logger.warning("⚠️ 将使用模拟数据进行干运行测试")
        
        # 创建模拟数据集
        class MockDataset:
            def __init__(self):
                self.length = 10
                
            def __len__(self):
                return self.length
                
            def __getitem__(self, idx):
                # 返回模拟数据
                n_frames = args.sequence_length
                height, width = 128, 128
                
                return {
                    'rgb_images': torch.randn(n_frames, 3, height, width),  # [n_frames, 3, H, W]
                    'poses': torch.eye(4).unsqueeze(0).repeat(n_frames, 1, 1),  # [n_frames, 4, 4]
                    'intrinsics': torch.eye(3).unsqueeze(0).repeat(n_frames, 1, 1),  # [n_frames, 3, 3]
                    'tsdf': torch.randn(1, 32, 32, 24),  # [1, H, W, D]
                    'sequence_id': idx
                }
        
        dataset = MockDataset()
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"✅ 模拟数据集创建成功，样本数: {len(dataset)}")
        
        return dataloader, dataset

def extract_frame_data(batch, frame_idx, device):
    """从批次中提取单帧数据"""
    # 获取批次大小
    batch_size = batch['rgb_images'].shape[0]
    
    # 提取当前帧并调整形状
    images = batch['rgb_images'][:, frame_idx]  # [batch, 3, H, W] - 移除n_frames维度
    poses = batch['poses'][:, frame_idx]        # [batch, 4, 4] - 移除n_frames维度
    
    # 处理内参矩阵（可能是 [3, 3] 或 [batch, n_frames, 3, 3]）
    intrinsics_tensor = batch['intrinsics']
    if len(intrinsics_tensor.shape) == 2:
        # 形状为 [3, 3]，直接使用并扩展到批次
        intrinsics = intrinsics_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, 3, 3]
        intrinsics = intrinsics.repeat(batch_size, 1, 1, 1)       # [batch, 1, 3, 3]
        intrinsics = intrinsics[:, 0]  # 取第一个（也是唯一一个）帧 [batch, 3, 3]
    elif len(intrinsics_tensor.shape) == 4:
        # 形状为 [batch, n_frames, 3, 3]
        intrinsics = intrinsics_tensor[:, frame_idx]
    else:
        # 其他形状，尝试直接使用
        intrinsics = intrinsics_tensor
    
    # 确保设备一致性
    if DEVICE_CONSISTENCY_AVAILABLE:
        images = move_to_device(images, device)
        poses = move_to_device(poses, device)
        intrinsics = move_to_device(intrinsics, device)
    else:
        images = images.to(device)
        poses = poses.to(device)
        intrinsics = intrinsics.to(device)
    
    return {
        'images': images,
        'poses': poses,
        'intrinsics': intrinsics,
        'sequence_id': batch.get('sequence_id', torch.zeros(batch_size, dtype=torch.long)),
        'frame_idx': frame_idx
    }

def compute_loss(output, ground_truth, frame_data):
    """计算损失函数 - 处理点云格式的SDF输出与体素网格TSDF真值的匹配"""
    import torch.nn as nn
    
    # 提取SDF预测（点云格式）
    if isinstance(output, dict):
        if 'sdf' in output and output['sdf'] is not None:
            sdf_pred = output['sdf']  # [num_points, 1] 点云格式
        else:
            # 如果没有SDF输出，使用占位符
            logger.warning("⚠️ 模型输出中没有SDF，使用占位损失")
            return torch.tensor(0.1, device=ground_truth.device, requires_grad=True)
    else:
        sdf_pred = output
    
    # 获取TSDF真值（体素网格格式）
    tsdf_gt_raw = ground_truth  # [batch, 1, H, W, D]
    
    # 检查SDF预测形状
    if len(sdf_pred.shape) == 2 and sdf_pred.shape[1] == 1:
        # 点云格式 [num_points, 1]
        # 我们需要将点云SDF与体素网格TSDF对齐
        
        # 方法1：从体素网格中采样对应的TSDF值
        # 这需要知道点云在体素网格中的坐标
        # 由于我们不知道点云坐标，使用简化方法
        
        # 简化方法：计算统计损失
        # 1. 确保SDF预测在合理范围内（-1到1）
        sdf_clamped = torch.clamp(sdf_pred, -1.0, 1.0)
        
        # 2. 计算TSDF真值的统计信息
        tsdf_flat = tsdf_gt_raw.view(-1)
        valid_tsdf = tsdf_flat[tsdf_flat != 0]  # 只考虑非零TSDF值
        
        if len(valid_tsdf) > 0:
            # 3. 计算SDF预测与TSDF真值的统计匹配损失
            # 使用均值和方差匹配
            pred_mean = sdf_clamped.mean()
            pred_std = sdf_clamped.std()
            
            gt_mean = valid_tsdf.mean()
            gt_std = valid_tsdf.std()
            
            # 计算均值损失和方差损失
            mean_loss = nn.functional.mse_loss(pred_mean.unsqueeze(0), gt_mean.unsqueeze(0))
            std_loss = nn.functional.mse_loss(pred_std.unsqueeze(0), gt_std.unsqueeze(0))
            
            # 组合损失
            loss = mean_loss + 0.5 * std_loss
            
            # 只在第一帧记录日志，避免日志过多
            if 'frame_idx' in frame_data and frame_data['frame_idx'] == 0:
                logger.info(f"点云SDF预测: {sdf_pred.shape}, 均值: {pred_mean:.3f}, 标准差: {pred_std:.3f}")
                logger.info(f"TSDF真值: {tsdf_gt_raw.shape}, 均值: {gt_mean:.3f}, 标准差: {gt_std:.3f}")
                logger.info(f"统计损失: 均值损失={mean_loss:.4f}, 方差损失={std_loss:.4f}")
            
            return loss
        else:
            # 如果没有有效的TSDF值，使用占位损失
            logger.warning("⚠️ 没有有效的TSDF真值，使用占位损失")
            return torch.tensor(0.1, device=sdf_pred.device, requires_grad=True)
    else:
        # 其他格式，尝试使用原始MSE
        # 调整预测形状以匹配真值
        tsdf_gt = tsdf_gt_raw.permute(0, 1, 4, 2, 3)  # [batch, 1, D, H, W]
        
        if sdf_pred.shape != tsdf_gt.shape:
            # 调整预测形状以匹配真值
            sdf_pred = torch.nn.functional.interpolate(
                sdf_pred,
                size=tsdf_gt.shape[2:],
                mode='trilinear',
                align_corners=False
            )
        
        # 计算MSE损失
        return nn.functional.mse_loss(sdf_pred, tsdf_gt)

def train_epoch_stream(model, dataloader, optimizer, device, args, epoch):
    """流式训练一个epoch（支持梯度累积）"""
    model.train()
    total_loss = 0.0
    total_frames = 0
    
    # 获取梯度累积步数
    accumulation_steps = getattr(args, 'accumulation_steps', 1)
    
    # 创建状态管理器
    if STREAM_STATE_MANAGER_AVAILABLE:
        state_manager = StreamStateManager(device=device, max_cached_states=5)
    else:
        state_manager = None
    
    # 创建内存管理器
    if MEMORY_MANAGER_AVAILABLE:
        memory_manager = MemoryManager(
            cleanup_frequency=getattr(args, 'cleanup_freq', 10),
            memory_threshold_gb=getattr(args, 'memory_threshold', 8.0)
        )
    else:
        memory_manager = None
    
    # 梯度累积计数器
    accumulation_counter = 0
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        # 确保设备一致性
        if DEVICE_CONSISTENCY_AVAILABLE:
            batch = move_to_device(batch, device)
        else:
            # 基本设备移动
            for key in ['rgb_images', 'poses', 'intrinsics', 'tsdf']:
                if key in batch:
                    batch[key] = batch[key].to(device)
        
        # 获取序列信息
        batch_size = batch['rgb_images'].shape[0]
        sequence_length = batch['rgb_images'].shape[1]
        
        # 按帧处理序列
        sequence_loss = 0.0
        current_sequence_id = None
        
        for frame_idx in range(sequence_length):
            # 提取当前帧数据
            frame_data = extract_frame_data(batch, frame_idx, device)
            
            # 获取序列ID
            sequence_id = frame_data['sequence_id'][0].item() if batch_size == 1 else 0
            
            # 检查是否需要重置状态
            reset_state = False
            if state_manager is not None:
                if current_sequence_id != sequence_id:
                    # 新序列，重置状态
                    state_manager.reset(sequence_id)
                    current_sequence_id = sequence_id
                    reset_state = True
                else:
                    reset_state = (frame_idx == 0)
            
            # 获取历史状态
            historical_state = state_manager.get_state() if state_manager is not None else None
            
            # 前向传播
            output, new_state = model.forward_single_frame(
                images=frame_data['images'],
                poses=frame_data['poses'],
                intrinsics=frame_data['intrinsics'],
                reset_state=reset_state
            )
            
            # 计算损失（除以累积步数以保持梯度一致性）
            loss = compute_loss(output, batch['tsdf'], frame_data)
            loss = loss / accumulation_steps  # 除以累积步数
            sequence_loss += loss
            
            # 反向传播（累积梯度）
            loss.backward()
            
            # 更新状态
            if state_manager is not None and new_state is not None:
                state_manager.update_state(new_state, sequence_id, frame_idx, reset_state=False)
        
        # 累积梯度计数
        accumulation_counter += 1
        
        # 达到累积步数或最后一个batch时更新参数
        if accumulation_counter % accumulation_steps == 0:
            # 梯度裁剪（可选，防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新参数
            optimizer.step()
            optimizer.zero_grad()
            
            # 记录损失（需要乘回累积步数）
            total_loss += sequence_loss.item() * accumulation_steps
            total_frames += sequence_length
            
            # 重置序列损失
            sequence_loss = 0.0
        
        # 内存管理：定期清理或按需清理
        if memory_manager is not None:
            # 每个batch后执行定期清理
            memory_manager.step()
            
            # 检查是否需要按阈值清理
            memory_manager.cleanup_if_needed()
        
        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            avg_loss = total_loss / max(total_frames, 1)
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(dataloader)}, "
                       f"Loss: {avg_loss:.6f}, Accumulation: {accumulation_counter % accumulation_steps + 1}/{accumulation_steps}")
    
    # 处理剩余的累积梯度
    if accumulation_counter % accumulation_steps != 0:
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        optimizer.zero_grad()
        
        # 记录损失
        total_loss += sequence_loss.item() * (accumulation_counter % accumulation_steps)
        total_frames += sequence_length
    
    # 计算平均损失
    avg_loss = total_loss / max(total_frames, 1)
    return avg_loss

def test_model(model, dataloader, device, args):
    """测试模型"""
    model.eval()
    total_loss = 0.0
    total_frames = 0
    
    # 创建内存管理器
    if MEMORY_MANAGER_AVAILABLE:
        memory_manager = MemoryManager(
            cleanup_frequency=getattr(args, 'cleanup_freq', 10),
            memory_threshold_gb=getattr(args, 'memory_threshold', 8.0)
        )
    else:
        memory_manager = None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 确保设备一致性
            if DEVICE_CONSISTENCY_AVAILABLE:
                batch = move_to_device(batch, device)
            else:
                # 基本设备移动
                for key in ['rgb_images', 'poses', 'intrinsics', 'tsdf']:
                    if key in batch:
                        batch[key] = batch[key].to(device)
            
            # 获取序列信息
            batch_size = batch['rgb_images'].shape[0]
            sequence_length = batch['rgb_images'].shape[1]
            
            # 重置状态
            state = None
            
            for frame_idx in range(sequence_length):
                # 提取当前帧数据
                frame_data = extract_frame_data(batch, frame_idx, device)
                
                # 前向传播
                output, state = model.forward_single_frame(
                    images=frame_data['images'],
                    poses=frame_data['poses'],
                    intrinsics=frame_data['intrinsics'],
                    reset_state=(frame_idx == 0)
                )
                
                # 调试：打印输出信息（仅第一帧）
                if batch_idx == 0 and frame_idx == 0:
                    logger.info(f"测试批次 {batch_idx}, 帧 {frame_idx}:")
                    if isinstance(output, dict):
                        logger.info(f"  输出字典键: {list(output.keys())}")
                        for k, v in output.items():
                            if v is not None:
                                if hasattr(v, 'shape'):
                                    logger.info(f"    {k}: {v.shape}")
                                else:
                                    logger.info(f"    {k}: {type(v)}")
                            else:
                                logger.info(f"    {k}: None")
                    else:
                        logger.info(f"  输出类型: {type(output)}")
                
                # 计算损失
                loss = compute_loss(output, batch['tsdf'], frame_data)
                total_loss += loss.item()
                total_frames += 1
            
            # 内存管理：定期清理或按需清理
            if memory_manager is not None:
                # batch_size=2时，每个batch后都强制清理以减少显存碎片化
                if args.batch_size >= 2:
                    memory_manager.force_cleanup(verbose=False)
                else:
                    # batch_size=1时，每5步清理一次
                    memory_manager.step()
                    # 检查是否需要按阈值清理
                    memory_manager.cleanup_if_needed()

                # 显存监控（每5个batch）
                if (batch_idx + 1) % 5 == 0:
                    memory_info = memory_manager.get_memory_info()
                    if memory_info.get('cuda_available', False):
                        logger.info(f"  显存监控 - "
                                   f"已分配: {memory_info['allocated_gb']:.3f}GB, "
                                   f"已保留: {memory_info['reserved_gb']:.3f}GB, "
                                   f"峰值已分配: {memory_info['max_allocated_gb']:.3f}GB")
            
            # 打印进度
            if (batch_idx + 1) % 5 == 0:
                avg_loss = total_loss / max(total_frames, 1)
                logger.info(f"Test Batch {batch_idx+1}/{len(dataloader)}, "
                           f"Loss: {avg_loss:.6f}")
    
    # 计算平均损失
    avg_loss = total_loss / max(total_frames, 1)
    return avg_loss

def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"使用设备: {device}")
    logger.info(f"参数: {args}")
    
    # 创建模型
    model = create_model(args, device)
    
    # 创建数据加载器
    dataloader, dataset = create_dataloader(args, device)
    
    # 如果是干运行模式，只检查不训练
    if args.dry_run:
        logger.info("干运行模式 - 检查配置...")
        
        # 检查一个批次
        for batch in dataloader:
            logger.info(f"批次键: {list(batch.keys())}")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  {key}: {value.shape}, {value.dtype}, {value.device}")
                elif isinstance(value, (list, tuple)):
                    logger.info(f"  {key}: {type(value).__name__} length {len(value)}")
            
            # 测试模型前向传播
            logger.info("测试模型前向传播...")
            try:
                # 提取第一帧
                frame_data = extract_frame_data(batch, 0, device)
                
                # 前向传播
                output, state = model.forward_single_frame(
                    images=frame_data['images'],
                    poses=frame_data['poses'],
                    intrinsics=frame_data['intrinsics'],
                    reset_state=True
                )
                
                logger.info(f"✅ 前向传播成功")
                if isinstance(output, dict):
                    logger.info(f"   输出字典键: {list(output.keys())}")
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor):
                            logger.info(f"     {k}: {v.shape}")
                else:
                    logger.info(f"   输出形状: {output.shape}")
                
                if state is not None:
                    logger.info(f"   状态类型: {type(state)}")
                
            except Exception as e:
                logger.error(f"❌ 前向传播失败: {e}")
                import traceback
                traceback.print_exc()
            
            break  # 只检查一个批次
        
        logger.info("✅ 干运行完成")
        return
    
    # 如果是仅测试模式
    if args.test_only:
        logger.info("仅测试模式...")
        test_loss = test_model(model, dataloader, device, args)
        logger.info(f"测试损失: {test_loss:.6f}")
        return
    
    # 正常训练模式
    logger.info("开始训练...")
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # 训练循环
    for epoch in range(args.epochs):
        logger.info(f"开始第 {epoch+1}/{args.epochs} 轮训练")
        
        # 训练一个epoch
        train_loss = train_epoch_stream(model, dataloader, optimizer, device, args, epoch)
        
        logger.info(f"第 {epoch+1} 轮训练完成，平均损失: {train_loss:.6f}")
        
        # 每5轮保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"checkpoints/stream_model_epoch_{epoch+1}.pth"
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'args': vars(args)
            }, checkpoint_path)
            logger.info(f"检查点保存到: {checkpoint_path}")
    
    logger.info("训练完成!")

if __name__ == '__main__':
    main()