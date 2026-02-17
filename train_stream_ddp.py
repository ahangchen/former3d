#!/usr/bin/env python3
"""
DDP版本的流式训练脚本
使用DistributedDataParallel实现单机多卡训练
"""

import os
import sys
import argparse
import time
import math
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
# 使TensorBoard成为可选项
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except (ImportError, AttributeError):
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
import numpy as np

# 确保torch.nn可用（compute_loss函数需要）
_ = nn  # 标记为已使用

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from distributed_utils import (
    setup_distributed, cleanup_distributed,
    create_distributed_dataloader, is_main_process,
    get_rank, get_world_size, save_on_master,
    AverageMeter, adjust_learning_rate, print_rank_0,
    synchronize
)
from former3d.pose_aware_stream_sdfformer_sparse import PoseAwareStreamSdfFormerSparse

# 导入MultiSequenceTartanAirDataset
try:
    from multi_sequence_tartanair_dataset import MultiSequenceTartanAirDataset
    DATASET_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unable to import MultiSequenceTartanAirDataset: {e}")
    print("Will use DummyDataset for testing")
    MultiSequenceTartanAirDataset = None
    DATASET_AVAILABLE = False

# 导入Rerun可视化器
try:
    from rerun_visualizer import RerunVisualizer
    RERUN_VIZ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unable to import RerunVisualizer: {e}")
    print("Will skip visualization features")
    RERUN_VIZ_AVAILABLE = False
    RerunVisualizer = None

# 导入实验配置管理
try:
    from experiment_config import create_experiment_directory
    EXPERIMENT_CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unable to import experiment_config: {e}")
    print("Will use default directory structure")
    EXPERIMENT_CONFIG_AVAILABLE = False
    create_experiment_directory = None


def parse_args():
    parser = argparse.ArgumentParser(description='DDP流式3D重建训练')

    # 训练参数
    parser.add_argument('--batch-size', type=int, default=4, help='总batch size（会被分配到各个GPU）')
    parser.add_argument('--epochs', type=int, default=100, help='训练epoch数')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='基础学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='warmup epoch数')
    parser.add_argument('--accumulation-steps', type=int, default=1, help='梯度累积步数')

    # 模型参数
    parser.add_argument('--attn-heads', type=int, default=2, help='注意力头数')
    parser.add_argument('--attn-layers', type=int, default=0, help='注意力层数')
    parser.add_argument('--use-proj-occ', action='store_true', help='使用投影占用')
    parser.add_argument('--voxel-size', type=float, default=0.0625, help='体素大小')
    parser.add_argument('--fusion-radius', type=float, default=0.0, help='融合半径')
    parser.add_argument('--crop-size', type=int, nargs=3, default=[10, 8, 6], help='裁剪空间大小')
    parser.add_argument('--use-checkpoint', action='store_true', help='使用gradient checkpointing')

    # 数据参数
    parser.add_argument('--data-path', type=str, default='/home/cwh/Study/dataset/tartanair', help='TartanAir数据根目录')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载worker数')
    parser.add_argument('--sequence-length', type=int, default=10, help='序列长度（每个片段的帧数）')
    parser.add_argument('--max-sequences', type=int, default=5, help='最大序列数')
    parser.add_argument('--target-image-size', type=int, nargs=2, default=[256, 256], help='目标图像大小')
    parser.add_argument('--max-depth', type=float, default=10.0, help='最大深度值（米）')
    parser.add_argument('--truncation-margin', type=float, default=0.2, help='TSDF截断边界')

    # 其他参数
    parser.add_argument('--log-dir', type=str, default='logs', help='日志基础目录（实验目录将创建在此下）')
    parser.add_argument('--save-frequency', type=int, default=1, help='checkpoint保存频率（每个epoch都保存）')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    # Rerun可视化参数
    parser.add_argument('--enable-rerun-viz', action='store_true', help='启用Rerun可视化')
    parser.add_argument('--rerun-viz-freq', type=int, default=1, help='可视化频率（每N个epoch记录一次）')

    return parser.parse_args()


def create_model(args):
    """创建模型"""
    model = PoseAwareStreamSdfFormerSparse(
        attn_heads=args.attn_heads,
        attn_layers=args.attn_layers,
        use_proj_occ=args.use_proj_occ,
        voxel_size=args.voxel_size,
        fusion_local_radius=args.fusion_radius,
        crop_size=tuple(args.crop_size),
        use_checkpoint=args.use_checkpoint
    )

    return model


def create_optimizer(model, args):
    """创建优化器"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    return optimizer


def prepare_visualization_data(batch, outputs, sequence_length):
    """
    准备可视化数据，将batch和模型输出转换为RerunVisualizer期望的格式

    Args:
        batch: 原始batch数据
        outputs: 模型输出
        sequence_length: 序列长度

    Returns:
        dict: 可视化数据字典
    """
    # 提取RGB图像并转换通道顺序：(batch, n_view, 3, H, W) -> (batch, n_view, H, W, 3)
    rgb_images = batch['rgb_images'].permute(0, 1, 3, 4, 2).cpu().numpy()  # (batch, n_view, H, W, 3)

    # 计算深度图（从TSDF的第一层提取作为近似）
    # TSDF格式：(batch, 1, D, H, W)，取D=0层作为深度近似
    tsdf = batch['tsdf'].cpu()  # (batch, 1, D, H, W)
    rgb_height, rgb_width = batch['rgb_images'].shape[3], batch['rgb_images'].shape[4]

    if tsdf.shape[2] > 0:  # 如果有深度维度
        depth_tsdf = tsdf[:, 0, 0, :, :]  # (batch, H_tsdf, W_tsdf)
        # 使用上采样或下采样匹配RGB分辨率
        from torch.nn.functional import interpolate
        depth_tsdf = depth_tsdf.unsqueeze(1)  # (batch, 1, H_tsdf, W_tsdf)
        depth_upsampled = interpolate(depth_tsdf, size=(rgb_height, rgb_width),
                                    mode='bilinear', align_corners=False)
        depth_upsampled = depth_upsampled.squeeze(1)  # (batch, H, W)
        # 扩展到n_view维度
        depth = depth_upsampled.unsqueeze(1).repeat(1, sequence_length, 1, 1)  # (batch, n_view, H, W)
    else:
        # 如果没有深度维度，创建零深度
        depth = torch.zeros(batch['rgb_images'].shape[0], sequence_length,
                         rgb_height, rgb_width)
    depth = depth.numpy()

    # 提取相机参数
    poses = batch['poses'].cpu().numpy()  # (batch, n_view, 4, 4)
    intrinsics = batch['intrinsics'].cpu().numpy()  # (batch, n_view, 3, 3)

    # 计算占用真值：从TSDF计算
    # 占用 = |TSDF| < 0.5
    tsdf_numpy = batch['tsdf'].cpu().numpy()
    occupancy = (np.abs(tsdf_numpy) < 0.5).astype(np.float32)  # (batch, 1, D, H, W)

    # 准备基础可视化数据
    viz_data = {
        'rgb_images': rgb_images,
        'depth': depth,
        'poses': poses,
        'intrinsics': intrinsics,
        'tsdf': tsdf_numpy,
        'occupancy': occupancy
    }

    # 尝试提取预测数据（如果可用）
    if outputs is not None:
        if isinstance(outputs, dict):
            # 处理字典格式的输出
            if 'sdf' in outputs and outputs['sdf'] is not None:
                sdf_pred = outputs['sdf']  # 可能格式：(batch, n_view, N, 1) 或其他
                # 转换为numpy
                if isinstance(sdf_pred, torch.Tensor):
                    sdf_pred = sdf_pred.detach().cpu().numpy()
                # 尝试reshape为(batch, N, 1)
                if len(sdf_pred.shape) == 4:  # (batch, n_view, N, 1)
                    # 取最后一个view的预测
                    sdf_pred = sdf_pred[:, -1, :, :]  # (batch, N, 1)
                elif len(sdf_pred.shape) == 3:  # (batch, N, 1) 或 (N, 1)
                    # 确保是(batch, N, 1)格式
                    if sdf_pred.shape[0] != batch['rgb_images'].shape[0]:
                        # 可能是(N, 1)格式，扩展到batch
                        sdf_pred = sdf_pred.unsqueeze(0).repeat(batch['rgb_images'].shape[0], 1, 1)
                viz_data['sdf_pred'] = sdf_pred

            # 其他预测格式（如occupancy prediction）
            # 根据实际模型输出添加
        else:
            # 处理tensor格式的输出
            if isinstance(outputs, torch.Tensor):
                outputs_np = outputs.detach().cpu().numpy()
                # 尝试判断这是什么类型的预测
                if len(outputs_np.shape) == 5:  # (batch, 1, D, H, W) - 可能是占用预测
                    viz_data['occ_pred'] = outputs_np

    return viz_data


def compute_loss(outputs, targets, frame_data):
    """
    计算损失函数 - 处理点云格式的SDF输出与体素网格TSDF真值的匹配

    Args:
        outputs: 模型输出字典，包含'sdf'键
        targets: TSDF真值 (batch, 1, D, H, W)
        frame_data: 当前帧数据字典

    Returns:
        torch.Tensor: 总损失
    """
    # 提取SDF预测（点云格式）
    if isinstance(outputs, dict):
        if 'sdf' in outputs and outputs['sdf'] is not None:
            sdf_pred = outputs['sdf']  # [num_points, 1] 点云格式
        else:
            # 如果没有SDF输出，使用占位符
            if is_main_process():
                print("Warning: No SDF in model outputs, using placeholder loss")
            return torch.tensor(0.1, device=targets.device, requires_grad=True)
    else:
        sdf_pred = outputs

    # 获取TSDF真值（体素网格格式）
    tsdf_gt_raw = targets  # [batch, 1, D, H, W]

    # 检查SDF预测形状
    if len(sdf_pred.shape) == 2 and sdf_pred.shape[1] == 1:
        # 点云格式 [num_points, 1]
        # 我们需要将点云SDF与体素网格TSDF对齐

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

            # 确保所有张量在同一设备上
            device = sdf_pred.device
            gt_mean = gt_mean.to(device)
            gt_std = gt_std.to(device)

            # 计算均值损失和方差损失
            mean_loss = nn.functional.mse_loss(pred_mean.unsqueeze(0), gt_mean.unsqueeze(0))
            std_loss = nn.functional.mse_loss(pred_std.unsqueeze(0), gt_std.unsqueeze(0))

            # 组合损失
            loss = mean_loss + 0.5 * std_loss

            # 只在第一帧记录日志，避免日志过多
            if 'frame_idx' in frame_data and frame_data['frame_idx'] == 0 and is_main_process():
                print(f"Point cloud SDF prediction: {sdf_pred.shape}, mean: {pred_mean:.3f}, std: {pred_std:.3f}")
                print(f"TSDF ground truth: {tsdf_gt_raw.shape}, mean: {gt_mean:.3f}, std: {gt_std:.3f}")
                print(f"Statistical loss: mean_loss={mean_loss:.4f}, std_loss={std_loss:.4f}")

            return loss
        else:
            # 如果没有有效的TSDF值，使用占位损失
            if is_main_process():
                print("Warning: No valid TSDF ground truth, using placeholder loss")
            return torch.tensor(0.1, device=sdf_pred.device, requires_grad=True)
    else:
        # 其他格式，尝试使用原始MSE
        # 调整预测形状以匹配真值
        tsdf_gt = tsdf_gt_raw.permute(0, 1, 4, 2, 3)  # [batch, 1, D, H, W]

        if sdf_pred.shape != tsdf_gt.shape:
            # 调整预测形状以匹配真值
            sdf_pred = nn.functional.interpolate(
                sdf_pred,
                size=tsdf_gt.shape[2:],
                mode='trilinear',
                align_corners=False
            )

        # 计算MSE损失
        return nn.functional.mse_loss(sdf_pred, tsdf_gt)


def train_one_epoch(model, dataloader, optimizer, epoch, args, logger=None):
    """训练一个epoch"""
    model.train()

    # 设置采样器epoch
    if hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)

    # 损失计算器
    loss_meter = AverageMeter()

    # 梯度累积
    accumulation_counter = 0
    optimizer.zero_grad()

    # 计时
    start_time = time.time()

    # 保存最后一个batch和outputs用于可视化
    last_batch = None
    last_outputs = None
    final_sequence_length = 0

    for batch_idx, batch in enumerate(dataloader):
        # 保存最后一个batch用于可视化
        last_batch = batch
        final_sequence_length = batch['rgb_images'].shape[1]

        # 提取数据
        images = batch['rgb_images']  # (batch, n_view, 3, H, W)
        poses = batch['poses']          # (batch, n_view, 4, 4)
        intrinsics = batch['intrinsics']  # (batch, n_view, 3, 3)

        # 移动到GPU
        device = torch.device('cuda')

        # 将数据移到GPU
        images = images.to(device)
        poses = poses.to(device)
        intrinsics = intrinsics.to(device)

        # 调用forward_sequence
        # DDP会自动将model.forward_sequence调用分发到各个GPU
        # 需要访问model.module来调用forward_sequence
        try:
            # forward_sequence内部会遍历所有帧
            outputs, states = model.module.forward_sequence(images, poses, intrinsics, reset_state=True)

            # 保存最后一个outputs用于可视化
            last_outputs = outputs

            # 计算损失
            targets = batch.get('tsdf', None)
            frame_data = {'frame_idx': batch_idx}  # 传递帧索引用于日志
            loss = compute_loss(outputs, targets, frame_data)

            # 梯度累积
            loss = loss / args.accumulation_steps
            loss.backward()

            accumulation_counter += 1

            # 达到累积步数时更新参数
            if accumulation_counter % args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                accumulation_counter = 0

            # 记录损失
            loss_meter.update(loss.item() * args.accumulation_steps)

            # 打印进度
            if is_main_process() and batch_idx % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time
                eta = elapsed / (batch_idx + 1) * (len(dataloader) - batch_idx - 1)

                print(f"Epoch [{epoch}/{args.epochs}] "
                      f"Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss_meter.avg:.6f} "
                      f"LR: {current_lr:.6f} "
                      f"ETA: {eta/60:.1f}min")

                # 记录到TensorBoard
                if logger is not None:
                    global_step = epoch * len(dataloader) + batch_idx
                    logger.add_scalar('train/loss', loss_meter.avg, global_step)
                    logger.add_scalar('train/learning_rate', current_lr, global_step)

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 同步损失（在所有进程间）
    loss_meter.all_reduce()

    return loss_meter.avg, last_batch, last_outputs, final_sequence_length


def validate(model, dataloader, args):
    """验证模型"""
    model.eval()

    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images = batch['rgb_images']
            poses = batch['poses']
            intrinsics = batch['intrinsics']

            try:
                outputs, states = model.forward_sequence(images, poses, intrinsics, reset_state=True)

                # 计算损失
                targets = batch.get('tsdf', None)
                frame_data = {'frame_idx': batch_idx}
                loss = compute_loss(outputs, targets, frame_data)

                loss_meter.update(loss.item())

            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                continue

    # 同步损失
    loss_meter.all_reduce()

    return loss_meter.avg


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed + get_rank())
    np.random.seed(args.seed + get_rank())

    # 初始化分布式环境
    local_rank = setup_distributed()

    # 打印配置（只在rank 0）
    print_rank_0("="*60)
    print_rank_0("DDP流式3D重建训练")
    print_rank_0("="*60)
    print_rank_0(f"配置信息:")
    print_rank_0(f"  - 总batch size: {args.batch_size}")
    print_rank_0(f"  - GPU数量: {get_world_size()}")
    print_rank_0(f"  - 每GPU batch size: {args.batch_size // get_world_size()}")
    print_rank_0(f"  - Epochs: {args.epochs}")
    print_rank_0(f"  - 基础学习率: {args.learning_rate}")
    print_rank_0(f"  - 日志基础目录: {args.log_dir}")
    print_rank_0("="*60)

    # 创建实验目录结构
    experiment_paths = None
    if is_main_process() and EXPERIMENT_CONFIG_AVAILABLE:
        print_rank_0("创建实验目录...")
        experiment_paths = create_experiment_directory(
            base_dir=args.log_dir,
            args=args,
            model_name="PoseAwareStreamSdfFormerSparse-DDP"
        )
        print_rank_0(f"实验目录: {experiment_paths['experiment_dir']}")
        print_rank_0(f"配置文件: {experiment_paths['config_file']}")
        print_rank_0(f"检查点目录: {experiment_paths['checkpoint_dir']}")
    elif is_main_process():
        # 回退到旧的目录结构
        save_dir = os.path.join(args.log_dir, 'ddp_default')
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_dir = os.path.join(save_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        experiment_paths = {
            'experiment_dir': save_dir,
            'checkpoint_dir': checkpoint_dir,
            'rrd_file': os.path.join(save_dir, 'training.rrd')
        }
        print_rank_0(f"使用默认目录结构: {save_dir}")

    # 创建TensorBoard logger
    logger = None
    if is_main_process() and TENSORBOARD_AVAILABLE and experiment_paths:
        log_dir = os.path.join(experiment_paths['experiment_dir'], 'tensorboard')
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir)
        print_rank_0(f"TensorBoard日志将保存到: {log_dir}")
    elif is_main_process():
        print_rank_0("TensorBoard不可用，跳过日志记录")

    # 创建模型
    print_rank_0("创建模型...")
    model = create_model(args)

    # 包装模型（DDP）
    model = model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False  # 设置为True如果有些参数没有用到
    )

    # 创建优化器
    optimizer = create_optimizer(model, args)

    # 恢复训练
    start_epoch = 0
    best_loss = float('inf')

    if args.resume:
        print_rank_0(f"从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))

    # 创建数据集和数据加载器
    print_rank_0("创建数据集...")

    # 根据数据集可用性选择数据集
    if DATASET_AVAILABLE:
        print_rank_0(f"使用MultiSequenceTartanAirDataset")
        print_rank_0(f"数据根目录: {args.data_path}")
        print_rank_0(f"序列长度: {args.sequence_length}")
        print_rank_0(f"最大序列数: {args.max_sequences}")

        train_dataset = MultiSequenceTartanAirDataset(
            data_root=args.data_path,
            n_view=args.sequence_length,
            max_sequences=args.max_sequences,
            crop_size=tuple(args.crop_size),
            voxel_size=args.voxel_size,
            target_image_size=tuple(args.target_image_size),
            max_depth=args.max_depth,
            truncation_margin=args.truncation_margin,
            augment=False,
            shuffle=True
        )

        val_dataset = MultiSequenceTartanAirDataset(
            data_root=args.data_path,
            n_view=args.sequence_length,
            max_sequences=max(1, args.max_sequences // 5),  # 验证集使用较少序列
            crop_size=tuple(args.crop_size),
            voxel_size=args.voxel_size,
            target_image_size=tuple(args.target_image_size),
            max_depth=args.max_depth,
            truncation_margin=args.truncation_margin,
            augment=False,
            shuffle=False
        )
    else:
        print_rank_0("Warning: MultiSequenceTartanAirDataset不可用，使用DummyDataset")
        print_rank_0("请确保multi_sequence_tartanair_dataset.py存在且可导入")

        # 回退到DummyDataset
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, num_samples=100):
                self.num_samples = num_samples

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx):
                H, W = 96, 128
                n_view = 2

                return {
                    'rgb_images': torch.randn(n_view, 3, H, W),
                    'poses': torch.eye(4).unsqueeze(0).repeat(n_view, 1, 1),
                    'intrinsics': torch.eye(3).unsqueeze(0).repeat(n_view, 1, 1),
                    'tsdf': torch.randn(1, 128, 128, 128)  # 示例目标
                }

        train_dataset = DummyDataset(num_samples=200)
        val_dataset = DummyDataset(num_samples=50)

    train_loader, train_sampler = create_distributed_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=MultiSequenceTartanAirDataset.collate_fn if DATASET_AVAILABLE and MultiSequenceTartanAirDataset is not None else None
    )

    val_loader, _ = create_distributed_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=MultiSequenceTartanAirDataset.collate_fn if DATASET_AVAILABLE and MultiSequenceTartanAirDataset is not None else None
    )

    # 创建Rerun可视化器
    visualizer = None
    if is_main_process() and RERUN_VIZ_AVAILABLE and args.enable_rerun_viz and experiment_paths:
        print_rank_0(f"✅ 启用Rerun可视化")
        print_rank_0(f"ℹ️  使用全局模式：所有epoch数据保存到单个文件")

        # 使用实验目录作为可视化输出目录
        viz_dir = experiment_paths['experiment_dir']

        # 初始化可视化器
        visualizer = RerunVisualizer(save_dir=viz_dir, global_mode=True)
        # 在训练开始前初始化recording（全局模式只需要初始化一次）
        visualizer.start_recording()
        print_rank_0(f"可视化输出目录: {viz_dir}")
        print_rank_0(f"Rerun文件: {experiment_paths['rrd_file']}")
    elif is_main_process() and args.enable_rerun_viz and not RERUN_VIZ_AVAILABLE:
        print_rank_0("⚠️ 请求启用Rerun可视化，但RerunVisualizer不可用")
        print_rank_0("⚠️ 请检查rerun_visualizer.py是否存在且可以导入")
    else:
        if is_main_process():
            print_rank_0("ℹ️  Rerun可视化已禁用")

    # 训练循环
    print_rank_0("\n开始训练...")
    print_rank_0("="*60)

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        # 调整学习率
        lr = adjust_learning_rate(
            optimizer, epoch, args.learning_rate,
            args.epochs, args.warmup_epochs
        )

        print_rank_0(f"\nEpoch {epoch+1}/{args.epochs} (LR: {lr:.6f})")
        print_rank_0("-" * 60)

        # 训练
        train_loss, last_batch, last_outputs, seq_len = train_one_epoch(
            model, train_loader, optimizer, epoch, args, logger
        )

        # 验证
        val_loss = validate(model, val_loader, args)

        epoch_time = time.time() - epoch_start_time

        # 打印结果
        print_rank_0(f"Epoch {epoch+1} 完成:")
        print_rank_0(f"  - 训练损失: {train_loss:.6f}")
        print_rank_0(f"  - 验证损失: {val_loss:.6f}")
        print_rank_0(f"  - 耗时: {epoch_time/60:.1f}min")
        print_rank_0("-" * 60)

        # 记录到TensorBoard
        if is_main_process() and logger is not None:
            logger.add_scalar('epoch/train_loss', train_loss, epoch)
            logger.add_scalar('epoch/val_loss', val_loss, epoch)
            logger.add_scalar('epoch/epoch_time', epoch_time, epoch)

        # 执行Rerun可视化（如果启用且达到频率）
        if visualizer and (epoch % args.rerun_viz_freq == 0):
            try:
                print_rank_0(f"正在记录epoch {epoch+1}的可视化数据...")

                # 准备可视化数据
                viz_data = prepare_visualization_data(last_batch, last_outputs, seq_len)

                # 记录可视化（全局模式：不需要start_recording和finish_recording）
                visualizer.log_sample(viz_data, epoch, n_view=seq_len)

                print_rank_0(f"✅ 可视化数据已记录（全局文件: {visualizer.output_path})")
            except Exception as e:
                print_rank_0(f"⚠️ 可视化记录失败: {e}")
                import traceback
                traceback.print_exc()

        # 保存检查点（每个epoch都保存）
        if experiment_paths:
            checkpoint_dir = experiment_paths['checkpoint_dir']
            # 每个epoch都保存checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1:03d}.pth')

            # 如果是最佳模型，额外保存一份
            if val_loss < best_loss:
                best_loss = val_loss
                best_checkpoint_path = os.path.join(experiment_paths['experiment_dir'], 'best_model.pth')
            else:
                best_checkpoint_path = None

            if is_main_process():
                # 保存当前epoch checkpoint
                save_on_master({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'args': args
                }, checkpoint_path)
                print_rank_0(f"✅ Checkpoint已保存: {checkpoint_path}")

                # 如果是最佳模型，额外保存
                if best_checkpoint_path:
                    save_on_master({
                        'epoch': epoch,
                        'model': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'val_loss': val_loss,
                        'train_loss': train_loss,
                        'args': args
                    }, best_checkpoint_path)
                    print_rank_0(f"🏆 最佳模型已保存: {best_checkpoint_path}")

    # 完成训练
    print_rank_0("\n" + "="*60)
    print_rank_0("训练完成！")
    print_rank_0(f"最佳验证损失: {best_loss:.6f}")
    print_rank_0("="*60)

    # 关闭TensorBoard logger
    if is_main_process() and logger is not None:
        logger.close()

    # 完成Rerun可视化（如果启用）
    if visualizer:
        try:
            print_rank_0("正在完成Rerun可视化记录...")
            visualizer.finish_recording()
            print_rank_0(f"✅ 可视化数据已全部保存到: {visualizer.output_path}")
        except Exception as e:
            print_rank_0(f"⚠️ 可视化完成失败: {e}")
            import traceback
            traceback.print_exc()

    # 清理分布式环境
    cleanup_distributed()


if __name__ == '__main__':
    main()