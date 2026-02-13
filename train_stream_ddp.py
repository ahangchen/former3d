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
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from distributed_utils import (
    setup_distributed, cleanup_distributed,
    create_distributed_dataloader, is_main_process,
    get_rank, get_world_size, save_on_master,
    AverageMeter, adjust_learning_rate, print_rank_0,
    synchronize
)
from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated


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
    parser.add_argument('--data-path', type=str, default='/home/cwh/coding/former3d/data', help='数据路径')
    parser.add_argument('--num-workers', type=int, default=4, help='数据加载worker数')

    # 其他参数
    parser.add_argument('--save-dir', type=str, default='./checkpoints/ddp', help='保存目录')
    parser.add_argument('--save-frequency', type=int, default=5, help='保存频率（epoch）')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')

    return parser.parse_args()


def create_model(args):
    """创建模型"""
    model = StreamSDFFormerIntegrated(
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


def compute_loss(outputs, targets, **kwargs):
    """
    计算损失函数

    Args:
        outputs: 模型输出
        targets: 目标值
        **kwargs: 其他参数

    Returns:
        torch.Tensor: 总损失
    """
    # 这里实现实际的损失计算
    # 根据您的具体需求修改

    # 示例：使用L1损失
    loss = torch.tensor(0.0, device=outputs['sdf'].device)

    if 'sdf' in outputs and targets is not None:
        loss = loss + torch.nn.functional.l1_loss(outputs['sdf'], targets)

    return loss


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

    for batch_idx, batch in enumerate(dataloader):
        # 提取数据
        images = batch['rgb_images']
        poses = batch['poses']
        intrinsics = batch['intrinsics']

        # 移动到GPU
        device = images.device

        # 调用forward_sequence
        # DDP会自动将model.forward_sequence调用分发到各个GPU
        try:
            outputs, states = model.forward_sequence(images, poses, intrinsics, reset_state=True)

            # 计算损失
            targets = batch.get('tsdf', None)
            loss = compute_loss(outputs, targets)

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

    return loss_meter.avg


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
                loss = compute_loss(outputs, targets)

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
    print_rank_0(f"  - 保存目录: {args.save_dir}")
    print_rank_0("="*60)

    # 创建保存目录
    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)

    # 创建TensorBoard logger
    logger = None
    if is_main_process():
        log_dir = os.path.join(args.save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        logger = SummaryWriter(log_dir)

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

    # 创建数据集和数据加载器（示例）
    # 注意：这里需要根据您的实际数据集进行修改
    print_rank_0("创建数据集...")

    # 示例数据集（需要替换为您的实际数据集）
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
        shuffle=True
    )

    val_loader, _ = create_distributed_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

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
        train_loss = train_one_epoch(
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

        # 保存检查点
        if (epoch + 1) % args.save_frequency == 0 or val_loss < best_loss:
            if val_loss < best_loss:
                best_loss = val_loss
                checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
            else:
                checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch{epoch+1}.pth')

            if is_main_process():
                save_on_master({
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'args': args
                }, checkpoint_path)

    # 完成训练
    print_rank_0("\n" + "="*60)
    print_rank_0("训练完成！")
    print_rank_0(f"最佳验证损失: {best_loss:.6f}")
    print_rank_0("="*60)

    # 关闭TensorBoard logger
    if is_main_process() and logger is not None:
        logger.close()

    # 清理分布式环境
    cleanup_distributed()


if __name__ == '__main__':
    main()