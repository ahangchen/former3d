"""
分布式训练工具函数
用于PyTorch DistributedDataParallel (DDP)训练
"""

import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader


def setup_distributed():
    """
    初始化分布式环境

    Returns:
        int: 本地进程rank (用于选择GPU)
    """
    # 初始化进程组
    dist.init_process_group(backend='nccl')

    # 获取本地rank并设置设备
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # 打印初始化信息（仅在rank 0）
    if dist.get_rank() == 0:
        world_size = dist.get_world_size()
        print(f"✅ 分布式环境初始化成功")
        print(f"   - 世界大小 (world_size): {world_size}")
        print(f"   - 本地进程数量 (nproc_per_node): {world_size}")
        print(f"   - 当前进程 rank: {dist.get_rank()}")
        print(f"   - 后端: nccl")

    return local_rank


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()
        if dist.get_rank() == 0:
            print("✅ 分布式环境已清理")


def create_distributed_dataloader(dataset, batch_size, num_workers=4, shuffle=True):
    """
    创建分布式数据加载器

    Args:
        dataset: 数据集对象
        batch_size: 总batch size（会被均匀分配到各个GPU）
        num_workers: 每个GPU的worker数量
        shuffle: 是否打乱数据

    Returns:
        DataLoader, DistributedSampler: 数据加载器和采样器
    """
    # 创建分布式采样器
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=shuffle
    )

    # 计算每个GPU的batch size
    world_size = dist.get_world_size()
    per_gpu_batch_size = batch_size // world_size

    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # 确保所有GPU的batch size相同
    )

    if dist.get_rank() == 0:
        print(f"✅ 分布式数据加载器创建成功")
        print(f"   - 总batch size: {batch_size}")
        print(f"   - 每GPU batch size: {per_gpu_batch_size}")
        print(f"   - GPU数量: {world_size}")
        print(f"   - 每GPU workers: {num_workers}")

    return dataloader, sampler


def reduce_dict(input_dict, average=True):
    """
    在所有进程间减少字典值

    Args:
        input_dict: 包含张量的字典
        average: 是否对值取平均

    Returns:
        dict: 减少后的字典
    """
    world_size = dist.get_world_size()

    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []

        for k, v in input_dict.items():
            names.append(k)
            values.append(v)

        # 减少所有张量
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size

        # 重构字典
        reduced_dict = {k: v for k, v in zip(names, values)}

        return reduced_dict


def all_gather_tensor(tensor):
    """
    在所有进程间收集张量

    Args:
        tensor: 要收集的张量

    Returns:
        torch.Tensor: 收集后的张量 [world_size, ...]
    """
    world_size = dist.get_world_size()

    if world_size < 2:
        return tensor.unsqueeze(0)

    # 获取张量大小
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    # 堆叠成单个张量
    gathered = torch.stack(tensor_list, dim=0)

    return gathered


def is_main_process():
    """检查当前进程是否为主进程 (rank 0)"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size():
    """获取世界大小（总进程数）"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """获取当前进程rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def synchronize():
    """
    同步所有进程
    用于确保所有进程在同一位置
    """
    if dist.is_initialized():
        dist.barrier()


def save_on_master(state, save_path):
    """只在主进程上保存状态"""
    if is_main_process():
        torch.save(state, save_path)
        print(f"✅ 检查点已保存: {save_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, lr_scheduler=None):
    """
    加载检查点

    Args:
        checkpoint_path: 检查点路径
        model: 模型
        optimizer: 优化器（可选）
        lr_scheduler: 学习率调度器（可选）

    Returns:
        dict: 加载的状态
    """
    if not os.path.exists(checkpoint_path):
        if is_main_process():
            print(f"⚠️ 检查点不存在: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 加载模型状态（处理DDP的module包装）
    model_state_dict = checkpoint.get('model', checkpoint)
    if 'module' in model_state_dict:
        model_state_dict = model_state_dict['module']
    model.load_state_dict(model_state_dict, strict=False)

    # 加载优化器状态
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # 加载学习率调度器状态
    if lr_scheduler is not None and 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if is_main_process():
        print(f"✅ 检查点已加载: {checkpoint_path}")
        if 'epoch' in checkpoint:
            print(f"   - Epoch: {checkpoint['epoch']}")
        if 'best_loss' in checkpoint:
            print(f"   - Best Loss: {checkpoint['best_loss']:.4f}")

    return checkpoint


class AverageMeter:
    """分布式平均计算器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n

    def all_reduce(self):
        """在所有进程间减少值"""
        if dist.is_initialized() and dist.get_world_size() > 1:
            tensor = torch.tensor([self.sum, self.count], dtype=torch.float32)
            dist.all_reduce(tensor)

            # 对值求和，对计数也求和（因为每个进程统计的是自己的部分）
            self.sum = tensor[0].item()
            self.count = tensor[1].item()

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0


def adjust_learning_rate(optimizer, epoch, base_lr, total_epochs, warmup_epochs=5):
    """
    调整学习率（带warmup的余弦退火）

    Args:
        optimizer: 优化器
        epoch: 当前epoch
        base_lr: 基础学习率
        total_epochs: 总epoch数
        warmup_epochs: warmup epoch数
    """
    world_size = get_world_size()

    # 线性缩放学习率
    lr = base_lr * world_size

    if epoch < warmup_epochs:
        # Warmup阶段：线性增加
        lr = base_lr * world_size * (epoch + 1) / warmup_epochs
    else:
        # 余弦退火
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793))) * base_lr * world_size

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def get_gpu_memory_usage():
    """获取当前GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        return allocated, reserved
    return 0.0, 0.0


def print_rank_0(message):
    """只在rank 0打印消息"""
    if is_main_process():
        print(message)