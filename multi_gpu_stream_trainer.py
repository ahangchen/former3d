"""
多GPU训练辅助函数
用于手动实现batch分发和结果合并，支持流式训练
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional


class MultiGPUStreamTrainer:
    """
    多GPU流式训练器
    手动分发batch到多个GPU，支持流式训练的序列依赖性
    """

    def __init__(self, model: nn.Module, gpu_ids: List[int]):
        """
        初始化多GPU训练器

        Args:
            model: 模型（原始模型，未包装DataParallel）
            gpu_ids: GPU ID列表
        """
        self.gpu_ids = gpu_ids
        self.num_gpus = len(gpu_ids)

        # 将模型复制到每个GPU（使用deepcopy确保完全独立）
        import copy
        self.models = nn.ModuleList()
        for gpu_id in gpu_ids:
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.to(f'cuda:{gpu_id}')
            self.models.append(model_copy)

        # 同步模型参数
        self._sync_parameters()

    def _sync_parameters(self):
        """同步所有GPU的模型参数"""
        if self.num_gpus <= 1:
            return

        # 以第一个GPU的模型为主，同步其他GPU
        src_model = self.models[0]
        for dst_model in self.models[1:]:
            for src_param, dst_param in zip(src_model.parameters(), dst_model.parameters()):
                dst_param.data.copy_(src_param.data)

    def parameters(self):
        """
        返回主GPU模型的参数迭代器
        用于创建优化器

        Returns:
            参数迭代器
        """
        return self.models[0].parameters()

    def forward_sequence(self,
                        images: torch.Tensor,
                        poses: torch.Tensor,
                        intrinsics: torch.Tensor,
                        reset_state: bool = True) -> Tuple[torch.Tensor, List[Dict]]:
        """
        多GPU前向传播（流式序列）

        Args:
            images: 图像序列 (batch, n_view, 3, H, W)
            poses: 位姿序列 (batch, n_view, 4, 4)
            intrinsics: 内参序列 (batch, n_view, 3, 3)
            reset_state: 是否在序列开始时重置状态

        Returns:
            Tuple[torch.Tensor, List[Dict]]: (输出序列 (batch, n_view, ...), 状态列表)
        """
        batch_size = images.shape[0]
        n_view = images.shape[1]

        # 计算每个GPU处理的batch大小
        batch_per_gpu = batch_size // self.num_gpus
        remainder = batch_size % self.num_gpus

        # 分发数据到各GPU
        batch_splits = []
        start_idx = 0
        for i in range(self.num_gpus):
            # 处理余数
            end_idx = start_idx + batch_per_gpu + (1 if i < remainder else 0)

            batch_split = {
                'images': images[start_idx:end_idx].to(f'cuda:{self.gpu_ids[i]}'),
                'poses': poses[start_idx:end_idx].to(f'cuda:{self.gpu_ids[i]}'),
                'intrinsics': intrinsics[start_idx:end_idx].to(f'cuda:{self.gpu_ids[i]}'),
            }
            batch_splits.append(batch_split)

            start_idx = end_idx

        # 在各GPU上执行前向传播
        gpu_outputs = []
        gpu_states = []

        for i, (model_gpu, batch_split) in enumerate(zip(self.models, batch_splits)):
            # 重置状态（只重置第一个GPU的状态，用于同步）
            if i == 0 and reset_state:
                model_gpu.clear_history()

            # 执行前向传播
            output_gpu, state_gpu = model_gpu.forward_sequence(
                batch_split['images'],
                batch_split['poses'],
                batch_split['intrinsics'],
                reset_state=(i == 0 and reset_state)  # 只在第一个GPU上重置状态
            )

            # 将输出移到主GPU（cuda:0）以便合并
            if isinstance(output_gpu, dict):
                # 字典类型，将每个张量移到主GPU
                output_main = {k: v.to(f'cuda:{self.gpu_ids[0]}') if isinstance(v, torch.Tensor) else v for k, v in output_gpu.items()}
            else:
                # 张量类型，直接移到主GPU
                output_main = output_gpu.to(f'cuda:{self.gpu_ids[0]}')
            gpu_outputs.append(output_main)
            gpu_states.append(state_gpu)

        # 合并结果
        if isinstance(gpu_outputs[0], dict):
            # 字典输出，合并每个key
            merged_output = {}
            for key in gpu_outputs[0].keys():
                if gpu_outputs[0][key] is None:
                    merged_output[key] = None
                elif isinstance(gpu_outputs[0][key], torch.Tensor):
                    # 沿batch维度拼接
                    merged_output[key] = torch.cat([out[key] for out in gpu_outputs], dim=0)
                else:
                    merged_output[key] = gpu_outputs[0][key]
        else:
            # 张量输出，直接拼接
            merged_output = torch.cat(gpu_outputs, dim=0)

        # 合并状态（只保留第一个GPU的状态）
        merged_states = gpu_states[0]

        return merged_output, merged_states

    def backward(self, loss: torch.Tensor):
        """
        多GPU反向传播

        Args:
            loss: 损失张量（必须在主GPU上）
        """
        # 在主GPU上执行反向传播
        loss.backward()

        # 同步梯度到所有GPU（如果需要）
        if self.num_gpus > 1:
            self._sync_gradients()

    def _sync_gradients(self):
        """同步所有GPU的梯度"""
        if self.num_gpus <= 1:
            return

        # 将第一个GPU的梯度广播到其他GPU
        src_model = self.models[0]
        for dst_model in self.models[1:]:
            for src_param, dst_param in zip(src_model.parameters(), dst_model.parameters()):
                if dst_param.grad is not None:
                    dst_param.grad.data.copy_(src_param.grad.data)

    def step(self, optimizer: torch.optim.Optimizer):
        """
        更新所有GPU的模型参数

        Args:
            optimizer: 优化器（只对主GPU的模型有效）
        """
        # 只更新主GPU的模型
        optimizer.step()

        # 同步参数到其他GPU
        self._sync_parameters()

    def zero_grad(self):
        """清空所有GPU的梯度"""
        for model in self.models:
            model.zero_grad()

    def train(self):
        """设置为训练模式"""
        for model in self.models:
            model.train()

    def eval(self):
        """设置为评估模式"""
        for model in self.models:
            model.eval()

    def to(self, device):
        """移动模型到指定设备"""
        for model in self.models:
            model.to(device)
        return self


class MultiGPULossAggregator:
    """
    多GPU损失聚合器
    用于计算和合并各GPU的损失
    """

    def __init__(self, compute_loss_fn):
        """
        初始化损失聚合器

        Args:
            compute_loss_fn: 损失计算函数，签名为: compute_loss(outputs, targets, ...) -> loss
        """
        self.compute_loss_fn = compute_loss_fn

    def __call__(self,
                outputs: torch.Tensor,
                targets: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """
        计算总损失

        Args:
            outputs: 模型输出
            targets: 目标值
            **kwargs: 其他参数

        Returns:
            总损失
        """
        return self.compute_loss_fn(outputs, targets, **kwargs)


def create_multi_gpu_trainer(model: nn.Module,
                           gpu_ids: List[int]) -> MultiGPUStreamTrainer:
    """
    创建多GPU训练器

    Args:
        model: 模型
        gpu_ids: GPU ID列表

    Returns:
        MultiGPUStreamTrainer实例
    """
    return MultiGPUStreamTrainer(model, gpu_ids)
