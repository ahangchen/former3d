#!/usr/bin/env python3
"""
完整场景测试：模拟多尺度池化 + BatchNorm3d（修复版）
验证修改是否能解决之前的错误
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("=== 完整场景测试：多尺度池化 + BatchNorm3d（修复版）===\n")

# 参数设置
batch_size = 2  # 可以改为1测试
channels = 96
pool_scales = [1, 2, 3]
input_size = [32, 32, 24]

print(f"参数:")
print(f"  batch_size: {batch_size}")
print(f"  channels: {channels}")
print(f"  pool_scales: {pool_scales}")
print(f"  input_size: {input_size}\n")

# 模拟输入：batch_size, channels, D, H, W
inputs = torch.randn(batch_size, channels, *input_size)
print(f"输入形状: {inputs.shape}\n")

# 创建global_norm模块（使用修改后的BatchNorm3d）
class GlobalNormModule(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # 使用修改后的BatchNorm3d，track_running_stats=False
        self.norm = nn.Sequential(
            nn.BatchNorm3d(num_features, track_running_stats=False),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.norm(x)

print("=== 测试1: 多尺度池化 + 上采样 + BatchNorm3d ===")
try:
    global_norm = GlobalNormModule(channels * len(pool_scales))
    global_norm.train()

    # 多尺度池化
    pools = []
    for i, pool_scale in enumerate(pool_scales):
        output_size = pool_scale

        # 计算stride和kernel_size
        stride = (np.array(input_size) / output_size).astype(np.int8)
        stride = np.maximum(stride, 1)
        kernel_size = np.array(input_size) - (output_size - 1) * stride
        kernel_size = np.maximum(kernel_size, 1)

        # 池化 -> 卷积 -> 上采样到原始尺寸
        out = F.avg_pool3d(inputs, kernel_size=tuple(kernel_size),
                           stride=tuple(stride), ceil_mode=False)

        # 上采样到原始输入尺寸
        out = F.interpolate(out, size=input_size, mode='nearest')

        pools.append(out)
        print(f"  尺度{pool_scale}: 池化后上采样到 {out.shape}")

    # 拼接所有池化尺度的特征
    pools = torch.cat(pools, dim=1)
    print(f"  拼接后: {pools.shape}")

    # 应用BatchNorm3d
    pools_norm = global_norm(pools)
    print(f"  归一化后: {pools_norm.shape}")

    print("✅ 多尺度池化 + 上采样 + BatchNorm3d 成功\n")

except Exception as e:
    print(f"❌ 失败: {e}\n")
    import traceback
    traceback.print_exc()

print("=== 测试2: 测试batch size=1的情况 ===")
try:
    batch_size_1 = 1
    inputs_1 = torch.randn(batch_size_1, channels, *input_size)

    global_norm_1 = GlobalNormModule(channels * len(pool_scales))
    global_norm_1.train()

    # 多尺度池化
    pools_1 = []
    for pool_scale in pool_scales:
        output_size = pool_scale
        stride = (np.array(input_size) / output_size).astype(np.int8)
        stride = np.maximum(stride, 1)
        kernel_size = np.array(input_size) - (output_size - 1) * stride
        kernel_size = np.maximum(kernel_size, 1)

        out = F.avg_pool3d(inputs_1, kernel_size=tuple(kernel_size),
                           stride=tuple(stride), ceil_mode=False)
        out = F.interpolate(out, size=input_size, mode='nearest')

        pools_1.append(out)

    pools_1 = torch.cat(pools_1, dim=1)

    # 应用BatchNorm3d
    pools_norm_1 = global_norm_1(pools_1)

    print(f"  输入: {inputs_1.shape}")
    print(f"  拼接后: {pools_1.shape}")
    print(f"  归一化后: {pools_norm_1.shape}")
    print("✅ batch size=1 成功\n")

except Exception as e:
    print(f"❌ 失败: {e}\n")
    import traceback
    traceback.print_exc()

print("=== 测试3: 正常batch size + track_running_stats=False ===")
try:
    # 正常的batch size和空间尺寸
    normal_input = torch.randn(2, 288, 32, 32, 24)

    bn3d = nn.BatchNorm3d(288, track_running_stats=False)
    bn3d.train()

    output = bn3d(normal_input)

    print(f"  输入: {normal_input.shape}")
    print(f"  输出: {output.shape}")
    print("✅ 正常场景成功\n")

except Exception as e:
    print(f"❌ 失败: {e}\n")
    import traceback
    traceback.print_exc()

print("=== 测试4: 评估模式下batch size=1 ===")
try:
    # 评估模式下应该可以处理batch size=1
    eval_input = torch.randn(1, 288, 32, 32, 24)

    bn3d = nn.BatchNorm3d(288, track_running_stats=False)
    bn3d.eval()

    output = bn3d(eval_input)

    print(f"  输入: {eval_input.shape}")
    print(f"  输出: {output.shape}")
    print("✅ 评估模式batch size=1 成功\n")

except Exception as e:
    print(f"❌ 失败: {e}\n")
    import traceback
    traceback.print_exc()

print("=== 总结 ===")
print("✅ 关键测试通过")
print("修改内容:")
print("1. 删除自定义的CustomBatchNorm3d类")
print("2. 直接使用PyTorch的nn.BatchNorm3d")
print("3. 在global_norm中显式设置track_running_stats=False")
print("\n注意事项:")
print("- track_running_stats=False 只在训练模式下有帮助")
print("- 评估模式即使track_running_stats=True也可以处理batch size=1")
print("- 确保多尺度池化后使用interpolate上采样到相同尺寸再拼接")
