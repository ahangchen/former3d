#!/usr/bin/env python3
"""
测试梯度累积逻辑的正确性
使用简单的线性模型验证梯度累积
"""

import torch
import torch.nn as nn
import torch.optim as optim

def test_gradient_accumulation_logic():
    """测试梯度累积逻辑的正确性"""

    print("="*70)
    print("测试梯度累积逻辑正确性")
    print("="*70)

    # 设置随机种子
    torch.manual_seed(42)

    # 配置
    batch_size_large = 4
    batch_size_small = 2
    accumulation_steps = batch_size_large // batch_size_small
    input_dim = 10
    output_dim = 1

    print(f"\n配置：")
    print(f"  - 大批次大小: {batch_size_large}")
    print(f"  - 小批次大小: {batch_size_small}")
    print(f"  - 累积步数: {accumulation_steps}")
    print(f"  - 输入维度: {input_dim}")
    print(f"  - 输出维度: {output_dim}")

    # 创建两个相同的线性模型
    model_large = nn.Linear(input_dim, output_dim)
    model_small = nn.Linear(input_dim, output_dim)

    # 复制权重
    model_small.load_state_dict(model_large.state_dict())

    # 创建优化器
    optimizer_large = optim.SGD(model_large.parameters(), lr=0.01)
    optimizer_small = optim.SGD(model_small.parameters(), lr=0.01)

    # 生成模拟数据（大批次）
    x_large = torch.randn(batch_size_large, input_dim)
    y_large = torch.randn(batch_size_large, output_dim)

    # 生成模拟数据（小批次）
    small_batches = []
    for i in range(accumulation_steps):
        x_small = x_large[i*batch_size_small:(i+1)*batch_size_small]
        y_small = y_large[i*batch_size_small:(i+1)*batch_size_small]
        small_batches.append((x_small, y_small))

    # 训练大批次
    print(f"\n训练大批次模型...")
    optimizer_large.zero_grad()

    y_pred_large = model_large(x_large)
    loss_large = nn.functional.mse_loss(y_pred_large, y_large)
    # 注意：大批次训练不需要除以accumulation_steps
    loss_large.backward()

    optimizer_large.step()

    print(f"  - 损失: {loss_large.item():.6f}")

    # 训练小批次（使用梯度累积）
    print(f"训练小批次模型（使用梯度累积）...")
    optimizer_small.zero_grad()
    accumulation_counter = 0
    accumulated_loss = 0.0

    for x_small, y_small in small_batches:
        y_pred_small = model_small(x_small)
        loss_small = nn.functional.mse_loss(y_pred_small, y_small)
        loss_small = loss_small / accumulation_steps  # 除以累积步数

        # 反向传播（累积梯度）
        loss_small.backward()

        accumulation_counter += 1
        accumulated_loss += loss_small.item()

        # 达到累积步数时更新参数
        if accumulation_counter % accumulation_steps == 0:
            optimizer_small.step()
            optimizer_small.zero_grad()

    print(f"  - 累积损失: {accumulated_loss * accumulation_steps:.6f}")

    # 比较参数
    print(f"\n比较模型参数...")
    params_large = list(model_large.parameters())
    params_small = list(model_small.parameters())

    max_diff = 0.0
    total_diff = 0.0
    param_count = 0

    for p_large, p_small in zip(params_large, params_small):
        diff = torch.abs(p_large - p_small).max().item()
        max_diff = max(max_diff, diff)
        total_diff += diff
        param_count += 1

        print(f"  参数形状: {p_large.shape}, 最大差异: {diff:.6e}")

    avg_diff = total_diff / param_count

    print(f"\n  - 最大差异: {max_diff:.6e}")
    print(f"  - 平均差异: {avg_diff:.6e}")
    print(f"  - 参数数量: {param_count}")

    # 比较梯度（训练前）
    print(f"\n验证梯度（大批次）：")
    for name, param in model_large.named_parameters():
        print(f"  {name}: {param.grad.norm().item():.6e}")

    # 验证结果
    if max_diff < 1e-6:  # 线性模型应该完全一致
        print(f"\n✅ 测试通过：梯度累积与大批次训练结果一致")
        print(f"   最大差异: {max_diff:.6e} < 1e-6")
    else:
        print(f"\n❌ 测试失败：梯度累积与大批次训练结果不一致")
        print(f"   最大差异: {max_diff:.6e} > 1e-6")
        raise AssertionError("梯度累积不正确")

    print(f"\n{'='*70}")
    print("梯度累积逻辑测试完成！")
    print(f"{'='*70}")

if __name__ == "__main__":
    test_gradient_accumulation_logic()
