#!/usr/bin/env python3
"""
测试维度对齐修复方案
"""
import torch

print("=== 测试repeat vs 列表乘法 ===\n")

# 模拟数据
num_historical = 500
num_current = 300
historical_features = torch.randn(num_historical, 16).cuda()

print(f"num_historical: {num_historical}")
print(f"num_current: {num_current}")
print(f"historical_features.shape: {historical_features.shape}")

# 原始错误方式（使用列表乘法）
repeat_count = (num_current + num_historical - 1) // num_historical  # 1
print(f"repeat_count: {repeat_count}")

projected_features_list = [historical_features] * repeat_count
print(f"projected_features_list length: {len(projected_features_list)}")
print(f"projected_features_list[0].shape: {projected_features_list[0].shape}")

concat_wrong = torch.cat(projected_features_list, dim=0)[:num_current]
print(f"concat_wrong.shape: {concat_wrong.shape}")  # 期望 [600, 128]
print()

# 修复方式（使用repeat）
repeat_times = (num_current + num_historical - 1) // num_historical  # 1
print(f"repeat_times: {repeat_times}")

concat_correct = historical_features.repeat(repeat_times, 1)[:num_current]
print(f"concat_correct.shape: {concat_correct.shape}")  # 期望 [300, 128]
print()

print("\\n=== 验证 ===\n")
print(f"错误方式结果: {concat_wrong.shape} (期望: torch.Size([300, 128])")
print(f"修复方式结果: {concat_correct.shape} (期望: torch.Size([300, 128]))")

if concat_correct.shape == torch.Size([num_current, 128]):
    print("✅ 修复方案正确！")
else:
    print("❌ 修复方案错误！")
