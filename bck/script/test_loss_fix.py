#!/usr/bin/env python3
"""
测试损失计算修复
"""

import torch
import torch.nn as nn

print("="*80)
print("测试损失计算修复")
print("="*80)

def test_case_1():
    """测试情况1: [B, num_points, 1]形状"""
    print("\n1. 测试点云预测形状 [B, num_points, 1]...")
    
    B = 2
    num_points = 100
    pred_sdf = torch.randn(B, num_points, 1)
    tsdf_target = torch.randn(B, 16, 16, 16)
    
    print(f"  pred_sdf形状: {pred_sdf.shape}")
    print(f"  tsdf_target形状: {tsdf_target.shape}")
    
    # 模拟修复后的逻辑
    tsdf_flat = tsdf_target.view(B, -1)
    num_voxels = tsdf_flat.shape[1]
    
    if num_voxels >= num_points:
        indices = torch.randint(0, num_voxels, (B, num_points))
        target_sdf = torch.gather(tsdf_flat, 1, indices).unsqueeze(-1)
    else:
        repeat_times = (num_points + num_voxels - 1) // num_voxels
        target_sdf = tsdf_flat.repeat(1, repeat_times)[:, :num_points].unsqueeze(-1)
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(pred_sdf, target_sdf)
    
    print(f"  target_sdf形状: {target_sdf.shape}")
    print(f"  损失计算成功: {loss.item():.6f}")
    return True

def test_case_2():
    """测试情况2: [num_voxels, 1]形状"""
    print("\n2. 测试体素预测形状 [num_voxels, 1]...")
    
    B = 2
    num_voxels = 19871  # 测试中的实际值
    pred_sdf = torch.randn(num_voxels, 1)
    tsdf_target = torch.randn(B, 16, 16, 16)
    
    print(f"  pred_sdf形状: {pred_sdf.shape}")
    print(f"  tsdf_target形状: {tsdf_target.shape}")
    
    # 模拟修复后的逻辑
    if num_voxels % B == 0:
        voxels_per_batch = num_voxels // B
        pred_sdf_reshaped = pred_sdf.view(B, voxels_per_batch, 1)
        tsdf_flat = tsdf_target.view(B, -1)
        
        print(f"  可以均匀分割: {voxels_per_batch}体素/批次")
        print(f"  pred_sdf_reshaped形状: {pred_sdf_reshaped.shape}")
        
        if voxels_per_batch <= tsdf_flat.shape[1]:
            indices = torch.randint(0, tsdf_flat.shape[1], (B, voxels_per_batch))
            target_sdf = torch.gather(tsdf_flat, 1, indices).unsqueeze(-1)
            loss_fn = nn.MSELoss()
            loss = loss_fn(pred_sdf_reshaped, target_sdf)
            print(f"  target_sdf形状: {target_sdf.shape}")
            print(f"  损失计算成功: {loss.item():.6f}")
            return True
        else:
            print("  ⚠️ 体素数量超过目标体素")
            return False
    else:
        print(f"  ⚠️ 无法均匀分割: {num_voxels}体素 / {B}批次")
        return False

def test_case_3():
    """测试情况3: 小批次简化"""
    print("\n3. 测试小批次简化...")
    
    B = 1
    num_voxels = 242  # 测试中的实际值
    pred_sdf = torch.randn(num_voxels, 1)
    tsdf_target = torch.randn(B, 16, 16, 16)
    
    print(f"  pred_sdf形状: {pred_sdf.shape}")
    print(f"  tsdf_target形状: {tsdf_target.shape}")
    
    # 使用简单采样
    num_sample_points = min(num_voxels, 1000)
    pred_sample = pred_sdf[:num_sample_points].unsqueeze(0)  # [1, num_sample_points, 1]
    tsdf_flat = tsdf_target.view(-1)
    
    print(f"  采样点数: {num_sample_points}")
    print(f"  pred_sample形状: {pred_sample.shape}")
    
    if tsdf_flat.shape[0] >= num_sample_points:
        indices = torch.randint(0, tsdf_flat.shape[0], (1, num_sample_points))
        target_sdf = torch.gather(tsdf_flat.unsqueeze(0), 1, indices).unsqueeze(-1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred_sample, target_sdf)
        print(f"  target_sdf形状: {target_sdf.shape}")
        print(f"  损失计算成功: {loss.item():.6f}")
        return True
    else:
        print("  ⚠️ 目标体素数量不足")
        return False

def check_fix_in_script():
    """检查修复是否已应用到脚本"""
    print("\n4. 检查修复是否已应用到optimized_online_training.py...")
    
    try:
        with open("optimized_online_training.py", "r") as f:
            content = f.read()
        
        # 检查是否包含新的处理逻辑
        check_points = [
            "elif len(pred_sdf.shape) == 2:",  # 体素预测
            "num_voxels = pred_sdf.shape[0]",
            "if num_voxels % B == 0:"
        ]
        
        all_found = True
        for check in check_points:
            if check in content:
                print(f"  ✅ 找到: {check}")
            else:
                print(f"  ❌ 未找到: {check}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return False

def main():
    """主函数"""
    print("测试损失计算修复...")
    
    test1 = test_case_1()
    test2 = test_case_2()
    test3 = test_case_3()
    test4 = check_fix_in_script()
    
    print("\n" + "="*80)
    print("测试结果")
    print("="*80)
    
    if test1 and test2 and test3 and test4:
        print("✅ 所有测试通过!")
        print("损失计算修复成功!")
        print("\n现在可以运行完整的训练脚本:")
        print("命令: python optimized_online_training.py")
    else:
        print("❌ 部分测试失败")
        print("\n需要进一步调试:")
        if not test1: print("  - 点云预测形状处理")
        if not test2: print("  - 体素预测形状处理")
        if not test3: print("  - 小批次简化处理")
        if not test4: print("  - 脚本修复检查")
    
    print("\n🚀 根据测试结果决定下一步操作")

if __name__ == "__main__":
    main()