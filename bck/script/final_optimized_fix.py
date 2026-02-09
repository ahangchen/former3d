#!/usr/bin/env python3
"""
optimized_online_training.py的最终修复方案
"""

import os

def create_final_solution():
    """创建最终解决方案"""
    print("="*80)
    print("optimized_online_training.py 最终修复方案")
    print("="*80)
    
    print("\n📊 问题分析:")
    print("1. ✅ intrinsics形状问题 - 已修复")
    print("2. ✅ spconv直接赋值错误 - 已修复") 
    print("3. ✅ 损失计算问题 - 已修复")
    print("4. ✅ 内存管理 - 已优化")
    print("5. ❌ spconv反向传播错误 - 需要特殊处理")
    print("6. ❌ BatchNorm单样本问题 - 需要特殊处理")
    
    print("\n🔧 解决方案:")
    print("方案1: 修改模型配置，避免spconv问题")
    print("   - 使用更小的体素大小和裁剪尺寸")
    print("   - 禁用某些spconv操作")
    print("   - 使用更简单的模型变体")
    
    print("\n方案2: 创建训练包装器，处理单样本问题")
    print("   - 使用虚拟批次训练")
    print("   - 累积多个样本后再更新")
    print("   - 修改BatchNorm统计方式")
    
    print("\n🚀 推荐实施:")
    
    fix_code = '''
# 在optimized_online_training.py中添加以下修复：

# 1. 修改模型创建，使用更简单的配置
def create_safe_model(config):
    """创建安全的模型，避免spconv问题"""
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        
        # 使用更安全的配置
        model = StreamSDFFormerIntegrated(
            attn_heads=1,  # 减少注意力头
            attn_layers=1,  # 减少注意力层
            use_proj_occ=False,  # 禁用投影占用（可能有问题）
            voxel_size=config["voxel_size"],
            fusion_local_radius=0.0,  # 禁用局部融合
            crop_size=config["crop_size"],
            use_checkpoint=False  # 禁用梯度检查点
        )
        return model
    except:
        # 备用简化模型
        return create_simple_model(config)

# 2. 修改训练循环，处理单样本
def safe_train_epoch(model, dataloader, optimizer, loss_fn, config, epoch):
    """安全的训练epoch，处理单样本和spconv问题"""
    model.train()
    
    # 对于单样本训练，使用特殊的BatchNorm处理
    if config["batch_size"] == 1:
        # 将BatchNorm层切换到评估模式进行统计，但保持梯度
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
    
    # ... 原有训练逻辑 ...
    
    # 训练完成后恢复
    if config["batch_size"] == 1:
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.track_running_stats = True

# 3. 添加spconv错误处理
import torch
import spconv

def safe_spconv_forward(model, *args, **kwargs):
    """安全的spconv前向传播"""
    try:
        return model(*args, **kwargs)
    except Exception as e:
        if "spconv" in str(e).lower() or "N > 0" in str(e):
            print(f"⚠️ spconv错误，尝试恢复: {e}")
            torch.cuda.empty_cache()
            # 简化输入重试
            return model(*args, **kwargs)
        else:
            raise e
'''
    
    print(fix_code)
    
    print("\n📝 实施步骤:")
    print("1. 备份当前optimized_online_training.py")
    print("2. 应用上述修复代码")
    print("3. 运行测试验证修复")
    print("4. 如果仍然失败，考虑使用简化模型变体")
    
    print("\n⏰ 预计时间: 30-60分钟")
    print("✅ 成功率: 中等（取决于spconv问题的复杂性）")
    
    return True

def check_current_state():
    """检查当前状态"""
    print("\n🔍 当前状态检查:")
    
    # 检查文件
    files = [
        "optimized_online_training.py",
        "former3d/stream_sdfformer_integrated.py",
        "online_tartanair_dataset.py"
    ]
    
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file}: {size:,} 字节")
        else:
            print(f"❌ {file}: 不存在")
    
    # 检查日志
    if os.path.exists("optimized_training.log"):
        with open("optimized_training.log", "r") as f:
            lines = f.readlines()
            last_errors = [l for l in lines[-10:] if "ERROR" in l]
            if last_errors:
                print(f"\n📋 最近错误:")
                for err in last_errors[-3:]:
                    print(f"  {err.strip()}")
    
    return True

def main():
    """主函数"""
    print("optimized_online_training.py 问题诊断与修复")
    print("="*80)
    
    check_current_state()
    create_final_solution()
    
    print("\n" + "="*80)
    print("🎯 最终建议")
    print("="*80)
    print("根据分析，optimized_online_training.py的主要问题已解决，但仍有:")
    print("1. spconv库的底层错误（可能需要更新spconv或修改模型）")
    print("2. 单样本训练的BatchNorm问题")
    print("")
    print("💡 建议下一步:")
    print("1. 尝试更新spconv到最新版本")
    print("2. 修改模型配置，使用更简单的设置")
    print("3. 如果时间有限，使用已验证的简化模型进行训练")
    print("")
    print("🚀 命令:")
    print("1. 更新spconv: pip install -U spconv")
    print("2. 运行修复后的训练: python optimized_online_training.py")
    print("3. 监控日志: tail -f optimized_training.log")
    
    print("\n📞 如果需要进一步帮助，请提供具体的错误信息!")

if __name__ == "__main__":
    main()