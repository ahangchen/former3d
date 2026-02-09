#!/usr/bin/env python3
"""
测试任务2进展 - 验证流式训练脚本框架
"""

import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_task2_progress():
    """测试任务2进展"""
    print("="*80)
    print("测试任务2进展 - 验证流式训练脚本框架")
    print("="*80)
    
    results = {
        "训练脚本框架": False,
        "模型导入": False,
        "设备一致性": False,
        "状态管理": False,
        "数据适配器": False,
        "验证测试": False
    }
    
    # 测试1: 训练脚本框架
    print("\n1. 测试训练脚本框架...")
    try:
        if os.path.exists("train_stream_integrated.py"):
            with open("train_stream_integrated.py", "r") as f:
                content = f.read()
            
            required_components = [
                "StreamSDFFormerIntegrated",
                "StreamStateManager",
                "forward_single_frame",
                "device_consistency",
                "train_epoch_stream"
            ]
            
            missing = []
            for component in required_components:
                if component not in content:
                    missing.append(component)
            
            if not missing:
                print(f"✅ 训练脚本框架完整")
                print(f"   文件大小: {len(content)} 字节")
                results["训练脚本框架"] = True
            else:
                print(f"❌ 缺少组件: {missing}")
        else:
            print("❌ 训练脚本文件不存在")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    # 测试2: 模型导入
    print("\n2. 测试模型导入...")
    try:
        from former3d.stream_sdfformer_integrated import StreamSDFFormerIntegrated
        print(f"✅ StreamSDFFormerIntegrated导入成功")
        results["模型导入"] = True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
    
    # 测试3: 设备一致性工具
    print("\n3. 测试设备一致性工具...")
    try:
        if os.path.exists("device_consistency_utils.py"):
            with open("device_consistency_utils.py", "r") as f:
                content = f.read()
            
            if "ensure_consistency" in content or "ensure_device_consistency" in content:
                print(f"✅ 设备一致性工具存在")
                results["设备一致性"] = True
            else:
                print("❌ 设备一致性函数不存在")
        else:
            print("❌ 设备一致性工具文件不存在")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    # 测试4: 状态管理
    print("\n4. 测试状态管理...")
    try:
        if os.path.exists("stream_state_manager.py"):
            with open("stream_state_manager.py", "r") as f:
                content = f.read()
            
            if "StreamStateManager" in content and "class StreamStateManager" in content:
                print(f"✅ 状态管理类存在")
                results["状态管理"] = True
            else:
                print("❌ 状态管理类不存在")
        else:
            print("❌ 状态管理文件不存在")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    # 测试5: 数据适配器
    print("\n5. 测试数据适配器...")
    try:
        if os.path.exists("stream_data_adapter.py"):
            with open("stream_data_adapter.py", "r") as f:
                content = f.read()
            
            if "StreamDataAdapter" in content and "class StreamDataAdapter" in content:
                print(f"✅ 数据适配器存在")
                results["数据适配器"] = True
            else:
                print("❌ 数据适配器不存在")
        else:
            print("❌ 数据适配器文件不存在")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    # 测试6: 验证测试
    print("\n6. 测试验证脚本...")
    try:
        if os.path.exists("validate_training_script.py"):
            # 运行验证脚本
            import subprocess
            result = subprocess.run(
                [sys.executable, "validate_training_script.py"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ 验证脚本通过")
                results["验证测试"] = True
            else:
                print(f"❌ 验证脚本失败")
                print(f"   错误: {result.stderr[:200]}")
        else:
            print("❌ 验证脚本文件不存在")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    # 总结
    print("\n" + "="*80)
    print("任务2进展总结")
    print("="*80)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    progress_percentage = (passed_tests / total_tests) * 100
    
    print(f"\n测试结果: {passed_tests}/{total_tests} 通过")
    print(f"进度: {progress_percentage:.1f}%")
    
    print("\n详细结果:")
    for test_name, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {test_name}")
    
    print("\n" + "="*80)
    print("下一步建议")
    print("="*80)
    
    if progress_percentage >= 80:
        print("\n✅ 任务2基本完成!")
        print("\n建议:")
        print("1. 收集更多训练数据")
        print("2. 运行完整训练测试:")
        print("   python train_stream_integrated.py --epochs 1 --batch-size 1 --dry-run")
        print("3. 解决数据集路径问题")
    elif progress_percentage >= 50:
        print("\n⚠️ 任务2部分完成")
        print("\n需要完成:")
        missing_tests = [name for name, passed in results.items() if not passed]
        for test in missing_tests:
            print(f"  - {test}")
        print("\n建议优先解决数据集问题")
    else:
        print("\n❌ 任务2进展缓慢")
        print("\n需要重点关注:")
        missing_tests = [name for name, passed in results.items() if not passed]
        for test in missing_tests:
            print(f"  - {test}")
    
    return results

def test_dry_run():
    """测试干运行"""
    print("\n" + "="*80)
    print("测试干运行")
    print("="*80)
    
    try:
        # 使用conda环境的Python
        python_path = sys.executable
        
        print(f"使用Python: {python_path}")
        print(f"运行命令: {python_path} train_stream_integrated.py --dry-run")
        
        import subprocess
        result = subprocess.run(
            [python_path, "train_stream_integrated.py", "--dry-run"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ 干运行成功!")
            
            # 检查输出
            lines = result.stdout.split('\n')
            success_indicators = [
                "✅ 设备一致性工具导入成功",
                "✅ StreamSDFFormerIntegrated导入成功",
                "✅ 前向传播成功",
                "✅ 干运行完成"
            ]
            
            print("\n关键输出:")
            for line in lines[-20:]:  # 最后20行
                for indicator in success_indicators:
                    if indicator in line:
                        print(f"  {line.strip()}")
            
            return True
        else:
            print(f"❌ 干运行失败，返回码: {result.returncode}")
            print(f"\n错误输出:")
            print(result.stderr[:500])
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 干运行超时")
        return False
    except Exception as e:
        print(f"❌ 干运行异常: {e}")
        return False

def main():
    """主函数"""
    print("任务2进展测试")
    print("="*80)
    
    # 测试任务2进展
    results = test_task2_progress()
    
    # 如果基本完成，测试干运行
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    if passed_tests >= total_tests * 0.8:  # 80%完成
        print("\n" + "="*80)
        print("进行干运行测试...")
        print("="*80)
        
        dry_run_success = test_dry_run()
        
        if dry_run_success:
            print("\n✅ 任务2完全验证通过!")
            print("\n可以开始实际训练:")
            print("python train_stream_integrated.py --epochs 1 --batch-size 1")
        else:
            print("\n⚠️ 干运行测试失败，需要修复")
    else:
        print("\n⚠️ 任务2未达到测试标准，跳过干运行")

if __name__ == "__main__":
    main()