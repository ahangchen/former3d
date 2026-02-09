#!/usr/bin/env python3
"""
测试TSDF生成脚本
"""

import os
import sys
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_pose_loading():
    """测试位姿加载"""
    print("测试位姿加载...")
    
    # 测试数据路径
    data_root = "/home/cwh/Study/dataset/tartanair"
    sequence_name = "abandonedfactory_sample_P001"
    sequence_path = Path(data_root) / sequence_name / "P001"
    
    pose_file = sequence_path / "pose_left.txt"
    
    if not pose_file.exists():
        print(f"❌ 位姿文件不存在: {pose_file}")
        return False
    
    poses = np.loadtxt(pose_file)
    print(f"✅ 加载位姿: {len(poses)} 帧")
    print(f"   位姿形状: {poses.shape}")
    print(f"   第一帧位姿: {poses[0]}")
    
    return True

def test_depth_loading():
    """测试深度图加载"""
    print("\n测试深度图加载...")
    
    data_root = "/home/cwh/Study/dataset/tartanair"
    sequence_name = "abandonedfactory_sample_P001"
    sequence_path = Path(data_root) / sequence_name / "P001"
    
    depth_dir = sequence_path / "depth_left"
    
    if not depth_dir.exists():
        print(f"❌ 深度图目录不存在: {depth_dir}")
        return False
    
    depth_files = sorted([f for f in depth_dir.glob("*.npy")])
    print(f"✅ 找到深度图: {len(depth_files)} 个")
    
    if depth_files:
        depth_path = depth_files[0]
        depth = np.load(depth_path)
        print(f"✅ 加载深度图: {depth_path.name}")
        print(f"   深度图形状: {depth.shape}")
        print(f"   深度图类型: {depth.dtype}")
        print(f"   深度范围: [{depth.min():.3f}, {depth.max():.3f}] 毫米")
        
        # 转换为米
        depth_meters = depth.astype(np.float32) / 1000.0
        print(f"   深度范围(米): [{depth_meters.min():.3f}, {depth_meters.max():.3f}] 米")
        
        return True
    else:
        print("❌ 没有找到深度图文件")
        return False

def test_tsdf_module():
    """测试TSDF模块"""
    print("\n测试TSDF模块...")
    
    try:
        from former3d.tsdf_fusion import TSDFVolume
        print("✅ TSDF模块导入成功")
        
        # 测试创建TSDF体积
        bounds = np.array([
            [-5.0, 5.0],  # X范围
            [-5.0, 5.0],  # Y范围
            [-5.0, 5.0]   # Z范围
        ])
        
        tsdf = TSDFVolume(
            vol_bnds=bounds,
            voxel_size=0.08,
            margin=5
        )
        
        print("✅ TSDF体积创建成功")
        return True
        
    except Exception as e:
        print(f"❌ TSDF模块测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sdf_generator():
    """测试SDF生成器"""
    print("\n测试SDF生成器...")
    
    try:
        from generate_tartanair_sdf import TartanAirSDFGenerator
        
        # 创建生成器（不实际运行）
        generator = TartanAirSDFGenerator(
            data_root="/home/cwh/Study/dataset/tartanair",
            sequence_name="abandonedfactory_sample_P001",
            voxel_size=0.08
        )
        
        print("✅ SDF生成器创建成功")
        print(f"   序列路径: {generator.sequence_path}")
        print(f"   内参矩阵:\n{generator.intrinsics}")
        
        return True
        
    except Exception as e:
        print(f"❌ SDF生成器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("=" * 60)
    print("TartanAir TSDF生成测试")
    print("=" * 60)
    
    tests = [
        ("位姿加载", test_pose_loading),
        ("深度图加载", test_depth_loading),
        ("TSDF模块", test_tsdf_module),
        ("SDF生成器", test_sdf_generator)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 测试: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append((test_name, False))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    print(f"\n📊 结果: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过! 可以运行完整的SDF生成脚本。")
        print("\n运行完整生成:")
        print("  python generate_tartanair_sdf.py --max_frames 10")
    else:
        print("\n⚠️  部分测试失败，请检查问题。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)