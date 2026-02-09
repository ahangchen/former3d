#!/usr/bin/env python3
"""
验证数据目录结构
"""

import os
import sys

def verify_tartanair_directory(data_root):
    """验证TartanAir数据目录结构"""
    print(f"验证数据目录: {data_root}")
    
    if not os.path.exists(data_root):
        print(f"❌ 目录不存在: {data_root}")
        return False
    
    print(f"✅ 目录存在")
    
    # 列出所有子目录
    items = os.listdir(data_root)
    print(f"\n目录内容 ({len(items)} 项):")
    
    sequences = []
    for item in items[:20]:  # 只显示前20个
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):
            sequences.append(item)
            print(f"  📁 {item}")
        else:
            print(f"  📄 {item}")
    
    if len(items) > 20:
        print(f"  ... 还有 {len(items)-20} 项")
    
    # 检查序列结构
    if sequences:
        print(f"\n检查序列结构...")
        sample_seq = sequences[0]
        seq_path = os.path.join(data_root, sample_seq)
        
        print(f"示例序列: {sample_seq}")
        
        # 检查P001子目录
        p001_path = os.path.join(seq_path, "P001")
        if os.path.exists(p001_path):
            print(f"✅ P001目录存在")
            
            # 检查必要的子目录
            subdirs = ["image_left", "depth_left"]
            files = ["pose_left.txt"]
            
            for subdir in subdirs:
                subdir_path = os.path.join(p001_path, subdir)
                if os.path.exists(subdir_path):
                    print(f"✅ {subdir} 目录存在")
                    # 统计文件数量
                    import glob
                    files_count = len(glob.glob(os.path.join(subdir_path, "*")))
                    print(f"   文件数量: {files_count}")
                else:
                    print(f"❌ {subdir} 目录不存在")
            
            for file in files:
                file_path = os.path.join(p001_path, file)
                if os.path.exists(file_path):
                    print(f"✅ {file} 文件存在")
                else:
                    print(f"❌ {file} 文件不存在")
        else:
            print(f"❌ P001目录不存在")
            
            # 检查是否有其他可能的子目录
            sub_items = os.listdir(seq_path)
            print(f"序列子目录: {sub_items}")
    
    return True

if __name__ == "__main__":
    data_root = "/home/cwh/Study/dataset/tartanair"
    
    print("=" * 60)
    print("TartanAir数据目录验证")
    print("=" * 60)
    
    success = verify_tartanair_directory(data_root)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 验证完成")
    else:
        print("❌ 验证失败")
    print("=" * 60)