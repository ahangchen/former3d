#!/usr/bin/env python3
"""
检查模块是否存在
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, '..'))

print("检查模块导入...")

modules_to_check = [
    'torch',
    'torchvision',
    'numpy',
    'numpy_indexed',
    'spconv',
    'numba'
]

for module_name in modules_to_check:
    try:
        __import__(module_name)
        print(f"✅ {module_name}")
    except ImportError as e:
        print(f"❌ {module_name}: {e}")

print("\n检查former3d模块...")
former3d_path = os.path.join(project_root, '..', 'former3d')
if os.path.exists(former3d_path):
    print(f"✅ former3d目录存在: {former3d_path}")
    
    # 检查关键文件
    key_files = [
        'stream_sdfformer_integrated.py',
        '__init__.py'
    ]
    
    for file in key_files:
        file_path = os.path.join(former3d_path, file)
        if os.path.exists(file_path):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} 不存在")
else:
    print(f"❌ former3d目录不存在")

print("\n检查stream_sdfformer_integrated.py的内容...")
stream_file = os.path.join(former3d_path, 'stream_sdfformer_integrated.py')
if os.path.exists(stream_file):
    with open(stream_file, 'r') as f:
        lines = f.readlines()[:20]  # 读取前20行
        print("文件前20行:")
        for i, line in enumerate(lines, 1):
            print(f"{i:3}: {line.rstrip()}")
else:
    print("文件不存在")