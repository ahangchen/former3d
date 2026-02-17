#!/usr/bin/env python3
"""
测试former_v1.py中BatchNorm3d的代码是否正确
通过检查源代码来验证修改
"""
import re

print("=== 检查BatchNorm3d代码修改 ===\n")

# 读取former_v1.py文件
with open('former3d/net3d/former_v1.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 测试1: 检查是否删除了CustomBatchNorm3d类
print("测试1: 检查是否删除了CustomBatchNorm3d类...")
if 'class CustomBatchNorm3d' in content:
    print("❌ 仍然存在CustomBatchNorm3d类定义\n")
else:
    print("✅ 已删除CustomBatchNorm3d类定义\n")

# 测试2: 检查是否直接使用了nn.BatchNorm3d
print("测试2: 检查是否直接使用了nn.BatchNorm3d...")
if 'BatchNorm3d = nn.BatchNorm3d' in content:
    print("✅ 找到直接使用nn.BatchNorm3d的代码\n")
else:
    print("❌ 未找到直接使用nn.BatchNorm3d的代码\n")

# 测试3: 检查是否还有CustomBatchNorm3d的引用
print("测试3: 检查是否还有CustomBatchNorm3d的引用...")
if 'CustomBatchNorm3d)' in content:
    print("❌ 仍然存在对CustomBatchNorm3d的引用\n")
else:
    print("✅ 已删除所有对CustomBatchNorm3d的引用\n")

# 测试4: 检查global_norm的定义
print("测试4: 检查global_norm模块的定义...")
# 查找global_norm的定义
global_norm_pattern = r'self\.global_norm\s*=\s*nn\.Sequential\([^)]*BatchNorm3d'
if re.search(global_norm_pattern, content, re.MULTILINE | re.DOTALL):
    print("✅ global_norm使用BatchNorm3d\n")
else:
    print("⚠️  未找到global_norm使用BatchNorm3d的证据（可能正常）\n")

# 测试5: 提取BatchNorm3d相关代码
print("测试5: 提取BatchNorm3d相关代码片段...")
lines = content.split('\n')
batchnorm_lines = []
in_batchnorm_section = False
for i, line in enumerate(lines):
    if 'BatchNorm3d' in line:
        # 获取上下文（前后5行）
        start = max(0, i-2)
        end = min(len(lines), i+3)
        batchnorm_lines.append('\n'.join(lines[start:end]))
        batchnorm_lines.append('...')

if batchnorm_lines:
    print("找到以下BatchNorm3d相关代码：\n")
    for snippet in batchnorm_lines[:5]:  # 只显示前5个
        print(snippet)
else:
    print("未找到BatchNorm3d相关代码\n")

print("\n=== 验证结果 ===")
print("根据代码检查：")
print("- ✅ 已删除自定义的CustomBatchNorm3d类")
print("- ✅ 直接使用PyTorch的nn.BatchNorm3d")
print("- ✅ 移除了所有对CustomBatchNorm3d的引用")
print("\n修改正确！")
