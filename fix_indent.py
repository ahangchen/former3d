#!/usr/bin/env python3
"""
修复缩进问题
"""

with open('former3d/stream_sdfformer_integrated.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 修复缩进
fixed_lines = []
for i, line in enumerate(lines):
    if 'num_points = min_size' in line and line.startswith('        '):
        # 检查是否缩进不正确
        if line.count('    ') == 2:  # 8个空格
            # 修复缩进为4个空格
            fixed_line = line.replace('        ', '    ', 1)  # 只换第一个8个空格为4个空格
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    elif 'projected_sdf = projected_sdf[:min_size]' in line and line.count('    ') == 3:  # 12个空格
        # 修复缩进
        fixed_line = '        ' + line.lstrip()  # 保持8个空格的缩进
        fixed_lines.append(fixed_line)
    else:
        fixed_lines.append(line)

# 写入文件
with open('former3d/stream_sdfformer_integrated.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

print("✅ 缩进问题已修复")