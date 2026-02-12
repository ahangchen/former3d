#!/usr/bin/env python3
"""
修改_create_legacy_state.py
添加current_voxel_indices参数
"""

import re

# 读取文件
with open('former3d/stream_sdfformer_integrated.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修改方法签名
old_signature = r'def _create_legacy_state\(self, output: Dict, current_pose: torch\.Tensor\) -> Dict:'
new_signature = 'def _create_legacy_state(self, output: Dict, current_pose: torch.Tensor, current_voxel_indices: Optional[torch.Tensor] = None) -> Dict:'

content = re.sub(old_signature, new_signature, content)

print("✅ 步骤3: _create_legacy_state签名已修改")

# 写入文件
with open('former3d/stream_sdfformer_integrated.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ _create_legacy_state修改完成")
