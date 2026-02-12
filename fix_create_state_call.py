#!/usr/bin/env python3
"""
修改_create_new_state调用.py
找到所有调用_create_new_state的地方，添加current_voxel_indices参数
"""

import re

# 读取文件
with open('former3d/stream_sdfformer_integrated.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修改第641行的调用
# 需要从voxel_outputs['fine']中提取indices
old_call = "        new_state = self._create_new_state(output, poses)"
new_call = '''        # 从voxel_outputs中提取当前体素索引（用于预投影）
        current_voxel_indices = None
        if 'voxel_outputs' in output and 'fine' in output['voxel_outputs']:
            fine_output = output['voxel_outputs']['fine']
            if hasattr(fine_output, 'indices'):
                current_voxel_indices = fine_output.indices  # [N, 4] (b, x, y, z)
                print(f"[StreamFusion] 提取当前体素索引: {current_voxel_indices.shape}")
        
        new_state = self._create_new_state(output, poses, current_voxel_indices)'''

content = content.replace(old_call, new_call)

print("✅ 步骤4: _create_new_state调用已修改")

# 检查是否还有其他调用_create_new_state的地方
if "self._create_new_state(output," in content:
    print("⚠️ 警告：可能还有其他调用_create_new_state的地方")
else:
    print("✅ 所有_create_new_state调用已修改")

# 写入文件
with open('former3d/stream_sdfformer_integrated.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 文件已更新")
