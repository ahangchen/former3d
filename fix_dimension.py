#!/usr/bin/env python3
"""
修复维度匹配问题
"""

with open('former3d/stream_sdfformer_integrated.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 找到3D卷积融合部分
start_marker = '# Concat: 历史fine [N, 128] + 当前 [N, 128] + 历史SDF [N, 1]'
end_marker = 'print(f"[StreamFusion] Concat特征: {concat_features.shape}")'

# 找到相关代码块
start_idx = content.find(start_marker)
if start_idx != -1:
    # 修复维度匹配问题
    old_code = '''        # Concat: 历史fine [N, 128] + 当前 [N, 128] + 历史SDF [N, 1]
        if projected_sdf is not None:
            concat_features = torch.cat([projected_fine, current_feats, projected_sdf], dim=1)  # [N, 257]
        else:
            concat_features = torch.cat([projected_fine, current_feats], dim=1)  # [N, 256]'''
    
    # 修复后的代码：统一特征维度
    new_code = '''        # Concat: 历史fine + 当前 + 历史SDF
        # 统一特征维度：将预投影的fine特征维度与当前特征维度对齐
        if projected_fine.shape[1] != current_feats.shape[1]:
            # 创建特征维度对齐层
            if not hasattr(self, '_feat_aligner'):
                import torch.nn as nn
                # 动态创建特征对齐层
                feat_in = projected_fine.shape[1]
                feat_out = current_feats.shape[1]
                self._feat_aligner = nn.Linear(feat_in, feat_out).to(projected_fine.device)
            
            # 对齐预投影特征维度
            projected_aligned = self._feat_aligner(projected_fine)
        else:
            projected_aligned = projected_fine
        
        # Concat: 对齐后的预投影fine [N, 128] + 当前 [N, 128] + 历史SDF [N, 1]
        if projected_sdf is not None:
            concat_features = torch.cat([projected_aligned, current_feats, projected_sdf], dim=1)  # [N, 257]
        else:
            concat_features = torch.cat([projected_aligned, current_feats], dim=1)  # [N, 256]'''
    
    content = content.replace(old_code, new_code)
    
    # 写入文件
    with open('former3d/stream_sdfformer_integrated.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 维度对齐问题已修复")
else:
    print("❌ 未找到相关代码块")
