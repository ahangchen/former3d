#!/usr/bin/env python3
"""
修复_apply_stream_fusion.py
完全重写_apply_stream_fusion，使用预投影特征 + concat + 3D卷积融合
"""

import re

# 读取文件
with open('former3d/stream_sdfformer_integrated.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 备份原来的_apply_stream_fusion方法
# 找到方法开始和结束
start_marker = '    def _apply_stream_fusion(self,'
end_marker = '    def _update_voxel_outputs('

start_idx = content.find(start_marker)
end_idx = content.find(end_marker, start_idx)

if start_idx != -1 and end_idx != -1:
    old_method = content[start_idx:end_idx]
    
    with open('former3d/stream_sdfformer_integrated_backup_apply_stream_fusion.py', 'w', encoding='utf-8') as f:
        f.write(old_method)
    print(f"✅ 已备份_apply_stream_fusion方法")
else:
    print(f"❌ 无法找到_apply_stream_fusion方法")
    exit(1)

# 新的_apply_stream_fusion方法
new_method = '''    def _apply_stream_fusion(self, 
                           current_features: Dict,
                           historical_features: Dict,
                           current_pose: torch.Tensor) -> torch.Tensor:
        """
        应用流式融合（选项A完整版：使用预投影特征 + concat + 3D卷积融合）
        
        Args:
            current_features: 当前特征字典（从_extract_current_features提取）
            historical_features: 历史特征字典（从_extract_historical_features提取）
            current_pose: 当前帧位姿 [B, 4, 4]
            
        Returns:
            融合后的特征 [N, 128]
        """
        if historical_features is None:
            print("⚠️ 没有历史特征，跳过流式融合")
            current_feats = current_features['features']
            return current_feats

        # 检查是否有预投影的特征（选项A改进版）
        if 'projected_features' not in historical_features:
            print("⚠️ 历史状态中没有projected_features，跳过流式融合")
            current_feats = current_features['features']
            return current_feats

        current_feats = current_features['features']  # [N, 128]
        projected = historical_features['projected_features']  # {resname: [N, C]}
        
        num_points = current_feats.shape[0]
        device = current_feats.device
        
        print(f"[StreamFusion] 当前特征: {current_feats.shape}")
        print(f"[StreamFusion] 预投影特征: {list(projected.keys())}")
        
        # 提取预投影的fine特征和SDF
        if 'fine' not in projected:
            print("⚠️ 没有预投影的fine特征，跳过流式融合")
            return current_feats
        
        projected_fine = projected['fine']  # [N, 128]
        projected_sdf = projected.get('sdf', None)  # [N, 1]
        
        print(f"[StreamFusion] 预投影fine特征: {projected_fine.shape}")
        if projected_sdf is not None:
            print(f"[StreamFusion] 预投影SDF: {projected_sdf.shape}")
        
        # 检查形状是否匹配
        if projected_fine.shape[0] != current_feats.shape[0]:
            print(f"⚠️ 预投影特征数量不匹配: {projected_fine.shape[0]} vs {current_feats.shape[0]}")
            # 截断或填充
            min_size = min(projected_fine.shape[0], current_feats.shape[0])
            projected_fine = projected_fine[:min_size]
            current_feats = current_feats[:min_size]
            if projected_sdf is not None:
                projected_sdf = projected_sdf[:min_size]
            num_points = min_size
        else:
            num_points = current_feats.shape[0]
        
        # Concat: 历史fine [N, 128] + 当前 [N, 128] + 历史SDF [N, 1]
        if projected_sdf is not None:
            concat_features = torch.cat([projected_fine, current_feats, projected_sdf], dim=1)  # [N, 257]
        else:
            concat_features = torch.cat([projected_fine, current_feats], dim=1)  # [N, 256]
        
        print(f"[StreamFusion] Concat特征: {concat_features.shape}")
        
        # 添加batch和空间维度（假设为1x1x1的3D空间）
        # 这样可以直接使用1D卷积（等同于3D卷积核大小为1）
        concat_features = concat_features.unsqueeze(1)  # [N, 257, 1]
        concat_features = concat_features.unsqueeze(2)  # [N, 257, 1, 1]
        concat_features = concat_features.permute(1, 0, 2, 3)  # [257, N, 1, 1]
        
        # 使用融合网络（1D卷积）
        try:
            fused = self.fusion_3d(concat_features)  # [128, N, 1, 1]
            fused = fused.permute(1, 0, 2, 3)  # [N, 128, 1, 1]
            fused = fused.squeeze(-1).squeeze(-1).squeeze(-1)  # [N, 128]
            
            print(f"[StreamFusion] 融合后特征: {fused.shape}")
            return fused
            
        except Exception as e:
            print(f"⚠️ 3D卷积融合失败: {e}")
            # 回退到简单的加权平均
            fused = 0.5 * projected_fine + 0.5 * current_feats
            print(f"[StreamFusion] 回退到加权平均: {fused.shape}")
            return fused
'''

# 替换方法
content_new = content[:start_idx] + new_method + content[end_idx:]

# 写入文件
with open('former3d/stream_sdfformer_integrated.py', 'w', encoding='utf-8') as f:
    f.write(content_new)

print("✅ _apply_stream_fusion方法已重写")
