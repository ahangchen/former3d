#!/usr/bin/env python3
"""
简单数据集测试 - 不依赖外部库
"""

import os
import sys
import glob

class SimpleMultiSequenceTartanAirDataset:
    """简化的多序列数据集（仅用于测试）"""
    
    def __init__(self, data_root, n_view=5, stride=2, max_sequences=3):
        self.data_root = data_root
        self.n_view = n_view
        self.stride = stride
        
        # 发现序列
        self.sequences = self._discover_sequences(max_sequences)
        
        # 构建片段
        self.segments = self._build_segments()
        
        print(f"数据集初始化完成:")
        print(f"  序列数: {len(self.sequences)}")
        print(f"  片段数: {len(self.segments)}")
        print(f"  片段长度: {n_view} 帧")
        print(f"  片段步长: {stride}")
    
    def _discover_sequences(self, max_sequences):
        """发现序列"""
        sequences = []
        
        for item in os.listdir(self.data_root):
            if len(sequences) >= max_sequences:
                break
                
            item_path = os.path.join(self.data_root, item)
            if os.path.isdir(item_path):
                p001_path = os.path.join(item_path, "P001")
                if os.path.exists(p001_path):
                    rgb_dir = os.path.join(p001_path, "image_left")
                    if os.path.exists(rgb_dir):
                        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
                        if len(rgb_files) >= self.n_view:
                            sequences.append({
                                'name': item,
                                'rgb_files': rgb_files,
                                'num_frames': len(rgb_files)
                            })
        
        return sequences
    
    def _build_segments(self):
        """构建片段"""
        segments = []
        
        for seq_idx, seq in enumerate(self.sequences):
            num_frames = seq['num_frames']
            num_segments = max(1, (num_frames - self.n_view) // self.stride + 1)
            
            for seg_idx in range(num_segments):
                start_frame = seg_idx * self.stride
                end_frame = start_frame + self.n_view
                
                if end_frame <= num_frames:
                    segments.append({
                        'seq_idx': seq_idx,
                        'seq_name': seq['name'],
                        'start_frame': start_frame,
                        'end_frame': end_frame
                    })
        
        return segments
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        seq = self.sequences[segment['seq_idx']]
        
        # 返回模拟数据
        return {
            'sequence_name': seq['name'],
            'segment_idx': idx,
            'start_frame': segment['start_frame'],
            'end_frame': segment['end_frame'],
            'rgb_files': seq['rgb_files'][segment['start_frame']:segment['end_frame']]
        }

def main():
    """主测试函数"""
    data_root = "/home/cwh/Study/dataset/tartanair"
    
    print("测试简化的多序列数据集...")
    print(f"数据根目录: {data_root}")
    
    try:
        # 创建数据集
        dataset = SimpleMultiSequenceTartanAirDataset(
            data_root=data_root,
            n_view=5,
            stride=2,
            max_sequences=3
        )
        
        # 测试长度
        print(f"\n数据集长度: {len(dataset)}")
        
        # 测试几个样本
        print(f"\n测试前3个样本:")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n样本 {i}:")
            print(f"  序列: {sample['sequence_name']}")
            print(f"  帧范围: {sample['start_frame']}-{sample['end_frame']}")
            print(f"  RGB文件数: {len(sample['rgb_files'])}")
            if sample['rgb_files']:
                print(f"  第一个文件: {os.path.basename(sample['rgb_files'][0])}")
        
        # 测试批量模拟
        print(f"\n模拟批量数据形状:")
        batch_size = 2
        indices = list(range(min(batch_size, len(dataset))))
        batch = [dataset[i] for i in indices]
        
        print(f"  批量大小: {len(batch)}")
        print(f"  每个样本帧数: {dataset.n_view}")
        
        # 验证数据一致性
        print(f"\n验证数据一致性:")
        all_valid = True
        for i, sample in enumerate(batch):
            expected_frames = dataset.n_view
            actual_frames = len(sample['rgb_files'])
            if actual_frames == expected_frames:
                print(f"  样本 {i}: ✅ 帧数正确 ({actual_frames})")
            else:
                print(f"  样本 {i}: ❌ 帧数错误 (期望 {expected_frames}, 实际 {actual_frames})")
                all_valid = False
        
        if all_valid:
            print(f"\n✅ 所有测试通过!")
            print(f"\n下一步:")
            print(f"1. 安装必要的Python包: pip install numpy torch imageio pillow")
            print(f"2. 运行完整的数据集测试: python multi_sequence_tartanair_dataset.py")
            print(f"3. 集成到训练脚本中")
        else:
            print(f"\n❌ 测试失败!")
            
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()