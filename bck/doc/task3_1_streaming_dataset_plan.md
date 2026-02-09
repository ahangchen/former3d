# Task 3.1: 单帧流式PyTorch Dataset 实现计划

## 📋 任务目标
创建一个支持流式推理的PyTorch Dataset，能够：
1. 按帧加载数据（单帧输入，单帧输出）
2. 管理历史状态（可选）
3. 支持真实数据集（ScanNet/TartanAir）
4. 与StreamSDFFormerIntegrated模型兼容

## 🎯 核心需求

### 1. 数据格式
```
单帧数据样本：
{
    "image": [3, H, W],           # 当前帧RGB图像
    "pose": [4, 4],               # 当前相机位姿（世界坐标系）
    "intrinsics": [3, 3],         # 相机内参矩阵
    "frame_id": int,              # 帧ID（用于序列跟踪）
    "sequence_id": str,           # 序列ID
    "timestamp": float,           # 时间戳（可选）
    "depth": [H, W] (可选),       # 深度图（用于监督）
    "sdf_gt": [N, 1] (可选)       # 地面真值SDF（稀疏表示）
}
```

### 2. 流式特性
- **按帧加载**：每次迭代返回一帧数据
- **序列连续性**：保持帧的顺序关系
- **状态管理**：可选的历史状态传递
- **随机访问**：支持按索引访问任意帧

### 3. 数据集支持
- **ScanNet**：室内场景重建数据集
- **TartanAir**：室外/室内合成数据集
- **自定义格式**：支持项目现有数据格式

## 🏗️ 架构设计

### 类结构
```python
class StreamingDataset(torch.utils.data.Dataset):
    """流式推理数据集"""
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'train',
                 sequence_ids: List[str] = None,
                 transform: Callable = None,
                 load_depth: bool = False,
                 load_sdf: bool = False,
                 max_sequence_length: int = None):
        pass
    
    def __len__(self) -> int:
        """返回总帧数"""
        pass
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单帧数据"""
        pass
    
    def get_sequence_info(self, sequence_id: str) -> Dict:
        """获取序列信息"""
        pass
    
    def get_frame_range(self, sequence_id: str) -> Tuple[int, int]:
        """获取序列的帧范围"""
        pass
```

### 数据组织
```
data_root/
├── scannet/
│   ├── scene0000_00/
│   │   ├── color/          # RGB图像
│   │   ├── pose/           # 相机位姿
│   │   ├── intrinsic/      # 相机内参
│   │   ├── depth/          # 深度图（可选）
│   │   └── sdf/            # SDF真值（可选）
│   └── scene0000_01/
└── tartanair/
    ├── abandonedfactory/
    └── carwelding/
```

## 📝 实现步骤

### 步骤1: 创建基础Dataset类
1. **文件**: `former3d/datasets/streaming_dataset.py`
2. **功能**: 
   - 基础数据加载
   - 帧索引管理
   - 简单数据转换

### 步骤2: 实现ScanNet支持
1. **数据解析**: 解析ScanNet原始格式
2. **位姿处理**: 处理ScanNet的相机轨迹
3. **内参处理**: 加载相机内参

### 步骤3: 实现TartanAir支持
1. **数据解析**: 解析TartanAir格式
2. **位姿处理**: 处理TartanAir的6DoF位姿
3. **数据增强**: 添加随机变换

### 步骤4: 添加数据预处理
1. **图像预处理**: 归一化、裁剪、缩放
2. **位姿预处理**: 坐标系转换
3. **数据增强**: 随机旋转、颜色抖动

### 步骤5: 添加流式状态管理
1. **历史状态**: 可选的历史帧缓存
2. **序列跟踪**: 跟踪帧的序列关系
3. **状态重置**: 支持序列边界的状态重置

### 步骤6: 单元测试
1. **数据加载测试**: 验证数据格式正确性
2. **序列连续性测试**: 验证帧顺序
3. **内存测试**: 验证大数据集下的内存使用

## 🔧 技术细节

### 1. 数据加载优化
```python
# 使用内存映射加速大文件读取
class MemoryMappedDataset:
    def __init__(self, data_root):
        self.image_cache = {}  # LRU缓存
        self.pose_cache = {}
```

### 2. 流式状态管理
```python
class StreamingState:
    """流式推理状态管理器"""
    def __init__(self, max_history=10):
        self.history = deque(maxlen=max_history)
    
    def update(self, frame_data, model_state):
        """更新历史状态"""
        pass
    
    def get_history(self, current_pose):
        """获取相关历史状态"""
        pass
```

### 3. 数据增强策略
```python
class StreamingTransform:
    """流式数据增强"""
    def __call__(self, data):
        # 保持序列一致性的增强
        # 同一序列的帧使用相同的随机种子
        pass
```

## 🧪 测试计划

### 单元测试
1. **test_streaming_dataset_basic.py**
   - 测试基础数据加载
   - 测试帧索引
   - 测试数据格式

2. **test_streaming_dataset_scannet.py**
   - 测试ScanNet数据解析
   - 测试位姿正确性
   - 测试序列连续性

3. **test_streaming_dataset_tartanair.py**
   - 测试TartanAir数据解析
   - 测试6DoF位姿
   - 测试数据增强

### 集成测试
1. **test_dataset_with_model.py**
   - 测试Dataset与StreamSDFFormerIntegrated的兼容性
   - 测试端到端数据流
   - 测试训练循环

## 📊 验收标准

### 必须完成
- [ ] 基础StreamingDataset类实现
- [ ] ScanNet数据集支持
- [ ] 正确的数据格式和类型
- [ ] 单元测试通过率 > 90%
- [ ] 与StreamSDFFormerIntegrated兼容

### 可选完成
- [ ] TartanAir数据集支持
- [ ] 数据增强功能
- [ ] 流式状态管理器
- [ ] 内存优化（LRU缓存）

## ⏱️ 时间安排

### Day 1: 基础框架
- 创建Dataset类结构
- 实现基础数据加载
- 编写基础单元测试

### Day 2: ScanNet支持
- 解析ScanNet格式
- 实现位姿和内参加载
- 测试数据正确性

### Day 3: 高级功能
- 添加数据预处理
- 实现数据增强
- 优化内存使用

### Day 4: 测试和验证
- 编写完整单元测试
- 集成测试与模型
- 性能基准测试

## 🚀 开始实施

首先创建基础Dataset类：

```bash
# 创建目录结构
mkdir -p former3d/datasets/
mkdir -p tests/unit/datasets/

# 创建基础文件
touch former3d/datasets/__init__.py
touch former3d/datasets/streaming_dataset.py
touch tests/unit/datasets/test_streaming_dataset.py
```

让我们开始实施！