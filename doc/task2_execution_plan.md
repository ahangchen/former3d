# 任务2执行计划：修改训练脚本框架以使用StreamSDFFormerIntegrated

## 🎯 目标
将现有的训练脚本框架从SDF3DModel改为使用StreamSDFFormerIntegrated模型，支持流式处理。

## 📋 任务分解

### 阶段1：分析现有代码（已完成）
- [x] 分析`final_multi_sequence_training_fixed_fixed.py`的结构
- [x] 理解SDF3DModel的使用方式
- [x] 了解StreamSDFFormerIntegrated的API

### 阶段2：创建新的训练脚本框架
- [ ] 创建`train_stream_integrated.py`脚本
- [ ] 实现StreamSDFFormerIntegrated的导入和初始化
- [ ] 修改数据准备函数以支持流式输入
- [ ] 实现流式训练循环

### 阶段3：测试和验证
- [ ] 创建测试脚本
- [ ] 验证模型可以正确加载和运行
- [ ] 确保设备一致性
- [ ] 验证训练循环可以执行

### 阶段4：集成和优化
- [ ] 集成到现有项目中
- [ ] 添加必要的工具函数
- [ ] 优化内存使用
- [ ] 添加日志和监控

## 🔧 技术要点

### 1. StreamSDFFormerIntegrated API
根据`test_simple_integration.py`的分析：
```python
# 模型初始化
model = StreamSDFFormerIntegrated(
    attn_heads=2,
    attn_layers=1,
    use_proj_occ=True,
    voxel_size=0.04,
    fusion_local_radius=2.0,
    crop_size=(32, 32, 24)
)

# 单帧处理
output, state = model.forward_single_frame(
    images=images,        # [batch, 1, 3, H, W]
    poses=poses,          # [batch, 1, 4, 4]
    intrinsics=intrinsics, # [batch, 1, 3, 3]
    reset_state=True      # 是否重置状态
)
```

### 2. 数据适配
需要将多帧数据集适配为单帧流式输入：
- 数据集应返回单帧数据
- 需要管理序列状态
- 需要处理序列边界（重置状态）

### 3. 训练循环结构
```python
def train_stream_epoch(model, dataloader, optimizer, device, state_manager):
    """流式训练一个epoch"""
    
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(dataloader):
        # 移动到设备
        batch = move_to_device(batch, device)
        
        # 获取序列信息
        sequence_id = batch['sequence_id']
        
        # 按帧处理序列
        sequence_loss = 0
        state_manager.reset(sequence_id)
        
        for frame_idx in range(batch['sequence_length']):
            # 获取当前帧
            frame_data = extract_frame(batch, frame_idx)
            
            # 获取历史状态
            historical_state = state_manager.get_state()
            
            # 前向传播
            output, new_state = model.forward_single_frame(
                images=frame_data['images'],
                poses=frame_data['poses'],
                intrinsics=frame_data['intrinsics'],
                reset_state=(frame_idx == 0)
            )
            
            # 计算损失
            loss = compute_loss(output, frame_data['ground_truth'])
            sequence_loss += loss
            
            # 更新状态
            state_manager.update(new_state, sequence_id, frame_idx)
            
            # 梯度累积
            if (frame_idx + 1) % gradient_accumulation_steps == 0:
                sequence_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                sequence_loss = 0
                
        total_loss += sequence_loss.item() if sequence_loss != 0 else 0
        
    return total_loss / len(dataloader)
```

## 🧪 测试计划

### 测试1：模型导入测试
```python
# 验证StreamSDFFormerIntegrated可以正确导入
python test_model_import.py
```

### 测试2：数据适配测试
```python
# 验证数据集可以生成正确的单帧数据
python test_data_adapter.py
```

### 测试3：训练循环测试
```python
# 验证训练循环可以执行
python train_stream_integrated.py --dry-run --test-mode
```

### 测试4：设备一致性测试
```python
# 验证所有张量在相同设备上
python test_device_consistency.py
```

## 📁 文件结构

```
former3d/
├── doc/
│   └── task2_execution_plan.md          # 本文件
├── test/
│   ├── test_model_import.py             # 模型导入测试
│   ├── test_data_adapter.py             # 数据适配测试
│   └── test_stream_training.py          # 流式训练测试
├── train_stream_integrated.py           # 主训练脚本
├── stream_state_manager.py              # 流式状态管理
└── stream_data_adapter.py               # 流式数据适配器
```

## ⏰ 时间估计

- 阶段1：已完成
- 阶段2：30分钟
- 阶段3：15分钟
- 阶段4：15分钟
- 总计：60分钟

## 🎯 成功标准

1. ✅ 创建`train_stream_integrated.py`脚本
2. ✅ 可以正确导入和使用StreamSDFFormerIntegrated
3. ✅ 训练脚本可以执行（--dry-run模式）
4. ✅ 无设备不匹配错误
5. ✅ 可以处理多序列数据

## 🆘 风险与应对

### 风险1：模型API变化
**应对**：检查最新的模型实现，确保API兼容性

### 风险2：数据格式不匹配
**应对**：创建数据适配器，转换数据格式

### 风险3：内存不足
**应对**：使用小批量，启用梯度累积

### 风险4：训练不稳定
**应对**：添加更多的监控和调试信息

---

**开始时间**：2026年2月9日 00:45  
**预计完成时间**：2026年2月9日 01:45