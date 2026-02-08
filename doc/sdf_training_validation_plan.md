# SDF真值训练验证计划

## 目标
创建一个使用TartanAir SDF真值的训练验证脚本，验证StreamSDFFormer模型可以使用生成的SDF数据进行训练。

## 背景
- 已成功生成TartanAir SDF真值：`tartanair_sdf_output/abandonedfactory_sample_P001_sdf_occ.npz`
- SDF网格形状: (291, 206, 206)，体素大小: 0.04米
- 需要将SDF真值集成到训练流程中

## 任务分解

### 1. 创建SDF真值加载器
- 加载NPZ文件中的SDF和occupancy网格
- 创建体素坐标到世界坐标的转换
- 实现采样函数从SDF网格中获取训练样本

### 2. 修改数据集类
- 扩展现有的`SimpleTartanAirDataset`类
- 添加SDF真值加载功能
- 实现SDF样本采样策略

### 3. 创建训练验证脚本
- 使用SDF真值作为监督信号
- 实现SDF损失函数（L1或Huber损失）
- 验证前向传播、反向传播和参数更新

### 4. 测试与验证
- 运行训练循环（3个epoch）
- 监控损失下降
- 检查梯度流动
- 验证内存使用

## 技术细节

### SDF真值格式
```python
{
    'sdf': (291, 206, 206) float32,      # SDF值网格
    'occupancy': (291, 206, 206) float32, # 占用网格
    'voxel_size': 0.04,                  # 体素大小（米）
    'bounds': (3, 2) float64,            # 场景边界
    'intrinsics': (3, 3) float32,        # 相机内参
    'sequence_name': 'abandonedfactory_sample_P001'
}
```

### 训练策略
1. **采样策略**：从SDF网格中随机采样点
   - 表面附近采样（SDF值接近0）
   - 自由空间采样（SDF值>0）
   - 占用空间采样（SDF值<0）

2. **损失函数**：
   - L1损失：`loss = |pred_sdf - gt_sdf|`
   - Huber损失：对异常值更鲁棒

3. **批次构建**：
   - 每个批次包含多个采样点
   - 平衡不同区域的采样

### 预期输出
1. 训练损失曲线
2. 验证损失曲线
3. 梯度统计信息
4. 内存使用报告

## 文件结构
```
former3d/
├── doc/
│   └── sdf_training_validation_plan.md  # 本文件
├── test_sdf_training.py                 # 测试脚本
├── sdf_training_validation.py           # 主训练脚本
└── former3d/datasets/
    └── tartanair_sdf_dataset.py         # SDF数据集类
```

## 时间安排
1. 创建SDF数据集类：30分钟
2. 创建训练脚本：30分钟
3. 测试与验证：30分钟
4. 调试与优化：30分钟

## 成功标准
- 训练损失在3个epoch内下降
- 梯度正常流动（无NaN或爆炸）
- GPU内存使用合理
- 模型参数正常更新

## 风险与缓解
1. **内存不足**：减少采样点数量，使用更小的批次
2. **训练不稳定**：使用梯度裁剪，调整学习率
3. **收敛缓慢**：调整采样策略，优化损失函数
4. **数据不匹配**：验证SDF真值与相机位姿的对齐

## 下一步
完成此验证后，可以：
1. 扩展到更多TartanAir序列
2. 实现完整的训练流程
3. 添加评估指标
4. 与其他方法对比