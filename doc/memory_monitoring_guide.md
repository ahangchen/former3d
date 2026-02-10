# PyTorch 显存监控工具使用指南

## 概述

本目录包含一组 PyTorch 显存监控工具，帮助你：

1. **监控不同层的显存占用**
2. **识别显存瓶颈**
3. **检测显存泄漏**
4. **优化模型配置**

## 文件说明

### 核心工具

- **`memory_monitor_layer.py`** - 核心显存监控类和工具函数
- **`monitor_stream_memory.py`** - StreamSDFFormer 模型专用监控脚本
- **`memory_monitor_examples.py`** - 完整使用示例

## 快速开始

### 1. 基本使用

```python
from memory_monitor_layer import MemoryMonitor, memory_context

# 创建监控器
monitor = MemoryMonitor()

# 监控代码块
with memory_context(monitor, "my_operation"):
    result = some_computation(input_data)

# 打印摘要
monitor.print_summary(top_n=10)
monitor.print_detailed()
```

### 2. 监控函数/方法

使用装饰器监控函数：

```python
from memory_monitor_layer import MemoryMonitor, monitor_layer

monitor = MemoryMonitor()

@monitor_layer(monitor, "layer_name")
def my_layer(x):
    # 你的代码
    return output

# 调用函数
result = my_layer(input_data)
```

### 3. 监控神经网络层

```python
import torch.nn as nn

model = MyModel().cuda()
monitor = MemoryMonitor()

# 逐层监控前向传播
with memory_context(monitor, "conv1"):
    x = model.conv1(input_data)

with memory_context(monitor, "batch_norm1"):
    x = model.batch_norm1(x)

with memory_context(monitor, "relu1"):
    x = torch.relu(x)

# 分析结果
monitor.print_summary(top_n=20)
```

## API 文档

### MemoryMonitor 类

#### 初始化

```python
monitor = MemoryMonitor(device=None)
```

- `device`: 要监控的设备（默认：cuda:0）

#### 主要方法

**`get_memory_info()`**
获取当前显存信息

返回：
```python
{
    'allocated': 已分配显存 (GB),
    'reserved': 已预留显存 (GB),
    'free': 可用显存 (GB),
    'total': 总显存 (GB)
}
```

**`record(layer_name, before, after)`**
记录某一层的显存变化

**`print_summary(top_n=10)`**
打印显存使用摘要，显示增量最大的前 N 层

**`print_detailed()`**
打印详细的逐层显存变化

**`clear()`**
清空所有记录

### 上下文管理器

```python
with memory_context(monitor, "context_name"):
    # 你的代码
    pass
```

会自动监控该代码块的显存变化。

### 装饰器

```python
@monitor_layer(monitor, layer_name=None)
def my_function(...):
    # 你的代码
    return result
```

## 示例

### 示例 1: 监控张量操作

```python
from memory_monitor_layer import MemoryMonitor, memory_context

monitor = MemoryMonitor()

# 创建张量
with memory_context(monitor, "create_tensors"):
    x = torch.randn(1000, 1000, device='cuda:0')
    y = torch.randn(1000, 1000, device='cuda:0')

# 矩阵乘法
with memory_context(monitor, "matrix_multiplication"):
    z = torch.matmul(x, y)

# 查看结果
monitor.print_summary()
```

**输出：**
```
================================================================================
显存使用摘要
================================================================================

显存增量最大的 5 层：
--------------------------------------------------------------------------------
层名                                               增量 (GB)         累计 (GB)
--------------------------------------------------------------------------------
create_tensors                                    0.0075          0.0075
matrix_multiplication                             0.0037          0.0112
cleanup                                          -0.0149          0.0000
--------------------------------------------------------------------------------
```

### 示例 2: 监控 CNN 模型

```python
from memory_monitor_layer import MemoryMonitor, memory_context
import torch.nn as nn

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2)
).cuda()

monitor = MemoryMonitor()

input_data = torch.randn(4, 3, 256, 256, device='cuda:0')

# 逐层监控
with memory_context(monitor, "conv1"):
    x = torch.relu(model[0](input_data))

with memory_context(monitor, "maxpool1"):
    x = model[2](x)

with memory_context(monitor, "conv2"):
    x = torch.relu(model[3](x))

with memory_context(monitor, "maxpool2"):
    x = model[5](x)

monitor.print_summary(top_n=10)
```

### 示例 3: 检测显存泄漏

```python
from memory_monitor_layer import MemoryMonitor, memory_context

monitor = MemoryMonitor()

# 运行多次，检查显存是否持续增长
for i in range(10):
    with memory_context(monitor, f"run_{i}"):
        result = my_function(input_data)
        del result

    torch.cuda.empty_cache()

monitor.print_summary()

# 分析：如果显存持续增长，可能存在泄漏
```

## 分析显存瓶颈

### 1. 识别最大显存消费者

```python
# 获取增量最大的层
sorted_records = sorted(monitor.records,
                       key=lambda x: x['delta_allocated'],
                       reverse=True)

print("显存占用最大的层:")
for record in sorted_records[:5]:
    print(f"  {record['layer']}: {record['delta_allocated']:.4f} GB")
```

### 2. 计算显存使用率

```python
final = monitor.records[-1]['after']
usage_pct = (final['allocated'] / final['total'] * 100)

if usage_pct > 90:
    print("⚠️  显存使用率超过90%！")
elif usage_pct > 80:
    print("⚠️  显存使用率较高。")
else:
    print("✅ 显存使用率合理。")
```

### 3. 按操作类型分类

```python
# 分类统计
conv_ops = sum(r['delta_allocated'] for r in monitor.records if 'conv' in r['layer'])
pool_ops = sum(r['delta_allocated'] for r in monitor.records if 'pool' in r['layer'])
fc_ops = sum(r['delta_allocated'] for r in monitor.records if 'fc' in r['layer'])

print(f"卷积操作: {conv_ops:.4f} GB")
print(f"池化操作: {pool_ops:.4f} GB")
print(f"全连接操作: {fc_ops:.4f} GB")
```

## 优化建议

### 显存不足时

1. **减小输入尺寸**
   ```python
   # 原来
   input_size = (256, 256)

   # 优化后
   input_size = (128, 128)
   ```

2. **减小 batch size**
   ```python
   # 原来
   batch_size = 8

   # 优化后
   batch_size = 4
   # 或者使用梯度累积
   accumulation_steps = 2
   ```

3. **使用混合精度训练**
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()

   with autocast():
       output = model(input)
       loss = criterion(output, target)

   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

4. **使用梯度检查点**
   ```python
   from torch.utils.checkpoint import checkpoint

   output = checkpoint(model.layer, input)
   ```

### 显存泄漏检测

如果显存持续增长：

1. **检查全局变量**
   ```python
   # 不好的做法
   global_cache = []
   global_cache.append(some_tensor)  # 泄漏！

   # 好的做法
   result = process_tensor(some_tensor)
   del some_tensor
   ```

2. **检查循环中的张量累积**
   ```python
   # 不好的做法
   all_outputs = []
   for x in data_loader:
       all_outputs.append(model(x))  # 可能泄漏

   # 好的做法
   for i, x in enumerate(data_loader):
       output = model(x)
       # 处理 output
       del output
       torch.cuda.empty_cache()
   ```

3. **使用显存监控工具定期检查**
   ```python
   # 每 N 步检查一次
   if step % 10 == 0:
       print(f"Step {step}: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
   ```

## StreamSDFFormer 专用监控

对于 StreamSDFFormer 模型，使用 `monitor_stream_memory.py`：

```bash
python monitor_stream_memory.py
```

这会：
1. 监控每个时间步的显存占用
2. 分析状态管理、输出操作等不同组件的显存使用
3. 提供优化建议

## 注意事项

1. **预热运行**
   - 第一次运行可能包含 CUDA 上下文初始化的开销
   - 建议运行 2-3 次预热后再开始监控

2. **清理显存**
   - 使用 `torch.cuda.empty_cache()` 释放未使用的缓存
   - 及时删除不需要的变量

3. **多次测量取平均**
   - 显存分配可能有波动
   - 建议多次运行取平均值

4. **考虑缓存效应**
   - PyTorch 会缓存已分配的显存
   - `reserved` 可能大于 `allocated`

## 故障排除

### 问题：显存报告不准确

**原因：**
- PyTorch 的显存缓存机制
- 多次运行后的累积效应

**解决：**
```python
# 在每次测试前清理
torch.cuda.empty_cache()
gc.collect()

# 或重启 Python 进程
```

### 问题：找不到模块

**解决：**
```bash
# 确保在项目根目录运行
cd /home/cwh/coding/former3d

# 或添加到 Python 路径
export PYTHONPATH="/home/cwh/coding/former3d:$PYTHONPATH"
```

## 参考资料

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
- [torch.cuda.memory_allocated](https://pytorch.org/docs/stable/generated/torch.cuda.memory_allocated.html)
- [torch.cuda.memory_reserved](https://pytorch.org/docs/stable/generated/torch.cuda.memory_reserved.html)
