# 文件清理计划

## 核心文件（保留在根目录）

### 1. 数据集文件
- `multi_sequence_tartanair_dataset.py` - 多序列数据集（核心）
- `online_tartanair_dataset.py` - 在线数据集（基础）

### 2. 训练脚本
- `final_multi_sequence_training_fixed.py` - 多序列训练脚本（核心）
- `successful_training.py` - 已验证成功的训练脚本

### 3. 配置文件
- `config.yml` - 主配置文件
- `config_tartanair.yml` - TartanAir配置文件

### 4. 文档
- `README.md` - 项目说明
- `dataset_modification_plan.md` - 数据集修改计划
- `MULTI_SEQUENCE_DATASET_SUMMARY.md` - 多序列数据集总结

### 5. 模型目录
- `former3d/` - 模型实现目录（完整保留）

## 移动到bck目录的文件

### 1. 调试脚本（bck/script/）
- 所有以 `debug_` 开头的文件
- 所有以 `test_` 开头的文件（除核心测试外）
- 所有以 `fix_` 开头的文件
- 所有梯度验证文件
- 所有中间训练脚本

### 2. 文档文件（bck/doc/）
- 所有实现计划文档
- 所有任务总结文档
- 所有流程图文档

### 3. 日志文件（bck/logs/）
- 所有 `.log` 文件
- 所有训练历史文件

### 4. 检查点（bck/checkpoints/）
- 所有检查点目录（除最新外）

## 清理步骤

1. 创建bck目录结构
2. 移动调试脚本到 bck/script/
3. 移动文档到 bck/doc/
4. 移动日志到 bck/logs/
5. 移动旧检查点到 bck/checkpoints/
6. 验证核心文件完整性