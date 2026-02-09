# 开发工作流程规范

## 核心原则
**每完成一个任务，必须遵循以下四步流程：**

### 1. 提交工作进度
- 完成核心功能实现后立即提交
- 提交信息清晰描述完成的工作
- 示例：`git commit -m "feat: 实现多序列数据集TSDF在线生成"`

### 2. 清理中间文件
- 删除所有调试文件（`debug_*.py`）
- 删除临时测试文件（`test_*.py`，除非是正式测试）
- 删除一次性脚本和实验代码
- 检查git状态：`git status --porcelain`

### 3. 提交清理结果
- 只保留最终有效的文件
- 提交清理后的干净状态
- 示例：`git commit -m "cleanup: 删除调试文件，保留核心功能"`

### 4. 保持项目简洁
- 避免积累未跟踪的临时文件
- 定期检查并清理`git status`中的未跟踪文件
- 保持项目结构清晰

## 文件命名规范

### 保留的文件
- **核心功能**：`train_*.py`, `dataset_*.py`, `model_*.py`
- **正式测试**：`tests/`目录下的测试文件
- **工具脚本**：`scripts/`目录下的工具
- **配置文件**：`config_*.yaml`, `*.json`

### 删除的文件
- **调试文件**：`debug_*.py`, `fix_*.py`
- **临时测试**：`test_*.py`（根目录下的临时测试）
- **实验代码**：`experiment_*.py`, `try_*.py`
- **备份文件**：`*_backup.py`, `*_old.py`

## 示例工作流程

### 场景：实现新功能
```bash
# 1. 实现核心功能
vim train_stream_integrated.py

# 2. 测试功能（创建临时测试文件）
python test_new_feature.py

# 3. 提交工作进度
git add train_stream_integrated.py
git commit -m "feat: 添加新的损失函数适配"

# 4. 清理中间文件
rm test_new_feature.py

# 5. 提交清理结果
git add -u
git commit -m "cleanup: 删除临时测试文件"
```

### 场景：调试问题
```bash
# 1. 创建调试文件
vim debug_shape_issue.py

# 2. 解决问题
vim multi_sequence_tartanair_dataset.py

# 3. 提交修复
git add multi_sequence_tartanair_dataset.py
git commit -m "fix: 修复数据集形状不匹配问题"

# 4. 清理调试文件
rm debug_shape_issue.py

# 5. 提交清理
git add -u
git commit -m "cleanup: 删除调试文件"
```

## Git提交信息规范

### 功能提交
```
feat: 简要描述新功能
示例: feat: 添加流式状态管理器
```

### 修复提交
```
fix: 简要描述修复的问题
示例: fix: 修复GPU内存泄漏问题
```

### 清理提交
```
cleanup: 简要描述清理内容
示例: cleanup: 删除调试文件和临时测试
```

### 文档提交
```
docs: 简要描述文档更新
示例: docs: 更新项目状态和开发流程
```

## 检查清单
每次完成任务后检查：
- [ ] 核心功能已提交
- [ ] 所有调试文件已删除
- [ ] `git status`显示干净或只有必要文件
- [ ] 提交信息清晰规范
- [ ] 项目结构保持简洁

## 好处
1. **版本清晰**：每个功能都有明确的提交记录
2. **项目整洁**：避免临时文件积累
3. **协作友好**：其他人可以轻松理解项目状态
4. **回滚容易**：如果需要，可以轻松回退到某个功能点
5. **专业形象**：体现系统化的开发习惯