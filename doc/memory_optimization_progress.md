# 显存泄露修复计划进度跟踪

## 📋 总览

**目标**：降低流式训练显存占用，防止内存泄漏，提高训练稳定性

**进度**：2/5 阶段完成（40%）

---

## ✅ 阶段1：状态管理器修复（已完成）

### 1.1 修改 `StreamStateManager.update_state()` 方法
**状态**：✅ 已完成（commit b04f005）
**修改内容**：
- ✅ 添加 `_release_state_tensors()` 方法，递归释放状态字典中的张量
- ✅ 在 `update_state()` 方法中，更新前先释放旧状态
- ✅ 在 `reset()` 方法中，清理前先释放张量

**测试**：✅ 已验证
- `test/test_stream_state_manager_memory.py`：验证内存稳定在51.25MB
- `test/test_lru_mechanism.py`：验证LRU机制正确工作
- `test/test_lru_state_preservation.py`：验证历史状态在序列内正确传递

### 1.2 添加状态清理方法
**状态**：✅ 已完成（commit b04f005）
**修改内容**：
- ✅ 实现 `clear_old_states()` 方法
- ✅ 添加 LRU 缓存机制（max_cached_states参数）
- ✅ 更新访问顺序，支持LRU清理

**测试**：✅ 已验证
- LRU机制在超过max_cached_states时自动清理最久未访问的状态
- 对当前batch的序列状态不会被误删

---

## ✅ 阶段2：历史状态优化（已完成）

### 2.1 修改 `StreamSDFFormerIntegrated._create_new_state()` 方法
**状态**：✅ 已完成（commit 09eedb5）
**修改内容**：
- ✅ 移除完整输出保存（`output`字段）
- ✅ 移除重复特征保存（`original_features`字段）
- ✅ 只保存必要信息（features, sdf, occupancy, coords, batch_inds, pose）

**测试**：✅ 已验证
- `test/test_lightweight_state.py`：验证轻量级状态模式启用

### 2.2 添加轻量级状态模式
**状态**：✅ 已完成（commit 09eedb5）
**修改内容**：
- ✅ 添加 `lightweight_state_mode` 标志（默认为True）
- ✅ 实现 `enable_lightweight_state()` 方法
- ✅ 非轻量级模式下显示警告

**分析文档**：✅ 已完成
- `doc/lru_impact_analysis.md`：分析LRU策略对流式训练的影响

---

## ⚠️ 阶段3：注意力计算优化（部分完成）

### 3.1 使用checkpointing减少显存
**状态**：✅ 已实现并测试
**预计工作量**：2小时
**风险**：中（可能影响精度）

**实现**：
- ✅ 在 `LocalCrossAttention` 中添加 `use_checkpoint` 参数
- ✅ 实现 `_compute_attention()` 方法用于checkpointing
- ✅ 更新 `StreamCrossAttention` 支持 `use_checkpoint`
- ✅ 更新 `StreamSDFFormerIntegrated` 传递 `use_checkpoint` 参数

**测试**：✅ 已验证
- ✅ `test/test_checkpointing_precision.py`：数值精度完全正确（差异为0）
- ⚠️  `test/test_checkpointing_memory.py`：当前模型规模下无显著收益

**分析**：
- Checkpointing数值精度完全正确，可以安全使用
- 对于当前模型规模（N=2000, 10000），checkpointing未降低显存
- 原因：计算规模较小，checkpointing开销大于收益
- 建议：只在更大规模或更深网络中使用checkpointing

### 3.2 实现分块注意力计算
**状态**：❌ 未开始（优先级降低）
**预计工作量**：1小时
**风险**：低

**原因**：
- Checkpointing在当前模型规模下未显示出优势
- 分块注意力主要用于超大注意力矩阵（N>50000）
- 当前模型的体素数量（N=2000, 10000）不需要分块
- **决定**：推迟实施，等待更大规模模型需求

**未来实施条件**：
- 单个注意力序列超过50000体素
- 显存超过6GB
- 检测到CUDA OOM错误


## ✅ 阶段4：显存清理机制（已完成）

### 4.1 添加定期清理函数
**状态**：✅ 已完成
**预计工作量**：1小时
**风险**：低

**实现**：
- ✅ 创建 `MemoryManager` 类
- ✅ 实现 `cleanup()` 方法（清理CUDA缓存和垃圾回收）
- ✅ 实现 `cleanup_if_needed()` 方法（基于阈值触发清理）
- ✅ 实现 `get_memory_info()` 方法（监控显存使用）
- ✅ 实现 `step()` 方法（定期清理）

**测试**：✅ 已验证
- ✅ `test/test_memory_manager.py`：验证显存清理功能
- ✅ 集成测试通过

### 4.2 集成到训练脚本
**状态**：✅ 已完成
**预计工作量**：1小时
**风险**：低

**实现**：
- ✅ 在 `train_stream_integrated.py` 中导入 `MemoryManager`
- ✅ 添加命令行参数：`--cleanup-freq` 和 `--memory-threshold`
- ✅ 在 `train_epoch_stream()` 中集成 `MemoryManager`
- ✅ 在 `test_model()` 中集成 `MemoryManager`

**功能**：
- ✅ 定期清理（每N步执行一次）
- ✅ 按需清理（显存超过阈值时自动清理）
- ✅ 显存信息监控
- ✅ 垃圾回收

**效果**：
- ✅ 防止显存泄漏
- ✅ 自动清理缓存
- ✅ 支持阈值控制
- ✅ 提高训练稳定性

---

## ⚠️ 阶段4：显存清理机制（未完成）

### 4.1 添加定期清理函数
**状态**：❌ 未开始
**预计工作量**：1小时
**风险**：低

**待实现**：
- [ ] 创建 `MemoryManager` 类
- [ ] 实现 `cleanup()` 方法（清理CUDA缓存和垃圾回收）
- [ ] 实现 `cleanup_if_needed()` 方法（基于阈值触发清理）

### 4.2 集成到训练脚本
**状态**：❌ 未开始
**预计工作量**：1小时
**风险**：低

**待实现**：
- [ ] 修改 `train_stream_integrated.py`
- [ ] 在训练循环中集成 `MemoryManager`
- [ ] 添加命令行参数：`--cleanup-freq`、`--memory-threshold`

---

## ❌ 阶段5：训练循环优化（未开始）

### 5.1 优化帧循环中的变量管理
**状态**：❌ 未开始
**预计工作量**：2小时
**风险**：低

**待实现**：
- [ ] 优化 `process_frame_sequence()` 函数
- [ ] 及时释放不再需要的帧数据
- [ ] 预分配输出列表，避免频繁分配

### 5.2 添加梯度累积支持
**状态**：❌ 未开始
**预计工作量**：1小时
**风险**：低

**待实现**：
- [ ] 实现 `train_with_gradient_accumulation()` 函数
- [ ] 添加命令行参数：`--accumulation-steps`
- [ ] 优化梯度更新逻辑

---

## 📊 当前状态总结

### 已完成的优化
| 优化项 | 状态 | 效果 |
|--------|------|------|
| 状态管理器张量释放 | ✅ 完成 | 防止内存泄漏 |
| LRU缓存策略 | ✅ 完成 | 限制状态数量 |
| 轻量级状态模式 | ✅ 完成 | 减少状态大小 |
| batch_idx归一化 | ✅ 完成 | 消除警告 |

### 待完成的优化
| 优化项 | 优先级 | 预计工作量 | 预计收益 |
|--------|--------|------------|----------|
| checkpointing | P1 | 2小时 | ↓20-30%显存 |
| 分块注意力 | P1 | 1小时 | ↓10-15%显存 |
| 内存管理器 | P1 | 2小时 | 稳定性提升 |
| 梯度累积 | P2 | 1小时 | 模拟大batch |

---

## 🎯 下一步行动

### 立即执行（今日）
1. ✅ 检查当前进度状态（已完成）
2. ⏭️ 执行阶段3：注意力计算优化
   - 实现checkpointing（2小时）
   - 实现分块注意力（1小时）
3. ⏭️ 执行阶段4：显存清理机制
   - 创建MemoryManager（1小时）
   - 集成到训练脚本（1小时）

### 后续执行（明日）
4. 执行阶段5：训练循环优化
   - 优化帧循环（2小时）
   - 添加梯度累积（1小时）
5. 综合测试和验证
   - 性能对比测试
   - 长时间训练稳定性测试

---

## 🧪 测试清单

### 已通过的测试 ✅
- [x] StreamStateManager内存泄漏测试
- [x] LRU机制正确性测试
- [x] LRU状态保护测试
- [x] batch_idx归一化测试

### 待编写的测试 ❌
- [ ] checkpointing数值精度对比测试
- [ ] 分块注意力性能对比测试
- [ ] MemoryManager清理功能测试
- [ ] 梯度累积正确性测试

---

## 📈 预期效果

### 优化前（当前状态）
- 峰值显存：~8.73 GB（batch_size=1）
- CUDA OOM：频繁发生
- 训练稳定性：低

### 优化后（阶段1-4完成后）
- 峰值显存：~6 GB（预计）↓ 30%
- CUDA OOM：减少
- 训练稳定性：显著提升

### 优化后（全部阶段完成）
- 峰值显存：~5 GB（预计）↓ 43%
- CUDA OOM：极少发生
- 训练稳定性：高

---

**创建时间**：2026-02-09 18:50
**创建者**：Frank
**状态**：执行中（40%完成）
**下一步**：执行阶段3：注意力计算优化
