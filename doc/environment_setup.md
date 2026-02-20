# 环境配置记录

## Spconv 版本升级

### 日期
2026-02-20

### 升级内容
- **升级前**: spconv-cu111 2.1.21
- **升级后**: spconv-cu111 2.1.25

### 升级原因
解决 spconv 底层 CUDA kernel 在某些稀疏卷积配置下的错误：
```
RuntimeError: /tmp/pip-build-env-r1c_rjmt/overlay/lib/python3.8/site-packages/cumm/include/tensorview/cuda/launch.h(53)
```

### 升级命令
```bash
source /home/cwh/miniconda3/etc/profile.d/conda.sh
conda activate former3d
pip uninstall spconv-cu111 -y
pip install spconv-cu111
```

### 验证
```python
import spconv
print(spconv.__version__)  # 应该输出: 2.1.25
```

## 当前环境关键包版本

```
spconv-cu111: 2.1.25
cumm: 0.7.11
torch: (通过conda安装)
```

## 注意事项
1. spconv-cu121 在 PyPI 上不可用，继续使用 cu111 版本
2. 如果遇到 CUDA 版本不兼容，需要重新编译 spconv
