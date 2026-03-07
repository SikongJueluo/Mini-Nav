# 开发者必读文档

## 代码规范

### Google风格代码
详细参阅：https://raw.githubusercontent.com/shendeguize/GooglePythonStyleGuideCN/refs/heads/master/README.md

### 代码编写原则
- 简洁，清晰易懂，最小化实现
- 条件或循环分支不能超过三层，提前Return以减少分支的出现
- 变量说明注释、条件或循环分支注释完全
- 无需向后兼容，避免添加过多功能
- 先编写测试集，再实现代码
- 实现测试集后，先询问用户意见，用户确认后才能继续
- 如非用户要求，无需编写基准测试代码
- 英文注释

### 测试编写原则
- 精简、干净、快速
- 核心关键逻辑或算法必须测试
- 无需编写测试集的情况
  - UI界面相关的代码
  - 过于复杂或耗时的逻辑
  - 基准测试相关

### 关键词说明
- 确认：用户认同当前的实现方案或测试集实现，即可以开始工作
- 继续：用户需要你重读上下文，继续未完成的工作

### 文档更新说明
仅在工程目录变化时，更新此文档的目录说明部分。
如需修改其他部分，请先询问，在进行修改。

## 工程说明
使用UV管理整个工程，pytest用于测试，justfile用于快捷命令。

### 目录说明

- mini-nav 为源代码目录
  - mini-nav/configs 为配置文件管理目录，使用python + yaml进行统一的配置管理
  - mini-nav/commands 为CLI命令管理目录，用于管理各种命令
  - mini-nav/tests 为pytest测试集目录，用于管理各项测试集
  - mini-nav/benchmarks 为pytest-benchmark基准测试目录，用于管理各项基准测试，包括速度、准确度等内容
  - mini-nav/visualizer 为plotly、dash的简单数据可视化APP
  - mini-nav/database.py 用于管理lancedb数据库
  - mini-nav/feature_retrieval.py 用于实现图像特征检索
  - mini-nav/main.py 为主程序入口
- outputs 为默认输出目录

### Python库
详细可查询pyproject.toml或使用`uv pip list`获取详细的库信息，请基于目前的库实现功能。
如需添加新库，请先询问，用户确认后才能使用`uv add <package>`新增库。
