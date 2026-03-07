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
- 英文注释，中文文档
- 完成代码编写后，在文档的框架不变的情况下更新文档，如CLAUDE.md

### 测试编写原则
- 精简、干净、快速
- 核心关键逻辑或算法必须测试
- 需要加载transformer模型进行验证的测试与无需加载模型的测试分离
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
使用UV管理整个工程，pytest用于测试，justfile用于快捷命令，jujutsu用于版本管理。

### 目录说明

**核心模块**
- mini-nav/main.py — CLI 入口 (Typer)
- mini-nav/database.py — LanceDB 单例管理，用于向量存储与检索
- mini-nav/feature_retrieval.py — DINOv2 图像特征提取与检索

**源代码目录 (mini-nav/)**
- mini-nav/configs/ — 配置管理 (Pydantic + YAML)
- mini-nav/commands/ — CLI 命令 (train, benchmark, visualize, generate)
- mini-nav/compressors/ — 特征压缩算法
  - hash_compressor.py — 哈希压缩器
  - dino_compressor.py — DINO 压缩器
  - segament_compressor.py — 分割压缩器
  - pipeline.py — 压缩流水线
  - train.py — 压缩器训练
- mini-nav/data_loading/ — 数据加载与合成
  - loader.py — 数据加载器
  - synthesizer.py — 场景合成器
- mini-nav/utils/ — 工具函数
  - feature_extractor.py — 特征提取工具
- mini-nav/tests/ — pytest 测试集
- mini-nav/benchmarks/ — 基准测试 (recall@k)
- mini-nav/visualizer/ — Dash + Plotly 可视化应用

**数据目录**
- datasets/ — 数据集目录
- outputs/ — 默认输出目录 (数据库、模型权重等)

### Python库
详细可查询pyproject.toml或使用`uv pip list`获取详细的库信息，请基于目前的库实现功能。
如需添加新库，请先询问，用户确认后才能使用`uv add <package>`新增库。
