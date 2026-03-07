# Project Spec & Rules

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
  - hash_compressor.py — 哈希压缩器与训练loss
  - pipeline.py — 压缩流水线（整合 DINO 特征提取）
  - train.py — 压缩器训练脚本
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

## 版本管理 (Jujutsu 特有)
本项目使用 Jujutsu (jj) 进行版本控制，并配套 Memorix MCP 作为架构决策与思维轨迹的持久化中心。

- 技能调用: 必须使用 jujutsu 相关工具技能来执行分支、提交、修改（describe）等操作，禁止直接通过 Shell 执行冗长的 Git 兼容指令。
- 描述规范 (jj desc):
  - 执行 jj desc 时，首行必须是精简的变更标题。
  - 空一行后，仅记录改动的核心业务点。
  - 语言使用英文进行描述
  - 禁忌: 禁止在 jj 描述中堆砌复杂的算法逻辑或长篇的设计决策。
- 记忆联动 (Memorix 优先):
  - 凡涉及架构变更、算法决策或重构逻辑，在执行 jj desc 之前，必须先调用 memorix_store (或对应的添加方法)。
  - 关联标记: 在 Memorix 的存储记录中，必须强制包含当前变更的 jj change ID，以便实现从代码变更到思维链的完美映射。
  - 检索逻辑: 在处理需要深入理解上下文的任务时，应主动调用 memorix_search 检索相关的历史 change_id 决策。
- 无感记录原则:
  - 严禁在工程目录下生成任何独立的 change_log.md 或 AI 自动化文档。
  - 所有关于“为什么这样改”的知识，应当流向 jj 的原子化提交描述或 Memorix 的知识图谱库。

### 描述示例
```text
refactor(compressors): Simplify module by removing SAM/DINO separation code

- Remove dino_compressor.py and segament_compressor.py
- Rewrite pipeline.py to inline DINO into HashPipeline
- Maintain backward compatibility: SAMHashPipeline alias
- Update tests and benchmark.py
```

### 提交步骤
- 执行`jj diff --no-pager`获取当前所有更改
- 根据更改内容，与openspec生成的相关文档进行总结，重点在于更改内容及其决策逻辑
- 调用记忆功能，如Memorix记忆先前总结的内容
- 遵循描述规范，使用jj进行更改的描述
- 执行`jj new`开启一个新的更改

## 记忆管理 (Memorix MCP)
本项目使用 Memorix 作为核心上下文引擎，用于存储架构决策、复杂逻辑关联和历史重构原因。

### 记忆写入准则
- 主动记录: 在完成以下操作后，必须调用 `memorix.store`：
  - 用户确认后的核心架构变更（例如：LanceDB 的索引策略）。
  - 复杂的 bug 修复逻辑（记录“为什么”这么修，防止回滚）。
  - 用户在对话中表达的明确偏好（例如：对特定 Python 库的厌恶）。
  - 代码的修改及其决策逻辑(例如：对于用户特定需求导致的更改)。
- 结构化存储: 存储时请使用 `[Category: Topic] Description` 的格式，确保检索效率。

### 记忆检索准则
- 冷启动检索: 每一轮新对话开始或切换到新任务时，优先调用 `memorix.search` 关键词（如 "project_architecture", "database_schema"），以确保不偏离既有设计。
- 防止幻觉: 如果对某个旧功能的实现细节不确定，先检索记忆，禁止凭空猜测。

### 内存与冗余控制
- 精简描述: 存入 Memorix 的信息必须精简，严禁存入整段代码块，仅存储“逻辑描述”和“决策依据”。
- 清理逻辑: 发现记忆库中存在与当前代码事实冲突的旧信息时，应主动提示用户进行更新或覆盖。
