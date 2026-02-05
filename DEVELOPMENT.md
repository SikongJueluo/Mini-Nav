# 开发者必看文档

## 代码规范

### Google风格代码
详细参阅：https://raw.githubusercontent.com/shendeguize/GooglePythonStyleGuideCN/refs/heads/master/README.md

### 代码编写要求
编写的代码要求：
- 简洁，清晰易懂，最小化实现
- 条件或循环分支不能超过三层，提前Return以减少分支的出现
- 变量说明注释、条件或循环分支注释完全
- 无需向后兼容，避免添加过多功能
- 先编写测试集，再实现代码
- 实现测试集后，先询问意见，在用户修改完成测试集后再实现代码

### 文档更新说明
仅在工程目录变化时，更新此文档的目录说明部分。
如需修改其他部分，请先询问，在进行修改。

## 工程说明
使用UV管理整个工程，pytest用于测试，justfile用于快捷命令。

### 目录说明

- mini-nav 为源代码目录
  - mini-nav/configs 为配置文件管理目录，使用python + yaml进行统一的配置管理
  - mini-nav/tests 为pytest测试集目录，用于管理各项测试集
  - mini-nav/visualizer 为plotly、dash的简单数据可视化APP
  - mini-nav/database.py 用于管理lancedb数据库
  - mini-nav/feature_retrieval.py 用于实现图像特征检索
  - mini-nav/main.py 为主程序入口
- outputs 为默认输出目录

### Python库
详细查询pyproject.toml或使用`uv pip list`获取详细的库信息，请基于目前的库实现功能。
如需添加新库，请先询问，再新增。