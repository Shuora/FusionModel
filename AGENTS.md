# Repository Guidelines

## 项目结构与模块组织
- `src/pipeline/`：数据预处理、会话切分与数据集构建。
- `src/fusion/`：训练与实验入口（attention、attention + stacking）。
- `configs/`：统一管理数据与训练 profile。
- `SourceData/` 放原始 `pcap`，`dataset/` 放可训练输入，`outputs/` 放日志、模型和评估图表，`doc/` 放方案文档。

## 构建、测试与开发命令
- 初始化环境：
- 数据预处理：
- 训练命令：

## 代码风格与命名约定
- Python 使用 4 空格缩进，遵循 PEP 8，新增函数尽量补类型注解。
- 变量/函数/文件使用 `snake_case`，类名使用 `PascalCase`，常量使用 `UPPER_SNAKE_CASE`。
- 配置项命名应与现有 profile 保持一致（如 `dataset_name`、`class_balance`、`attention_dim`）。

## 测试与验证规范
- 当前仓库没有独立 `tests/` 目录，提交前至少完成最小验证链路。
- 运行一个小规模预处理 profile，检查 `dataset/<name>/image_data` 与 `pcap_data` 是否成对生成。
- 启动一次训练冒烟，确认 `outputs/` 下生成 `logs/*.log`、`report_*.md`、`confusion_matrix_*.png`、`metrics_curve_*.png`。

## 提交与 Pull Request 规范
- 提交信息建议沿用历史风格：`feat:`、`fix:`、`docs:`、`chore:`（可接中文说明）。
- 每个 commit 聚焦单一目的；功能改动与配置改动尽量分开提交。
- PR 需写明：变更摘要、影响的 profile/数据集、复现命令、关键指标变化（Acc/Macro-F1），必要时附 `outputs/archive/...` 路径。

## 数据与安全提示
- 不提交原始数据、缓存和大体积训练产物；仅提交可复现所需代码与配置。
- 路径优先使用相对路径，避免硬编码本机绝对路径。
- 修改 `configs/*.yaml` 时注明默认行为变化，避免破坏既有实验流程。


## Planning with Files

<IMPORTANT>

对于复杂任务（3个以上步骤、研究或项目）

1. 阅读技能文档：`cat ~/.codex/skills/planning-with-files/SKILL.md`
2. 在项目目录中创建以下文件：
   - doc/planning-with-files/task_plan.md（任务计划）
   - doc/planning-with-files/findings.md（发现 / 调研结果）
   - doc/planning-with-files/progress.md（进度）
3. 在整个任务过程中始终遵循这三个文件的模式

</IMPORTANT>