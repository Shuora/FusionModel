# Paper-Compatible Metrics Design

## Goal

在保留当前工程主口径（`sklearn` 风格 `macro_f1`）的前提下，为评估产物补充一套与 MVTBA 论文更接近的兼容指标，便于论文对照和工程使用并存。

## Scope

- `src/evaluate.py`
- `src/report.py`
- `src/ablation.py`
- `tests/pipeline/test_train_eval_report.py`
- `tests/pipeline/test_ablation_plan.py`

## Design

### 主口径保持不变

- 继续保留现有字段：
  - `top1`
  - `macro_precision`
  - `macro_recall`
  - `macro_f1`
- 训练阶段的 `val_macro_f1`、best checkpoint 选择逻辑不变，避免影响当前训练/调参行为。

### 新增论文兼容口径

- 在评估输出中新增：
  - `paper_precision`
  - `paper_recall`
  - `paper_f1`
  - `paper_macro_precision`
  - `paper_macro_recall`
  - `paper_macro_f1`
- 定义：
  - 二分类时，`paper_precision / paper_recall / paper_f1` 使用正类二分类口径。
  - 多分类时，`paper_macro_precision / paper_macro_recall` 为宏平均 precision/recall。
  - `paper_macro_f1 = 2 * paper_macro_precision * paper_macro_recall / (paper_macro_precision + paper_macro_recall)`。

### 报告与汇总

- `report.py` 优先展示工程主口径，并额外展示 paper-compatible 指标。
- `ablation.py` 读取并输出新增 paper-compatible 字段，但保持旧列兼容。

### 测试策略

- 先补测试验证新增字段存在。
- 增加一个多分类夹具，验证 `paper_macro_f1` 与当前 `macro_f1` 在特定样例下可不同。
- 回归 `report` / `ablation` 读取逻辑，确保旧字段不回归。
