# 实验与基线 (Experiments & Baselines)

本目录用于存放主模型（V4 MobileViT + CharBERT）以外的所有对比工作。

## 目录指南

- **`baselines/`**: 存放通用基线模型（如 ResNet, LSTM, Transformer）。
- **`reproduction/`**: 存放针对特定论文（如 ATVITSC）的完整复现。
- **`comparisons/`**: 存放用于生成对比图表和汇总指标的分析脚本。

## 运行约定

1. **数据**: 请统一读取根目录下的 `ProcessedData/`。
2. **输出**: 实验产物请输出至 `outputs/experiments/<experiment_name>/`。
3. **指标**: 确保每个实验最后能产出符合项目规范的 `metrics.json`。
