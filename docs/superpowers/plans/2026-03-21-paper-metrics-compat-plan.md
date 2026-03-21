# Paper-Compatible Metrics Implementation Plan

1. 在评估测试中先声明新字段与 paper-compatible `macro_f1` 差异预期，并跑出失败。
2. 在 `src/evaluate.py` 中实现 paper-compatible 指标计算与 JSON 输出。
3. 更新 `src/report.py` 和 `src/ablation.py` 读取/展示新增字段。
4. 跑针对性 pytest，确认评估、报告、ablation 回归通过。
5. 更新 `docs/planning-with-files` 记录实现与验证结果。
