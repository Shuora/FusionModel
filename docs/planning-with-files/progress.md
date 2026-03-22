# Progress

## 2026-03-18

- 读取并核对了 `README.md`、`docs/commands/session-full-experiments.md` 与当前实现代码（MobileViT / ET-BERT / stage1 / stage2）。
- 更新 `README.md`：
  - 改为“当前架构 + 环境 + 最小流程命令”口径。
  - 明确 MobileViT 本地 checkpoint 复用行为与 ET-BERT adapter 能力边界。
  - 修正仓库路径为 `/home/shuora/Traffic/FusionModel`，补充 `pip install -r requirements.txt`。
- 更新 `docs/commands/session-full-experiments.md`：
  - 命令参数与当前 `stage1_binary.py` / `stage2_multiclass.py` 保持一致。
  - 明确当前管线为 MobileViT + ET-BERT Adapter，并注明“非原始 UER ET-BERT 完整实现”。
- 更新 `docs/planning-with-files/findings.md`，记录本轮文档同步结论与当前 46 项测试验证范围。
- 执行并通过指定回归命令：`python -m pytest -q tests/data/test_etbert_feature_encoder.py tests/models/test_pretrained_backbones.py tests/models/test_fusion_model.py tests/pipeline/test_pipeline_data_protocol.py tests/pipeline/test_train_eval_report.py tests/pipeline/test_train_stage_dispatch.py tests/pipeline/test_stacking_pipeline.py tests/pipeline/test_moe_pipeline.py tests/pipeline/test_stage1_binary_protocol.py tests/pipeline/test_stage2_multiclass_protocol.py tests/pipeline/test_protocol_execution.py -q`，结果 `46 passed`。

## 2026-03-19

- 读取并更新了 `docs/planning-with-files/task_plan.md`，为本轮运行时能力改造补充计划。
- 按 TDD 增加并跑红了两类测试：
  - `train` 记录 `device` / `num_workers`
  - `evaluate` 在 CUDA 不可用时记录 CPU fallback
- 新增共享运行时解析模块：`src/runtime_device.py`。
- 更新 `src/train.py`：
  - 增加 `--device {auto,cpu,cuda}`
  - 增加 `--num-workers`
  - 训练与验证 batch 按解析后的 device 迁移
  - 将 `device_requested` / `device` / `device_fallback` / `num_workers` 写入 `config.yaml`
- 更新 `src/evaluate.py`：
  - 增加 `--device`
  - 默认复用训练配置中的设备偏好
  - CUDA 不可用时自动回退到 CPU，并在 `eval_*.json` 中记录
- 更新 `src/stacking.py` 与 `src/moe.py`，使其支持 `device` / `num-workers`。
- 更新 `src/experiments/stage1_binary.py` 与 `src/experiments/stage2_multiclass.py`，使 `--execute` 路径也能透传 `device` / `num-workers` 到 train/evaluate。
- 更新 `docs/commands/session-full-experiments.md`：
  - 主命令显式加入 `--device auto`
  - 主训练命令显式加入 `--num-workers 4`
  - 增加针对 `RTX 4060 Laptop 8GB + i7-13700 + 8GB RAM` 的推荐参数说明
- 执行并通过回归命令：`pytest -q tests/pipeline/test_train_eval_report.py tests/pipeline/test_train_stage_dispatch.py tests/pipeline/test_protocol_execution.py tests/pipeline/test_stacking_pipeline.py tests/pipeline/test_moe_pipeline.py`，结果 `23 passed`。

## 2026-03-20

- 读取并核对论文原文 `docs/paper/MVTBA A Novel Hybrid Deep Learning Model for Encrypted Malicious Traffic Identification.pdf` 第 10-12 页。
- 确认论文 stage1 / Exp. I 协议为：
  - `ISCX + MTA + MFCP`
  - 不含 `USTC`
  - 按 Table 1-3 的类别/家族与 train/test 配额构造数据
- 确认当前 `src/experiments/stage1_binary.py` 仍是近似论文的白名单实现，尚未严格按表 1-3 构造 manifest。
- 新增规格文档：
  - `docs/superpowers/specs/2026-03-20-stage1-paper-protocol-design.md`
- 新增实现计划：
  - `docs/superpowers/plans/2026-03-20-stage1-paper-protocol.md`
- 更新 `src/experiments/stage1_binary.py`：
  - 引入论文 Table 1-3 驱动的协议配置
  - 按 `capture_id` / `family` 与 `split=train|test` 精确裁样
  - 样本不足时直接报错并输出缺口信息
  - 删除“paper subset 匹配不到则 fallback 到全量”的行为
- 更新 `tests/pipeline/test_stage1_binary_protocol.py`：
  - 新增 `PUA` 与精确配额测试
  - 使用最小论文协议夹具更新旧测试
- 更新 `tests/pipeline/test_protocol_execution.py`：
  - 将 stage1 execute smoke tests 改为最小论文协议配额
  - 显式传入 `--num-workers 0`，避免无关的多进程 worker 干扰执行测试
- 执行并通过回归命令：
  - `pytest -q tests/pipeline/test_stage1_binary_protocol.py`，结果 `10 passed`
  - `pytest -q tests/pipeline/test_protocol_execution.py -k 'stage1_binary_execute_runs_train_eval_report or stage1_binary_execute_stacking_reports_stacking_metrics or stage1_binary_execute_moe_reports_moe_metrics'`，结果 `3 passed`
- 继续完善数据层口径兼容：
  - `pcap_sessionizer` 支持 `raw IP` linktype 与 `session_full` 下的 `UDP`
  - `session_splitcap` 支持 `raw IP` / `UDP` session 切分，并保留原始 linktype
  - `preprocess` 在 `session_full` 下用 `include_udp=True`
- 新增数据层回归测试：
  - `tests/data/test_pcap_sessionizer.py`
  - `tests/data/test_session_splitcap.py`
- 引入 `paper_balanced` 协议：
  - `src/experiments/stage1_binary.py` 增加 `--protocol-mode {paper_strict,paper_balanced}`
  - 默认切换到 `paper_balanced`
  - `paper_balanced` 下对超大组做上限裁样，对不足组全保留，对缺失组跳过并输出 summary
- 更新 `tests/pipeline/test_stage1_binary_protocol.py`，增加 `paper_balanced` 缺失组/裁样/不足组测试。
- 更新 `tests/pipeline/test_protocol_execution.py`，兼容 `protocol_mode` 参数后的默认执行行为。
- 执行并通过回归命令：
  - `/home/shuora/miniconda3/envs/FusionModel/bin/python -m pytest -q tests/data/test_pcap_sessionizer.py tests/data/test_session_splitcap.py tests/data/test_preprocess_pipeline.py tests/data/test_preprocess_runner.py tests/data/test_session_full_filtering.py`，结果 `20 passed`
  - `/home/shuora/miniconda3/envs/FusionModel/bin/python -m pytest -q tests/pipeline/test_stage1_binary_protocol.py tests/pipeline/test_protocol_execution.py -k 'stage1_binary'`，结果 `19 passed`

## 2026-03-21

- 对 `runs/stage1-binary` 做只读排查，核对了：
  - `src/experiments/stage1_binary.py`
  - `src/train.py`
  - `src/evaluate.py`
  - `src/pipeline_data.py`
  - `src/data/preprocess.py`
  - `src/data/dataset_inventory.py`
  - `runs/stage1-binary/{config.yaml,eval_test.json,metrics.csv,train.log,report.md}`
  - `runs/stage1-binary/figures/confusion_matrix_test.csv`
  - `outputs/protocol/stage1_binary_manifest.csv`
- 确认 `0.9642` 的来源：
  - 来自 `eval_test.json` 的 `top1`
  - 对应 `src/evaluate.py` 中的 `accuracy_score`
  - 不是 `macro_f1`
- 确认 checkpoint 选择协议：
  - `best.ckpt` 按 `val_macro_f1` 选择
  - best epoch 为 `26`
  - best val macro-F1 为 `0.9556`
- 汇总当前 manifest / run 分布：
  - manifest 总样本 `69144`
  - 总类别分布 `0:21290, 1:47854`
  - test 样本 `13815`
  - test 混淆矩阵 `[[4121,124],[371,9199]]`
- 识别出当前 run 的关键解释边界：
  - 口径是 `session_full + paper_balanced`
  - train 时无显式 val split，`src/train.py` 从 train 内再切 `10%` 做 val
  - manifest 中存在 `42` 个 `dataset+capture_id` 同时落在 train/test，说明结果含有 capture 级 leakage 风险
- 结论：
  - 没发现直接把结果打坏的实现 bug
  - 更需要提醒的是指标误读风险与协议不严格带来的可比性问题

## 2026-03-21

- 为“论文指标计算方式是否与仓库一致”建立专门核对任务，更新了 `docs/planning-with-files/task_plan.md`。
- 已确认当前仓库主评估入口 `src/evaluate.py` 使用的是分类指标而非回归指标：
  - `top1`
  - `macro_precision`
  - `macro_f1`
  - `macro_recall`
  - `confusion_matrix`
- 已确认 `src/stacking.py` 与 `src/moe.py` 也沿用分类任务口径：
  - `top1`
  - `macro_f1`
  - `macro_recall`
- 已从论文 4.2/4.3/4.5 节抽取出指标定义、实验设置与表 5/6/7 的展示口径。
- 已完成论文与仓库指标对照，当前结论是：
  - `accuracy` ≈ 仓库 `top1`
  - `macroP / macroR` ≈ 仓库 `macro_precision / macro_recall`
  - 论文 `macroF1` 公式与仓库 `sklearn macro_f1` 不完全相同
  - 仓库二分类阶段仍输出 macro 指标，与论文 Exp. I 的 `Precision / Recall / F1` 口径不严格一致
  - 仓库默认评估 `best.ckpt`，论文描述为固定训练 `30` 轮，评估流程也不完全一致
- 在独立 worktree `paper-metrics-compat` 中实现了“双口径指标”并已同步回当前 `dev` 工作区。
- 按 TDD 先新增并跑红：
  - `compute_classification_metrics_includes_paper_compatible_macro_f1`
  - `write_ablation_summary_collects_run_metrics`
- 已实现：
  - `src/evaluate.py` 新增双口径指标计算 helper 与 `paper_*` 输出
  - `src/report.py` 新增 `Paper-Compatible Metrics` 展示
  - `src/ablation.py` 新增 `paper_macro_*` 汇总列
- 已验证通过：
  - `pytest -q tests/pipeline/test_train_eval_report.py -k compute_classification_metrics_includes_paper_compatible_macro_f1`
  - `pytest -q tests/pipeline/test_ablation_plan.py -k write_ablation_summary_collects_run_metrics`
  - `python -m py_compile src/evaluate.py src/report.py src/ablation.py`
  - 手工最小化 `report_main` 验证：`report.md` 已包含 `Paper-Compatible Metrics` 与 `Paper Macro-F1`

## 2026-03-21

- 针对用户反馈“run 报告里没有混淆矩阵分类表和 classification report 表格”，已完成只读根因定位：
  - `src/evaluate.py` 只会输出 summary json 与 confusion matrix csv/png
  - `src/report.py` 只会把 artifact 路径列进 `report.md`
  - 当前缺失表格是实现空缺，不是 run 失败
- 本轮实现目标收敛为：
  - 新增 `classification_report_<split>.csv/json`
  - 在 `report.md` 中直接渲染 confusion matrix 与 classification report Markdown 表格
- 按 TDD 先补了快速单测并跑红：
  - `test_evaluate_writes_classification_report_artifacts`
  - `test_evaluate_fallback_writes_classification_report_with_effective_split`
  - `test_report_renders_confusion_matrix_and_classification_tables`
  - `test_report_discovers_eval_val_and_renders_tables`
- 更新 `src/evaluate.py`：
  - 新增 `classification_report` 计算
  - 输出 `classification_report_<split>.csv/json`
- 更新 `src/report.py`：
  - 将 confusion matrix 渲染为 Markdown 表
  - 将 classification report 渲染为 Markdown 表
  - artifact 列表中加入 classification report csv/json
- 执行并通过快速回归命令：
  - `/home/shuora/miniconda3/envs/FusionModel/bin/python -m pytest -q tests/pipeline/test_train_eval_report.py -k 'evaluate_writes_classification_report_artifacts or evaluate_fallback_writes_classification_report_with_effective_split or report_renders_confusion_matrix_and_classification_tables or report_discovers_eval_val_and_renders_tables or report_falls_back_to_stacking_metrics_when_eval_missing or report_falls_back_to_moe_metrics_when_eval_missing'`
  - 结果：`6 passed, 6 deselected`
- 用新代码重跑现有 `runs/stage1-binary` 的 `evaluate + report`：
  - 成功生成 `runs/stage1-binary/figures/classification_report_test.csv`
  - 成功生成 `runs/stage1-binary/figures/classification_report_test.json`
  - 成功更新 `runs/stage1-binary/report.md`，新增：
    - `## Confusion Matrix`
    - `## Classification Report`

## 2026-03-22

- 在独立 worktree `/.worktrees/chinese-logs` 中执行日志中文化改造，避免影响主工作区。
- 更新 `src/common/structured_logging.py`：
  - level/module 文案改为中文。
  - 增加 event 中文映射，并输出 `中文说明 (event_code)` 双语样式。
  - 未命中的 module 使用 `模块:<name>` 统一展示。
- 更新 `src/experiments/stage1_binary.py`：
  - 主要流程日志翻译为中文（构建 manifest、训练/评估/报告步骤、跳过原因等）。
  - 日志前缀调整为 `[Stage1Binary][阶段1协议]`。
- 更新 `src/ablation.py`：
  - `plan/summary` 输出日志翻译为中文。
- 同步更新断言测试：
  - `tests/common/test_structured_logging.py`
  - `tests/data/test_preprocess_pipeline.py`
  - `tests/pipeline/test_stage1_binary_protocol.py`
- 执行并通过针对性回归：
  - `pytest -q tests/common/test_structured_logging.py tests/data/test_preprocess_pipeline.py::test_preprocess_source_writes_expected_outputs tests/pipeline/test_stage1_binary_protocol.py::test_stage1_main_emits_progress_logs`
  - 结果：`4 passed`
- 额外执行快速兼容性检查，确认 `config_summary` 等英文 event code 仍在日志文本中保留。

- 在独立 worktree `.worktrees/stage1-acc-fix` 创建分支 `fix-stage1-acc-report`，隔离本次修复。
- 已完成问题复盘：
  - 最新二分类 test 指标约 `96.5%`；
  - 训练日志缺少阈值口径 acc；
  - `report` best row 忽略 `best_metric` 配置。
- 已建立本轮实施计划，下一步按 TDD 先补失败测试再改实现。
- 已按 TDD 新增并跑红：
  - `test_report_selects_best_epoch_by_configured_best_metric`
  - `test_train_records_val_acc_at_decision_threshold`
- 已完成实现：
  - `src/train.py` 增加 `val_acc_at_decision_threshold` 计算、日志与 CSV 输出；
  - `src/train.py` 增加 `--best-metric` 与 best checkpoint 选择逻辑；
  - `src/report.py` 按 `config.best_metric` 选 best epoch，并展示阈值口径 acc。
- 已完成验证：
  - `pytest -q tests/pipeline/test_train_eval_report.py -k 'report_selects_best_epoch_by_configured_best_metric or train_records_val_acc_at_decision_threshold'` 通过；
  - `python -m py_compile src/train.py src/report.py` 通过；
  - 端到端快速链路：
    - `python -m src.train ... --run-id stage1-binary-e2e-accfix-fast --train-max-samples 256 --num-workers 0`
    - `python -m src.evaluate --run-dir runs/2026-03-22/stage1-binary-e2e-accfix-fast --split test --device cpu`
    - `python -m src.report --run-dir runs/2026-03-22/stage1-binary-e2e-accfix-fast`
  - 产物核对通过：train.log 与 report.md 均出现新增字段。

## 2026-03-22

- 新建 worktree `.worktrees/attention-fusion` 与分支 `feat-attention-fusion`，用于注意力融合改造。
- 已完成现状确认：
  - `fusion_model` 当前为 gate 加权融合，不是 attention 融合；
  - `stacking/moe/train/evaluate` 均依赖 `out["gate"]` 字段，改造需保持兼容输出。
- 已建立改造计划，下一步开始落地 attention 融合实现与参数透传。
- 已完成代码改造：
  - `fusion_model` 从 gate 线性融合切换为 query+cross-attention 融合；
  - 保留 `gate` 输出兼容下游；
  - `num_heads` 参数同步到 train/evaluate/stacking/moe 模型构造。
- 已完成验证：
  - `python -m py_compile src/models/fusion_model.py src/train.py src/evaluate.py src/stacking.py src/moe.py tests/models/test_fusion_model.py`
  - `pytest -q tests/models/test_fusion_model.py`（`4 passed`）
  - `pytest -q tests/pipeline/test_train_eval_report.py -k train_records_val_acc_at_decision_threshold`（`1 passed`）
  - 自定义 smoke（`num_workers=0`）：
    - train + stacking 全链路通过
    - train + moe 全链路通过
