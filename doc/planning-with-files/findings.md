# Findings & Decisions

## Requirements
- 用户要求按 `executing-plans` 开始实施，按批次交付并等待反馈。
- 当前批次目标：完成 Day4~Day6 核心功能并补齐训练日志/进度条规范。

## Research Findings
- 当前 worktree 基于已提交快照，不包含主工作区的未跟踪 `SourceData` 与 `doc/plans`。
- 通过复制计划文件到 worktree，确保执行上下文一致。
- `pytest -q` 在该仓库无默认测试；需要按任务级测试文件执行。
- 环境依赖：`dpkt` 与 `pandas` 可用，`pyarrow/fastparquet` 不可用，无法直接写 parquet。
- `SourceData` 实际包含 `CICAndMal2017`、`MFCP`、`USTC-TFC2016` 三个目录，存在 `*.pcap:Zone.Identifier` 需忽略。
- `numpy` 与 `PIL` 可用，满足 `npz` 特征落盘与抽样图扩展需求。
- `torch/yaml/matplotlib/sklearn` 可用，满足 Day3 训练评估报告骨架实现。
- `MultiheadAttention` 在当前 CPU 环境可稳定运行，满足 Tiny 融合模型单测与 smoke test。
- `xgboost` 在 `FusionModel` conda 环境可用，满足 stacking 元学习器训练。
- `pytest -q` 在 `FusionModel` conda 环境全量通过（31 passed, warnings 非阻塞）。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| `format_log_line` 统一输出 `{time}|{level}|{module}|{event}|kv` | 对齐计划中的结构化日志要求 |
| `classify_session_as_tls` 返回 `(bool, reason)` | 直接支持 `non_tls_dropped.parquet` 的 drop reason |
| `build_output_paths` 固定生成计划路径模板 | 确保产物目录与计划一致 |
| `split_tls_and_non_tls` 输出 accepted/dropped | 先打通过滤与审计链路 |
| `scan_source_pcaps + split_by_capture + detect_capture_leakage` 形成 Day1 主链 | 满足 strict 评估前置条件 |
| `read_tcp_sessions` 使用 5-tuple 双向归并 | 满足 session 级处理粒度 |
| 表格落盘统一 parquet 优先、csv 回退 | 在当前环境保持可运行 |
| `preprocess_source` 统一编排数据阶段并输出 icon 日志 | 与计划中 5.4.1/5.4.2 对齐 |
| 新增 `feature_encoder` 生成 `rgb[3,28,28]` 与 `token_ids/mask/segment` | 对齐计划 Day2 输出格式 |
| `preprocess_source` 增加 `filter_mode` 参数 | 支持 policy 标签与过滤策略分离 |
| 新增 `run_preprocess_policies` 支持 `strict/full` 批处理 | 满足双口径跑批需求 |
| 新增 `pipeline_data.load_policy_data` 聚合 rgb/seq/manifest | 统一 train/evaluate 数据入口 |
| 训练阶段先使用 `nn.Linear` 基线 | 快速打通 run 产物与评估报告闭环 |
| 报告阶段默认生成 learning curve 图 | 先满足最小可视化要求，再扩展混淆图等图表 |
| 新增 `load_policy_multimodal_data` 返回 `rgb/token_ids/attention_mask` | 支撑融合模型端到端训练 |
| `TinyFusionClassifier` 采用 image conv + token embed + 双向 cross-attn + gate | 对齐 Day4 融合机制核心要求 |
| `train.py` 用 `stage` 切换 warmup/fusion 损失 | 保持与计划中的分阶段训练一致 |
| `train.py` 新增启动日志（git commit/config 摘要/数据集统计） | 对齐用户提出的 5.4.1 启动日志规范 |
| `train.py` 新增 train/val 双 `tqdm` | 对齐用户提出的 5.4.2 实时进度展示规范 |
| 引入 NaN/梯度异常显式处理 | 满足“异常显式报错并终止或降级处理”要求 |
| `stage=stacking|moe` 在 train 结束后自动调度子流程 | 对齐计划中的统一入口设计 `src/train.py --stage ...` |
| `ablation` 新增 summary 模式，自动聚合各 run 指标 | 对齐计划“消融输出自动汇总总表”要求 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| `ModuleNotFoundError: src` during pytest | 新增 `tests/conftest.py` 添加 repo root 到 `sys.path` |
| 无 parquet 引擎导致 `to_parquet` 失败风险 | 增加统一 fallback，输出 `*.csv` |
| 新增 `full` policy 时过滤模式语义不清 | 用 `DEFAULT_POLICY_FILTER_MAP` 显式映射处理 |
| `torch` 在本机触发 CUDA 初始化 warning | 保持 CPU 路径执行，测试通过后记录 warning 即可 |
| 融合模型替换后旧 smoke test 约束不够 | 扩展测试断言 `model_type` 与 `gate_mean` |
| `stage=stacking|moe` 原实现仅训练主模型、不产出子流程结果 | 新增阶段调度测试并在 `train.py` 增加 dispatch 逻辑 |

## Resources
- `doc/plans/2026-02-23-tls-family-classification-delivery-plan.md`
- `src/common/structured_logging.py`
- `src/data/tls_filter.py`
- `src/data/build_dataset.py`
- `src/data/dataset_inventory.py`
- `src/data/pcap_sessionizer.py`
- `src/data/preprocess.py`
- `src/data/feature_encoder.py`
- `src/data/preprocess_runner.py`
- `src/pipeline_data.py`
- `src/train.py`
- `src/evaluate.py`
- `src/report.py`
- `src/models/fusion_model.py`
- `src/stacking.py`
- `src/moe.py`
- `src/ablation.py`
- `tests/models/test_fusion_model.py`
- `tests/pipeline/test_stacking_pipeline.py`
- `tests/pipeline/test_moe_pipeline.py`
- `tests/pipeline/test_ablation_plan.py`
- `tests/pipeline/test_train_stage_dispatch.py`
