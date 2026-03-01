# Findings & Decisions

## Requirements
- 用户要求按 `executing-plans` 开始实施，按批次交付并等待反馈。
- 当前批次目标：完成 Day4~Day6 核心功能并补齐训练日志/进度条规范。
- 新增目标：引入论文口径 `session_full` 预处理与两阶段评估（阶段1混合二分类 + 阶段2三数据集独立多分类）。

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
- 现有 `preprocess` 仅支持 `strict/relaxed` 过滤语义，`full` 仍映射到 strict，不满足“非 TLS 保留”。
- 现有流水线不落盘 session pcap，也没有自动清理 `tmp_sessions` 机制。
- 现有输出虽然有 `debug/preview_png` 目录约定，但尚未在编码阶段真实写入抽检 PNG。
- `session_full` 落地后，预处理将为每个原始 pcap 先生成 `tmp_sessions/<capture>/...pcap`，提特征后默认自动清理。
- `session_full` manifest 已新增 `is_tls_ssl` 与 `tls_ssl_reason` 字段，可追踪非TLS保留样本。
- 入口兼容性：`python src/data/preprocess_runner.py ...` 在无 `PYTHONPATH` 时会导入失败，需要在脚本入口兜底 `sys.path`。
- 原 `README` 中存在与当前 `load_policy_multimodal_data` 目录约定不一致的命令（`--processed-root` 指向数据集子目录）；已统一为“指向包含数据集目录的一层根目录”。

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
| 新增策略名 `session_full` | 明确表达“按会话全量保留”的论文口径，不与 `full` 混淆 |
| `session_full` 数据链采用 `PCAP -> Session PCAP -> 特征提取` | 与论文处理链保持一致，并满足用户要求真实落盘 session pcap |
| `session_full` 默认 `cleanup_sessions=True` | 提特征后清理 session pcap，降低磁盘占用 |
| 保留抽检 `preview_png` | 保障可解释性和质量抽检，不做全量图像存储 |
| 阶段1标签固定：`ISCX=normal`，`MFCP/MTA/USTC=malicious` | 与已确认协议一致，缺任一数据集直接失败 |
| 阶段2固定三任务：`MTA-7`、`MFCP-6`、`USTC-10` | 避免动态口径导致论文复现实验不可比 |
| 新增 `src.data.session_splitcap` | 独立封装 Session PCAP 切分与清理，避免在 `preprocess.py` 堆积复杂逻辑 |
| `save_feature_shards` 增加 preview 抽样落盘能力 | 满足“保留抽检RGB图像”并限制每类样本数 |
| 新增 `src.experiments.stage1_binary/stage2_multiclass` | 将两阶段协议固化为可执行 CLI，而非仅文档描述 |
| 在 `preprocess_runner.py/preprocess.py` 添加脚本入口 `sys.path` 兜底 | 兼容用户常用的脚本路径运行方式，消除 `ModuleNotFoundError: src` |
| 重写 `README` 为 `session_full` 主线命令手册 | 保证用户按文档可直接复现，不再混淆 strict/full 老口径 |

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
- `docs/plans/2026-02-23-session-full-mvtba-design.md`
- `docs/plans/2026-02-23-session-full-mvtba-implementation-plan.md`
- `src/data/session_splitcap.py`
- `tests/data/test_session_splitcap.py`
- `tests/data/test_session_full_schema.py`
- `tests/data/test_session_full_filtering.py`
- `tests/data/test_preview_and_cleanup.py`
- `src/experiments/stage1_binary.py`
- `src/experiments/stage2_multiclass.py`
- `tests/pipeline/test_stage1_binary_protocol.py`
- `tests/pipeline/test_stage2_multiclass_protocol.py`
- `docs/commands/session-full-experiments.md`

---

## 2026-03-01 仓库运行标准体检（进行中）

### 新发现
- 根目录未发现 `requirements.txt`、`environment.yml`、`pyproject.toml` 等依赖锁定文件。
- `README.md` 已提供完整运行命令，当前项目默认依赖 `conda activate FusionModel`。
- `src/` 下训练与预处理入口均存在：`src.data.preprocess_runner`、`src.train`、`src.evaluate`、`src.report`。
- `SourceData` 已存在且包含 ISCX 相关目录（此前核验为 `ISCX-VPN-NonVPN-2016`）。

### 待确认
- README 里的 `SourceData/ISCX` 与实际目录命名是否一致（可能存在别名或不一致风险）。
- 当前 shell 所在环境是否确为 `FusionModel` conda 环境，及关键依赖版本是否满足。

### 运行体检结果（已验证）
- `python` 实际解释器：`/home/shuora/miniconda3/envs/FusionModel/bin/python`（3.9.23）。
- 关键依赖均可导入：`dpkt/numpy/pandas/PIL/torch/sklearn/matplotlib/yaml/xgboost/tqdm`。
- 关键 CLI 均可启动：
  - `python -m src.data.preprocess_runner --help`
  - `python -m src.train --help`
  - `python -m src.evaluate --help`
  - `python -m src.report --help`
  - `python -m src.experiments.stage1_binary --help`
  - `python -m src.experiments.stage2_multiclass --help`
  - `python -m src.stacking --help`
  - `python -m src.moe --help`
  - `python -m src.ablation --help`
- 全量测试通过：`pytest -q` -> `41 passed`（存在非阻塞 warning）。
- 真实 smoke（数据+训练链路）通过：
  1) `preprocess_runner` 在 `MTA` 上成功生成 `manifest/rgb/seq`；
  2) `train --stage warmup`（1 epoch）成功产出 checkpoint 和 metrics；
  3) `evaluate` 成功生成 `eval_test.json`；
  4) `report` 成功生成 `report.md` 与 `learning_curve.png`。

### 关键风险/不一致
- **数据目录命名不一致（高优先级）**：
  - README 声明阶段1需要 `SourceData/ISCX`；
  - 实际目录为 `SourceData/ISCX-VPN-NonVPN-2016`；
  - `src/experiments/stage1_binary.py` 中 `REQUIRED_STAGE1_DATASETS` 硬编码为 `("ISCX", "MFCP", "MTA", "USTC-TFC2016")`。
- 结论：若不做“目录别名/重命名/代码映射”，阶段1流程在真实全量运行时会因 `ISCX` 缺失失败。

### 数据集现状快照
- `SourceData` 子目录：`CICAndMal2017`、`MFCP`、`MTA`、`USTC-TFC2016`、`ISCX-VPN-NonVPN-2016`。
- `SourceData/ISCX` 不存在。

### 2026-03-01 修复落地：ISCX 兼容 + 环境锁定
- `stage1_binary` 已支持目录别名：`ISCX` 与 `ISCX-VPN-NonVPN-2016`。
- 别名加载后会标准化 `dataset=ISCX`，并保留 `dataset_raw` 追溯原始目录名。
- 新增测试 `test_stage1_accepts_iscx_alias_directory`，验证仅存在 `ISCX-VPN-NonVPN-2016` 目录时阶段1仍可构建清单。
- 新增 `environment.yml`（`python=3.9 + pip` 依赖列表），用于跨环境复现。
- `README` 已同步：
  - 环境准备增加 `conda env create -f environment.yml`；
  - 数据目录说明增加 `ISCX-VPN-NonVPN-2016` 别名；
  - 阶段1章节注明 ISCX 别名兼容。

### 2026-03-01 review后修正（torch项忽略）
- README 已补充阶段1输出字段说明：`dataset` 标准化、`dataset_raw` 原始目录追溯。
- 新增测试覆盖别名优先级：当 `ISCX` 与 `ISCX-VPN-NonVPN-2016` 同时存在时，优先使用 `ISCX`。
