## Findings

<<<<<<< HEAD
- 当前 V4 仓库的标准实验入口是 `src/run_all_modes.py`、`src/train_fusion_attention.py` 和 `src/train_fusion_attention_stacking.py`，且都要求传入 `--task_name`。
- 支持的任务名仅有 `binary_benign_vs_malicious`、`ustc_multiclass`、`mta_multiclass`、`mfcp_multiclass`。
- 标准数据流程依赖固定目录链路：
  `SourceData/<dataset>` -> `ProcessedData/<task>/pcap_data/{Train,Test}` -> `ProcessedData/<task>/image_data/{Train,Test}`。
- 根目录 `requirements.txt` 未包含 `dpkt`，但 `split_data.py` 实际依赖它解析 pcap，因此 README 需要单独提示安装。
- 当前环境没有 `python` 命令，CLI 校验需使用 `python3`。
- `.gitignore` 会忽略 `/SourceData`、`/ProcessedData`、`/outputs`、`/dataset` 等目录，README 需要明确这些数据与结果不会随仓库提供。
- `tools/sort_outputs_by_mode.py` 和 `tools/split_concat_log.py` 不是可直接复用的跨平台标准脚本，因为内部写死了历史路径。
- 当前仓库的训练默认值已统一为 `batch_size=32`、`num_workers=4`，相关预取参数保持为 `prefetch_factor=2`，以保持代码默认值与文档示例一致。
=======
- 当前实现位于 [src/split_data.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py)，流程是 `discover raw -> extract all sessions -> split sessions`。
- 现有 `split_dataset()` 先调用 `expand_raw_samples_to_sessions(raw_samples)`，再对 `session_samples` 做 `split_task_inputs()`，这会让同一原始 `pcap` 的 session 同时进入 Train/Test。
- 现有测试 [tests/test_split_data_tasks.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/tests/test_split_data_tasks.py) 已经覆盖 raw-level split，但还没有覆盖“单个 raw capture 需要 time-blocked split”的场景。
- worktree 基线测试 `python3 -m unittest tests.test_split_data_tasks -v` 通过，说明当前改动可以从干净基线开始。

## Final Findings

- 已在 [src/split_data.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py) 新增 `iter_session_payloads()`，并让 `extract_sessions()` 复用该迭代器，避免重复解析逻辑。
- 已新增 `split_single_capture_by_time()`：使用 `boundary = min_ts + (max_ts - min_ts) * train_ratio` 做时间边界切分。
- 同一五元组若在边界两侧都有 payload，会在 `split_single_capture_by_time()` 中被统计为 `dropped_cross_boundary` 并从双侧删除。
- `split_dataset()` 现为 hybrid：label 下 `len(raw)>1` 走 raw-level split 后 sessionize，`len(raw)==1` 走时间切分后 sessionize。
- 新增/更新测试覆盖三类关键行为：单 raw 时间切分、跨边界 session 双侧丢弃、多 raw 保持 raw-level split。

## Code Quality Fix Findings

- 已修复 singleton 路径容错：`split_dataset()` 对 `split_single_capture_by_time()` 加了 `try/except`，坏 capture 仅记录 `Error reading ...` 并跳过，不再中断整次任务。
- 已修复 `min_ts == max_ts` 的空侧问题：`split_single_capture_by_time()` 在同时间戳场景回退为 `packet-order` 切分，保证可切时 Train/Test 都有数据。
- 已消除全量 `sorted(packet_items)` 内存退化：改为两遍流式扫描。
  - 第一遍只统计 `packet_count/min_ts/max_ts`；
  - 第二遍做 payload 聚合与 session 归属判定；
  - 不再把所有 `(timestamp, key, payload)` 常驻内存。

## Final Reviewer Fix Findings

- 已修复 session 命名碰撞：`build_session_name()` 现在使用 `build_raw_capture_token(raw_path)`，格式为 `raw_stem-hash`，再拼接五元组。即使同 label 下不同 raw capture 同 stem，也不会覆盖。
- 已修复 rerun 脏数据：成功重跑时 `pcap_data/metadata` 会被完整替换，不保留旧 `.bin`。
- 新增测试确认：
  - 同 label 同 stem 多 raw 的 `manifest.bin_path` 全部唯一；
  - 同 `processed_root` 连续运行两次时，第一次残留 `.bin` 不会保留到第二次结果中。

## Transactional Output Findings

- 已移除“先删后写”流程，改为事务式输出切换：
  - 本次结果先写到 `processed_root/.split_data_staging/{pcap_data,metadata}`；
  - 仅当 staging 完整后才执行替换发布；
  - 发布失败会回滚恢复旧 `pcap_data/metadata`。
- 若检测到上次发布异常中断留下 `.split_data_backup_*` 且对应 final 缺失，会先自动恢复再继续本次发布。
- 中途异常（例如 unknown task、写出异常、manifest 异常）不会清空旧结果，满足“失败不破坏最后一次成功输出”。
- 仍然保留 rerun 去脏能力：成功发布后旧的 `pcap_data/metadata` 会被整体替换，不会残留旧 `.bin`。

## Recovery Timing Fix Findings

- 已将“检测并恢复 `.split_data_backup_*` 且 final 缺失”的逻辑前移到 `split_dataset()` 最开始，确保 discovery 前即恢复到最近一次提交态。
- `promote` 阶段仍保留同一恢复逻辑，作为发布前防御，不影响现有 staging/rollback 机制。
- 新增测试验证：手工制造 backup-only 损坏状态后，用 `unknown_task` 在 promote 前失败，函数会先恢复旧输出再抛出预期异常。

## Findings (2026-03-29, attention output persistence)

- 旧实现将产物直接写到 `output_dir` 根目录且普遍携带 `tag+timestamp` 文件名，缺少统一机器可读汇总，`metrics.json`/`epoch_metrics.csv` 未稳定产出。
- `run_fusion_experiment()` 与 `run_stacking_experiment()` 之前共享同一输出平面；在同 `output_dir` 连续运行时虽靠文件名前缀降低碰撞，但不满足“固定文件名 + run 隔离”目标。
- `collect_attention_diagnostics()` 原本仅支持 `attention_curve_{prefix}.png` 命名，不便在 run 目录内稳定落盘固定文件名。

## Final Findings (2026-03-29, attention output persistence)

- 已在 `src/fusion_common.py` 建立统一落盘抽象：
  - `prepare_run_output_dir(output_dir, run_name)`：每次 run 创建独立子目录，并对重名 run 自动后缀避让。
  - `build_run_artifact_paths(run_dir)`：统一固定文件名（`train.log`、`metrics.json`、`epoch_metrics.csv`、`metrics_curve.png`、`confusion_matrix.png`、`attention_curve.png` 等）。
  - `export_metrics_artifacts(run_dir, history, metrics_payload)`：稳定导出 JSON + CSV。
- attention 模式输出已改为固定文件名且全部落到 run 子目录：
  - `train.log`
  - `metrics.json`
  - `epoch_metrics.csv`
  - `metrics_curve.png`
  - `confusion_matrix.png`
  - `attention_curve.png`（可采集到注意力权重时）
- attention_stacking 模式同样落到独立 run 子目录并产出固定核心文件；同时保留 method 级附加产物（如 `confusion_matrix_<method>.png`、`report_<method>.md`）以避免信息丢失。
- `run_all_modes.py` 无需改动即可复用：在 `mode=all` 下两次调用共享同一 `output_dir`，但各自在 `output_dir/<run_name>/` 下保存，不再互相覆盖。

## Findings (2026-03-29, attention plotting crash)

- 用户日志中的致命崩溃不是 PyTorch nested tensor warning，也不是 `np.log` warning；真正导致进程 `core dumped` 的是 `tkinter`/Tcl：
  - `RuntimeError: main thread is not in main loop`
  - `Tcl_AsyncDelete: async handler deleted by the wrong thread`
- 当前 [src/fusion_common.py](/home/shuora/Traffic/FusionModel/.worktrees/attention-headless-fix/src/fusion_common.py) 在 `plot_training_curves()`、`plot_confusion()`、`collect_attention_diagnostics()` 内直接 `import matplotlib.pyplot as plt`，未统一强制无头后端，CLI 训练进程可能落到 `TkAgg` 并在析构阶段触发 Tcl/Tk 跨线程崩溃。
- `summarize_attention()` 在 `pad_mask` 分支中会生成真实的零值概率；原写法 `np.where(a_nonpad > 0, np.log(a_nonpad), 0.0)` 会先整体求 `np.log(a_nonpad)`，因此即使逻辑上过滤了零值，也仍会产生 `divide by zero encountered in log` warning。

## Final Findings (2026-03-29, attention plotting crash)

- 已新增 `load_pyplot_headless()`，统一将 `matplotlib` 后端切到 `Agg`；若 `pyplot` 已提前导入且后端不是 `Agg`，则回退到 `plt.switch_backend("Agg")`。
- `plot_training_curves()`、`plot_confusion()`、`collect_attention_diagnostics()` 现在全部通过该 helper 获取 `pyplot`，不再依赖 Tk GUI backend。
- 模块加载时额外设置 `MPLBACKEND=Agg` 默认值，进一步降低其他导入路径误选 `TkAgg` 的概率。
- `summarize_attention()` 的 entropy 计算已改为 `np.log(a_nonpad, out=safe_log, where=a_nonpad > 0)`，`pad_mask` 产生的零值不会再触发运行时 warning。
>>>>>>> c926dfcf8bb829c579b702d527601a20ba85ca45
