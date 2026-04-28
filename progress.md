## Progress

- 2026-04-28: 完成 MTA 泄露参数 Task 1：在 `src/split_data.py` 中新增 `--mta_leakage_ratio` 参数，并在 `split_task_inputs` 中实现跨 split 注入逻辑。
- 2026-04-28: 修复 `src/split_data.py` 中的逻辑冲突：为 `score_chasing_v1` 恢复提前返回（Early Return），确保该模式仅支持 `mfcp_multiclass` 并避免与 `mta_leakage_ratio` 产生二次注入；同步更新测试用例。
- 2026-04-28: 同步修复并更新 `tests/test_split_data_tasks.py`，适配 `score_chasing_v1` 的最新分布比例（10:1）与 `mta_multiclass` 支持，并新增 leakage 注入参数测试（20 passed）。

- 2026-04-22: 按 spec/plan 审计结果补齐 runbook 缺口：在 README 的 score-chasing 命令块中新增 A1.5 `ssl_tls_rgb_image.py`，避免 A1 后直接训练导致缺 `image_data`。
- 2026-04-22: 在 README 新增 A3 验收脚本（读取最新 `metrics.json`，断言 `acc>=0.97`，不达标提示触发方案 C）。
- 2026-04-22: 修复 `src/fusion_common.py` MFCP 动态 pair 的健壮性问题：当 `dynamic_pair/fallback_pair` 都为空时跳过 pair 后处理，避免 `pair_class_a/pair_class_b` 未定义风险。

- 2026-04-21: 进入 MFCP A 方案 Inline 实施，完成 `src/split_data.py` 的 `score_chasing_v1` 分布落地（仅 `mfcp_multiclass` 可用），并写入 `metadata/split_profile_summary.json`（含 `max_min_ratio` 与跨 split 近重复计数）。
- 2026-04-21: 完成 `src/fusion_common.py` accuracy-first 改造：`--stacking_threshold_objective` 新增 `accuracy`，MFCP pair 校正改为动态混淆对选择（优先 `Dridex/Trickbot`，无混淆则回退最大混淆对，再回退 `Artemis/Ursnif`）。
- 2026-04-21: 新增 `--preset mfcp_score_chasing`（class_balance/loss/early_stop/threshold objective 默认值对齐冲分目标），并加测试覆盖 preset 默认行为。
- 2026-04-21: 完成 README 同步：新增“Score-Chasing 冲分口径（宽松评估）”命令段，补充 `distribution_profile` 与输出目录说明。
- 2026-04-21: 回归验证通过：
  - `python3 -m unittest tests.test_attention_entrypoints tests.test_fusion_output_artifacts tests.test_stacking_improvements tests.test_split_data_tasks -v`（72 tests, OK）
  - `python3 -m unittest tests.test_rebalance_processed -v`（1 test, OK）

- 2026-04-21: 根据你的 `superpowers:brainstorming` 指令，读取并启用 brainstorming 技能流程，确认本轮先做设计讨论、不做实现改动。
- 2026-04-21: 读取并恢复 `task_plan.md / findings.md / progress.md` 上下文，新增“MFCP 训练优化 Brainstorming”任务阶段与状态。
- 2026-04-21: 复核最新 MFCP 与 MTA 训练日志、`metrics.json`、`report*.md`，确认两者均无崩溃报错，问题集中在效果结构（类别混淆、stacking 退化）而非训练中断。
- 2026-04-21: 补充代码与文档上下文检查：复核 `README.md` 的 MFCP 命令与 `src/fusion_common.py` 的 two-level / MFCP pair 后处理路径，准备进入一问一答澄清与方案设计。
- 2026-04-21: 完成一问一答澄清（目标硬性 `acc>=97`、允许宽松口径 `C`、保留 `max:min=2.5~3.0` 不均衡、预算 `12~24h`）。
- 2026-04-21: 完成 3 路方案比较并确认路线（先方案A，未达标切方案C），并完成四段式详细设计确认。
- 2026-04-21: 已写入设计文档 `docs/superpowers/specs/2026-04-21-mfcp-accuracy97-design.md`，并执行 spec 自检（修正 pair 后处理目标从固定类名改为按混淆矩阵动态选 top-confusion pair）。
- 2026-04-21: 用户已确认 spec 无问题，进入 writing-plans 阶段。
- 2026-04-21: 已读取 `superpowers/writing-plans` 技能，并按要求生成实现计划文档 `docs/superpowers/plans/2026-04-21-mfcp-accuracy97.md`。
- 2026-04-21: 已完成实现计划自检（spec 覆盖、无占位、函数/参数命名一致）。

- 2026-04-18: 读取 `using-superpowers`、`systematic-debugging`、`planning-with-files`、`test-driven-development` 技能，并恢复当前 planning 文件上下文。
- 2026-04-18: 复核 `src/rebalance_processed.py` 实现与 `mfcp_multiclass` 目录规模，确认 `PUA` 类约 67.9 万个 session，是本次命令慢路径的主要数据面。
- 2026-04-18: 用 `timeout 20s` 复现 `rebalance_processed.py`，确认脚本已开始工作但在图片关联阶段极慢；初步 root cause 为每个 session 重复全量扫描 `image_data/<split>/<label>`。
- 2026-04-18: 按 TDD 新增 `tests/test_rebalance_processed.py`，先让“同一 label/split 的图片目录只扫描一次”测试红灯，确认当前实现会重复扫描。
- 2026-04-18: 在 `src/rebalance_processed.py` 增加 `collect_image_index()`，将图片查找改为一次建索引、按 stem 命中；回归测试已转绿。
- 2026-04-18: 使用真实 `mfcp_multiclass` 再次短时复现原命令，20 秒内日志已从 `Artemis` 推进到 `Cobalt`，证明热点修复生效；脚本剩余耗时来自正常的大量 link/copy。
- 2026-04-19: 读取 `outputs/mfcp_multiclass/attention_stacking/attention_stacking_20260418_232645` 的 run 目录、`train.log` 和 `report.md`，确认只有 base attention artefact，没有任何 stacking artefact。
- 2026-04-19: 对照 `run_stacking_experiment()` 控制流，确认 `report.md` 是 base_eval 阶段就会写出的文件；`metrics.json`、各 `report_<method>.md`、`done stacking` 只会在 stacking 完整执行后出现。
- 2026-04-19: 形成结论：该 run 在 base attention 训练完成并保存基础报告后即中断，未进入或未完成后续 meta feature / meta learner / soft-voting / two-level 路径。

- 2026-03-31: 读取 `using-superpowers`、`brainstorming`、`writing-plans`、`test-driven-development`、`using-git-worktrees`、`verification-before-completion` 技能，确认本次任务边界为“只修改默认输出根目录，不迁移历史产物”。
- 2026-03-31: 检查仓库输出路径实现，确认 `src/fusion_common.py` 是本次代码修改核心，`README.md` 需要同步把 `src/outputs` 改为根目录 `outputs`。
- 2026-03-31: 使用并行子代理复核影响面，确认 `tests/test_fusion_output_artifacts.py` 适合补默认路径测试，`src/run_all_modes.py` 无需改动。
- 2026-03-31: 在 `.worktrees/codex-output-root` 创建隔离 worktree，并确认 `.worktrees` 已被 git ignore。
- 2026-03-31: 先为默认 `output_dir` 与默认日志目录补回归测试，再运行红灯验证，确认当前行为仍指向 `src/outputs`。
- 2026-03-31: 在 `src/fusion_common.py` 引入共享的 `DEFAULT_OUTPUT_ROOT`，将默认输出与默认日志目录统一迁移到仓库根目录 `outputs/`。
- 2026-03-31: 同步更新 `README.md` 的训练输出路径说明，并修复 `AGENTS.md` 的 merge conflict，保留仓库当前有效约束。
- 2026-03-31: 发现 conda 环境测试会被 Windows 用户 site-packages 污染；后续验证命令统一显式清理相关 Python 环境变量。

- 2026-03-31: aligned MFCP paper/source/processed counts; found Cobalt missing only in processed data.
- 2026-03-31: checked capinfos and git history; identified truncation plus old parser behavior as root cause.
- 2026-03-31: verified Cobalt raw pcap still has many payload sessions; recommend regenerate ProcessedData with current split_data.

- 2026-03-31: 统一 `src/fusion_common.py` 内早停默认耐心轮次为 8（CLI / train_fusion_model / EarlyStopping）。
- 2026-03-31: 新增 `_resolve_early_stop_mode` 与 `_select_monitor_value`，并在训练循环中加上监控值 `NaN/Inf` 保护与 scheduler 安全更新。
- 2026-03-31: 在 `tests/test_fusion_output_artifacts.py` 补充早停默认值与模式校验测试，并同步 README 早停参数说明。
- 2026-04-01: 修复 `src/fusion_common.py` 早停逻辑：非有限监控值按“未改善”推进 early-stop 计数，并在达到 `patience` 时恢复 best weights 后停止训练。
- 2026-04-01: 在 `tests/test_fusion_output_artifacts.py` 新增 NaN 场景回归测试，验证不会在 NaN 后继续长时间训练。
- 2026-04-01: 在 `train_fusion_model` 训练 batch 增加 `torch.isfinite(loss)` 检查，NaN/Inf batch 跳过并记录 warning。
- 2026-04-01: 新增回归测试 `test_non_finite_train_batch_loss_is_skipped` 并通过全量 `tests.test_fusion_output_artifacts` 验证。
- 2026-04-01: 完成训练记录全量盘点，确认 `outputs/` 下 11 个 run 中 5 个完整、6 个中断，并对每个中断 run 提取了 batch_size 与终止 epoch。
- 2026-04-01: 复核二分类 run `attention_dim256_20260401_021405`，确认第 9 轮后 NaN 连续传播且最终单类塌缩，属于无效训练产物。
- 2026-04-01: 复核 mta/mfcp 完整 run 的最终报告，识别到 `mta_multiclass` 在极小类上的系统性召回缺失，accuracy 与 macro_f1 口径分离明显。
- 2026-04-02: 盘点最新全量训练的 8 个 run，确认四个任务的 attention / attention_stacking 均已完整产出到 `outputs/<task>/<mode>/<run_name>/`。
- 2026-04-02: 提取全部 `metrics.json` / `epoch_metrics.csv` / `train.log` 关键指标，确认 `binary` 与 `mta` 的 stacking run 在后半程分别出现 30513 / 3366 个连续 `NaN/Inf batch`，最终由 early stop 恢复 best weights 后收尾。
- 2026-04-02: 对照 `src/fusion_common.py` 与 `src/run_all_modes.py`，确认当前默认配置缺少梯度裁剪和学习率调度，且 `--mode all` 只在整次进程开始时设 seed，一次 attention 完成后不会为 stacking 重新设 seed。
- 2026-04-02: 完成任务级结论归纳：`ustc` 当前表现最好且稳定；`binary` 指标高但 stacking 训练过程不稳定；`mfcp` 主要短板在 `Ursnif`；`mta` 仍受少数类 `Dridex` 完全失效拖累。

- 2026-04-03: 用户确认“集成学习优化全部落地”；已在 `.worktrees/codex-stacking-improvements` 创建隔离分支 `codex/stacking-improvements`。
- 2026-04-03: 完成基线验证 `pytest -q tests/test_attention_entrypoints.py`（2 passed），准备进入 stacking 改造的 TDD 阶段。
- 2026-04-03: 已明确本轮改造范围：OOF stacking、元特征扩展、class-weighted XGBoost、多模型 soft-voting、mta/mfcp 任务定向校正。
- 2026-04-03: 新增失败测试 `tests/test_stacking_improvements.py`（初始 6 failed），覆盖 OOF、soft-voting、class gain、pair correction 等核心能力。
- 2026-04-03: 完成 `src/fusion_common.py` 改造：新增 stacking 纯函数工具集、OOF 训练流程、扩展元特征、加权软投票、任务定向后处理与 OOF 指标落盘。
- 2026-04-03: 新测试已转绿（`tests/test_stacking_improvements.py` 6 passed），并通过相关回归测试（`test_attention_entrypoints/test_run_all_modes/test_fusion_output_artifacts` 共 21 passed）。
- 2026-04-03: 已同步更新 README 的 stacking 行为与输出说明，补充 soft-voting 与 OOF 指标文档。

- 2026-04-04: 新建隔离 worktree `.worktrees/codex-fix-invalid-grad`（branch: `codex/fix-invalid-grad`），用于修复“梯度无效”问题，避免干扰主工作区。
- 2026-04-04: 已完成根因定位：AMP 路径把 scaler overflow 当作无效梯度事件统计与告警；准备先补回归测试，再最小改动修复。
- 2026-04-04: 已按 TDD 新增失败测试 `test_amp_overflow_is_not_counted_as_invalid_grad_batch`，红灯确认当前逻辑会将 AMP 场景误计入 `invalid_grad_batches`。
- 2026-04-04: 已完成最小修复：删除 AMP 分支中的手动无效梯度判定与跳过分支，改由 `GradScaler` 自适应处理 overflow。
- 2026-04-04: 验证通过：
  - `pytest -q tests/test_fusion_output_artifacts.py -k amp_overflow_is_not_counted_as_invalid_grad_batch`（1 passed）
  - `pytest -q tests/test_fusion_output_artifacts.py`（16 passed）
  - `pytest -q tests/test_attention_entrypoints.py tests/test_run_all_modes.py`（6 passed）
- 2026-04-04: 在 `.worktrees/codex-mta-mfcp-improve` 启动仅面向 `mta/mfcp` 的定向增强，复核确认仓库已具备 OOF stacking、soft-voting、mta gain 与 mfcp 0/4 校正。
- 2026-04-04: 完成 `mta` 后处理增强：gain 调优目标类从固定 `[0,1]` 改为按 `meta_labels` 样本数自动选择最少的两个类别，减少类顺序耦合。
- 2026-04-04: 完成 `mfcp` 后处理增强：新增 `tune_binary_correction_alpha_for_pair`，基于 OOF macro-F1 自动选择 `alpha` 并用于测试集 `0/4` 二分类校正。
- 2026-04-04: 新增单测覆盖 `alpha=0` 恒等性与 `alpha` 调参不劣于基线场景，等待回归测试执行结果。
- 2026-04-04: 回归测试通过：`pytest -q tests/test_stacking_improvements.py tests/test_fusion_output_artifacts.py`（24 passed）。
- 2026-04-04: 根据你“效果不理想”反馈，在 `.worktrees/codex-mfcp-postprocess-tuning` 启动二次优化，目标聚焦 `mfcp` 的 `0/4` 混淆问题。
- 2026-04-04: 已按 TDD 新增失败测试（2 failed），覆盖 `pair_f1` 目标与 pair 校准/阈值搜索链路。
- 2026-04-04: 已在 `fusion_common.py` 实现 `score_pair_f1`、`tune/apply_pair_temperature`、`tune/apply_pair_threshold`，并扩展 `tune_binary_correction_alpha_for_pair(objective=...)`。
- 2026-04-04: 已接入 `mfcp` method 与 soft-voting 后处理链路，新增 `mfcp_pair_temperature/mfcp_pair_threshold` 落盘字段。
- 2026-04-04: 新增测试转绿，目标回归 `pytest -q tests/test_stacking_improvements.py -k 'pair_f1_objective or pair_calibration'`（2 passed）。
- 2026-04-04: 在 `.worktrees/codex-paper-mta-mfcp-distribution` 创建隔离分支 `codex/paper-mta-mfcp-distribution`，按 TDD 新增 `paper_mvtba` 分布模式测试并验证红灯（`split_task_inputs` 不支持 profile 参数）。
- 2026-04-04: 已实现 `src/split_data.py` 的 `--distribution_profile paper_mvtba`（仅作用 `mta_multiclass/mfcp_multiclass`），支持固定类集合与固定 Train/Test 计数抽样，缺类/样本不足会 fail-fast。
- 2026-04-04: 回归通过 `python3 -m unittest tests.test_split_data_tasks -v`（12 tests, OK）。
- 2026-04-04: 已同步更新 README 的 MTA/MFCP 预处理命令与参数说明，新增 `--distribution_profile` 用法。
- 2026-04-04: 已重建主仓库 `ProcessedData/mta_multiclass` 与 `ProcessedData/mfcp_multiclass`（含 `image_data`），并完成逐类计数核验；MTA 与 MFCP 的 Train/Test 计数已按论文目标对齐。

- 2026-04-04: 针对你反馈的 MTA stacking 指标偏低，在 `.worktrees/codex-mta-stacking-boost` 创建隔离分支 `codex/mta-stacking-boost` 并建立回归基线。
- 2026-04-04: 完成问题定位：MTA 最新 run 存在显著 OOF-test gap（约 0.216），并确认 class-name 硬编码未覆盖 `IcedID` 导致 MTA 定向后处理失效。
- 2026-04-04: 按 TDD 新增失败测试 `test_build_deterministic_meta_loader_ignores_weighted_sampler` 与 `test_detect_stacking_special_tasks_supports_mta_with_icedid`，确认红灯后再实现修复。
- 2026-04-04: 在 `src/fusion_common.py` 新增 `build_deterministic_meta_loader`、`detect_stacking_special_tasks` 与 task hint 解析逻辑；`run_stacking_experiment` 已切换到 deterministic meta loader 并加入 OOF-test gap 诊断日志。
- 2026-04-04: 回归验证通过：`pytest -q -s tests/test_stacking_improvements.py tests/test_attention_entrypoints.py tests/test_fusion_output_artifacts.py`（30 passed）。
- 2026-04-04: 已同步更新 README 的 stacking 默认行为说明（元特征提取数据流与 MTA 7 类识别）。
- 2026-04-05: 完成预处理体验优化：`src/ssl_tls_rgb_image.py` 移除逐条图片保存日志，统一为进度条 + processed/skipped 实时统计。
- 2026-04-05: 完成预处理日志落盘：`src/split_data.py` 与 `src/ssl_tls_rgb_image.py` 均新增 `--log_file`，默认落盘到任务 `metadata/` 目录。
- 2026-04-05: 已同步 README 预处理章节，补充日志文件默认路径与 `--log_file` 用法。
- 2026-04-05: 验证结果：`python3 -m py_compile src/split_data.py src/ssl_tls_rgb_image.py` 通过；`python3 -m unittest tests.test_split_data_tasks tests.test_ssl_tls_rgb_image` 中 split_data 通过，ssl_tls_rgb_image 因本机 numpy 环境污染失败。

- 2026-04-06: 对照 2026-04-04 MTA 修复链路复核 `src/fusion_common.py`，确认 MFCP 后处理仍存在 `class_a=0,class_b=4` 固定索引依赖。
- 2026-04-06: 已将 MFCP pair 校正改为按类名解析（`Artemis/Ursnif`），并在 `method` 与 `soft_voting` 两条路径同时替换；新增 `mfcp_pair_classes` 落盘字段。
- 2026-04-06: 已扩展 MFCP 任务签名识别，兼容含 `Cobalt` 的 6 类分布。
- 2026-04-06: 回归验证通过：`python3 -m unittest tests.test_stacking_improvements -v`（14 tests, OK）；`python3 -m unittest tests.test_attention_entrypoints -v`（2 tests, OK）。
- 2026-04-06: 按你的要求更新 `README.md` 中 `mfcp_multiclass` 的 `attention_stacking` 命令，与 `mta_multiclass` 当前推荐稳定参数对齐。

- 2026-04-07: 按“是否真正 CharBERT”问题完成代码审计，确认当前实现为轻量 byte Transformer，缺少 char-aware 关键机制。
- 2026-04-07: 与你确认改造方向为“兼容式升级”（入口脚本保持不变，默认仍注意力融合）。
- 2026-04-07: 完成方案对比并确定推荐方案：分层门控 `char-aware byte encoder`（`legacy/charaware` 双模式）。
- 2026-04-07: 已输出设计 spec：`docs/superpowers/specs/2026-04-07-charaware-byte-charbert-design.md`，包含架构、参数兼容、checkpoint 兼容、验证与回滚策略。

- 2026-04-07: 根据你“开始实现”指令进入 build 阶段，先按 TDD 新增测试：
  - `tests/test_attention_entrypoints.py`：校验新增 char-aware 参数默认值；
  - `tests/test_fusion_output_artifacts.py`：校验 `build_common_kwargs` 透传；
  - `tests/test_charbert_loader.py`：校验 `charaware` 模式可构建并前向。
- 2026-04-07: 完成 `src/CharBERT/src/config.py` 扩展（`mode/char_vocab/char_emb_dim/char_cnn_channels/char_fusion/char_fusion_layers`）。
- 2026-04-07: 完成 `src/CharBERT/src/model.py` 升级：新增 char lookup、char embedding + Conv1d、分层 token/char 融合和 `encode_tokens` 接口。
- 2026-04-07: 完成 `src/fusion_common.py` 升级：新增 CLI 参数、kwargs 透传、`initialize_fusion_model`/`run_*_experiment` 参数链路与 `CharBERTTextEncoder` 新接口适配。
- 2026-04-07: 完成 `README.md` 同步：新增 char-aware 可选参数列表、模式说明与最小启用示例。
- 2026-04-07: 目标回归通过：`python3 -m unittest tests.test_attention_entrypoints tests.test_run_all_modes tests.test_fusion_output_artifacts tests.test_charbert_loader`（26 tests, OK）。

- 2026-04-07: 按你“inline + worktree”要求在 `.worktrees/codex-two-level-stacking` 启动二层 stacking 改造，先完成基线校验：`attention_entrypoints + fusion_output_artifacts + stacking_improvements` 共 34 tests, OK。
- 2026-04-07: 按 TDD 新增 two-level 相关测试（参数透传、校准、Level-2 特征、阈值优化、编排降级、报告字段），红灯验证失败点为新增函数/参数缺失。
- 2026-04-07: 完成参数接线：`add_common_args/build_common_kwargs/run_stacking_experiment` 已支持 five new stacking args，并输出 stacking config 日志。
- 2026-04-07: 完成能力实现：新增 multiclass calibration（temp/isotonic）、ECE/Brier、Level-2 特征构造、per-class threshold、two-level blender OOF 路径、effective-level 自动降级。
- 2026-04-07: 完成主链路接入：`run_stacking_experiment` 在可用 learner>=2 时新增 `two_level_blender` 结果落盘；保留单层 method 与 soft-voting 对照。
- 2026-04-07: 完成 README 同步：四个任务的 `attention_stacking` 命令和参数说明已加入 two-level 相关参数与默认行为说明。
- 2026-04-07: 完成回归验证：`python3 -m unittest tests.test_attention_entrypoints tests.test_fusion_output_artifacts tests.test_stacking_improvements tests.test_run_all_modes -v`（47 tests, OK）。
- 2026-04-07: 根据你“还有什么没实现”反馈补齐设计缺口：新增 pairwise-KL 特征、hard-sample factor 加权、two-level oof-test gap 字段与告警。
- 2026-04-07: 补齐后再次回归通过：`python3 -m unittest tests.test_attention_entrypoints tests.test_fusion_output_artifacts tests.test_stacking_improvements tests.test_run_all_modes -v`（49 tests, OK）。
- 2026-04-07: 根据你“再仔细检查”反馈继续补齐：修复 two-level postprocess 参数接线（`threshold_objective/objective_value`），并把 `single_layer_baseline` 正式接入 `method_results` 主流程。
- 2026-04-07: 增补测试并回归通过：`python3 -m unittest tests.test_attention_entrypoints tests.test_fusion_output_artifacts tests.test_stacking_improvements tests.test_run_all_modes -v`（52 tests, OK）。
- 2026-04-07: 按你要求增强预处理可观测性：`split_data.py` 新增预处理汇总日志与按家族 Train/Test/Total 样本统计输出。
- 2026-04-07: 按 TDD 补充 `tests/test_split_data_tasks.py` 两个新测试（family summary 统计、日志输出校验）并回归通过：`python3 -m unittest tests.test_split_data_tasks -v`（15 tests, OK）。
- 2026-04-07: 同步更新 README 预处理说明，明确 `split_data.py` 日志新增家族级统计输出。

- 2026-04-07: 根据你的新问题完成 MTA/MFCP 训练产物审计，读取并对比 `metrics.json`、`report*.md`、`train.log` 与 `ProcessedData/*/metadata/manifest.json`。
- 2026-04-07: 通过脚本汇总关键证据：MTA `20260404 -> 20260405` 的 OOF-test gap 从约 0.214 降至约 0.074；MFCP `20260406` gap 约 0.037，且 health 指标无 invalid 事件。
- 2026-04-07: 完成根因排序：MTA 以数据分布极不均衡（max:min≈66:1）为主，MFCP 以类间可分性/特定配对混淆为主；当前集成学习链路整体有效，不是主故障源。

- 2026-04-07: 按你的要求联网抓取 MTA/MFCP 官方页面并生成“尽量均衡”的下载清单 CSV。
- 2026-04-07: 复用仓库内 `mta_direct_links_2021plus.txt`，按 7 类各 20 条写出 `outputs/balanced_pcap_links_mta_20_per_family.csv`。
- 2026-04-07: 解析 `stratosphereips.org/datasets-malware` 的 capture 链接并二次解析各 capture 页面 `.pcap` 直链，按 6 类各 2 条写出 `outputs/balanced_pcap_links_mfcp_target2_per_family.csv`。
- 2026-04-07: 额外生成合并文件 `outputs/balanced_pcap_links_combined.csv`，总计 152 条链接。

- 2026-04-10: 读取 planning-with-files 约束并执行 session-catchup；确认本次请求为“MTA 最新训练日志复盘”。
- 2026-04-10: 定位最新 MTA 训练产物，确认 `attention_stacking_v2/attention_stacking_20260408_234642` 仅存在 `train.log`。
- 2026-04-10: 复核最新 `train.log`，记录末尾止于 `Epoch 40/40` 起始行，缺少最终 epoch 结果与 done/metrics 产物；整理 val_f1 平台期与 early-stop 计数情况。
- 2026-04-10: 按用户要求回滚“精确 1w/类”版本，重新整理为“每类约 1w（可上下浮动）”。
- 2026-04-10: 实施方式为从原始备份重建 `mta_multiclass`，设置每类总量上限 12000，不做上采样，保持原始 Train/Test 比例。
- 2026-04-10: 完成一致性校验：`manifest total=81708`，各类分布在 `10176~12000`，并保留两个回滚快照目录。
- 2026-04-13: 收到新的时序预处理增强需求，目标是引入 packet boundary / direction / length / delta_t / hierarchical input，并尽量保持现有训练入口不变。
- 2026-04-13: 新增红灯测试，当前实现还没有 `split_data.PacketRecord` 和 `fusion_common.build_temporal_pcap_token_ids`，说明 packet 级元数据链路尚未接入。
- 2026-04-13: 已完成时序预处理增强实现，`split_data.py` 现在写出 packet sidecar，`fusion_common.py` 现在优先读 sidecar 并生成分层 byte 序列。
- 2026-04-13: 继续补完模型侧 packet-aware hierarchy，`CharBERTTextEncoder` 现在会基于 packet block、packet 元数据和 packet encoder 融合 CLS 与 packet summary。
- 2026-04-13: 已完成回归验证：`python3 -m unittest tests.test_split_data_tasks tests.test_temporal_pcap_hierarchy tests.test_fusion_output_artifacts tests.test_attention_entrypoints tests.test_run_all_modes -v`。
- 2026-04-13: 已完成文档措辞对齐，明确 `charaware` 路径不是简单平均池化，而是 packet 级元数据 + packet encoder 的分层时序摘要；再次回归确认 `49 tests, OK`。
- 2026-04-13: 已补 `FusionDataset.load_pcap_data()` 的 sidecar/legacy 双路径回归测试，并再次验证 `49 tests, OK`。
- 2026-04-13: 已补 sidecar 版本门控与回退测试，避免未来 sidecar 格式升级后被静默误读。
- 2026-04-13: 最新完整回归通过 `50 tests, OK`，包含 sidecar 版本回退场景。
- 2026-04-14: 读取 `using-superpowers`、`planning-with-files`、`figures-diagram`、`brainstorming` 技能，并恢复现有 planning 文件上下文。
- 2026-04-14: 复核 `src/split_data.py`、`src/fusion_common.py`、`README.md` 与 `AGENTS.md` 中的时序增强链路，确认流程图范围限定为“预处理产物 + 训练前 temporal token 生成/回退”。
- 2026-04-14: 提取参考 `mobilevit.drawio` 主配色，准备生成仓库内可编辑的时序预处理 `.drawio` 源文件。
- 2026-04-14: 根据你的新要求改为仅输出 Nano Banana prompt，不保留新建 `.drawio` 源文件。
- 2026-04-23: 收到“为什么 MTA 最新日志效果好而 MFCP 不行”问题后，完成最新完整 run 定位与同口径对比：MTA `20260423_183745` vs MFCP `20260422_234257`。
- 2026-04-23: 解析 `metrics.json` 与 `train.log`，确认两者训练稳定性均正常（`run_status=ok`、无 invalid batch 异常），排除“训练崩坏”作为主因。
- 2026-04-23: 完成数据分布与策略对比：MTA 当前为严格均衡分布且用 `val_f1/focal/weighted sampler`；MFCP 为 3:1 轻不均衡且以 `val_acc/accuracy objective` 为主。
- 2026-04-23: 完成 MFCP 误差归因：核心问题是 `Trickbot` 与 `Dridex` 可分性不足（`Trickbot -> Dridex` 大规模混淆），导致总体指标上限受限。
- 2026-04-23: 按你的要求重建 `mta_multiclass` 为最多 `3:1`。已先备份旧目录，再从 `SourceData/MTA` 重新切分到 `mta_multiclass_raw_unbalanced`。
- 2026-04-23: 执行 `rebalance_processed.py` 回写 `ProcessedData/mta_multiclass`，并在遇到 numpy 用户站点污染后通过 `unset PYTHONPATH PYTHONHOME PYTHONUSERBASE && PYTHONNOUSERSITE=1` 重跑图片生成成功。
- 2026-04-23: 完成最终校验：`manifest` 与 `image_data` 计数一致，`Train/Test` 类别比例均满足 `<=3:1`。
