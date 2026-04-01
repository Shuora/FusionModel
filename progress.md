## Progress

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
