## Progress

- 2026-03-29: 读取 `using-superpowers`、`brainstorming`、`writing-plans`、`using-git-worktrees`、`systematic-debugging`、`test-driven-development` 技能。
- 2026-03-29: 确认用户批准“单 capture 先按时间切分，再分别 sessionize”的设计。
- 2026-03-29: 创建 worktree `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split`，分支 `codex/time-blocked-split`。
- 2026-03-29: 在 worktree 运行 `python3 -m unittest tests.test_split_data_tasks -v`，基线通过。
- 2026-03-29: 在 [tests/test_split_data_tasks.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/tests/test_split_data_tasks.py) 增加 3 个 TDD 用例（单 raw 时间切分、跨边界丢弃、多 raw raw-level）。
- 2026-03-29: 运行 `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python -m unittest tests.test_split_data_tasks -v`，得到 `FAILED (failures=1, errors=2)`，进入 RED。
- 2026-03-29: 修改 [src/split_data.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py)，实现 hybrid split policy 与跨边界丢弃逻辑。
- 2026-03-29: 复跑同一命令，结果 `Ran 10 tests ... OK`，进入 GREEN。
- 2026-03-29: 更新 task artifacts 与 superpowers spec/plan，记录最终策略与验证结果。
- 2026-03-29: 在 [tests/test_split_data_tasks.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/tests/test_split_data_tasks.py) 新增质量回归测试（singleton 读包失败容错、同时间戳 packet-order fallback）。
- 2026-03-29: 运行同一 unittest 命令，得到 `FAILED (failures=1, errors=1)`，确认 code review 问题可复现。
- 2026-03-29: 修改 [src/split_data.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py)，完成 singleton 容错、同时间戳 fallback 与流式两遍扫描改造。
- 2026-03-29: 复跑同一 unittest 命令，结果 `Ran 12 tests ... OK`。
- 2026-03-29: 在 [tests/test_split_data_tasks.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/tests/test_split_data_tasks.py) 新增碰撞与 rerun 清理回归测试。
- 2026-03-29: 运行同一 unittest 命令，得到 `FAILED (failures=2)`，确认存在 session 文件覆盖与 rerun 残留。
- 2026-03-29: 修改 [src/split_data.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py)，加入 `raw_path hash` 命名策略与 rerun 去脏流程。
- 2026-03-29: 复跑同一 unittest 命令，结果 `Ran 14 tests ... OK`。
- 2026-03-29: 新增事务回归测试（成功后再次失败应保留旧输出），用 `unknown_task` 触发失败路径。
- 2026-03-29: 运行同一 unittest 命令，得到 `FAILED (errors=1)`，确认失败 rerun 会清空旧 manifest。
- 2026-03-29: 修改 [src/split_data.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py)，实现 `.split_data_staging` staging + commit/rollback 事务发布。
- 2026-03-29: 复跑同一 unittest 命令，结果 `Ran 15 tests ... OK`。
- 2026-03-29: 增强事务发布健壮性：若存在中断遗留的 `.split_data_backup_*` 且 final 缺失，先恢复再发布。
- 2026-03-29: 复跑同一 unittest 命令，结果仍为 `Ran 15 tests ... OK`。
- 2026-03-29: 新增恢复时机回归测试（backup-only 损坏 + promote 前失败），确认当前实现会失败。
- 2026-03-29: 运行同一 unittest 命令，得到 `FAILED (errors=1)`。
- 2026-03-29: 修改 [src/split_data.py](/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py)，将恢复逻辑前移到 `split_dataset()` 起始阶段。
- 2026-03-29: 复跑同一 unittest 命令，结果 `Ran 16 tests ... OK`。

## Progress (2026-03-29, attention output persistence)

- 2026-03-29: 创建 worktree `/home/shuora/Traffic/FusionModel/.worktrees/attention-run-output-isolation`，分支 `codex/attention-run-output-isolation`。
- 2026-03-29: 阅读 `src/fusion_common.py`、`src/run_all_modes.py`、`src/train_fusion_attention.py`、`src/train_fusion_attention_stacking.py` 与相关 tests，确认现有输出命名与冲突风险。
- 2026-03-29: 新增 `tests/test_fusion_output_artifacts.py`，先写两个 TDD 用例（metrics 导出 + run 隔离）。
- 2026-03-29: 运行 RED：`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python -m unittest tests.test_fusion_output_artifacts -v`，结果 `FAILED (errors=2)`（缺少 `export_metrics_artifacts`、`prepare_run_output_dir`）。
- 2026-03-29: 修改 `src/fusion_common.py`，新增 run 目录和指标导出 helper，并将 attention / stacking 训练主流程接入固定文件名落盘。
- 2026-03-29: 更新 `collect_attention_diagnostics()`，支持固定 `attention_curve.png` 输出（兼容旧前缀逻辑）。
- 2026-03-29: 运行 GREEN：同命令复跑，结果 `Ran 2 tests ... OK`。
- 2026-03-29: 回归验证：`python -m unittest tests.test_attention_entrypoints tests.test_run_all_modes tests.test_fusion_output_artifacts -v`，结果 `Ran 6 tests ... OK`。
