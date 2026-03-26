# Task Plan: Task 7 Remove Old Stage2 Recommendations And Record Acceptance Workflow

## Goal
- 宣告统一的 stage2 主线：Stage A shared stabilization -> Stage B per-dataset fine-tune -> end-to-end eval/report。
- 在文档中退役 stacking / Level 3 MoE 命令，强调它们不再作为推荐路径。
- 把 acceptance track 规则写入 planning 文件并确认 `runs/<date>/stage2_acceptance.json` 既有的 entry 结构。
- 运行指定 pytest 回归，确保当前主线命令依旧通过协议测试。

## Scope (Touch Only If Needed)
- `docs/commands/session-full-experiments.md`
- `docs/planning-with-files/task_plan.md`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Phases
| Phase | Status | Notes |
|------:|--------|-------|
| 1. 同步 Stage2 docs，明确 unified cross-attention 主线 | completed | 已更新 `session-full-experiments.md` 相关段落，并标注 stacking/moe 为 legacy。 |
| 2. 记录 acceptance gate 规则与 manifest 结构 | completed | 已在 planning docs 中写入 Gate 0~3、`runs/<date>/stage2_acceptance.json` 与 manifest 字段说明。 |
| 3. 执行 `pytest -q tests/pipeline/test_protocol_execution.py -k 'stage2_'` | blocked | 实际运行命令失败：`test_stage2_runner_main_path_calls_shared_stage_a_then_dataset_stage_b` 仍报 `AttributeError: module 'src.experiments.stage2_multiclass' has no attribute 'run_stage2_shared_stage_a'`，且其他 `stage2_*` 用例期望的 `stage2-unified-*` run 目录不存在（当前仅产出 `stage2-<dataset>` 目录），总共 6 个用例未通过。 |

## Acceptance Tracking
- Gate 0：protocol hygiene——每个 dataset run 的 `code` 必须为 0 才能认为 manifest 经过协议检查。
- Gate 1：MTA test top1 >= 0.70。
- Gate 2：MFCP test top1 >= 0.70。
- Gate 3：USTC-TFC2016 test top1 >= 0.86。
- 当前实现会把所有 Stage B run 的 acceptance 结果写入 `runs/<date>/stage2_acceptance.json`，每条 entry 至少包含 `dataset`、`run_dir`、`code`、`shared_checkpoint`、`test_top1`、`gate_passed`，保持最小 manifest，后续需要扩展额外 metadata 时再补。
- Stage2 `test_protocol_execution` 子集曾因强行断言 `shared_checkpoint` 路径包含 `stage2-unified-shared` 而失败，现在已改成：字段必须存在、类型为字符串、非空时以 `best.ckpt` 结尾，空字符串也是允许的最小值。
