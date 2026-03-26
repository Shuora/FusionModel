# Progress Log

## 2026-03-26
- Switched the planning docs to a narrow Task 5 code-quality review focused on the minimal stage2 runner main-path replacement.
- Read skills: `using-superpowers`, `planning-with-files`, `dispatching-parallel-agents`.
- Read only the 3 requested files in worktree `stage2-runner-arg-forwarding`.
- Confirmed the intended minimal main path is `run_stage2_shared_stage_a(...) -> run_stage2_stage_b(...) -> stage2_acceptance.json`.
- Identified a must-fix contract gap: `shared_checkpoint` is forwarded through wrappers but discarded inside `_run_stage2_task(...)`.
- Identified a second contract weakness: `gate_passed` is effectively always true when `eval_test.json` is missing, so acceptance output is not a trustworthy gate signal.
- Final review conclusion: `❌ Issues`; current Task 5 replacement is not strong enough to treat the shared checkpoint contract as established for Task 6/7.
- Re-reviewed the same 3 files after the Task 5 fix round.
- Confirmed `shared_checkpoint` now flows into `_run_stage2_task(...)` and is emitted as `--warmup-checkpoint` to the real training argv.
- Confirmed `run_stage_b_dataset_finetune(...)` now reports `test_top1=None` and `gate_passed=False` when `eval_test.json` is missing.
- Confirmed the main-path protocol test now covers stage A kwargs plus acceptance-manifest checkpoint persistence.
- Updated review result to `✅ Approved` for Task 5 minimal main-path replacement quality.
- 重新聚焦到 Task 7：Stage2 文档已经转为 shared Stage A -> per-dataset Stage B -> eval/report 主线，并表明 stacking / Level 3 MoE 已退役。
- Acceptance gate 规则（Gate 0 protocol hygiene，Gate 1/2/3 verifing `MTA>=0.70`、`MFCP>=0.70`、`USTC-TFC2016>=0.86`）与 `runs/<date>/stage2_acceptance.json` manifest 结构已写入 planning docs。
- 准备运行 `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_protocol_execution.py -k 'stage2_'` 检查新的命令链。
- 实际运行 `env -u PYTHONPATH PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest -q tests/pipeline/test_protocol_execution.py -k 'stage2_'`，`test_stage2_runner_main_path_calls_shared_stage_a_then_dataset_stage_b` 报 `AttributeError: module ... has no attribute 'run_stage2_shared_stage_a'`，其余用例通过。
- 进一步调整 `stage2_` 用例：只留下一套字段存在/`best.ckpt` 后缀的契约，因此旧的 `shared_checkpoint` 路径依赖不再造成失败。
- 最新 pytest 运行仍未过：`stage2_*` 测试找不到 `stage2-unified-*` run 目录（代码只输出 `stage2-<dataset>`），且 `run_stage2_shared_stage_a` 仍未导出，导致 6 条测试失效。
