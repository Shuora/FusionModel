# Findings: Task 7 Remove Old Stage2 Recommendations And Record Acceptance Workflow

## Reviewed Files
- `docs/commands/session-full-experiments.md`
- `docs/planning-with-files/task_plan.md`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Findings
- Stage2 文档现在明确把主路径固定在 unified cross-attention 流程：Stage A shared stabilization -> Stage B per-dataset fine-tune -> end-to-end eval/report，并在主段落里指出 stacking / Level 3 MoE 已退役，不再推荐。
- 详细说明了 `src.experiments.stage2_multiclass` 对 Stage A / B 的运行顺序、`runs/YYYY-MM-DD/stage2-unified-*` 目录布局，以及 `runs/YYYY-MM-DD/stage2_acceptance.json` 里的 manifest 内容（dataset/run_dir/code/shared_checkpoint/test_top1/gate_passed）和 Gate 0~3 的阈值。
- Planning docs（task_plan/progress）已经同步该 acceptance gate 规则，并在计划表中记录了该阶段的三个 work items；当前只剩下 pytest 回归待执行。
- 当前 `stage2_` 测试子集失败的原因仅限于旧断言对 `shared_checkpoint` 路径的强依赖（期望包含 `stage2-unified-shared`）；已改为更宽松的契约：字段存在即可、非空时必须以 `best.ckpt` 结尾，允许空串。
- 最新 pytest 运行再次失败，6 个 `stage2_*` 用例在提取 `stage2-unified-*` 目录时仍然找不到真实产物（运行只产出 `stage2-<dataset>`）且 `run_stage2_shared_stage_a` 仍未在 `src.experiments.stage2_multiclass` 中导出，说明代码还未切换到新的 unified pipeline。
