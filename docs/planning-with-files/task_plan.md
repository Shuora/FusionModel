# Task Plan: Canonical Final Metric Handling Follow-up Fix

## Goal
统一“canonical final metric”的来源与优先级规则，确保 report discovery、stage2 execution summary、以及 tests 使用同一套语义；并兼容可选 `level3_router=moe` 时后续阶段产物可覆盖 stacking 阶段的 final。

## Scope (Touch Only If Needed)
- src/experiments/stage2_multiclass.py
- src/report.py
- src/train.py (only if testing non-default meta_artifacts_dir override is easy)
- tests/pipeline/test_protocol_execution.py
- tests/pipeline/test_train_stage_dispatch.py
- tests/pipeline/test_stacking_pipeline.py
- tests/pipeline/test_moe_pipeline.py (only if needed)

## Phases
| Phase | Status | Notes |
|------:|--------|-------|
| 1. Baseline discovery (code + tests) | in_progress | 找出当前 summary/report/test 对 final metric 文件的分歧点 |
| 2. Define canonical rule | pending | 明确优先级：later-stage final > stacking final > meta_metrics(compat) |
| 3. Implement narrow fixes | pending | 只改必要处，避免扩散 |
| 4. Update/add tests | pending | 语义断言，不绑定具体内部实现细节 |
| 5. Run focused tests | pending | 只跑相关 tests，避免跑全套 |
| 6. Commit follow-up fix | pending | 输出 SHA/commit message/变更文件/测试结果 |

## Canonical Rule (Draft)
- Prefer the latest-stage `final_metrics.json` when present (e.g. MoE / level3 router stage can supersede stacking).
- Otherwise use stacking `final_metrics.json` if present.
- Fall back to legacy `meta_metrics.json` only for backward compatibility (read-only).

## Risks / Open Questions
- 不同 stage 的 artifacts 目录结构是否一致（stacking vs moe vs protocol runner）。
- report discovery 当前是扫目录还是直接 hardcode 到 stacking/meta_metrics.json。

