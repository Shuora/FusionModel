# Stage1 Paper Protocol Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `stage1_binary` 改为按论文表 1-3 严格构造 stage1 binary manifest。

**Architecture:** 保留现有 `session_full` 预处理产物，不修改底层 session 切分。协议层在 `src/experiments/stage1_binary.py` 中引入论文表驱动配置，按 group/family 和精确 train/test 配额从 `session_manifest` 稳定裁样；若数量不足则报错。

**Tech Stack:** Python, pandas, pytest

---

### Task 1: 更新 planning-with-files 文档

**Files:**
- Modify: `docs/planning-with-files/task_plan.md`
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`

- [ ] 记录本轮任务目标与实现边界。
- [ ] 在 findings 中写入论文表 1-3 的关键结论。
- [ ] 在 progress 中记录后续测试与实现结果。

### Task 2: 先写失败测试

**Files:**
- Modify: `tests/pipeline/test_stage1_binary_protocol.py`
- Optional: `tests/pipeline/test_protocol_execution.py`

- [ ] 新增测试，验证 `torrent` 与 `PUA` 被纳入论文协议。
- [ ] 新增测试，验证按论文配额精确裁样。
- [ ] 新增测试，验证样本不足时报错。
- [ ] 运行 `pytest` 观察红灯，确认失败原因是旧实现不符合论文协议。

### Task 3: 实现论文表驱动协议

**Files:**
- Modify: `src/experiments/stage1_binary.py`

- [ ] 定义论文表 1-3 的协议配置。
- [ ] 实现 ISCX group 匹配逻辑。
- [ ] 实现 MTA / MFCP family 配额裁样逻辑。
- [ ] 删除旧的“匹配不到时 fallback 到未过滤数据”行为。
- [ ] 让错误信息带上缺口详情。

### Task 4: 更新文档

**Files:**
- Modify: `docs/commands/session-full-experiments.md`
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`

- [ ] 更新 stage1 文档为“论文 Table 1-3 严格复现”。
- [ ] 说明当前复现边界是“类别 + 数量”，不是原作者逐 session 列表还原。

### Task 5: 验证

**Files:**
- Test: `tests/pipeline/test_stage1_binary_protocol.py`
- Test: `tests/pipeline/test_protocol_execution.py`

- [ ] 跑目标测试并确认通过。
- [ ] 记录验证命令与结果。
