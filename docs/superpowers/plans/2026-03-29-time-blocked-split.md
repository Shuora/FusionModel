# Time-Blocked Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove session-level train/test leakage by time-splitting singleton raw captures before session extraction while preserving raw-level splitting for labels with multiple captures.

**Architecture:** `split_data.py` keeps the same processed output layout, but the split policy becomes hybrid. Multi-capture labels continue to split at raw-file granularity; singleton labels are split inside the capture by timestamp, then sessionized independently with boundary-overlap sessions dropped.

**Tech Stack:** Python 3.12, `dpkt`, standard-library `unittest`

---

### Task 1: Add Regression Tests For Singleton Capture Time Splitting

**Files:**
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/tests/test_split_data_tasks.py`
- Test: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/tests/test_split_data_tasks.py`

- [x] **Step 1: Write the failing tests**

```python
def test_split_dataset_time_splits_single_raw_capture(self) -> None:
    ...

def test_split_dataset_drops_boundary_crossing_sessions(self) -> None:
    ...
```

- [x] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_split_data_tasks.SplitDataTaskTests.test_split_dataset_time_splits_single_raw_capture tests.test_split_data_tasks.SplitDataTaskTests.test_split_dataset_drops_boundary_crossing_sessions -v`
Expected: FAIL because `split_data.py` currently extracts all sessions first and cannot time-split a single raw capture.

- [x] **Step 3: Write minimal implementation support in tests**

```python
with patch("split_data.iter_session_payloads", return_value=[...]):
    processed_root = split_dataset(...)
```

- [x] **Step 4: Run test to verify failure mode is correct**

Run: `python3 -m unittest tests.test_split_data_tasks.SplitDataTaskTests.test_split_dataset_time_splits_single_raw_capture tests.test_split_data_tasks.SplitDataTaskTests.test_split_dataset_drops_boundary_crossing_sessions -v`
Expected: FAIL on wrong Train/Test bin counts before production code changes.

### Task 2: Implement Hybrid Split Policy In `split_data.py`

**Files:**
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py`
- Test: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/tests/test_split_data_tasks.py`

- [x] **Step 1: Introduce a reusable packet-to-session payload iterator**

```python
def iter_session_payloads(
    capture_path: os.PathLike[str] | str,
) -> Iterable[tuple[float, tuple[str, str, int, str, int], bytes]]:
    ...
```

- [x] **Step 2: Add singleton raw-capture time split helper**

```python
def split_single_capture_by_time(
    sample: RawSample,
    train_ratio: float,
) -> dict[str, list[SessionSample]]:
    ...
```

- [x] **Step 3: Keep raw-level split for labels with multiple captures**

```python
grouped = _group_samples_by_label(raw_samples)
if len(label_samples) == 1:
    ...
else:
    ...
```

- [x] **Step 4: Run targeted tests**

Run: `python3 -m unittest tests.test_split_data_tasks -v`
Expected: PASS

### Task 3: Update Task Docs And Progress Artifacts

**Files:**
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/task_plan.md`
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/findings.md`
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/progress.md`
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/docs/superpowers/specs/2026-03-29-time-blocked-split-design.md`
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/docs/superpowers/plans/2026-03-29-time-blocked-split.md`

- [x] **Step 1: Record final split policy and verification results**

```markdown
- singleton raw capture: time-blocked split before session extraction
- multi raw capture: raw-level split before session extraction
- boundary-crossing sessions: dropped from both splits
```

- [x] **Step 2: Run verification summary command**

Run: `git diff -- src/split_data.py tests/test_split_data_tasks.py task_plan.md findings.md progress.md docs/superpowers/specs/2026-03-29-time-blocked-split-design.md docs/superpowers/plans/2026-03-29-time-blocked-split.md`
Expected: diff only contains the intended split-policy, test, and documentation changes.

### Task 4: Address Code Review Quality Findings

**Files:**
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py`
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/tests/test_split_data_tasks.py`

- [x] **Step 1: Add failing tests for robustness and fallback**

```python
def test_split_dataset_singleton_read_failure_does_not_abort_whole_task(self) -> None:
    ...

def test_split_dataset_same_timestamp_stream_falls_back_to_packet_order(self) -> None:
    ...
```

- [x] **Step 2: Run tests to confirm RED**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python -m unittest tests.test_split_data_tasks -v`
Expected: FAIL due to singleton exception bubbling and same-timestamp stream all going to Train.

- [x] **Step 3: Implement fixes**

```python
try:
    singleton_splits = split_single_capture_by_time(...)
except Exception:
    logger.error(...)
    continue
```

```python
# pass-1: packet_count/min_ts/max_ts
# pass-2: aggregate with time-boundary or packet-order fallback
```

- [x] **Step 4: Run tests to confirm GREEN**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python -m unittest tests.test_split_data_tasks -v`
Expected: PASS

### Task 5: Fix Path Collision And Rerun Cleanup Findings

**Files:**
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py`
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/tests/test_split_data_tasks.py`

- [x] **Step 1: Add failing tests for collision and rerun cleanup**

```python
def test_split_dataset_same_label_same_stem_raws_have_unique_bin_paths(self) -> None:
    ...

def test_split_dataset_rerun_cleans_previous_outputs(self) -> None:
    ...
```

- [x] **Step 2: Run tests to confirm RED**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python -m unittest tests.test_split_data_tasks -v`
Expected: FAIL due to bin-path overwrite collisions and old files lingering across reruns.

- [x] **Step 3: Implement fixes**

```python
def build_raw_capture_token(raw_path: Path) -> str:
    ...

def _promote_staged_outputs(processed_root: Path, staging_root: Path) -> None:
    ...
```

- [x] **Step 4: Run tests to confirm GREEN**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python -m unittest tests.test_split_data_tasks -v`
Expected: PASS

### Task 6: Make Output Publication Transactional

**Files:**
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py`
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/tests/test_split_data_tasks.py`

- [x] **Step 1: Add failing test for failed-rerun safety**

```python
def test_split_dataset_failed_rerun_keeps_previous_outputs(self) -> None:
    ...
```

- [x] **Step 2: Run tests to confirm RED**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python -m unittest tests.test_split_data_tasks -v`
Expected: FAIL because current flow clears final outputs before running discovery/write.

- [x] **Step 3: Implement staging + commit/rollback publish**

```python
# write to .split_data_staging first
# promote to pcap_data/metadata only when complete
# rollback old outputs if promotion fails
```

- [x] **Step 4: Run tests to confirm GREEN**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python -m unittest tests.test_split_data_tasks -v`
Expected: PASS

### Task 7: Move Interrupted-Backup Recovery To Preflight

**Files:**
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/src/split_data.py`
- Modify: `/home/shuora/Traffic/FusionModel/.worktrees/time-blocked-split/tests/test_split_data_tasks.py`

- [x] **Step 1: Add failing test for recovery timing gap**

```python
def test_split_dataset_recovers_interrupted_backup_before_pre_promote_failure(self) -> None:
    ...
```

- [x] **Step 2: Run tests to confirm RED**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python -m unittest tests.test_split_data_tasks -v`
Expected: FAIL because interrupted-backup recovery only runs during promote.

- [x] **Step 3: Implement preflight recovery**

```python
# call interrupted-backup recovery at split_dataset start
# keep existing promote-time recovery as defensive guard
```

- [x] **Step 4: Run tests to confirm GREEN**

Run: `source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python -m unittest tests.test_split_data_tasks -v`
Expected: PASS

---

## Execution Result

- RED 阶段已验证：`source /home/shuora/miniconda3/etc/profile.d/conda.sh && conda activate FusionModel && python -m unittest tests.test_split_data_tasks -v` 返回 `FAILED (failures=1, errors=2)`。
- GREEN 阶段已验证：同一命令复跑返回 `Ran 10 tests ... OK`。
- 代码已实现 hybrid split policy，并通过新增测试确认：
  - 单 raw capture 先时间切分再 sessionize；
  - 跨边界 session 双侧丢弃；
  - 多 raw capture 标签保持 raw-level split。
- 第二轮 RED 阶段已验证：同一命令返回 `FAILED (failures=1, errors=1)`，命中 code review 指出的问题。
- 第二轮 GREEN 阶段已验证：同一命令复跑返回 `Ran 12 tests ... OK`。
- 代码质量修复已完成，并通过新增测试确认：
  - singleton 读包失败仅跳过该 capture，不中断整体；
  - 同时间戳流使用 packet-order fallback 做真实 Train/Test 切分；
  - singleton 切分已改为流式两遍扫描，不再 `sorted(...)` 全量驻留 payload。
- 第三轮 RED 阶段已验证：同一命令返回 `FAILED (failures=2)`，命中 session 文件覆盖与 rerun 残留问题。
- 第三轮 GREEN 阶段已验证：同一命令复跑返回 `Ran 14 tests ... OK`。
- 最终修复已完成，并通过新增测试确认：
  - 同 label 下即使 raw capture 同 stem，`manifest.bin_path` 仍唯一；
  - 同一 `processed_root` 连续成功运行时会整体替换 `pcap_data/metadata`，不会残留旧 `.bin`。
- 第四轮 RED 阶段已验证：同一命令返回 `FAILED (errors=1)`，命中“失败 rerun 清空旧输出”的高危问题。
- 第四轮 GREEN 阶段已验证：同一命令复跑返回 `Ran 15 tests ... OK`。
- 事务发布修复已完成，并通过新增测试确认：
  - 先写 staging，再发布 final；
  - 第二次运行失败时，第一次成功输出保持可用；
  - 成功重跑仍不会保留旧 bin。
- 额外健壮性：若检测到中断遗留 `.split_data_backup_*` 且 final 缺失，会先恢复再发布；复跑测试仍为 `Ran 15 tests ... OK`。
- 第五轮 RED 阶段已验证：同一命令返回 `FAILED (errors=1)`，命中“恢复仅在 promote 执行”的时机缺口。
- 第五轮 GREEN 阶段已验证：同一命令复跑返回 `Ran 16 tests ... OK`。
- 时机缺口修复已完成，并通过新增测试确认：
  - `split_dataset()` 启动即恢复 backup-only 损坏状态；
  - 即使随后在 promote 前失败（如 unknown task），旧输出仍可用；
  - staging/rollback 机制保持不变。
