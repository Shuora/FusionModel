# Session Full MVTBA Protocol Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在现有工程中落地 `session_full` 论文口径预处理与两阶段评估流程，保证可复现、可测试、可命令化执行。

**Architecture:** 在不破坏现有 `strict/full/relaxed` 的前提下，新增 `session_full` 分支：`PCAP -> session pcap -> RGB+时序 -> 清理 session pcap`。评估层新增两类编排器：阶段1混合二分类（ISCX normal vs 其余 malicious）和阶段2三数据集独立多分类（MTA-7 / MFCP-6 / USTC-10）。

**Tech Stack:** Python 3.9、dpkt、numpy、pandas、Pillow、pytest、现有 `src/train.py`/`src/evaluate.py` 训练评估链路。

---

### Task 1: 新增 `session_full` 策略与 manifest 字段骨架

**Files:**
- Modify: `src/data/preprocess_runner.py`
- Modify: `src/data/build_dataset.py`
- Modify: `src/data/preprocess.py`
- Test: `tests/data/test_preprocess_runner.py`
- Create: `tests/data/test_session_full_schema.py`

**Step 1: Write the failing test**

```python
def test_session_full_policy_emits_tls_flags(tmp_path):
    results = run_preprocess_policies(
        source_root=source_root,
        output_root=output_root,
        policies=["session_full"],
        show_progress=False,
    )
    assert "session_full" in results
    df = pd.read_csv(output_root / "DemoSet" / "session_full" / "manifest" / "session_manifest.csv")
    assert {"is_tls_ssl", "tls_ssl_reason"}.issubset(df.columns)
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/data/test_session_full_schema.py::test_session_full_policy_emits_tls_flags -v`
Expected: FAIL（找不到 `session_full` 或字段缺失）。

**Step 3: Write minimal implementation**

```python
DEFAULT_POLICY_FILTER_MAP = {
    "strict": "strict",
    "full": "strict",
    "relaxed": "relaxed",
    "session_full": "session_full",
}
```

```python
def make_manifest_row(..., is_tls_ssl: bool | None = None, tls_ssl_reason: str | None = None):
    row = {...}
    if is_tls_ssl is not None:
        row["is_tls_ssl"] = bool(is_tls_ssl)
        row["tls_ssl_reason"] = tls_ssl_reason or ""
    return row
```

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/data/test_preprocess_runner.py tests/data/test_session_full_schema.py -v`
Expected: PASS。

**Step 5: Commit**

```bash
git add tests/data/test_session_full_schema.py tests/data/test_preprocess_runner.py src/data/preprocess_runner.py src/data/build_dataset.py src/data/preprocess.py
git commit -m "feat(data): add session_full policy skeleton and manifest tls flags"
```

### Task 2: 实现 Session PCAP 切分与临时目录管理

**Files:**
- Create: `src/data/session_splitcap.py`
- Modify: `src/data/preprocess.py`
- Test: `tests/data/test_session_splitcap.py`

**Step 1: Write the failing test**

```python
def test_split_pcap_to_sessions_and_cleanup(tmp_path):
    session_files = split_pcap_to_session_pcaps(pcap_path, tmp_dir)
    assert len(session_files) == 2
    assert all(p.exists() for p in session_files)
    removed = cleanup_session_pcaps(session_files)
    assert removed == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/data/test_session_splitcap.py::test_split_pcap_to_sessions_and_cleanup -v`
Expected: FAIL（模块/函数不存在）。

**Step 3: Write minimal implementation**

```python
def split_pcap_to_session_pcaps(pcap_path: Path, out_dir: Path) -> list[Path]:
    # 5-tuple 聚合后按 session 写入独立 pcap
    ...
    return written_paths

def cleanup_session_pcaps(paths: Sequence[Path]) -> int:
    removed = 0
    for p in paths:
        if p.exists():
            p.unlink()
            removed += 1
    return removed
```

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/data/test_session_splitcap.py -v`
Expected: PASS。

**Step 5: Commit**

```bash
git add tests/data/test_session_splitcap.py src/data/session_splitcap.py src/data/preprocess.py
git commit -m "feat(data): add session pcap splitcap and cleanup helpers"
```

### Task 3: `session_full` 保留非 TLS 流并写入 TLS 标记

**Files:**
- Modify: `src/data/tls_filter.py`
- Modify: `src/data/build_dataset.py`
- Modify: `src/data/pcap_sessionizer.py`
- Modify: `src/data/preprocess.py`
- Test: `tests/data/test_preprocess_pipeline.py`
- Create: `tests/data/test_session_full_filtering.py`

**Step 1: Write the failing test**

```python
def test_session_full_keeps_non_tls_and_marks_reason(tmp_path):
    summary = preprocess_source(..., policy="session_full", show_progress=False)
    assert summary["accepted_sessions"] == 2
    assert summary["dropped_sessions"] == 0
    df = pd.read_csv(manifest_csv)
    assert set(df["is_tls_ssl"].astype(int).tolist()) == {0, 1}
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/data/test_session_full_filtering.py -v`
Expected: FAIL（目前 non-TLS 会被 dropped）。

**Step 3: Write minimal implementation**

```python
def split_tls_and_non_tls(sessions, mode="strict"):
    if mode == "session_full":
        accepted = []
        for session in sessions:
            ok, reason = classify_session_as_tls(..., mode="relaxed")
            accepted.append({**session, "is_tls_ssl": bool(ok), "tls_ssl_reason": reason})
        return accepted, []
    ...
```

```python
if policy == "session_full":
    # accepted 包含 TLS+non-TLS，manifest 必写 is_tls_ssl/tls_ssl_reason
    ...
```

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/data/test_preprocess_pipeline.py tests/data/test_session_full_filtering.py -v`
Expected: PASS。

**Step 5: Commit**

```bash
git add tests/data/test_session_full_filtering.py tests/data/test_preprocess_pipeline.py src/data/tls_filter.py src/data/build_dataset.py src/data/pcap_sessionizer.py src/data/preprocess.py
git commit -m "feat(data): keep non-tls flows in session_full with tls flags"
```

### Task 4: 特征提取后自动清理 Session PCAP，并保留抽检 RGB 图

**Files:**
- Modify: `src/data/feature_encoder.py`
- Modify: `src/data/preprocess.py`
- Test: `tests/data/test_feature_encoder.py`
- Create: `tests/data/test_preview_and_cleanup.py`

**Step 1: Write the failing test**

```python
def test_session_full_cleanup_and_preview_png(tmp_path):
    preprocess_source(..., policy="session_full", show_progress=False)
    assert not (out_root / "DemoSet" / "session_full" / "tmp_sessions").exists()
    preview = list((out_root / "DemoSet" / "session_full" / "debug" / "preview_png").rglob("*.png"))
    assert len(preview) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/data/test_preview_and_cleanup.py::test_session_full_cleanup_and_preview_png -v`
Expected: FAIL（未清理 tmp_sessions 或无 preview png）。

**Step 3: Write minimal implementation**

```python
def save_feature_shards(..., preview_dir: Path | None = None, preview_per_family: int = 20):
    ...
    if preview_dir is not None:
        _save_preview_png(rgb_arr, session_ids, labels, preview_dir, preview_per_family)
```

```python
if policy == "session_full" and cleanup_sessions:
    removed = cleanup_session_pcaps(session_files)
    log_fn(format_log_line(..., event="tmp_session_cleanup", kv={"removed": removed}))
```

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/data/test_feature_encoder.py tests/data/test_preview_and_cleanup.py -v`
Expected: PASS。

**Step 5: Commit**

```bash
git add tests/data/test_feature_encoder.py tests/data/test_preview_and_cleanup.py src/data/feature_encoder.py src/data/preprocess.py
git commit -m "feat(data): auto-clean tmp session pcaps and keep rgb previews"
```

### Task 5: 预处理 CLI 增加 `session_full` 参数与日志/进度条细节

**Files:**
- Modify: `src/data/preprocess_runner.py`
- Modify: `src/data/preprocess.py`
- Test: `tests/data/test_preprocess_runner.py`

**Step 1: Write the failing test**

```python
def test_runner_passes_cleanup_and_preview_flags(tmp_path):
    results = run_preprocess_policies(
        ...,
        policies=["session_full"],
        cleanup_sessions=True,
        preview_per_family=5,
        show_progress=False,
    )
    assert results["session_full"]["policy"] == "session_full"
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/data/test_preprocess_runner.py::test_runner_passes_cleanup_and_preview_flags -v`
Expected: FAIL（函数签名或透传参数缺失）。

**Step 3: Write minimal implementation**

```python
def run_preprocess_policies(..., cleanup_sessions: bool = True, preview_per_family: int = 20, ...):
    ...
    preprocess_source(..., cleanup_sessions=cleanup_sessions, preview_per_family=preview_per_family)
```

```python
parser.add_argument("--cleanup-sessions", action="store_true")
parser.add_argument("--keep-sessions", action="store_true")
parser.add_argument("--preview-per-family", type=int, default=20)
```

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/data/test_preprocess_runner.py -v`
Expected: PASS。

**Step 5: Commit**

```bash
git add tests/data/test_preprocess_runner.py src/data/preprocess_runner.py src/data/preprocess.py
git commit -m "feat(data): expose session_full cleanup and preview options in runner"
```

### Task 6: 阶段1混合二分类编排器（ISCX normal vs 其他 malicious）

**Files:**
- Create: `src/experiments/stage1_binary.py`
- Create: `tests/pipeline/test_stage1_binary_protocol.py`
- Modify: `src/pipeline_data.py`

**Step 1: Write the failing test**

```python
def test_stage1_requires_all_datasets(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_stage1_manifest(processed_root=tmp_path, policy="session_full")
```

```python
def test_stage1_label_mapping(tmp_path):
    df = build_stage1_manifest(...)
    assert set(df["label_binary"].unique()) == {0, 1}
    assert (df[df["dataset"] == "ISCX"]["label_binary"] == 0).all()
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/pipeline/test_stage1_binary_protocol.py -v`
Expected: FAIL（模块不存在）。

**Step 3: Write minimal implementation**

```python
REQUIRED = {"ISCX", "MFCP", "MTA", "USTC-TFC2016"}

def build_stage1_manifest(processed_root: Path, policy: str) -> pd.DataFrame:
    present = set(_discover_datasets(processed_root, policy))
    missing = REQUIRED - present
    if missing:
        raise FileNotFoundError(f"stage1 missing datasets: {sorted(missing)}")
    df = _concat_manifests(...)
    df["label_binary"] = np.where(df["dataset"] == "ISCX", 0, 1)
    return df
```

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/pipeline/test_stage1_binary_protocol.py -v`
Expected: PASS。

**Step 5: Commit**

```bash
git add tests/pipeline/test_stage1_binary_protocol.py src/experiments/stage1_binary.py src/pipeline_data.py
git commit -m "feat(exp): add strict stage1 binary protocol runner"
```

### Task 7: 阶段2三数据集独立多分类编排器

**Files:**
- Create: `src/experiments/stage2_multiclass.py`
- Create: `tests/pipeline/test_stage2_multiclass_protocol.py`

**Step 1: Write the failing test**

```python
def test_stage2_tasks_are_fixed():
    tasks = build_stage2_tasks()
    assert tasks == [
        {"dataset": "MTA", "num_classes": 7},
        {"dataset": "MFCP", "num_classes": 6},
        {"dataset": "USTC-TFC2016", "num_classes": 10},
    ]
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/pipeline/test_stage2_multiclass_protocol.py -v`
Expected: FAIL（模块不存在）。

**Step 3: Write minimal implementation**

```python
STAGE2_TASKS = [
    {"dataset": "MTA", "num_classes": 7},
    {"dataset": "MFCP", "num_classes": 6},
    {"dataset": "USTC-TFC2016", "num_classes": 10},
]

def build_stage2_tasks() -> list[dict]:
    return [dict(x) for x in STAGE2_TASKS]
```

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/pipeline/test_stage2_multiclass_protocol.py -v`
Expected: PASS。

**Step 5: Commit**

```bash
git add tests/pipeline/test_stage2_multiclass_protocol.py src/experiments/stage2_multiclass.py
git commit -m "feat(exp): add stage2 multiclass protocol task builder"
```

### Task 8: 命令文档与 smoke 验证

**Files:**
- Modify: `README.md`
- Create: `docs/commands/session-full-experiments.md`

**Step 1: Write the failing doc test (manual checklist)**

```text
Checklist:
1) `python -m src.data.preprocess_runner --help` 包含 session_full 相关参数
2) `python -m src.experiments.stage1_binary --help` 可运行
3) `python -m src.experiments.stage2_multiclass --help` 可运行
```

**Step 2: Run checklist command**

Run: `python -m src.data.preprocess_runner --help && python -m src.experiments.stage1_binary --help && python -m src.experiments.stage2_multiclass --help`
Expected: 三条命令均返回 0。

**Step 3: Write docs**

```markdown
## Session Full 预处理
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --cleanup-sessions \
  --preview-per-family 20
```

**Step 4: Re-run checklist**

Run: `python -m src.data.preprocess_runner --help`
Expected: 输出参数完整，文档命令可直接复制运行。

**Step 5: Commit**

```bash
git add README.md docs/commands/session-full-experiments.md
git commit -m "docs: add session_full preprocessing and stage1/stage2 run commands"
```

### Final Verification Gate

1. `pytest -q tests/data tests/pipeline -k "session_full or stage1 or stage2"`
2. `pytest -q`
3. 试跑（小样本）：
`python src/data/preprocess_runner.py --source-root SourceData --output-root outputs/processed --policies session_full --datasets USTC-TFC2016 --cleanup-sessions`
4. 确认产物：
   - `outputs/processed/USTC-TFC2016/session_full/manifest/session_manifest.csv`
   - `outputs/processed/USTC-TFC2016/session_full/debug/preview_png/`
   - `outputs/processed/USTC-TFC2016/session_full/tmp_sessions/`（默认已清理）

