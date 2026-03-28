# Preprocess Workers Design

## Goal

Make dataset preparation resumable beyond SplitCap, observable during long runs, and able to use multiple CPU cores for post-SplitCap work.

## Scope

- Add fine-grained path filtering so users can run a dataset root or a narrower subtree such as `MTA/Dridex`.
- Preserve existing SplitCap checkpoint behavior.
- Add resume semantics for `sessions_clean` and `cache` outputs.
- Add lightweight progress logging.
- Add configurable multiprocessing for post-processing only.

## Decisions

### Filtering

`prepare_dataset.py` will accept repeatable `--include-path` arguments. Each raw capture is included when its path relative to `SourceData` contains any provided fragment. No filter means current behavior.

### Resume Semantics

- If `sessions_clean/<sample_id>.pcap` already exists, cleaning is skipped.
- If `cache/<sample_id>.npz` already exists, tokenization and cache writing are skipped.
- Sample-level duplicate filtering remains correct by computing fingerprint decisions in the parent process before heavy work is dispatched.

### Parallelism

- SplitCap remains single-process and checkpointed as-is.
- Post-processing uses a worker pool with a new `--num-workers` option.
- The same worker count is reused for planning-stage payload inspection so the expensive `read_session_bytes(...)` pass no longer stays single-process on large runs.
- Each worker handles cleaning, byte normalization, tokenization, and cache writing for one sample at a time.
- Tokenizer instances are initialized inside workers to avoid cross-process sharing issues.

### Progress Logging

- Add `--progress-every` with a conservative default.
- Log total scheduled items, completed items, skipped items, deduped items, and cache hits.
- The parent-process planning pass must emit its own `[plan]` heartbeat before any worker output exists, because full binary runs can spend tens of minutes deduplicating `sessions_raw` serially.

## Risks

- Multiprocessing increases memory use because each worker loads a tokenizer.
- Duplicate filtering must stay centralized to avoid inconsistent results across workers.
- Existing scripts should stay compatible when new flags are omitted.
