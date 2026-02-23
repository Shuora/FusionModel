# Progress Log

## Session: 2026-02-23

### Batch 1: 数据入口核心模块
- **Status:** complete
- **Started:** 2026-02-23 14:27
- **Finished:** 2026-02-23 14:36
- Actions taken:
  - 创建 `src/common`, `src/data`, `tests/common`, `tests/data` 目录。
  - 通过 TDD 实现 `structured_logging`。
  - 通过 TDD 实现 `tls_filter` strict/relaxed 过滤核心。
  - 通过 TDD 实现 `build_dataset` 路径与切分骨架。
  - 创建并更新 `doc/planning-with-files` 三文件。
- Files created/modified:
  - `src/common/structured_logging.py`
  - `src/data/tls_filter.py`
  - `src/data/build_dataset.py`
  - `tests/conftest.py`
  - `tests/common/test_structured_logging.py`
  - `tests/data/test_tls_filter.py`
  - `tests/data/test_build_dataset.py`
  - `doc/planning-with-files/task_plan.md`
  - `doc/planning-with-files/findings.md`
  - `doc/planning-with-files/progress.md`

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Task1 RED | `pytest -q tests/common/test_structured_logging.py` | import 失败 | 失败（符合预期） | ✓ |
| Task1 GREEN | `pytest -q tests/common/test_structured_logging.py` | 通过 | 2 passed | ✓ |
| Task2 RED | `pytest -q tests/data/test_tls_filter.py` | import 失败 | 失败（符合预期） | ✓ |
| Task2 GREEN | `pytest -q tests/data/test_tls_filter.py` | 通过 | 5 passed | ✓ |
| Task3 RED | `pytest -q tests/data/test_build_dataset.py` | import 失败 | 失败（符合预期） | ✓ |
| Task3 GREEN | `pytest -q tests/data/test_build_dataset.py` | 通过 | 3 passed | ✓ |
| Batch1 Full | `pytest -q tests/common/test_structured_logging.py tests/data/test_tls_filter.py tests/data/test_build_dataset.py` | 全绿 | 10 passed | ✓ |

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-02-23 14:31 | `ModuleNotFoundError: src` in pytest | 1 | 新增 `tests/conftest.py` 修复导入路径 |

## Next Batch Candidates
1. 实现 pcap 读取与会话双向聚合。
2. 实现 manifest 持久化（parquet/csv fallback）。
3. 实现预处理进度条与 icon 日志对接。

### Batch 2: Day1 + Day2 前半落地
- **Status:** complete
- **Started:** 2026-02-23 14:45
- **Finished:** 2026-02-23 15:00
- Actions taken:
  - 通过 TDD 实现 `dataset_inventory`（数据盘点、capture 分组 split、泄漏检查、split 落盘）。
  - 通过 TDD 实现 `pcap_sessionizer`（TCP 会话双向聚合、UDP 忽略、TLS/非TLS 分类接入）。
  - 通过 TDD 实现 `preprocess_source`（扫描->split->过滤->manifest 输出 + icon 日志 + tqdm）。
  - 全量回归测试，确认 17 个测试全部通过。
- Files created/modified:
  - `src/data/dataset_inventory.py`
  - `src/data/pcap_sessionizer.py`
  - `src/data/preprocess.py`
  - `tests/data/test_dataset_inventory.py`
  - `tests/data/test_pcap_sessionizer.py`
  - `tests/data/test_preprocess_pipeline.py`
  - `doc/planning-with-files/task_plan.md`
  - `doc/planning-with-files/findings.md`
  - `doc/planning-with-files/progress.md`

## Test Results (Batch 2)
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Task4 RED | `pytest -q tests/data/test_dataset_inventory.py` | import 失败 | 失败（符合预期） | ✓ |
| Task4 GREEN | `pytest -q tests/data/test_dataset_inventory.py` | 通过 | 4 passed | ✓ |
| Task5 RED | `pytest -q tests/data/test_pcap_sessionizer.py` | import 失败 | 失败（符合预期） | ✓ |
| Task5 GREEN | `pytest -q tests/data/test_pcap_sessionizer.py` | 通过 | 2 passed | ✓ |
| Task6 RED | `pytest -q tests/data/test_preprocess_pipeline.py` | import 失败 | 失败（符合预期） | ✓ |
| Task6 GREEN | `pytest -q tests/data/test_preprocess_pipeline.py` | 通过 | 1 passed | ✓ |
| Batch1+2 Full | `pytest -q` | 全绿 | 17 passed | ✓ |

## Error Log (Batch 2)
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-02-23 14:48 | `src.data.dataset_inventory` 不存在 | 1 | 新增模块并实现最小功能 |
| 2026-02-23 14:52 | `src.data.pcap_sessionizer` 不存在 | 1 | 新增模块并实现 dpkt 会话聚合 |
| 2026-02-23 14:56 | `src.data.preprocess` 不存在 | 1 | 新增模块并实现预处理总流程 |

## Next Batch Candidates
1. 实现 `RGB/TLS token` 产物生成（`rgb_shard_*.npz`、`seq_shard_*.npz`）。
2. 接入 `strict/full` 双策略跑批命令与配置文件。
3. 实现训练入口 `train.py` 与最小 smoke test。

### Batch 3: Day2 后半落地
- **Status:** complete
- **Started:** 2026-02-23 15:05
- **Finished:** 2026-02-23 15:18
- Actions taken:
  - 通过 TDD 实现 `feature_encoder`：`encode_session_rgb`、`encode_tls_tokens`、`save_feature_shards`。
  - 将 `feature_encoder` 接入 `preprocess_source`，实际生成 `rgb_shard_00000.npz` 与 `seq_shard_00000.npz`。
  - 实现 `preprocess_runner`，支持 `strict/full` 双策略批处理并映射 filter mode。
  - 回归全量测试，确认 21 个测试全部通过。
- Files created/modified:
  - `src/data/feature_encoder.py`
  - `src/data/preprocess.py`
  - `src/data/preprocess_runner.py`
  - `tests/data/test_feature_encoder.py`
  - `tests/data/test_preprocess_pipeline.py`
  - `tests/data/test_preprocess_runner.py`
  - `doc/planning-with-files/task_plan.md`
  - `doc/planning-with-files/findings.md`
  - `doc/planning-with-files/progress.md`

## Test Results (Batch 3)
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Task7 RED | `pytest -q tests/data/test_feature_encoder.py` | import 失败 | 失败（符合预期） | ✓ |
| Task7 GREEN | `pytest -q tests/data/test_feature_encoder.py` | 通过 | 3 passed | ✓ |
| Task8 GREEN | `pytest -q tests/data/test_preprocess_pipeline.py tests/data/test_feature_encoder.py` | 通过 | 4 passed | ✓ |
| Task9 RED | `pytest -q tests/data/test_preprocess_runner.py` | import 失败 | 失败（符合预期） | ✓ |
| Task9 GREEN | `pytest -q tests/data/test_preprocess_runner.py` | 通过 | 1 passed | ✓ |
| Batch1+2+3 Full | `pytest -q` | 全绿 | 21 passed | ✓ |

## Error Log (Batch 3)
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-02-23 15:07 | `src.data.feature_encoder` 不存在 | 1 | 新增模块并实现编码与落盘 |
| 2026-02-23 15:14 | `src.data.preprocess_runner` 不存在 | 1 | 新增 runner 并接入策略映射 |

### Batch 4: Day3 训练骨架落地
- **Status:** complete
- **Started:** 2026-02-23 15:22
- **Finished:** 2026-02-23 15:31
- Actions taken:
  - 通过 TDD 新增 `tests/pipeline/test_train_eval_report.py` 端到端 smoke test。
  - 实现 `pipeline_data` 聚合加载 `rgb/seq/manifest`。
  - 实现 `train.py`（run 目录、config/train.log/metrics/checkpoints 落盘）。
  - 实现 `evaluate.py`（`eval_<split>.json` + confusion matrix csv）。
  - 实现 `report.py`（`report.md` + `learning_curve.png`）。
  - 回归测试通过，当前共 22 个测试。
- Files created/modified:
  - `src/pipeline_data.py`
  - `src/train.py`
  - `src/evaluate.py`
  - `src/report.py`
  - `tests/pipeline/test_train_eval_report.py`
  - `doc/planning-with-files/task_plan.md`
  - `doc/planning-with-files/findings.md`
  - `doc/planning-with-files/progress.md`

## Test Results (Batch 4)
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Task9 RED | `pytest -q tests/pipeline/test_train_eval_report.py` | import 失败 | 失败（符合预期） | ✓ |
| Task9 GREEN | `pytest -q tests/pipeline/test_train_eval_report.py` | 通过 | 1 passed | ✓ |
| Batch1~4 Full | `pytest -q` | 全绿 | 22 passed | ✓ |

## Error Log (Batch 4)
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-02-23 15:24 | 缺少 `src.evaluate/src.train/src.report` | 1 | 新增三模块并实现最小可运行链路 |
| 2026-02-23 15:29 | `torch/matplotlib` warning 输出 | 1 | 记录 warning，确认不影响通过 |

### Batch 5: Day4 融合训练最小实装
- **Status:** complete
- **Started:** 2026-02-23 15:36
- **Finished:** 2026-02-23 15:45
- Actions taken:
  - 通过 TDD 新增融合模型测试 `tests/models/test_fusion_model.py`。
  - 实现 `TinyFusionClassifier`（image 分支 + tls 分支 + 双向 cross-attn + gate + 三头输出）。
  - 新增 `load_policy_multimodal_data` 并保留旧 `load_policy_data` 兼容。
  - 重构 `train.py`：切换融合模型训练，支持 warmup/fusion 损失策略，日志增加 `gate_mean`。
  - 重构 `evaluate.py`：按融合模型推理并输出 `gate_mean` 指标。
  - 扩展 smoke test 断言 `model_type: TinyFusionClassifier` 与 `gate_mean` 日志。
  - 全量回归通过（24 tests）。
- Files created/modified:
  - `src/models/fusion_model.py`
  - `src/pipeline_data.py`
  - `src/train.py`
  - `src/evaluate.py`
  - `tests/models/test_fusion_model.py`
  - `tests/pipeline/test_train_eval_report.py`
  - `doc/planning-with-files/task_plan.md`
  - `doc/planning-with-files/findings.md`
  - `doc/planning-with-files/progress.md`

## Test Results (Batch 5)
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Task10 RED | `pytest -q tests/models/test_fusion_model.py` | import 失败 | 失败（符合预期） | ✓ |
| Task10 GREEN | `pytest -q tests/models/test_fusion_model.py` | 通过 | 2 passed | ✓ |
| Smoke RED | `pytest -q tests/pipeline/test_train_eval_report.py` | 断言不满足 | 失败（符合预期） | ✓ |
| Smoke GREEN | `pytest -q tests/models/test_fusion_model.py tests/pipeline/test_train_eval_report.py` | 通过 | 3 passed | ✓ |
| Batch1~5 Full | `pytest -q` | 全绿 | 24 passed | ✓ |

## Error Log (Batch 5)
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-02-23 15:38 | `src.models` 不存在 | 1 | 新建 `src/models` 并实现 `fusion_model.py` |
| 2026-02-23 15:40 | smoke test 未断言融合模型信息 | 1 | 扩展 test 断言 `model_type` 与 `gate_mean` |

### Batch 6: Day5 Stacking 落地
- **Status:** complete
- **Started:** 2026-02-23 15:47
- **Finished:** 2026-02-23 15:54
- Actions taken:
  - 通过 TDD 新增 `tests/pipeline/test_stacking_pipeline.py`。
  - 实现 `src/stacking.py`：OOF 元特征构建、XGBoost meta-learner 训练、测试集评估与产物落盘。
  - 输出 `run_dir/stacking/{oof_meta_train.npz,meta_test.npz,meta_metrics.json,meta_model.json}`。
  - 回归测试通过，覆盖 stacking 全链路。
- Files created/modified:
  - `src/stacking.py`
  - `tests/pipeline/test_stacking_pipeline.py`
  - `doc/planning-with-files/task_plan.md`
  - `doc/planning-with-files/findings.md`
  - `doc/planning-with-files/progress.md`

## Test Results (Batch 6)
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Task11 RED | `pytest -q tests/pipeline/test_stacking_pipeline.py` | import 失败 | 失败（符合预期） | ✓ |
| Task11 GREEN | `pytest -q tests/pipeline/test_stacking_pipeline.py` | 通过 | 1 passed | ✓ |
| Batch1~6 Full | `pytest -q` | 全绿 | 26 passed | ✓ |

## Error Log (Batch 6)
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-02-23 15:49 | `src.stacking` 不存在 | 1 | 新增模块并实现 OOF + meta 训练 |

### Batch 7: Day6 MoE + 消融清单落地
- **Status:** complete
- **Started:** 2026-02-23 15:55
- **Finished:** 2026-02-23 15:59
- Actions taken:
  - 通过 TDD 新增 `tests/pipeline/test_moe_pipeline.py`。
  - 实现 `src/moe.py`：冻结 fusion 主干，训练 router 并输出评估指标。
  - 新增 `src/ablation.py` 与 `tests/pipeline/test_ablation_plan.py`，自动生成论文消融矩阵清单。
  - 回归测试通过，覆盖 moe/ablation 流程。
- Files created/modified:
  - `src/moe.py`
  - `src/ablation.py`
  - `tests/pipeline/test_moe_pipeline.py`
  - `tests/pipeline/test_ablation_plan.py`
  - `doc/planning-with-files/task_plan.md`
  - `doc/planning-with-files/findings.md`
  - `doc/planning-with-files/progress.md`

## Test Results (Batch 7)
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Task12 RED | `pytest -q tests/pipeline/test_moe_pipeline.py` | import 失败 | 失败（符合预期） | ✓ |
| Task12 GREEN | `pytest -q tests/pipeline/test_moe_pipeline.py` | 通过 | 1 passed | ✓ |
| Task13 GREEN | `pytest -q tests/pipeline/test_ablation_plan.py` | 通过 | 1 passed | ✓ |
| Batch1~7 Full | `pytest -q` | 全绿 | 28 passed | ✓ |

## Error Log (Batch 7)
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-02-23 15:56 | `src.moe` 不存在 | 1 | 新增模块并实现 router 训练与评估 |

### Batch 8: 训练日志规范 + Stage 调度补齐
- **Status:** complete
- **Started:** 2026-02-23 16:00
- **Finished:** 2026-02-23 16:09
- Actions taken:
  - 先 RED：扩展 `tests/pipeline/test_train_eval_report.py`，新增日志字段断言；新增 `tests/pipeline/test_train_stage_dispatch.py` 验证 `stage=stacking|moe` 自动调度。
  - 实现 `src/train.py`：
    - 启动日志新增 `git_commit`、`config_summary`、`dataset_stats(class_dist)`。
    - epoch 日志新增 `train_macroF1`，并保留 `val_macroF1/lr/time`。
    - 新增 NaN loss、invalid grad norm、gradient explosion 显式报错；梯度裁剪告警降级处理。
    - 新增 train/val 双 `tqdm` 进度条（epoch/batch、ETA、loss/acc/lr）。
    - 新增 `--stage stacking|moe` 子流程 dispatch（自动调用 `src/stacking.py`/`src/moe.py`）。
  - 目标测试与全量回归均通过。
- Files created/modified:
  - `src/train.py`
  - `tests/pipeline/test_train_eval_report.py`
  - `tests/pipeline/test_train_stage_dispatch.py`
  - `doc/planning-with-files/task_plan.md`
  - `doc/planning-with-files/findings.md`
  - `doc/planning-with-files/progress.md`

## Test Results (Batch 8)
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Batch8 RED | `pytest -q tests/pipeline/test_train_eval_report.py tests/pipeline/test_train_stage_dispatch.py` | 失败 | 3 failed（符合预期） | ✓ |
| Batch8 GREEN | `pytest -q tests/pipeline/test_train_eval_report.py tests/pipeline/test_train_stage_dispatch.py` | 通过 | 3 passed | ✓ |
| Batch1~8 Full | `pytest -q` | 全绿 | 30 passed | ✓ |

## Error Log (Batch 8)
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-02-23 16:01 | 日志字段断言缺失（`git_commit/config_summary/dataset_stats/train_macroF1`） | 1 | 扩展 train 启动/epoch 日志 |
| 2026-02-23 16:02 | `stage=stacking|moe` 不生成子流程产物 | 1 | 新增 `train.py` stage dispatch 逻辑 |

### Batch 9: 消融结果总表聚合
- **Status:** complete
- **Started:** 2026-02-23 16:10
- **Finished:** 2026-02-23 16:15
- Actions taken:
  - 先 RED：扩展 `tests/pipeline/test_ablation_plan.py`，新增 `write_ablation_summary` 汇总测试。
  - 在 `src/ablation.py` 增加：
    - `collect_ablation_summary(run_root)` 聚合 `runs/ablation/<group>/<run_id>/` 指标；
    - `write_ablation_summary(run_root, output_csv)` 写出 summary CSV；
    - CLI 新增 `--mode summary` 与 `--run-root` 参数。
  - 按优先级读取指标来源：`stacking/meta_metrics.json` > `moe/moe_metrics.json` > `eval_test.json`，并补充 `metrics.csv` 的 best val 指标。
  - 目标测试与全量回归通过。
- Files created/modified:
  - `src/ablation.py`
  - `tests/pipeline/test_ablation_plan.py`
  - `doc/planning-with-files/task_plan.md`
  - `doc/planning-with-files/findings.md`
  - `doc/planning-with-files/progress.md`

## Test Results (Batch 9)
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Batch9 RED | `pytest -q tests/pipeline/test_ablation_plan.py` | 失败 | import error（符合预期） | ✓ |
| Batch9 GREEN | `pytest -q tests/pipeline/test_ablation_plan.py` | 通过 | 3 passed | ✓ |
| Batch1~9 Full | `pytest -q` | 全绿 | 31 passed | ✓ |

## Error Log (Batch 9)
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-02-23 16:11 | `write_ablation_summary` 不存在导致收集失败 | 1 | 在 `src/ablation.py` 实现 summary 聚合与 CLI 模式 |
