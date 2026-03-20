# Documentation Sync Plan (MobileViT + ET-BERT Adapter)

## Goal

将项目文档与当前代码实现对齐，重点覆盖架构口径、环境准备、实验命令和验证记录。

## Status

- Completed on 2026-03-18（documentation sync）

## Scope

- `README.md`
- `docs/commands/session-full-experiments.md`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 核对当前代码中的 MobileViT 与 ET-BERT 相关实现细节（主干类型、checkpoint/vocab/config 接入能力、能力边界）。
2. 更新 `README.md` 的架构与环境说明，并修正路径与依赖口径。
3. 更新 `session-full-experiments.md` 命令与描述，保持与当前 pipeline 行为一致。
4. 更新 findings/progress，记录当前状态与验证范围（46 项测试）。

---

# Runtime Support Plan (CUDA + num-workers)

## Goal

为 `train/evaluate` 增加可用的 CUDA 运行时支持与 `--num-workers` 参数，并把推荐参数和命令文档同步到实验文档。

## Status

- In progress on 2026-03-19

## Scope

- `src/train.py`
- `src/evaluate.py`
- `tests/pipeline/test_train_eval_report.py`
- `docs/commands/session-full-experiments.md`
- `docs/planning-with-files/findings.md`
- `docs/planning-with-files/progress.md`

## Plan

1. 先补测试，覆盖 `device=auto/cpu/cuda` 解析、训练配置持久化、评估阶段 CUDA 不可用时的安全回退。
2. 在 `src.train` 中增加 `--device` 与 `--num-workers`，实现自动选择设备并将模型/张量迁移到目标设备。
3. 在 `src.evaluate` 中复用保存配置中的设备偏好，但当 CUDA 不可用时回退到 CPU，避免评估阶段崩溃。
4. 更新实验命令文档，补充 CUDA / `num-workers` 命令示例，以及针对 `RTX 4060 Laptop 8GB + i7-13700 + 8GB RAM` 的推荐参数。
5. 运行针对性测试并记录结果。
