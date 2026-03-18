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
