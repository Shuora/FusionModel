# Findings & Decisions

## Requirements
- 基于 `doc/恶意tls家族分类方案计划书.md` 的“初步计划”，产出“可执行的详细计划”。
- 计划应覆盖完整任务链路：数据、表示、模型、训练、集成、评估、报告、交付。
- 计划需能落地到当前仓库结构与产物目录（`configs/`, `src/`, `dataset/`, `outputs/`）。

## Research Findings
- 方案边界明确：只做恶意 TLS 家族多分类，不做 benign/malicious 二分类，不解密 payload。
- 数据策略明确：必须按 capture/场景做分组切分，重点防止会话泄漏。
- 表示策略明确：保留 RGB 图像分支，新增 TLS-Field-BERT 序列分支。
- 融合主线明确：双向 cross-attn + gating + 辅助监督，解决单分支塌缩。
- 集成升级明确：优先增强 stacking（meta-features + GBDT），MoE 作为高级增强。
- 输出要求明确：训练日志、进度条、run_id 目录、自动报告、图表与消融清单。
- 当前仓库 `src/` 下尚未发现已实现文件，意味着需从骨架开始规划全链路交付。
- `configs/`、`dataset/`、`outputs/` 目前也未看到现成配置与脚本文件，计划需包含初始化与模板产物创建步骤。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 先给出“阶段化执行计划 + 每阶段验收标准” | 用户请求是“详细计划”，优先保证可执行性而非代码改动 |
| 采用“主线必做 + 增强可选”双层规划 | 保证先交付可用主模型，再迭代 MoE/蒸馏 |
| 统一将日志、图表、报告收敛到 `outputs/runs/<run_id>/` | 便于复现实验与后续论文取材 |
| 详细计划文档使用 `doc/plans/` 而非 `docs/plans/` | 仓库既有文档目录为 `doc/`，保持一致性更利于维护 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| `src/` 文件扫描为空，无法直接映射现有实现细节 | 将计划按“目录职责 + 新增模块”方式编排，后续实现时再映射到具体文件 |
| 启动执行前仓库存在大量已暂存删除与未暂存改动（108 files changed） | 切换为 `executing-plans` Step 1 阻塞处理，先确认隔离工作区策略再执行 |
| 未发现 `.worktrees/` 或 `worktrees/`，且 `.gitignore` 未忽略这两个目录 | 按 `using-git-worktrees` 规则需要用户确认目录选择（或改用全局目录） |

## Resources
- `doc/恶意tls家族分类方案计划书.md`
- `AGENTS.md`
- `task_plan.md`
- `progress.md`
- `doc/plans/2026-02-21-malicious-tls-family-classification.md`

## Visual/Browser Findings
- 本次任务未使用浏览器或图像查看；主要依据本地文档与目录结构。
