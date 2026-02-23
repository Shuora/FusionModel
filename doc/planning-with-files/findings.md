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
- 在 `conda activate FusionModel` 环境内，`pytest` 初始缺失；补装后可执行 TDD 流程。
- 首批 Task 1-3 已完成最小可用实现，并通过对应 3 个单测与一次聚合回归。
- 第二批 Task 4-6 已完成最小可用实现：泄漏控制双轨、TLS-RGB 编码器、TLS token 编码器。
- `encode_tls_tokens` 当前实现保持 `[CLS]` 与 `[SEP]` 锚点并以 `PAD` 填充中间位，满足最小测试预期。
- 第三批 Task 7-9 已完成最小可用实现：数据构建 CLI、跨模态融合前向链路、三阶段训练冒烟链路。
- `build_dataset` 当前通过 `samples` 字段生成成对 `.npy`/`.json` 占位样本，便于后续替换为真实解析流水线。
- `run_train` 已确保输出 `config.yaml`、`train.log`、`metrics.csv`、`checkpoints/best.pt`，满足最小产物验收。
- 第四批 Task 10-12 已完成最小可用实现：stacking 元特征与 meta-learner、评估与报告自动化、消融汇总脚本。
- 全量测试已扩展到 12 个用例并全部通过，主线 Task 1-12 的“失败测试 -> 最小实现 -> 回归通过”链路完整闭环。
- 可选 Task 13 已完成 smoke 级实现：MoE 路由器与蒸馏损失函数，并新增独立冒烟测试。
- 可选任务的性能退出条件（F1 提升/推理提速、蒸馏精度损失阈值）尚未在真实数据训练上验证。
- 用户反馈“预处理无控制台输出”，已补齐实时可观测性：数据集级/样本级进度条 + 摘要输出 + 预处理日志。
- `build_dataset` 在缺少 `samples` 时可基于 `source_root` 自动扫描 `*.pcap|*.pcapng|*.cap` 生成样本。
- 日志路径统一为 `outputs/logs/<category>/`，其中 `<category>` 包含 `preprocess`、`train`、`evaluate`、`report`。
- 当前真实数据目录可直接复现多数据集预处理：`CICAndMal2017`（2126）、`MFCP`（7）、`USTC-TFC2016`（10）。

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 先给出“阶段化执行计划 + 每阶段验收标准” | 用户请求是“详细计划”，优先保证可执行性而非代码改动 |
| 采用“主线必做 + 增强可选”双层规划 | 保证先交付可用主模型，再迭代 MoE/蒸馏 |
| 统一将日志、图表、报告收敛到 `outputs/runs/<run_id>/` | 便于复现实验与后续论文取材 |
| 详细计划文档使用 `doc/plans/` 而非 `docs/plans/` | 仓库既有文档目录为 `doc/`，保持一致性更利于维护 |
| 执行环境固定为 `conda activate FusionModel` | 用户明确指定环境依赖，避免 Python 解释器不一致 |
| 按任务粒度提交 3 个独立 commit | 与计划中“每个 task 单独提交”保持一致，便于回滚和审阅 |
| 第二批继续保持“先失败测试再最小实现”节奏 | 降低一次性改动规模，便于快速定位回归 |
| 第三批将融合模型实现限定为可验证最小骨架 | 先锁定模块接口与产物格式，再迭代真实训练细节 |
| 第四批优先保证产物可落盘与接口闭合 | 先满足 run/report/ablation 的工程可复现性，再细化指标质量 |
| 可选 Task 13 先以 smoke 验证接口正确性 | 真实收益评估留给后续基于数据集的长跑实验 |
| 日志根目录统一到 `outputs/logs/` | 满足“预处理与训练验证日志统一归档且分类摆放”的运维需求 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| `src/` 文件扫描为空，无法直接映射现有实现细节 | 将计划按“目录职责 + 新增模块”方式编排，后续实现时再映射到具体文件 |
| 启动执行前仓库存在大量已暂存删除与未暂存改动（108 files changed） | 切换为 `executing-plans` Step 1 阻塞处理，先确认隔离工作区策略再执行 |
| 未发现 `.worktrees/` 或 `worktrees/`，且 `.gitignore` 未忽略这两个目录 | 按 `using-git-worktrees` 规则需要用户确认目录选择（或改用全局目录） |
| 沙箱网络限制导致 pip 安装 `pytest` 失败 | 请求提权安装，依赖补齐后恢复测试流程 |

## Resources
- `doc/恶意tls家族分类方案计划书.md`
- `AGENTS.md`
- `task_plan.md`
- `progress.md`
- `doc/plans/2026-02-21-malicious-tls-family-classification.md`

## Visual/Browser Findings
- 本次任务未使用浏览器或图像查看；主要依据本地文档与目录结构。
