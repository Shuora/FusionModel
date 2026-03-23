# Bidirectional Fusion Encoder Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前 `MobileViT + ET-BERT + gate fusion` 模型升级为 `MobileViT + ET-BERT + 2-layer bidirectional fusion encoder`，并让训练/评估链路消费新的模型输出契约。

**Architecture:** 保留 `MobileViT` 与 `ET-BERT` 作为 backbone，但让二者都暴露 token-level 表示。新增由 2 个 `BidirectionalFusionBlock` 组成的融合编码器，在 text/image 两个流之间反复执行 cross-attention、残差、LayerNorm 与 FFN，最终输出三路 logits。旧 `gate` 从主模型接口中删除，并同步清理训练与评估链路对它的依赖。

**Tech Stack:** Python, PyTorch, transformers, pytest

---

### Task 1: 固化 planning 文档与设计输入

**Files:**
- Modify: `docs/planning-with-files/task_plan.md`
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`
- Reference: `docs/superpowers/specs/2026-03-23-bidirectional-fusion-encoder-design.md`

- [ ] 确认 planning 文件已记录本轮目标、边界与设计结论。
- [ ] 在开始实现前重新阅读 spec，避免实现阶段偏离。

### Task 2: 创建隔离 worktree

**Files:**
- Modify if needed: `.gitignore`
- Create: `.worktrees/bidirectional-fusion-encoder/` 或项目约定的 worktree 路径

- [ ] 检查 `.worktrees/` 或 `worktrees/` 是否存在。
- [ ] 若使用项目内 worktree 目录，先运行 `git check-ignore` 确认已被忽略；若未忽略，补 `.gitignore`。
- [ ] 创建新分支，分支名前缀使用 `codex/`。
- [ ] 切换到 worktree 后确认 `git status --short` 基线干净。

### Task 3: 先写失败测试，定义新 backbone token 接口

**Files:**
- Modify: `tests/models/test_fusion_model.py`
- Optional: `tests/models/test_pretrained_backbones.py`
- Test: `tests/models/test_fusion_model.py`

- [ ] 新增测试：`MobileViTBackbone.forward_features` 返回 `tokens` 和 `pooled`。
- [ ] 新增测试：`ETBertBackbone.forward_features` 返回 `tokens`、`mask` 和 `pooled`。
- [ ] 新增测试：新 fusion model 输出只包含 `logits_fuse / logits_img / logits_tls`，不再包含 `gate`。
- [ ] 新增测试：部分 `attention_mask` 下前向可运行。
- [ ] 运行：
  - `pytest -q tests/models/test_fusion_model.py`
- [ ] 确认测试先失败，失败原因是当前实现尚未提供 token-level 接口和新输出契约。

### Task 4: 实现 backbone token-level 特征接口

**Files:**
- Modify: `src/models/mobilevit_backbone.py`
- Modify: `src/models/etbert_backbone.py`
- Test: `tests/models/test_fusion_model.py`
- Optional Test: `tests/models/test_pretrained_backbones.py`

- [ ] 在 `MobileViTBackbone` 中新增 `forward_features`，返回：
  - `tokens`
  - `pooled`
- [ ] 启用 `output_hidden_states=True`，从多个中后期 hidden states 提取多尺度 image tokens。
- [ ] 将多尺度空间特征整理成 `[B, I, D]` token 序列，并投影到统一维度。
- [ ] 在 `ETBertBackbone` 中新增 `forward_features`，返回：
  - `tokens`
  - `mask`
  - `pooled`
- [ ] 保持原有 `attention_mask` 全 0 时的安全处理。
- [ ] 运行：
  - `pytest -q tests/models/test_fusion_model.py -k 'forward_features or partial_attention_mask'`
- [ ] 确认这些测试通过。

### Task 5: 实现 bidirectional fusion encoder 并替换 gate fusion

**Files:**
- Modify: `src/models/fusion_model.py`
- Test: `tests/models/test_fusion_model.py`

- [ ] 新增 `BidirectionalFusionBlock`，每层包含：
  - `text <- image` cross-attention
  - `image <- text` cross-attention
  - residual
  - `LayerNorm`
  - FFN
- [ ] 新增 `BidirectionalFusionEncoder`，初版固定 2 层。
- [ ] 将主模型前向改为：
  - backbone 提供 token-level 表示
  - 经过 fusion encoder
  - 分别池化为 `img_ctx` / `txt_ctx`
  - 输出 `logits_fuse / logits_img / logits_tls`
- [ ] 删除旧的 `gate` 模块与相关返回值。
- [ ] 运行：
  - `pytest -q tests/models/test_fusion_model.py`
- [ ] 确认测试通过。

### Task 6: 对齐训练、评估与集成入口

**Files:**
- Modify: `src/train.py`
- Modify: `src/evaluate.py`
- Modify: `src/stacking.py`
- Modify: `src/moe.py`
- Optional Test: `tests/pipeline/test_train_eval_report.py`
- Optional Test: `tests/pipeline/test_stacking_pipeline.py`
- Optional Test: `tests/pipeline/test_moe_pipeline.py`

- [ ] 搜索并删除对 `out["gate"]` 的硬依赖。
- [ ] 保证训练阶段仍以三路 logits 完成 loss 与预测。
- [ ] 更新评估、stacking、moe，使其与新模型输出契约一致。
- [ ] 如现有报告或配置写入依赖 `gate` 统计，移除或替换为新字段。
- [ ] 运行最小链路测试：
  - `pytest -q tests/pipeline/test_train_eval_report.py -k fusion_model`
  - 若无对应子测试，则运行整个文件中与 fusion 输出契约相关的测试集合。

### Task 7: 更新文档

**Files:**
- Modify: `README.md`
- Modify: `docs/planning-with-files/task_plan.md`
- Modify: `docs/planning-with-files/findings.md`
- Modify: `docs/planning-with-files/progress.md`

- [ ] 将 README 中的“融合头”描述从 gate fusion 更新为 bidirectional fusion encoder。
- [ ] 在 findings/progress 中记录新模型结构、接口变化和验证结果。

### Task 8: 完整验证

**Files:**
- Test: `tests/models/test_fusion_model.py`
- Test: `tests/models/test_pretrained_backbones.py`
- Test: `tests/pipeline/test_train_eval_report.py`
- Test: `tests/pipeline/test_stacking_pipeline.py`
- Test: `tests/pipeline/test_moe_pipeline.py`

- [ ] 运行模型与 pipeline 的定向回归：
  - `pytest -q tests/models/test_fusion_model.py tests/models/test_pretrained_backbones.py tests/pipeline/test_train_eval_report.py tests/pipeline/test_stacking_pipeline.py tests/pipeline/test_moe_pipeline.py`
- [ ] 若出现失败，按 TDD 方式逐个修复，不跳过红灯。
- [ ] 将最终验证命令和结果写入 `docs/planning-with-files/progress.md`。
