# Cross-Attention Stabilization Design

## Goal

修复当前 `MobileViTETBertFusionClassifier` 在 `stage1 binary` 上快速塌缩到多数类的问题，同时为训练入口增加通用 early stopping，避免长时间空转训练。

## Context

当前 cross-attention 版完整 run `runs/stage1-binary` 在第 2 个 epoch 起就稳定退化到多数类预测，`train/val/test` 都约为 `69.2%`，与 malicious 类占比一致。只读复算表明 `fuse/img/tls` 三个头全部退化为全预测恶意类，因此问题不在评估口径，而在模型训练路径本身。

与旧高分 gate run 相比，当前实现有两个关键退化点：

1. `head_img` 和 `head_tls` 在 `use_fusion=True` 时不再监督 backbone 的 pre-fusion pooled feature，而是监督 fusion 后的 context，失去了稳定的单模态锚点。
2. fusion 主路径绕开了预训练最稳定的 pooled feature，改为依赖随机初始化的 `token_proj + fusion_encoder + fusion_proj` 直接承担主要判别任务，在当前不平衡 binary 任务上更容易塌缩。

## Requirements

### 1. 恢复稳定单模态监督

- `head_img` 必须始终作用于 image backbone 的 pre-fusion pooled feature。
- `head_tls` 必须始终作用于 text backbone 的 pre-fusion pooled feature。
- 辅助头不再依赖 fusion 后 context。

### 2. 保留 fusion 能力，但加入 pooled shortcut

- `use_fusion=True` 时，cross-attention 仍然生成 `img_ctx_fused` 与 `txt_ctx_fused`。
- `head_fuse` 的输入不能只依赖 fusion 后 context，必须保留来自 `img_pooled/text_pooled` 的 shortcut。
- 目标是让 fusion 学增益，而不是覆盖 backbone 已有判别信息。

### 3. 兼容 warmup / non-fusion 语义

- `use_fusion=False` 或 `stage=warmup` 时，fusion encoder 仍应被绕开。
- 该路径下模型行为应继续使用 pooled feature，且不触发 fusion encoder。

### 4. 增加通用 early stopping

- `src.train` 新增 `--early-stopping-patience`，默认关闭。
- 关闭 sentinel 明确为 `0`：
  - `0` 表示禁用 early stopping
  - `> 0` 表示启用
- 监控指标复用现有 `--best-metric`。
- “提升”语义与 best checkpoint 保持一致：
  - 只有 `current_best_value > best_value` 才算 improvement
  - 持平不算 improvement，并计入 `epochs_without_improvement`
- 当连续 `patience` 个 epoch 未提升时，提前停止训练。
- 提前停止必须保留已经写出的 `best.ckpt` 与 `metrics.csv`。
- 日志里要明确记录 early stopping 触发。
- `metrics.csv` 只应包含实际执行过的 epoch，不应预填充未执行的 epoch。

### 5. 入口透传

- `src.experiments.stage1_binary --execute` 必须支持并透传 `--early-stopping-patience`。

### 6. 外部接口保持稳定

- `MobileViTETBertFusionClassifier.forward()` 的输出键保持不变：
  - `logits_fuse`
  - `logits_img`
  - `logits_tls`
- 当 `return_features=True` 时，仍继续暴露：
  - `img_tokens`
  - `txt_tokens`
- 本轮不新增或删除现有输出 key，也不改变各 logits 的 batch/num_classes shape 契约。

## Proposed Design

### Model Path

保留当前 token-level bidirectional fusion encoder，但显式拆分三类表示：

- `img_pooled_pre`: `MobileViTBackbone.forward_features(rgb)["pooled"]`
- `txt_pooled_pre`: `ETBertBackbone.forward_features(... )["pooled"]`
- `img_ctx_fused/txt_ctx_fused`: token 经过 fusion encoder 后得到的 pooled context

输出策略改为：

- `logits_img = head_img(img_pooled_pre)`
- `logits_tls = head_tls(txt_pooled_pre)`
- `logits_fuse = head_fuse(fusion_shortcut_proj([img_ctx_fused, txt_ctx_fused, img_pooled_pre, txt_pooled_pre]))`

其中 fusion head 输入保留 pooled shortcut，确保即使新加的 token/fusion 层尚未学稳，也不会立刻把 backbone 的判别信息冲掉。

### Training Path

保持现有 loss 形式不变：

- `warmup`: 只用 `logits_img/logits_tls`
- `fusion`: `logits_fuse + alpha * logits_img + beta * logits_tls`

early stopping 状态机：

- `best_value` 更新时，`epochs_without_improvement = 0`
- 否则 `epochs_without_improvement += 1`
- 当 `epochs_without_improvement >= patience` 时：
  - 记录 early stopping 日志
  - 跳出 epoch 循环
  - 返回 `0`

### CLI

- `src.train`:
  - 新增 `--early-stopping-patience`
  - 写入 `config.yaml`
- `src.experiments.stage1_binary`:
  - 新增同名参数并透传

## Testing Strategy

### Model Tests

- 断言 aux heads 使用 pre-fusion pooled feature，而不是 fused context。
- 断言 fusion head 输入包含 pooled shortcut。
- 断言 `use_fusion=False` 时继续绕开 fusion encoder。

### Training / Pipeline Tests

- 断言 early stopping 在验证指标停滞时提前结束。
- 断言 `0` 会禁用 early stopping。
- 断言持平不会重置 patience 计数。
- 断言 `best.ckpt` 仍按 `best_metric` 保存。
- 断言 `metrics.csv` 只包含实际执行 epoch。
- 断言日志记录 `early_stopping_triggered`。
- 断言 `stage1_binary --execute` 透传 `--early-stopping-patience`。

## Non-Goals

- 本轮不引入 class weight、focal loss、weighted sampler。
- 本轮不修改现有 `stage1_binary` 数据协议。
- 本轮不强制把训练改成“两阶段 warmup -> fusion”调度，只保留兼容语义。
