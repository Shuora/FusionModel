# Bidirectional Fusion Encoder Design

## Goal

将当前 `MobileViTETBertFusionClassifier` 从浅层 gate feature fusion 升级为面向性能上限的双向多层 cross-attention 融合模型，同时继续使用 `MobileViT` 作为图像 backbone、`ET-BERT` 作为时序 backbone。

## Decision Summary

本轮明确采用以下设计约束：

- 保留 backbone：
  - 图像分支继续使用 `MobileViT`
  - 时序分支继续使用 `ET-BERT`
- 不再优先兼容旧融合接口
- 目标优先级为：
  - 最终效果
  - 在 `RTX 4060 Laptop 8GB` 上可训练
  - 单次完整训练预算控制在 `2-4` 小时

## Background

当前实现的问题不在于 backbone 明显失效，而在于多模态交互过浅：

- `MobileViTETBertFusionClassifier` 目前只拿到两个 pooled feature
- 使用单标量 `gate` 执行：
  - `gate * img_feature + (1 - gate) * tls_feature`
- 图像和时序之间没有 token-level 信息交换

在用户已有经验里：

- `MobileViT` 单分支在 `USTC` 上能跑到约 `98.5% acc`
- `ET-BERT` 本身也是面向流量任务的方向性 backbone

因此本轮不优先替换 backbone，而是把主要改造集中在融合层。

## Non-Goals

本轮不做以下事情：

- 不把 backbone 切换为更大的通用 vision / language model
- 不引入统一超长 multimodal transformer 序列
- 不为兼容旧 gate 语义而保留“假 gate”
- 不强求 `train/evaluate/stacking/moe` 与旧模型输出键完全一致

## High-Level Architecture

目标结构：

1. `MobileViTBackbone` 输出 image token 序列与 pooled image feature
2. `ETBertBackbone` 输出 text token 序列、有效 mask 与 pooled text feature
3. 新增 `BidirectionalFusionEncoder`
4. `BidirectionalFusionEncoder` 由 `N=2` 层 `BidirectionalFusionBlock` 组成
5. 每个 block 都执行双向 cross-attention：
   - `text <- image`
   - `image <- text`
6. 经过多层融合后分别做池化，得到：
   - `img_ctx`
   - `txt_ctx`
7. 通过融合头生成：
   - `logits_fuse`
   - `logits_img`
   - `logits_tls`

## Why Not A Unified Multimodal Transformer

更激进的方案是把 image tokens 与 text tokens 拼成统一长序列，再跑多层 transformer。这个方案表达力更完整，但在本项目约束下不作为初版：

- 显存压力更高
- 训练时延更高
- 在 `RTX 4060 Laptop 8GB` 上更容易被 batch size 限制
- 当前任务预算只有 `2-4` 小时，先采用更可控的双流 bidirectional fusion 更合理

## Component Design

### 1. MobileViT token interface

`MobileViTBackbone` 需要从“只返回 pooled feature”升级为“可返回 token-level 表示”。

建议新增：

- `forward_features(rgb) -> dict`

返回：

- `tokens`: `[B, I, D]`
- `pooled`: `[B, D]`

实现原则：

- 启用 `output_hidden_states=True`
- 从多个中后期 hidden states 提取多尺度空间特征，而不是只取最终层
- 原因是当前 RGB 默认 `28x28`，最终层在该分辨率下会退化成 `1x1`，只保留单 image token
- 将选中的空间特征 reshape 为 token 序列
- 对每个尺度分别投影到统一 `hidden_dim` 后拼接
- pooled feature 继续取最终层语义表示，用于单模态辅助头

### 2. ET-BERT token interface

`ETBertBackbone` 需要从“只返回 masked mean pooled feature”升级为“返回 token-level 表示 + pooled feature + mask”。

建议新增：

- `forward_features(input_ids, attention_mask, token_type_ids) -> dict`

返回：

- `tokens`: `[B, T, D]`
- `mask`: `[B, T]`
- `pooled`: `[B, D]`

实现原则：

- 继续使用现有 embedding 与 `TransformerEncoder`
- 保留当前 masked mean pooling 作为单模态 pooled feature

### 3. BidirectionalFusionBlock

每层 block 包含两条更新流：

- text stream
- image stream

每层执行顺序：

1. `text <- CrossAttention(query=text, key=image, value=image)`
2. text residual + norm
3. text FFN residual + norm
4. `image <- CrossAttention(query=image, key=text, value=text)`
5. image residual + norm
6. image FFN residual + norm

实现细节：

- 使用 `nn.MultiheadAttention(batch_first=True)`
- 文本侧传入 `key_padding_mask`
- 图像侧默认全有效，不额外引入 image mask
- 初版不输出 attention weights 到训练主链路，避免显存与 I/O 负担

### 4. BidirectionalFusionEncoder

由多个 `BidirectionalFusionBlock` 堆叠组成。

初版参数：

- `num_layers = 2`
- `num_heads = 4` 或与 `hidden_dim` 相容的安全值
- `dropout = 0.1`

原因：

- 比单层交互更强
- 比 4 层以上更适合当前 8GB 显存与时长预算

### 5. Pooling and Heads

融合编码完成后：

- `img_tokens_fused -> img_ctx`
- `txt_tokens_fused -> txt_ctx`

池化方式：

- 文本：沿用 masked mean pooling
- 图像：mean pooling

最终头部：

- `logits_img = head_img(img_ctx)`
- `logits_tls = head_tls(txt_ctx)`
- `logits_fuse = head_fuse(fusion_proj(concat(img_ctx, txt_ctx)))`

其中 `fusion_proj` 为：

- `Linear(2D -> 2D)`
- `GELU`
- `Dropout`
- `Linear(2D -> D)`

## Forward Contract

目标输出改为：

- `logits_fuse`
- `logits_img`
- `logits_tls`

可选调试输出：

- `img_tokens`
- `txt_tokens`

默认训练链路不依赖 attention weights 持久化。

`gate` 从主模型接口中移除。

## Training Implications

### Loss

继续保留辅助监督，但主目标转为真正的融合表示：

- `loss = CE(logits_fuse) + alpha * CE(logits_img) + beta * CE(logits_tls)`

建议：

- `alpha = 0.2`
- `beta = 0.2`

原因：

- 保留单模态分支的可学习性
- 避免辅助头权重过高，反过来弱化融合层价值

### Warmup

当前 `warmup` 阶段以单模态头为主。由于新模型更依赖 token-level 融合，可以保留 warmup 概念，但应重新解释：

- warmup：只训练 backbone + 单模态 heads
- fusion：解冻 fusion encoder 与 fusion head，联合训练

若现有训练逻辑实现成本较高，也可以在第一版中保持：

- warmup 时不使用 `logits_fuse`
- fusion 时启用完整 loss

## Data Flow

输入仍保持当前四路张量：

- `rgb`
- `input_ids`
- `attention_mask`
- `token_type_ids`

数据预处理与 dataloader 不需要新增字段。

模型内部流程变为：

1. image backbone 编码出 `image tokens`
2. text backbone 编码出 `text tokens + text mask`
3. 通过 `2-layer bidirectional fusion encoder`
4. 分别池化为 `img_ctx` 与 `txt_ctx`
5. 计算三路 logits

## Error Handling

需要显式处理以下边界：

- `attention_mask` 全 0：
  - 继续沿用当前逻辑，至少保留一个 token 避免全 masked 崩溃
- `hidden_dim` 与 `num_heads` 不整除：
  - 继续采用安全归一化策略或显式降头数
- 单样本 batch：
  - 保持当前 `MobileViT` 对 batch size 1 的兼容处理

## Testing Strategy

实现前先补失败测试，至少覆盖：

1. `MobileViTBackbone.forward_features` 返回：
   - `tokens`
   - `pooled`
2. `ETBertBackbone.forward_features` 返回：
   - `tokens`
   - `mask`
   - `pooled`
3. 新 fusion model 前向输出：
   - `logits_fuse`
   - `logits_img`
   - `logits_tls`
4. 部分 `attention_mask` 下前向可运行
5. 多层 fusion encoder 输出 shape 稳定
6. 模型输出中不再包含旧 `gate`

## Expected Code Changes

核心文件：

- `src/models/mobilevit_backbone.py`
- `src/models/etbert_backbone.py`
- `src/models/fusion_model.py`
- `tests/models/test_fusion_model.py`

很可能需要同步调整：

- `src/train.py`
- `src/evaluate.py`
- `src/stacking.py`
- `src/moe.py`

原因：

- 这些入口当前默认仍会读取 `gate`
- 新模型输出契约变更后，需要统一清理旧依赖

## Risks

### 1. Image token extraction quality

如果 `MobileViT` 暴露的 `last_hidden_state` 语义或形状与预期不一致，image token 的表达质量可能不稳定，需要在实现阶段核对真实输出形状。

### 2. Memory pressure

双向 cross-attention 会显著提升显存占用，尤其在：

- token 数过大
- head 数过多
- fusion 层数过深

因此初版固定为 2 层，并避免额外保存大块 attention map。

### 3. Downstream contract breakage

当前训练、评估、stacking、moe 都围绕旧输出结构构建。移除 `gate` 后，下游调用点需要系统性检查。

## Rollout Plan

1. 先写并确认本设计文档
2. 用 TDD 补失败测试
3. 创建隔离 git worktree
4. 实现 backbone token 接口
5. 实现 `BidirectionalFusionEncoder`
6. 替换旧 gate fusion
7. 对齐训练/评估/stacking/moe
8. 跑定向测试并记录结果

## Acceptance Criteria

满足以下条件视为第一版完成：

- 主模型已不再使用 gate fusion
- `MobileViT + ET-BERT + 2-layer bidirectional fusion encoder` 路径可前向运行
- 训练链路可以消费新模型输出
- 模型相关定向测试通过
- README 与 planning 文档已同步到位
