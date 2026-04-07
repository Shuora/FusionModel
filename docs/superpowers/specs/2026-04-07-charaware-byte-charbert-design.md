# Char-Aware Byte CharBERT Compatible Upgrade Design

## Goal

在保持现有训练入口与主流程不变的前提下，将当前 `src/CharBERT/src/model.py` 的轻量 byte Transformer 升级为更接近 CharBERT 思想的 `char-aware` 文本分支，实现“token(byte) + char(feature) + 分层融合”，并确保 `attention` / `attention_stacking` 可以直接复用。

## Scope

- 保持入口脚本不变：`src/train_fusion_attention.py`、`src/train_fusion_attention_stacking.py`。
- 保持数据目录与预处理流程不变：`ProcessedData/<task>/pcap_data/{Train,Test}`。
- 保持融合框架不变：图像分支 `MobileViT` + 文本分支 + cross-attention。
- 升级文本分支内部实现与配置体系，补齐 checkpoint 兼容、回滚、测试。
- 不在本轮强制引入独立 pretrain 脚本（可作为后续增量）。

## Current Gap vs Real CharBERT Idea

当前实现本质为 `Embedding + PositionalEncoding + TransformerEncoder`，缺少 CharBERT 核心思想中的字符增强路径与层级融合机制：

- 无独立 char encoder。
- 无 token/char 融合门控。
- 无分层融合（仅单路 token 表征）。
- 训练目标仅监督分类，未保留可扩展的辅助对齐目标接口。

## Approaches Considered

### A. Input-Only Char Fusion (Not Recommended)

仅在输入 embedding 层做一次 `token + char` 融合，然后进入原 Transformer。

- 优点：实现快、风险低。
- 缺点：Char 信息在深层易衰减，提升上限有限。

### B. Layer-Wise Gated Char-Aware ByteBERT (Recommended)

新增 char 分支，并在每层 encoder 前做门控融合，保持主训练入口与输出接口兼容。

- 优点：更接近真正 CharBERT 思想，兼容性与效果上限平衡最好。
- 缺点：实现复杂度中等，需要更细的 checkpoint 兼容处理。

### C. Compatible Shell + Optional Pretraining

在 B 基础上再增加可选 byte-MLM 预训练路径。

- 优点：效果上限更高。
- 缺点：训练成本与工程复杂度显著提升，不符合本轮“兼容式升级优先”。

本设计选择 **B**。

## Target Architecture

### 1) Byte Token Path

- 输入仍为现有 `input_ids`（`0..255` + `PAD/CLS/SEP`）。
- 保持 `token_embedding` 与 `positional_encoding`。

### 2) Char Feature Path

对每个 byte token 构造固定长度字符序列，默认 `hex` 映射：

- 例：`0xAF -> ['A','F']`。
- `char_embedding -> Conv1d -> nonlinearity -> pooling` 得到 `char_vec`。
- 通过 `char_proj` 投影到与 token hidden 相同维度。

支持 `char_vocab` 可选值：

- `hex`（默认，稳定且对 byte 语义直接）；
- `ascii`（可选，增强可打印字符局部模式感知）。

### 3) Layer-Wise Gated Fusion

在每层 Transformer 前进行融合：

- `gate = sigmoid(MLP([token_vec; char_vec]))`
- `fused = gate * token_vec + (1 - gate) * char_vec`

融合层位可配置：

- `first` / `last` / `all`（默认 `all`）。

### 4) Encoder Output for Fusion Model

- 保留 sequence hidden states（`B,S,H`）供现有 cross-attention 使用（`K/V` 来自文本序列）。
- 保留 pooled feature（`B,H`）供兼容路径使用。
- 推荐池化：`masked_mean + cls_concat` 后线性投影到目标维度。

### 5) Failure Fallback

当 char 分支异常时：

- 自动回退 token-only 路径（不中断训练）；
- 打印 warning，并记录一次性计数指标（便于诊断）。

## Compatibility Plan

### CLI / Config

新增参数（保持默认兼容旧行为）：

- `--charbert_mode {legacy,charaware}`，默认 `legacy`。
- `--char_vocab {hex,ascii}`，默认 `hex`。
- `--char_emb_dim`，默认 `32`。
- `--char_cnn_channels`，默认 `64`。
- `--char_fusion {gated,add,concat}`，默认 `gated`。
- `--char_fusion_layers {first,last,all}`，默认 `all`。

`legacy` 下行为与当前版本一致，确保旧命令可直接复现。

### Code Interface

- `src/fusion_common.py` 中 `CharBERTTextEncoder` 继续提供现有调用接口。
- cross-attention 主体保持 `Q=image, K/V=text` 不变。
- `attention` / `attention_stacking` 默认仍为注意力融合，不引入双向注意力新拓扑。

## Checkpoint Compatibility

- 加载旧 checkpoint 使用 `strict=False`。
- 旧模型到 `charaware` 时，新参数（`char_*`、`fusion_gate_*`）随机初始化。
- metadata 增加：
  - `text_encoder_arch` (`legacy` / `charaware`)
  - `char_vocab`
  - `char_fusion`
  - `char_fusion_layers`
- 训练日志打印 `missing_keys/unexpected_keys` 摘要，避免 silent mismatch。

## Training Strategy

- 主任务继续使用分类 `cross_entropy`。
- 可选辅助项（默认关闭）：`char-token consistency` 正则。
- 早停、采样、stacking、后处理逻辑不改，仅提升文本表征质量。

## Migration and Rollout

分三步上线：

1. 上线代码与参数，默认 `legacy`，零风险兼容。
2. 四任务各跑 `charaware` 对照实验（attention + attention_stacking）。
3. 依据 macro-F1 与少数类召回结果，决定是否将默认切换到 `charaware`。

回滚路径：任意异常时使用 `--charbert_mode legacy` 即可恢复旧行为。

## Verification Plan

### Unit Tests

- char 映射与 tensor shape 测试（`hex/ascii` 两模式）。
- 门控融合输出范围与数值稳定性测试。
- mask 与 padding 行为测试。
- `legacy` / `charaware` checkpoint 加载兼容测试。

### Integration Smoke

- 四个 `task_name` 各跑至少 1 epoch：
  - `attention`
  - `attention_stacking`
- 验证训练可完成、日志无结构性错误、产物目录完整。

### Metrics Review

重点比较 `charaware` 相对 `legacy` 的：

- `macro_f1`
- 少样本类 `recall`
- 训练稳定性（`NaN/Inf batch` 是否增加）

若多数任务无增益或稳定性退化，则保留 `legacy` 为默认。

## Documentation Updates Required

实现阶段若落地本设计，需同步更新：

- `README.md`：新增 char-aware 参数说明与示例命令（保持四任务独立命令）。
- `AGENTS.md`：仅在协作流程或约束变化时更新；本设计本身不强制改动。

## Non-Goals

- 不在本轮改为 HuggingFace BERT/Tokenizer 全栈。
- 不在本轮引入强制 pretrain pipeline。
- 不改变现有输出目录约定与 run 组织结构。
