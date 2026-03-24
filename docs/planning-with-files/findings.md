# Findings

## Cross-Attention Stage1 Acc 回退排查结论（2026-03-23）

- 与旧 gate 模型相比，当前最可疑的结构性问题不是“warmup 逻辑写错”，而是“实际 stage1 run 根本没有走 warmup”：
  - `stage1_binary` 默认 `--stage fusion`，见 `src/experiments/stage1_binary.py`
  - 当前 `runs/stage1-binary/config.yaml` 也明确记录 `stage: fusion`
  - `src/train.py` 没有任何 `resume / preload / requires_grad` 冻结解冻逻辑，说明 fusion run 会从随机初始化的 fusion 模块直接开训
  - 这与设计文档里“warmup 先训 backbone + 单模态头，再进入 fusion”的语义并不一致
- 当前实现里，真正新增且未预训练的关键模块规模不小：
  - `image_backbone.token_proj`
  - `fusion_encoder`
  - `fusion_proj`
  其中 `MobileViTBackbone` 先加载 `self.model` checkpoint，再创建 `token_proj / proj`，因此这两类投影层必然随机初始化。
  - 使用项目 conda 环境实测，`fusion_encoder + fusion_proj + image token_proj` 参数量约为 `94` 万；对比旧 gate 版本只新增了一个很浅的 `gate`
- 当前文本 backbone 也并没有在训练入口使用任何 ET-BERT 预训练 checkpoint：
  - `ETBertBackbone` 支持 `checkpoint_path`
  - 但 `src/train.py` 实例化 `MobileViTETBertFusionClassifier` 时没有向 `text_backbone` 传入任何 checkpoint/config
  - 因而在 fusion run 中，强预训练的图像分支会从第一步开始反复与随机初始化的文本 token 以及随机初始化的 cross-attention 交互，这比旧 gate 模型更容易污染图像主干表示
- token 数本身不是最直接的问题，但它放大了上述风险：
  - 设计文档明确因为 `28x28` 输入下最终层只剩单 image token，所以改成多尺度 image tokens
  - 实际使用项目 conda 环境验证，当前 `MobileViTBackbone.forward_features()` 在 `28x28` 输入下会产出 `21` 个 image tokens（来自 `4x4 + 2x2 + 1x1`）
  - 这解决了“单 token”瓶颈，但也让随机初始化的 fusion 模块在每个 batch 都要处理更多跨模态交互
- 辅助头“还能学”，但语义已明显偏离旧 gate 模型，对稳定训练不利：
  - 旧 gate 模型中 `logits_img/logits_tls` 直接监督各自 backbone 的 pooled feature
  - 新模型与测试约束都明确要求 `head_img/head_tls` 吃的是融合后的 `img_ctx/txt_ctx`
  - 这意味着辅助监督不再是“保护单模态主干”，而是在给已经被 cross-attention 改写过的表示继续加损失；在文本侧未预训练时，这种监督更可能把噪声反向灌进图像侧
- warmup “是否真的绕过 fusion”的答案是：
  - 是，代码层面确实绕过了
  - 但这只能说明 `stage=warmup` 时行为正确，不能缓解当前 `stage1-binary` 实际 run 直接使用 `stage=fusion` 的问题
- `loss` 配比不是当前 `runs/stage1-binary` 回退到约 `70%` 的首要嫌疑：
  - 设计文档建议 `alpha=0.2 / beta=0.2`
  - 当前 CLI 默认值仍是 `0.3 / 0.3`
  - 但 `runs/stage1-binary/config.yaml` 实际记录的是 `alpha: 0.2 / beta: 0.2`
  - 因此“loss 配比不匹配”是默认入口的隐患，不是这次已生成 run 的最强解释
- 综上，最可能伤害 `stage1 acc` 的差异排序是：
  1. 实际 fusion run 没有任何真正的 warmup/分阶段衔接，随机初始化 fusion 模块从 step 1 就参与训练
  2. 新增 cross-attention/token projection 大量未预训练参数，且直接作用于原本强势的 MobileViT 表征
  3. ET-BERT 侧仍未预训练，但新结构不再允许模型像旧 gate 那样轻易“忽略弱文本分支”
  4. 辅助头从单模态监督变成融合后监督，削弱了 aux loss 原本的稳定器作用
  5. `alpha/beta` 默认值与 spec 不一致，但不是当前 `runs/stage1-binary` 的主因

## Stage1 Evaluate OOM 修复结论（2026-03-23）

- `src/evaluate.py` 的根因不是模型加载，而是评估时把整个目标 split 一次性搬上设备并单次前向。
- 当前已将评估路径改为 mini-batch 推理：
  - 新增 CLI 参数 `--eval-batch-size`
  - 未显式传入时，默认回退到 `config.yaml` 中训练期 `batch_size`
  - 因此像 `runs/stage1-binary` 这类未带新参数的旧命令，也会自动按训练 batch size 分批评估
- 新实现会逐批：
  - 从 numpy 切片构造 tensor
  - `.to(device)` 后前向
  - 立刻把 `logits_*` 拉回 CPU 聚合
  这样能显著降低新版 token-level fusion 模型在评估阶段的显存峰值。
- 新增回归测试 `test_evaluate_batches_forward_pass_to_avoid_full_split_oom`：
  - 使用 5 条 test 样本
  - 显式传入 `--eval-batch-size 2`
  - 断言模型前向批次为 `[2, 2, 1]`
  - 断言 `eval_test.json` 中 `num_samples == 5`

## Evaluate Checkpoint 兼容性排查发现（2026-03-23）

- 当前 `src/evaluate.py` 会根据 `config.yaml` 实例化新版 `MobileViTETBertFusionClassifier`：
  - `hidden_dim`
  - `vocab_size`
  - `fusion_layers`
  - `fusion_heads`
  - `fusion_dropout`
  见 `src/evaluate.py:152-159`。
- 当前主模型已经不是旧 `gate` 融合，而是双向 cross-attention 风格融合：
  - `fusion_encoder`
  - `fusion_proj`
  - `head_fuse/head_img/head_tls`
  见 `src/models/fusion_model.py:121-138`。
- 当前图像分支也已从“只返回 pooled vector”升级为“多尺度 tokens + pooled”：
  - 新增 `token_proj`
  - `forward_features()` 返回 `tokens` 与 `pooled`
  见 `src/models/mobilevit_backbone.py:25-28`、`src/models/mobilevit_backbone.py:52-66`。
- `runs/stage1-binary/checkpoints/best.ckpt` 实际包含新版 key：
  - `fusion_encoder.layers.*`
  - `fusion_proj.*`
  - `image_backbone.token_proj.*`
  - `head_fuse/head_img/head_tls`
  说明当前 `runs/stage1-binary` 不是旧 gate checkpoint。
- 因此用户贴出的第二个 `state_dict` 报错，大概率不是来自 `runs/stage1-binary`，而是来自另一个旧 run 目录：
  - 用户手工命令使用的是 `runs/${RUN_DATE}/${RUN_ID}`
  - traceback 中出现 `Unexpected key(s): gate.*`
  - 同时缺失 `token_proj.* / fusion_encoder.* / fusion_proj.*`
  这与 2026-03-23 之前的旧 gate 模型完全一致。
- Git 历史证据明确显示：`2026-03-23` 提交 `d72aec7` 将模型从 gate 融合切换为 bidirectional fusion encoder：
  - 旧版 `fusion_model.py` 只有 `self.gate`
  - 新版新增 `fusion_encoder/fusion_proj`
  - 旧版 `evaluate.py` 还在消费 `out["gate"]`
- 旧版 `MobileViTBackbone` 也没有 `token_proj`，只保留 `proj` 和 `forward()` pooled 输出，见 `git show d72aec7^:src/models/mobilevit_backbone.py`。
- 这说明 `Missing key(s)` / `Unexpected key(s)` 的根因是“当前代码已升级为新版融合结构，但你手工指定的某个旧 run checkpoint 仍是 gate 时代产物”，属于历史 checkpoint 兼容性断裂，不是单纯 checkpoint 损坏。
- 与此分离的另一个问题是 CUDA OOM：
  - `stage1_binary` 的协议执行会直接调用 `evaluate_main(["--run-dir", ..., "--split", "test", ...])`，见 `src/experiments/stage1_binary.py:332-339`
  - `evaluate.py` 会把整个 eval split 一次性 `.to(device)` 并整批前向，见 `src/evaluate.py:144-169`
  - 这会在新版 token-level fusion 模型上显著抬高显存占用，因此与 checkpoint mismatch 不是同一个根因。

## Cross-Attention 融合改造发现（2026-03-23）

- 用户明确表示本轮优化目标是“最终效果优先”，不是“尽量兼容当前训练/输出接口”。
- 用户最终确认采用的设计路线是：
  - 保留 `MobileViT + ET-BERT`
  - 不替换 backbone
  - 将融合层升级为 `2-layer bidirectional fusion encoder`
  - 删除旧 `gate` 语义，不再为兼容性保留占位输出
- 当前 `MobileViTETBertFusionClassifier` 不是 attention-level 多模态融合，而是 feature-level gate 融合：
  - 先取 `img_feature` 与 `tls_feature`
  - 再经 `Linear -> ReLU -> Linear -> Sigmoid` 得到单标量 `gate`
  - 最后做 `gate * img + (1 - gate) * tls`
- 当前多模态对外接口很稳定：
  - 输入仍是 `rgb / input_ids / attention_mask / token_type_ids`
  - 输出固定包含：
    - `logits_fuse`
    - `logits_img`
    - `logits_tls`
    - `gate`
- 文本分支内部已存在 self-attention，但它只发生在 `ETBertBackbone` 内部，不是图文 cross-attention。
- `MobileViTBackbone` 当前只返回 pooled image feature，不返回 spatial tokens；如果要做真正的 image-text cross-attention，图像分支大概率需要暴露 token-level 序列而不只是 pooled vector。
- 当前数据与测试里的 RGB 输入默认是 `28x28`；实际验证发现：
  - `MobileViT` 最后一层 `last_hidden_state` 在 `28x28` 下只有 `1x1`
  - 直接拿最终层做 image tokens 会退化成单 token，严重削弱 co-attention 价值
  - 因此实现上更合理的路线是：启用 `output_hidden_states=True`，从多个中后期 hidden states 提取多尺度 image tokens 并投影后拼接
- 现有模型测试只覆盖：
  - 输出 logits shape
  - `gate` shape/range
  - `attention_mask` 部分有效时的前向可运行
  说明测试还没有约束 cross-attention 的中间表示或新输出字段。
- `planning-with-files` skill 推荐的 `session-catchup.py` 路径 `/home/shuora/.codex/skills/planning-with-files/scripts/session-catchup.py` 在当前环境不存在；本机可读到的 skill 文件位于 `/mnt/c/Users/11098/.codex/skills/planning-with-files/SKILL.md`，后续需要按实际路径适配。
- 本轮实现结果：
  - `MobileViTBackbone` 现在通过多个中后期 hidden states 构造多尺度 image tokens，再投影到统一 `hidden_dim`
  - `ETBertBackbone` 现在同时输出 `tokens / mask / pooled`
  - `MobileViTETBertFusionClassifier` 已移除 gate fusion，改为 `2-layer bidirectional fusion encoder`
  - 辅助头继续保留三路 logits，但 `gate` 已不再是模型输出的一部分
  - `train/evaluate` 的观测统计从 `gate_mean` 改为 `fuse_conf_mean`
  - `stacking/moe` 不再依赖 `gate`，而是使用基于专家概率的模态一致性特征
- 用户进一步要求：
  - 删除 `docs/commands/stage2-multiclass-e2e.sh`
  - 将该脚本中的 stage2 命令并入总文档 `docs/commands/session-full-experiments.md`
- 本轮补齐的缺口：
  - `logits_img` / `logits_tls` 已从“融合前 pooled feature”切到“融合后 `img_ctx` / `txt_ctx`”
  - `forward(..., return_features=True)` 现在可选返回 `img_tokens` / `txt_tokens`
  - `stacking` 现在会复用主 run 的 `hidden_dim / fusion_layers / fusion_heads / fusion_dropout / lr / alpha / beta`
  - `src.train` / `stage1_binary` / `stage2_multiclass` 现已暴露并透传：
    - `--fusion-layers`
    - `--fusion-heads`
    - `--fusion-dropout`
  - `docs/commands/stage2-multiclass-e2e.sh` 已删除，命令已并入总文档
- 后续 code review 暴露出的关键偏差也已修正：
  - `warmup` 现在会显式绕开 fusion encoder
  - `warmup` 下 `logits_img / logits_tls` 回退到未融合的 pooled feature
  - `evaluate` 在 `stage=warmup` 的 run 上也会禁用融合路径
  - `warmup` 的置信度统计不再假设融合头是主输出

## Session Full 命令重写发现（2026-03-22）

- 当前仓库工作区是干净的，`git status --short` 无未提交改动。
- `docs/planning-with-files/` 已存在，可直接沿用，不需要新建重复目录。
- 当前 shell 环境里默认没有 `python` 命令，只有 `python3`；旧文档里的 `python` 需要按仓库实际运行环境重新表述。
- 直接运行系统 `python3 -m ... --help` 无法获得 CLI 帮助，因为当前基础环境缺失项目依赖：
  - `numpy`
  - `pandas`
  - `matplotlib`
- 因此本轮命令文档重写不能依赖“裸环境 help 输出”，需要结合：
  - 仓库源码中的 argparse 定义
  - 现有文档与脚本
  - 项目已有 conda 环境路径（若存在）
- 使用 `/home/shuora/miniconda3/envs/FusionModel/bin/python -m ... --help` 复核后，当前真实 CLI 情况是：
  - `src.train` 支持 `--best-metric`，但不支持任何 `early-stopping` 参数
  - `src.evaluate` 仅支持 `--run-dir / --split / --checkpoint / --device / --allow-split-fallback`
  - `src.report` 仅支持 `--run-dir`
- `src.experiments.stage1_binary --execute` 当前只会向训练/评估链路透传：
  - `--device`
  - `--num-workers`
  - `--best-metric`
- `src.experiments.stage2_multiclass --execute` 当前默认不仅跑 3 个基础任务，还会追加：
  - `USTC-TFC2016 train4000`
  - `USTC-TFC2016 train3000`
  - `USTC-TFC2016 train2000`
- `src.report` 当前 `Best Validation` 区块固定按 `metrics.csv` 中的 `val_macro_f1` 选最佳 epoch；当训练使用 `--best-metric val_acc` 时，报告展示与 `best.ckpt` 选择依据可能不一致。

## 模型结构调研发现（2026-03-22）

- 模型代码主入口位于 `src/models/`，包含：
  - `fusion_model.py`
  - `mobilevit_backbone.py`
  - `etbert_backbone.py`
- 训练入口在 `src/train.py`，模型相关测试位于：
  - `tests/models/test_fusion_model.py`
  - `tests/models/test_pretrained_backbones.py`
- 当前工作区存在未提交改动，集中在：
  - `src/common/structured_logging.py`
  - `src/train.py`
  - `tests/common/test_structured_logging.py`
  - `tests/data/test_preprocess_pipeline.py`
- 本次任务以只读调研为主，不改动模型代码；阅读时需要注意与用户现有未提交改动并存，不做回滚。
- `MobileViTETBertFusionClassifier` 是当前核心融合模型：
  - 图像分支：`MobileViTBackbone`
  - 序列分支：`ETBertBackbone`
  - 融合方式：先拼接两个 `hidden_dim` 特征，经过 `Linear -> ReLU -> Linear -> Sigmoid` 得到单标量 gate，再做 `gate * img + (1 - gate) * tls`
  - 输出头包含三路：
    - `logits_fuse`
    - `logits_img`
    - `logits_tls`
- `MobileViTBackbone` 不是自写 CNN，而是直接包了一层 `transformers.MobileViTForImageClassification`：
  - 取 `self.model.mobilevit(...)` 的 `pooler_output`
  - 若 `pooler_output` 缺失，则退化为 `last_hidden_state.mean(dim=(-1, -2))`
  - 最后经 `proj` 线性层投影到统一 `out_dim`
  - 默认会尝试加载本地 checkpoint：`/tmp/Shuora-MobileViT/malicious_traffic_mobilevit_model.pth`
  - checkpoint 加载时会过滤掉 `classifier.*`，说明主要复用 backbone 权重而不是原分类头
- `ETBertBackbone` 本质上是一个 ET-BERT 风格 adapter：
  - embedding = token + type + position
  - 编码器 = `nn.TransformerEncoder`
  - 输出 = 按 `attention_mask` 做 masked mean pooling
  - 支持 `config/config_path/num_layers/checkpoint_path`
  - checkpoint 加载逻辑较强，能适配多种外部 key 格式并生成诊断报告 `checkpoint_report`
- 测试约束显示当前模型接口契约很明确：
  - fusion model 输出三路 logits + gate
  - gate 范围必须在 `[0, 1]`
  - ET-BERT 支持层数截断、缺失 checkpoint 报告、BERT/Transformer 风格权重映射、QKV 拼接与不完整组诊断
- 训练主链路位于 `src/train.py`：
  - 先通过 `load_policy_multimodal_data(...)` 读取四路输入：
    - `rgb`
    - `input_ids`
    - `attention_mask`
    - `token_type_ids`
  - 再用 `TensorDataset/DataLoader` 组织 batch
  - 模型实例化参数目前核心是：
    - `num_classes`
    - `hidden_dim`
    - `vocab_size`
- 训练存在两种 loss 模式：
  - `warmup`：
    - loss = `0.5 * CE(logits_img) + 0.5 * CE(logits_tls)`
    - 预测也取两路平均
  - `fusion`：
    - loss = `CE(logits_fuse) + alpha * CE(logits_img) + beta * CE(logits_tls)`
    - 预测主用 `logits_fuse`
- 训练/验证阶段都会统计 `gate` 的均值，说明门控值被当作可观察训练信号，而不只是内部中间变量。
- `pipeline_data` 显示当前多模态样本对齐方式是按 `session_id` 对齐：
  - RGB 来自 `policy/rgb/rgb_shard_*.npz`
  - ET-BERT 序列输入来自 `policy/etbert/etbert_shard_*.npz`
  - 二者按同一 `session_id` 拼成一个样本

## 文档对齐结论（2026-03-18）

- `MobileViTBackbone` 当前为真实 `transformers.MobileViTForImageClassification` 主干，并在默认路径存在时复用本地 checkpoint：`/tmp/Shuora-MobileViT/malicious_traffic_mobilevit_model.pth`。
- ET-BERT 侧已具备工程化接入基础设施：
  - `vocab` 文件加载（`src/data/etbert_tokenizer.py`）
  - `config` / `config_path` 注入（`src/models/etbert_backbone.py`）
  - `num_layers` 截断
  - checkpoint 加载与映射诊断报告（`last_checkpoint_report` / `checkpoint_report`）
- 能力边界已在文档明确：
  - 当前 ET-BERT 侧为 ET-BERT 风格兼容 adapter，不是原始 UER ET-BERT 预训练模型的完整实现。
- 训练/评估/协议管线状态：
  - `train/evaluate/report + stage1/stage2 + stacking/moe` 相关测试链路可通过。
  - 用户指定回归命令已在本次文档更新后复验通过：`46 passed`。

## 运行时支持结论（2026-03-19）

- `src.train` 现已支持显式设备选择：
  - `--device auto`
  - `--device cpu`
  - `--device cuda`
- `src.train` 现已支持 `--num-workers`，并将解析后的 `device` / `num_workers` 写入 `config.yaml`。
- `src.evaluate` 现已支持：
  - CLI `--device`
  - 若未显式传入，则优先复用训练时保存的 `device_requested`
  - 当请求 `cuda` 但当前环境不可用时，自动回退到 `cpu`
- 当前实现仍有一条重要边界：
  - 数据在训练/评估前会整体载入内存，因此在 `8GB RAM` 机器上，内存往往比显存更早成为瓶颈。

## Stage1 论文协议核对结论（2026-03-20）

- 论文原文依据：`docs/paper/MVTBA A Novel Hybrid Deep Learning Model for Encrypted Malicious Traffic Identification.pdf` 第 10-12 页。
- 论文 Exp. I 使用的是：
  - `ISCX VPN-nonVPN`
  - `MTA`
  - `MFCP`
  - 不包含 `USTC`
- 论文 Table 1 为 ISCX 的 9 个 normal traffic group，并给出每组固定的 train/test 配额。
- 论文 Table 2 为 MTA 的 7 个家族，并给出每家族固定的 train/test 配额。
- 论文 Table 3 为 MFCP 的 6 个家族，并给出每家族固定的 train/test 配额。
- 论文明确写到 `MFCP` 做过 `trimmed some of the traffic`，因此论文协议不是“原始数据集全量样本直接混合”。
- 当前仓库 `stage1_binary` 的实现问题：
  - 仅通过 ISCX 文件名前缀白名单与 MTA/MFCP 家族白名单近似论文子集
  - 没有严格按论文表 1-3 的每组 train/test 配额构造 manifest
  - 保留了“匹配不到论文子集时 fallback 到未过滤数据”的行为，不符合严格复现要求
- 本轮实现边界：
  - 严格复现论文的类别/家族集合与每组 train/test 数
  - 使用仓库现有 `session_full` session 样本按稳定排序裁样
  - 不承诺与论文作者原始逐 session 列表一一对应

## Paper Balanced 方向结论（2026-03-20）

- 用户已接受“不必逐表精确复刻，大致像论文即可”的目标调整。
- 当前代码与数据现实表明，论文 Table 1-3 的样本数与仓库现有 `session_full` session 统计口径并不一致：
  - `ISCX` 多个组明显大于论文
  - `MTA` 多个组明显小于论文
  - `MFCP` 多个组明显大于论文
- 因此新增 `paper_balanced` 协议作为默认推荐模式：
  - 保留论文类别集合
  - 不再硬卡论文绝对配额
  - 对超大组做上限裁样
  - 对不足组全保留
  - 对缺失组跳过并记录
- 当前实现约定：
  - `paper_balanced` 的平衡上限为论文配额的 `120%`
  - 缺失组状态记为 `missing`
  - 不足组状态记为 `undersupplied`
  - 超大组状态记为 `capped`

## 论文指标口径对照结论（2026-03-21）

- 论文 `MVTBA A Novel Hybrid Deep Learning Model for Encrypted Malicious Traffic Identification` 在 4.2 节明确使用：
  - `accuracy`
  - `precision`
  - `recall`
  - `F1-score`
- 论文对多分类（Exp. II）进一步明确使用：
  - `macroP`
  - `macroR`
  - `macroF1`
- 仓库当前主评估实现 `src/evaluate.py` 使用：
  - `top1`（等价于 accuracy）
  - `macro_precision`
  - `macro_f1`
  - `macro_recall`
  - `confusion_matrix`
- 仓库与论文的“一致项”：
  - `accuracy` 与仓库 `top1` 本质一致，都是预测正确比例。
  - `macro_precision` / `macro_recall` 与论文的 `macroP` / `macroR` 一致，都是按类别先算再做宏平均。
- 仓库与论文的“关键差异”：
  - 论文把 `macroF1` 写成 `2 * macroP * macroR / (macroP + macroR)`。
  - 仓库使用 `sklearn.metrics.f1_score(..., average="macro")`，即“先逐类算 F1，再对各类 F1 做平均”。
  - 这两种 `macroF1` 在多分类下通常不完全相等，只有部分特殊分布下才会数值一致。
  - 通过本地示例验证，二者可出现非零差值，例如：
    - `macro_p=0.583333`
    - `macro_r=0.527778`
    - sklearn `macro_f1=0.530159`
    - 论文公式 `macro_f1=0.554167`
- 另一个重要差异是二分类 Exp. I 的口径：
  - 论文 Table 5 标的是 `Precision / Recall / F1`，不是 `macroP / macroR / macroF1`。
  - 仓库即使在二分类阶段，也统一输出 `macro_precision / macro_f1 / macro_recall`。
  - 因此 Stage1 binary 的指标口径与论文表 5 不是严格一一对应。
- 评估流程层面的差异：
  - 论文 4.3 节写明训练 epoch 为 `30`。
  - 仓库训练阶段用 `val_macro_f1` 选 `best.ckpt`，评估默认读取 `best.ckpt`，不是固定取最后 30 轮后的模型。
  - 因此即使数据和模型相同，仓库最终报告值也未必与论文实验表格可直接横向比较。
- 工程化扩展差异：
  - 仓库额外生成 confusion matrix csv/png 与 report 汇总，这属于论文展示方式的扩展，不影响 accuracy / precision / recall 的基本定义。

## Stage1 Binary 结果排查结论（2026-03-21）

- `runs/stage1-binary/eval_test.json` 中的 `0.9641693811074918` 不是 `macro_f1`，而是 `src/evaluate.py` 里的 `top1 = accuracy_score(y_eval, pred)`。
- 同一次 test 评估对应的其余指标为：
  - `macro_precision = 0.9521`
  - `macro_f1 = 0.9586`
  - `macro_recall = 0.9660`
- 该 run 的 best checkpoint 不是按 test accuracy 选的，而是按 `val_macro_f1` 选的：
  - `src/train.py` 用 `val_macro_f1 >= best_f1` 保存 `best.ckpt`
  - `runs/stage1-binary/report.md` 记录 best epoch 为 `26`
  - `runs/stage1-binary/metrics.csv` / `train.log` 对应 best val macro-F1 为 `0.9556`
- 从当前代码和产物看，`0.9642` 本身没有显示出明显实现错误：
  - `evaluate.py` 直接在 `test` split 上评估 `best.ckpt`
  - `eval_test.json` 记录 `fallback_used=false`
  - 混淆矩阵 `[[4121,124],[371,9199]]` 与 `num_samples=13815`、`accuracy=0.964169...` 一致
  - test `macro_f1=0.9586` 与 best val `macro_f1=0.9556` 接近，数值走势合理
- 但当前 run 的协议边界很重要，容易导致误解：
  - 该 run 使用的是 `session_full + paper_balanced`，不是严格论文原始 split 的逐样本复现
  - `paper_balanced` 默认允许“不足组全保留、超大组裁到论文配额的 120%”，因此不是严格 Table 1-3 配额
  - 生成的 `stage1_binary_manifest.csv` 只有 `train/test`，训练时 `src/train.py` 会再从 train 内随机切出 `10%` 作为 val；这不是预先固定的 val protocol
  - report 默认优先展示 `Top-1`，用户很容易把 `0.9642` 误读成 `F1`
- manifest 现实分布说明这个 run 也不是“论文强对照”：
  - 总样本 `69144`
  - 类别分布 `normal=21290`、`malicious=47854`，存在明显类不平衡
  - test 分布 `normal=4245`、`malicious=9570`
  - `MTA` 在当前产物中仅有 `1557` 个样本，远低于论文 strict 配额总量；malicious 类主要由 `MFCP` 支撑
- 当前协议里还存在明显的“结果解释风险”：
  - `session_full` 的底层 split 是 `split_by_session`，按 `dataset+family` 内部随机切 session，不是按 capture 切
  - 本次 `stage1_binary_manifest.csv` 中有 `42` 个 `dataset+capture_id` 同时出现在 train/test
  - 由于每个 capture 下往往含有大量 session，这会让 test 更像“同 capture 下的 session 泛化”，结果可能偏乐观
- 如果只问“有没有明显会压低性能的 bug/config”：
  - 没看到直接把结果打坏的实现问题
  - 可能轻微压低/增加波动的因素主要是：
    - 二分类训练使用未加权 `CrossEntropyLoss`，面对 `0:21290 / 1:47854` 的不平衡数据，minority 类 precision/F1 会受影响
    - val 是从 train 内随机切出且未做分层抽样，会带来一定验证波动
- 如果问“有没有会导致误解的协议因素”：
  - 有，而且比“压低性能”更显著：
    - `0.9642` 是 accuracy，不是 F1
    - run 口径是 `paper_balanced`，不是 strict paper reproduction
    - train/test 存在 capture 级混杂，不能把这个数直接当成更严格的跨 capture 泛化能力

## 论文兼容指标设计结论（2026-03-21）

- 项目主口径继续保留 `sklearn` 风格：
  - `top1`
  - `macro_precision`
  - `macro_recall`
  - `macro_f1`
- 为对齐论文展示，评估结果额外补充一套 `paper_*` 字段更合适，而不是直接替换现有指标：
  - 不破坏训练/选模逻辑
  - 保持与当前测试和报表兼容
  - 允许论文复现与工程调参共存

## 论文兼容指标实现结论（2026-03-21）

- `src/evaluate.py` 已新增统一 helper `compute_classification_metrics(...)`：
  - 保留原有 `top1 / macro_precision / macro_recall / macro_f1`
  - 新增 `paper_macro_precision / paper_macro_recall / paper_macro_f1`
  - 二分类额外输出 `paper_precision / paper_recall / paper_f1`
- `paper_macro_f1` 按论文 4.2 节公式计算：
  - `2 * paper_macro_precision * paper_macro_recall / (paper_macro_precision + paper_macro_recall)`
- `src/report.py` 已新增 `Paper-Compatible Metrics` 区块，用于展示新增 `paper_*` 指标。
- `src/ablation.py` 已扩展 summary 列：
  - `paper_macro_precision`
  - `paper_macro_recall`
  - `paper_macro_f1`
- 当前实现边界：
  - 未修改训练阶段的 `val_macro_f1` 与 best checkpoint 选择逻辑
  - `stacking/moe` 产物暂未主动补 `paper_*` 字段；ablation 仅在存在时读取

## Evaluation / Report 表格补齐结论（2026-03-21）

- 当前仓库原实现里，`src/evaluate.py` 只会输出：
  - `eval_<split>.json`
  - `confusion_matrix_<split>.csv`
  - `confusion_matrix_<split>.png`
- 当前仓库原实现里，`src/report.py` 只会在 `report.md` 里列出 artifact 路径，不会渲染表格。
- 本轮实现后：
  - `src/evaluate.py` 会额外输出：
    - `classification_report_<split>.csv`
    - `classification_report_<split>.json`
  - `src/report.py` 会在 `report.md` 中直接渲染：
    - `## Confusion Matrix`
    - `## Classification Report`
- 新增 classification report 的行结构为：
  - 每类一行：`label / precision / recall / f1 / support`
  - 额外包含：
    - `accuracy`
    - `macro avg`
    - `weighted avg`
- 已用新代码重刷现有 `runs/stage1-binary`：
  - 新增 `runs/stage1-binary/figures/classification_report_test.csv`
  - 新增 `runs/stage1-binary/figures/classification_report_test.json`
  - `runs/stage1-binary/report.md` 现已直接显示混淆矩阵与分类指标表
- 当前 `runs/stage1-binary` 刷新后的 test classification report 关键值为：
  - `label 0`: precision `0.9174` / recall `0.9706` / f1 `0.9432` / support `4245`
  - `label 1`: precision `0.9866` / recall `0.9612` / f1 `0.9737` / support `9570`
  - `accuracy`: `0.9641`
  - `macro avg f1`: `0.9585`

## 日志中文化结论（2026-03-22）

- 结构化日志主模板已中文化：
  - level：`成功/警告/错误/信息`
  - module：`数据/模型/评估/保存/时间/指标`
- event 展示改为“中文说明 + 英文 event code”：
  - 例如：`配置摘要 (config_summary)`
  - 这样既满足中文可读性，也保留了英文 event code 的可检索性和测试兼容性。
- `src/experiments/stage1_binary.py` 的直出日志已翻译为中文语义（如“评估步骤开始”“Manifest 已保存”等）。
- `src/ablation.py` 的 CLI 输出日志已翻译为中文（如“ablation 计划已保存”）。
- 现有与日志文案直接耦合的测试已同步更新，并完成针对性回归通过。

## Timeline Run Dir 解析结论（2026-03-21）

- `src/train.py` 的默认 run 布局是：
  - `runs/YYYY-MM-DD/<auto_run_id>`
- 但 `src/evaluate.py` / `src/report.py` 之前要求 `--run-dir` 必须已经是完整目录，因此输入：
  - `runs/stage1-binary`
  - 或仅输入短 run id
  在“真实目录位于日期分区下”时会直接报 `config.yaml` 不存在。
- 本轮已新增 `src/run_dir.py`，并让 `evaluate/report` 统一复用：
  - 如果 `--run-dir` 本身就是完整目录，直接使用
  - 如果不是，则自动在 `runs/**/<run-id>` 下查找
  - 若存在多个同名时间分区 run，则优先选择最新日期分区
- 因此现在可以直接用短路径：
  - `python -m src.evaluate --run-dir runs/<run-id> ...`
  - `python -m src.report --run-dir runs/<run-id>`
  无需手工先展开到 `runs/YYYY-MM-DD/<run-id>`

## 最新 run 结果排查结论（2026-03-21）

- 当前最新 run 是：
  - `runs/2026-03-21/stage1-binary-195511`
- 这次 run 的三个关键数值分别是：
  - 训练最后一轮：`train_acc=0.9623`，`train_macro_f1=0.9566`
  - 最佳验证：`epoch=29`，`val_acc=0.9626`，`val_macro_f1=0.9564`
  - 测试：`top1=0.9639`，`macro_f1=0.9583`
- 因此“只有 96%”不是单一异常点，而是：
  - train / val / test 三侧都稳定落在 `95.6% - 96.4%`
  - 更像当前协议与模型组合下的稳定平台，而不是单次评估偶然波动
- 当前 run 没看到明显把结果打坏的实现错误：
  - `best.ckpt` 仍按 `val_macro_f1` 选择
  - test confusion matrix `[[4130,115],[384,9186]]` 与 `accuracy=0.963879...` 一致
  - 训练末轮并未明显高于验证/测试，缺少典型过拟合特征
- 更可能限制结果的因素有三类：
  - 类不平衡：manifest 总分布为 `malicious=47854`、`normal=21290`，训练端仍使用未加权 `CrossEntropyLoss`
  - 验证切分粗糙：manifest 只有 `train/test`，`src/train.py` 会再从 train 随机切 `10%` 作为 val，且不是分层抽样
  - 协议异质性：`paper_balanced` 下 test 的 `67.0%` 来自 `MFCP`，`30.7%` 来自 `ISCX`，`MTA` 仅 `2.25%`
- 从分类明细看，当前主要短板不是整体崩掉，而是类别两侧仍有稳定残余误差：
  - `label 0 (normal)`：precision `0.9149`，recall `0.9729`
  - `label 1 (malicious)`：precision `0.9876`，recall `0.9599`
  - 说明模型更容易把一部分样本判成 `malicious`，normal 侧 precision 被拉低
- 另一个重要事实是：这个 `96.39%` 不能视为“严格跨 capture 泛化”的保守成绩：
  - 当前 manifest 中有 `42` 个 `dataset+capture_id` 同时出现在 train/test
  - 这会带来一定 capture 级 overlap
  - 因此如果用户觉得“96 已经偏低”，从协议角度看它甚至更可能是略偏乐观，而不是被评估脚本压低

## Accuracy-Oriented Training 结论（2026-03-21）

- 为优先提升当前协议下的 binary `accuracy`，本轮实现了三项训练策略改动：
  - 派生 validation 从“全局随机切分”改为“按 label 分层切分”
  - `best.ckpt` 选择指标改为可配置，支持 `val_macro_f1` 与 `val_acc`
  - 二分类 validation 会自动搜索最佳 decision threshold，并在评估时复用
- 当前实现边界：
  - 默认 `best_metric` 仍是 `val_macro_f1`，保持兼容
  - 只有 binary 任务会启用阈值校准；多分类仍然走 `argmax`
  - 未改动 `stage1_binary` 协议本身，也未引入 class weight / sampler
- 这组改动更偏向“提高当前协议下的最终 acc”，不是更严格泛化评估。

## Stage1 Execute 透传结论（2026-03-22）

- 用户 2026-03-22 的新 run `runs/2026-03-22/stage1-binary-133158` 虽然已经写入 `decision_threshold`，但 `best_metric` 仍为 `val_macro_f1`。
- 根因不是 `src.train` 忽略了 `--best-metric`，而是 `src.experiments.stage1_binary --execute` 路径此前：
  - CLI 不接受 `--best-metric`
  - `run_stage1_protocol(...)` 也没有把 `best_metric` 透传给 `train_main(...)`
- 因此如果用户通过 `stage1_binary --execute` 一键跑实验，即使预期想走 accuracy-first，也会默默退回默认 `val_macro_f1`。
- 本轮修复后：
  - `stage1_binary` CLI 已支持 `--best-metric {val_macro_f1,val_acc}`
  - `--execute` 路径会将该参数继续传给 `src.train`

## Cross-Attention Stage1 70% Acc 排查结论（2026-03-23）

- 当前 cross-attention 改造后的完整失败 run 是：
  - `runs/stage1-binary`
  - `train.log` 记录的 git commit 为 `d72aec7`
  - `config.yaml` 已包含 `fusion_layers / fusion_heads / fusion_dropout`
  - 指标观测字段也从旧版 `gate_mean` 变为 `fuse_conf_mean`
- 这次失败不是评估口径问题，也不是中途崩溃，而是训练在第 2 个 epoch 起快速塌缩为“全预测恶意类”：
  - `train_acc / val_acc / test_top1` 长期稳定在 `~0.69`
  - 该数值与整体 malicious 占比 `47854 / 69144 = 0.692` 基本一致
  - `eval_test.json` 中：
    - `top1 = 0.692725...`
    - `macro_precision = 0.3463`
    - `macro_recall = 0.5`
    - `macro_f1 = 0.4092`
  - 这组数值与“二分类里全预测为正类/恶意类”的行为一致
- 进一步用 `runs/stage1-binary/checkpoints/best.ckpt` 做分支级只读复算后，确认不是只有融合头坏掉，而是三路头全部塌缩：
  - `fuse`：`acc=0.6927`，`positive_rate=1.0`
  - `img`：`acc=0.6927`，`positive_rate=1.0`
  - `tls`：`acc=0.6927`，`positive_rate=1.0`
  - `warmup_avg`：`acc=0.6927`，`positive_rate=1.0`
- 和旧高分 run 对比：
  - `runs/2026-03-22/stage1-binary-153827`（gate 时代）可以稳定收敛到：
    - `val_acc ≈ 0.9617`
    - `test_top1 ≈ 0.9655`
  - 说明数据协议与评估链路本身没有突然变成只能跑 70%，回退点在模型/训练路径
- 当前 cross-attention 实现里，最可能直接伤害 stage1 收敛的结构性变化有三项：
  - 旧 gate / attention 版本中，`head_img` 和 `head_tls` 直接吃 backbone 的 pooled feature；当前版本在 `use_fusion=True` 时，两个辅助头吃的是 fusion 后的 `img_ctx/txt_ctx`，不再承担“稳定单模态监督”的作用
  - `MobileViTBackbone` 在 fusion 路径下不再直接使用已有的 pooled image feature，而是改走多尺度 hidden state -> 随机初始化 `token_proj` -> fusion encoder；这等于把最强的图像预训练表征绕开了
  - 当前 stage1 run 直接以 `stage=fusion` 从头训练，没有 warmup 阶段；因此 `token_proj + fusion_encoder + fusion_proj + 三个 head` 在不平衡二分类上同步从随机状态优化，最容易收敛到多数类解
- 当前 run 的超参也比旧高分 run 更激进，放大了上述不稳定性：
  - `hidden_dim: 128 -> 192`
  - `batch_size: 16 -> 32`
  - `alpha/beta: 0.3/0.3 -> 0.2/0.2`
  - 新增 `fusion_layers=3`
  - 新增 `fusion_dropout=0.2`
- 综合判断：
  - 主因不是“cross-attention 天生比 gate 差”，而是当前这版 bidirectional multi-token fusion 把预训练表征的稳定路径切断了，同时又没有 warmup/残差兜底，导致在类不平衡 stage1 binary 上迅速掉进多数类塌缩解。

## Cross-Attention Stabilization + Early Stopping 实现结论（2026-03-23）

- 本轮已在隔离 worktree `./.worktrees/codex-cross-attn-stabilization` 中完成实现。
- 模型侧修复点：
  - `src/models/fusion_model.py` 中，`head_img` 与 `head_tls` 已改回监督 pre-fusion pooled feature
  - fusion 主头现在吃 `fused image context + fused text context + pre-fusion pooled image + pre-fusion pooled text`
  - `return_features` 与三路 logits 的外部接口保持不变
- 训练侧修复点：
  - `src/train.py` 新增 `--early-stopping-patience`
  - sentinel 为 `0`，表示禁用
  - early stopping 监控指标复用 `--best-metric`
  - 只有严格更优才重置 patience；持平不算 improvement
  - 触发时会写 `early_stopping_triggered` 日志，并保留已写出的 `best.ckpt` / `metrics.csv`
- 协议入口修复点：
  - `src.experiments.stage1_binary.py` 已支持并透传 `--early-stopping-patience`
- 测试侧新增/修正约束：
  - 模型测试已从“aux heads 吃 fused context”改为“aux heads 吃 pre-fusion pooled feature”
  - 新增 fusion head pooled shortcut 测试
  - 新增 early stopping 的 trigger、`best-metric` 绑定、tie 语义、默认关闭、stage1 协议透传测试
- 本轮验证命令需要显式清空环境污染的 Python 路径：
  - `PYTHONPATH= PYTHONNOUSERSITE=1 /home/shuora/miniconda3/envs/FusionModel/bin/pytest ...`
  - 原因是当前桌面环境默认把 Python 3.12 的 `~/.py-user` 注入到 Python 3.9 conda 环境中
- 已完成定向回归：
  - `tests/models/test_fusion_model.py`
  - `tests/pipeline/test_train_eval_report.py`
  - `tests/pipeline/test_protocol_execution.py`
  - 结果：`48 passed`

## 实验命令文档补丁结论（2026-03-23）

- `docs/commands/session-full-experiments.md` 已补回当前真实可用的 `--early-stopping-patience` 参数说明。
- 已同步更新两处 stage1 示例命令：
  - 手动 `src.train` 训练命令
  - `src.experiments.stage1_binary --execute` 一键命令
- 文档中的 early stopping 语义已明确：
  - `0` 表示禁用
  - `> 0` 表示启用
  - 监控指标复用 `--best-metric`
- 同时修正了底部“旧内容替换”区块中的过期表述：
  - 从“训练支持 early-stopping 参数”改为“训练完全不支持 early-stopping 参数”
  - 以避免与当前代码实现冲突

## Python 环境污染排查结论（2026-03-23）

- conda 本身不是根因；真正污染来源是 shell 启动链路里把 Windows 侧 `.py-user` 的 Python 3.12 site-packages 全局注入到了所有 shell：
  - `PYTHONPATH=/mnt/c/Users/11098/.py-user/lib/python3.12/site-packages:...`
  - `PYTHONUSERBASE=/mnt/c/Users/11098/.py-user`
- 初始状态下，即使使用 `/home/shuora/miniconda3/envs/FusionModel/bin/python`，`sys.path` 仍会把：
  - `/mnt/c/Users/11098/.py-user/lib/python3.12/site-packages`
  放在前面，导致 Python 3.9 conda 环境错误加载到 Python 3.12 的 `numpy`
- 直接原因有两层：
  - `~/.zshrc` / `~/.zprofile` 里曾经显式 `export PYTHONPATH=...python3.12...`
  - 当前桌面终端会话本身也会继承这条 `PYTHONPATH`
- 本轮已完成修复：
  - 从 `~/.zshrc` / `~/.zprofile` 中删除了直接导出的 `PYTHONPATH`
  - 改为在 zsh 启动时主动过滤掉：
    - `/mnt/c/Users/11098/.py-user/lib/python3.12/site-packages`
  - 保留了 `PYTHONUSERBASE` 与 `.py-user/bin` 的 PATH，不影响用户级工具命令
- 验证结果：
  - 在新的 login zsh 中，模拟污染输入：
    - `PYTHONPATH=/mnt/c/Users/11098/.py-user/lib/python3.12/site-packages:/tmp/demo`
  - 启动后实际变为：
    - `PYTHONPATH=/tmp/demo`
  - `sys.path` 中不再包含 `.py-user/lib/python3.12/site-packages`
  - conda Python 3.9 的 site-packages 顺序恢复正常

## Stage1 98+ 目标设计前提（2026-03-24）

- 用户对这轮目标的要求已从“修复 cross-attention 退化”升级为：
  - `stage1 binary` 指标冲击 `98%+`
- 用户明确允许的改动边界为：
  - 训练策略可改
  - `stage1_binary` 数据协议可改
  - 当前 fusion 结构可改
- 因此这轮不再适合继续按“小修补”思路推进，而应按“高分方案重设计”处理。
- 设计侧当前共识：
  - 单靠当前 protocol 下的小幅调参与稳定化，无法有把握从 `95.2%` 直接拉到 `98%+`
  - 更合理的路线是同时重做：
    - protocol / sampling
    - warmup + fusion 的训练阶段设计
    - fusion 在整体判别中的角色分工

## Stage1 High-Score Redesign 当前实现结论（2026-03-24）

- 已落地的协议层能力：
  - `src.experiments.stage1_binary.py` 新增 `protocol_mode=score_optimized`
  - `score_optimized` 会生成显式 `train/val/test`
  - binary class 在各 split 内被重新平衡
  - malicious 侧会约束 `MFCP/MTA` 的相对平衡，避免单源主导
- 已落地的数据传播能力：
  - `src.pipeline_data.py` 现在会让 `session_filter_manifest` 覆盖 `split/dataset/family`
  - 因而显式 `val` split 不会在数据加载阶段被抹掉
- 已落地的训练/执行能力：
  - `src.train.py` 新增：
    - `--checkpoint-selection`
    - `--warmup-checkpoint`
  - `src.experiments.stage1_binary.py` 新增：
    - `--holdout-eval {always,final_only}`
    - `--two-stage`
    - `--warmup-epochs`
  - `score_optimized` 默认会向训练透传：
    - `--checkpoint-selection score_optimized`
  - `--two-stage` 会先跑 warmup 子 run，再把其 `best.ckpt` 作为 fusion 阶段初始化
- 已落地的 fusion 角色重构：
  - `src.models.fusion_model.py` 现支持：
    - `fusion_mode=legacy`
    - `fusion_mode=residual_enhancer`
  - `legacy` 保持旧 checkpoint 的主前向语义
  - `residual_enhancer` 下：
    - `use_fusion=False` 时，`logits_fuse` 回退为 image 主路径
    - `use_fusion=True` 时，fusion 通过有界 residual 做增强
    - `text_shortcut_scale` 已进入 `state_dict`，不再是纯代码硬编码
- 已落地的 report 修正：
  - `src/report.py` 的 `Best Validation` 优先读取 `best.ckpt` 记录的最佳 epoch
  - 不再固定按 `metrics.csv` 中的 `val_macro_f1` 重新排序
- 当前回归证据：
  - `tests/models/test_fusion_model.py`
  - `tests/pipeline/test_stage1_binary_protocol.py`
  - `tests/pipeline/test_pipeline_data_protocol.py`
  - `tests/pipeline/test_train_eval_report.py`
  - `tests/pipeline/test_protocol_execution.py`
  - 汇总结果：`77 passed`
- 尚未完成的部分不是代码正确性，而是高分验收实验本身：
  - protocol-only baseline
  - protocol + two-stage baseline
  - final 98% 验收 run

## Protocol 默认日期分区现状（2026-03-24）

- `src.train.py` 已支持默认日期分区：
  - 当不传 `--run-id` 时，产物会写到 `runs/YYYY-MM-DD/<HHMMSS-ffffff>`
- 但协议执行脚本绕开了这套默认行为：
  - `src.experiments.stage1_binary.py` 会固定把训练产物写到 `run_root / run_id`
  - `src.experiments.stage2_multiclass.py` 也同样固定使用 `run_root / run_id`
- 因此用户即使只传 `--run-root runs`，一键协议执行的结果仍是扁平目录：
  - `runs/stage1-binary`
  - `runs/stage2-mta`
  - `runs/stage2-mfcp`
- `src.run_dir.resolve_run_dir()` 已经能把 `runs/<run-id>` 自动解析到最新 `runs/YYYY-MM-DD/<run-id>`，所以如果协议执行层改成日期分区，后续 `evaluate/report` 的短路径兼容性可以直接复用现有能力。
- `stage2_execution_summary.json` 当前写在 `run_root` 根目录，且 summary item 只有 `run_id` / `code` / `dataset` 等信息，不包含实际 `run_dir` 或 `run_date`，不利于定位当天执行产物。

## Protocol 默认日期分区实现结果（2026-03-24）

- 已将协议执行层默认 run 目录改为：
  - `runs/YYYY-MM-DD/stage1-binary`
  - `runs/YYYY-MM-DD/stage2-mta`
  - `runs/YYYY-MM-DD/stage2-mfcp`
- 改动边界保持在协议执行层和共享 helper：
  - `src.train.py` 的显式 `--run-id` 通用语义未被改成“强制日期分区”
  - 仅 `src.experiments.stage1_binary.py` / `src.experiments.stage2_multiclass.py` 会在内部自动把 `--run-root` 展开到当天日期分区
- `stage2_execution_summary.json` 现已随当天批次一起写到：
  - `runs/YYYY-MM-DD/stage2_execution_summary.json`
- summary item 现已补充：
  - `run_date`
  - `run_dir`
  这样即使同一个 `run_id` 跨天重复，也能直接定位到真实产物目录。
- 现有短路径兼容能力保持不变：
  - `src.run_dir.resolve_run_dir()` 仍可把 `runs/<run-id>` 自动解析到最新日期分区下的同名 run
  - 因而后续 `evaluate/report` 无需改接口
