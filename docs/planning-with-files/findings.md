# Findings

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
