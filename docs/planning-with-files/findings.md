# Findings

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

## Stage1 Binary Acc 可解释性修复发现（2026-03-22）

- 最新 run `runs/2026-03-22/stage1-binary-153827` 的 `test top1` 实际为 `0.9655`，并非固定 `0.95`。
- 训练日志当前只输出 `val_acc(argmax)` 与 `val_decision_threshold`，缺少“阈值校准后 acc”字段，易造成口径误读。
- `report.py` 当前 best row 固定按 `val_macro_f1` 排序，忽略了 `config.best_metric`，在 `best_metric=val_acc` 时会出现“配置与报告不一致”。
- 在当前沙箱环境，`num_workers>0` 的真实训练会触发 multiprocessing socket 权限错误（`PermissionError: [Errno 1] Operation not permitted`），端到端验证需使用 `--num-workers 0`。

## Stage1 Binary Acc 可解释性修复实现结论（2026-03-22）

- `src/train.py` 已新增：
  - `choose_best_binary_threshold(...)`，在验证集上搜索使 accuracy 最大的阈值；
  - `val_acc_at_decision_threshold` 指标，写入 `metrics.csv` 且输出到 `epoch_done` 日志；
  - `--best-metric {val_macro_f1,val_acc}` 参数，best checkpoint 选择不再固定 `val_macro_f1`；
  - `best.ckpt` 与 `config.yaml` 写入 `decision_threshold`。
- `src/report.py` 已调整：
  - `Best Validation` 的 best row 由 `config.best_metric` 决定（缺失时回退 `val_macro_f1`）；
  - 当 `metrics.csv` 含 `val_acc_at_decision_threshold` 时，报告展示 `Val Acc @ Decision Threshold`。
- 端到端快速验证 run：
  - `runs/2026-03-22/stage1-binary-e2e-accfix-fast`
  - 训练日志已出现 `val_acc_at_decision_threshold=...`
  - 报告已出现 `Best Metric: val_acc` 与 `Val Acc @ Decision Threshold: ...`

## Attention Fusion 改造发现（2026-03-22）

- 当前 `MobileViTETBertFusionClassifier` 融合方式是单标量门控：
  - `gate = MLP([img_feature, tls_feature])`
  - `fused = gate * img + (1-gate) * tls`
- 该实现不属于注意力层面的融合；要改成 attention，最小可行方案是：
  - 将 `img_feature` 与 `tls_feature` 视作两个 modality token；
  - 使用 learnable query 对这两个 token 做 cross-attention，输出 fused feature。
- 下游依赖兼容要求：
  - `train/evaluate/stacking/moe` 都读取 `out["gate"]`，因此新实现仍需返回 `gate`，可定义为 attention 对 image token 的权重。

## Attention Fusion 改造实现结论（2026-03-22）

- `src/models/fusion_model.py` 已改为注意力融合：
  - 输入 token：`[img_feature, tls_feature]`
  - 融合机制：learnable query + `MultiheadAttention` cross-attention
  - 融合后：`LayerNorm + MLP + LayerNorm`
- 输出接口保持兼容：
  - `logits_fuse/logits_img/logits_tls` 保持不变
  - `gate` 保持 `(B,1)`，定义为 attention 分配到 image token 的权重
- `num_heads` 已从训练配置透传到各入口：
  - `train.py`
  - `evaluate.py`
  - `stacking.py`
  - `moe.py`
- 为防止错误配置，模型初始化新增约束：
  - `hidden_dim % num_heads == 0`，否则抛 `ValueError`

## ETBERT Nested Tensor Warning 修复结论（2026-03-22）

- warning 根因位于 `ETBertBackbone`：
  - `nn.TransformerEncoder(...)` 默认 `enable_nested_tensor=True`
  - 在部分 PyTorch 运行路径下会触发 `nested tensors is in prototype stage` 的用户警告
- 本次修复采用最小策略：
  - 显式设置 `enable_nested_tensor=False`
  - 不修改 encoder 层数、注意力头数、mask 逻辑或 pooling 逻辑
- 该修复的目标是消除日志污染，而不是调整模型行为；前向语义保持不变。
