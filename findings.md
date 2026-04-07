## Findings

- `src/fusion_common.py` 里有两处默认路径原本仍指向 `src/outputs`：`setup_logging()` 默认日志目录、`add_common_args()` 的 `--output_dir` 默认值。
- `src/run_all_modes.py` 只是透传 `fusion_common.py` 生成的 `output_dir`，本次无需单独修改。
- `README.md` 的四个实验命令和“训练输出”章节原本写成 `src/outputs/<task_name>/...`，已同步改为根目录 `outputs/<task_name>/...`。
- `tests/test_fusion_output_artifacts.py` 原先不覆盖“默认路径在仓库根目录”的行为，本次新增了默认 `output_dir` 与默认日志目录的回归测试。
- `AGENTS.md` 原先存在未解决的 merge conflict 标记；本次已合并有效约束并补充默认输出目录说明。
- 本机测试环境存在 `PYTHONPATH=/mnt/c/Users/11098/.py-user/lib/python3.12/site-packages` 污染；运行 conda 环境下的验证命令时，需要显式 `unset PYTHONPATH PYTHONHOME PYTHONUSERBASE` 并设置 `PYTHONNOUSERSITE=1`。


- 2026-03-31 MFCP: SourceData has 6 families/7 pcaps; processed mfcp currently misses Cobalt due old pre-fix truncation handling and should be regenerated with current split_data.

- 2026-03-31 EarlyStopping: CLI 默认 `--patience=8`，但 `EarlyStopping.__init__` 与 `train_fusion_model` 内部默认仍是 7，存在默认值不一致。
- 2026-03-31 EarlyStopping: 旧实现允许 `early_stop_metric` 与手动 `early_stop_mode` 方向冲突（如 `val_f1 + min`），会造成静默错误早停；现改为显式报错。
- 2026-03-31 EarlyStopping: 旧实现在监控值为 NaN/Inf 时会累加 early-stop 计数并喂给 ReduceLROnPlateau；现改为跳过该轮并记录 warning。
- 2026-04-01 NaN早停: `src/fusion_common.py` 在 `monitor_is_finite=False` 时原先仅 warning 并跳过 early-stop 更新；这会让计数器停滞，导致 `patience` 失效并持续训练至 `num_epochs`。
- 2026-04-01 NaN训练防护: 训练循环此前对 batch-level 非有限 loss 缺少保护；现新增 finite-check，NaN/Inf batch 直接跳过，避免无效梯度污染参数。
- 2026-04-01 训练记录审计: `outputs/` 共有 11 个 run 目录，其中仅 5 个完整产出（含 `done attention|done stacking` 与 `metrics.json`），其余 6 个 run 仅启动到 `Epoch 1~8` 即终止，无异常栈信息，形态上更接近人为重启/中断。
- 2026-04-01 训练记录审计: `binary_benign_vs_malicious/attention_dim256_20260401_021405` 在第 9 轮开始出现 `train_loss/val_loss=nan`，后续持续到第 30 轮，最终评估塌缩为单类预测（acc=0.4000, macro_f1=0.2857，恶意类召回=0）。
- 2026-04-01 训练记录审计: `mta_multiclass` 在 attention 与 attention_stacking 两次完成 run 中都出现极小类完全失效（Dridex support=6, recall=0），accuracy 高但 macro_f1 明显偏低（0.5943/0.4779），存在“被大类掩盖”的风险。
- 2026-04-02 全量训练日志审计: 最新 8 个 run 全部完成，但 `binary_benign_vs_malicious/attention_stacking_20260402_013619` 与 `mta_multiclass/attention_stacking_20260402_062100` 在后半程出现连续 `NaN/Inf batch`；前者从 epoch 7 batch 1267/15900 开始，累计 30513 个 batch 被跳过，后者从 epoch 21 batch 940/4305 开始，累计 3366 个 batch 被跳过。
- 2026-04-02 全量训练日志审计: 当前默认训练配置仍是 `lr=1e-3`、`weight_decay=0`、`grad_clip_norm=0`、`lr_scheduler=none`、`early_stop_metric=val_loss`；这两次崩坏都在该配置下发生，attention-only run 未复现，形态更像数值稳定性被随机状态触发，而不是固定脏数据样本。
- 2026-04-02 全量训练日志审计: `run_all_modes.py --mode all` 只在入口通过 `build_common_kwargs()` 调一次 `set_seed(args.seed)`；随后 attention 与 attention_stacking 在同一进程顺序执行，stacking 并不是从同样随机初始状态起跑，削弱了模式间可比性与复现性。
- 2026-04-02 全量训练日志审计: `mta_multiclass` 仍是本轮最弱任务。attention `macro_f1=0.5511`，stacking base `0.5724`，xgboost `0.6105`；但 `Dridex` 召回仍为 `0.0000 -> 0.0224`，说明主要瓶颈仍是严重类别不均衡与少数类学习不足，而不是 stacking 元学习器本身。
- 2026-04-02 全量训练日志审计: `mfcp_multiclass` 无 NaN，但 attention / stacking base 提升很小（0.7756 -> 0.7769），主要弱类是 `Ursnif`（recall≈0.509），xgboost 后整体 `macro_f1=0.7867`，说明堆叠收益主要来自后端分类器而非 base fusion 明显增强。
- 2026-04-02 全量训练日志审计: `ustc_multiclass` 是当前最稳定任务，attention `macro_f1=0.9719`，xgboost `0.9757`；但 attention 在 epoch 15 出现 `val_loss=0.9455` 的孤立尖峰，而同时 `val_acc/val_f1` 仍约 `0.969`，说明仅盯 `val_loss` 对高置信多分类任务并不稳健。

- 2026-04-03 stacking 实现复核: 当前 `run_stacking_experiment` 用训练集元特征直接训练 meta learner，再在测试集评估；无 OOF 机制，存在元学习器过拟合风险。
- 2026-04-03 stacking 实现复核: 当前元特征仅由 `text_prob + image_prob` 构成，缺少 fusion 分支输出和不确定性统计特征（entropy、margin、分支一致性）。
- 2026-04-03 stacking 实现复核: XGBoost 目前未使用类别权重，且仅单模型报告；尚未做多元学习器融合与任务定向后处理。
- 2026-04-03 stacking 改造结果: 已新增 OOF 训练与 OOF 指标落盘（`oof_acc` / `oof_macro_f1`），并在多 meta learner 可用时自动生成 `soft_voting` 结果。
- 2026-04-03 stacking 改造结果: 元特征已扩展为 `text/image/fusion` 概率 + entropy + margin + 分支一致性；对应纯函数已补单测覆盖。
- 2026-04-03 stacking 改造结果: `xgboost/lightgbm/catboost` 训练路径已接入 class-balanced sample weight；`mta` 增加类增益调优，`mfcp` 增加 `0/4` 二分类校正头。

- 2026-04-04 梯度无效排查: 训练日志显示 `attention_stacking` 在 `epoch=1,batch=1/2` 即出现“梯度无效（NaN/Inf）”，且主要发生在 `AMP + weighted_sampler_loss + focal` 组合下。
- 2026-04-04 梯度无效排查: 当前 AMP 分支在 `scaler.unscale_` 后自行扫描梯度并将非有限梯度计入 `invalid_grad_batches`，会把 `GradScaler` 可恢复的 overflow 也记为“梯度无效”并触发跳过告警。
- 2026-04-04 梯度无效修复: 已移除 AMP 路径中的 `_has_non_finite_gradients` 强制跳过逻辑，改为交给 `GradScaler.step/update` 统一处理 overflow；CPU/非AMP 路径的梯度有限值保护保持不变。
- 2026-04-04 回归验证: 新增测试 `test_amp_overflow_is_not_counted_as_invalid_grad_batch`，确保 AMP overflow 不再记为 `invalid_grad_batches` 且 `scaler.step()` 会被调用。
- 2026-04-04 mta/mfcp 现状审计: 仓库已实现 `mta` 类增益调优与 `mfcp` `0/4` 二分类校正，但 `mta` 目标类硬编码为 `[0,1]`，对类顺序/映射有隐式依赖。
- 2026-04-04 mta 增强: 已改为按 `meta_labels` 样本数自动选择最少的两个类别做 gain 调优，仅在 `mta_multiclass` 触发。
- 2026-04-04 mfcp 增强: 已新增 OOF 驱动的 `alpha` 自动调参（`0~1`）用于控制 `0/4` 二分类校正强度，避免固定强校正导致过修正；仅在 `mfcp_multiclass` 触发。
- 2026-04-04 mfcp 新一轮诊断: 最新 run（`attention_stacking_20260404_145855`）虽稳定无无效梯度，但 `0<->4` 互混仍重，且 `mfcp_binary_pair_alpha=0.0`，说明现有“按全局 macro-f1 选 alpha”在该批数据上未激活 pair 校正。
- 2026-04-04 mfcp 二次调优: 已将 `alpha` 调参目标扩展为 `objective=\"pair_f1\"`，并新增 pair 概率温度校准与阈值搜索（均基于 OOF 调参后迁移到测试集）。
- 2026-04-04 输出可观测性: `metrics.json` 的 `postprocess` 新增 `mfcp_pair_temperature` 与 `mfcp_pair_threshold`，用于复盘后处理是否真正生效。
- 2026-04-04 论文分布对齐: 当前 `split_data.py` 仅支持按标签比例随机切分，无法表达论文 Table 2/3 的固定 Train/Test 计数，因此新增了 `distribution_profile` 抽样路径。
- 2026-04-04 论文分布对齐: `SourceData/MTA` 实际家族目录名为 `IcedID`（大写 D），与论文写法 `Icedid` 存在大小写差异；本次按仓库现有目录命名保留为 `IcedID`。
- 2026-04-04 论文分布对齐: `SourceData/MFCP` 含 `Cobalt`，之前 `ProcessedData/mfcp_multiclass` 缺失并非源数据不存在，而是旧处理产物未按论文口径重建。
- 2026-04-04 论文分布对齐: MFCP 的 PUA 可提取 session 为 6737，低于论文目标 7017；为满足目标计数，本次对 PUA 启用了有放回补齐，并为补齐样本加 `__dupN` 后缀避免文件覆盖。

- 2026-04-04 MTA 指标复盘: 论文分布下 `mta_multiclass` 类别极不均衡（Train/Test 均约 `27.34:1`），明显高于 `ustc_multiclass` 的 `1.69:1`，直接限制“接近 USTC 指标”的可达上限。
- 2026-04-04 根因1 (实现偏差): stacking 元特征此前直接使用训练 `train_loader` 提取，在 `weighted_sampler_loss` 下会继承 `WeightedRandomSampler + drop_last`，导致 OOF 评估与真实分布偏离并放大乐观偏差。
- 2026-04-04 根因2 (任务识别): `run_stacking_experiment` 对 MTA 的 class-name 硬编码未包含 `IcedID`，导致 MTA 定向 gain 后处理在 7 类 MTA 数据上未触发。
- 2026-04-04 证据: 最新 MTA run 的 meta learner 出现 `oof_macro_f1≈0.948` 但 `test_macro_f1≈0.733`（gap≈0.216），已在本次修复中加入日志级告警。
- 2026-04-04 修复: 新增 deterministic meta loader（顺序、全量、无 sampler、无 drop_last）并接入 stacking；同时改为 task hint + class signature 的任务识别，恢复 MTA 后处理触发。
- 2026-04-05 预处理输出改造: `ssl_tls_rgb_image.py` 原先每张图片都会 `logger.info("Saved: ...")`，在大数据集上会刷屏；现改为仅显示 `tqdm` 进度条并附带 processed/skipped 统计。
- 2026-04-05 预处理日志落盘: `split_data.py` 与 `ssl_tls_rgb_image.py` 新增 `--log_file`，默认分别写入 `<processed_root>/metadata/split_data.log` 与 `<dataset_root>/metadata/ssl_tls_rgb_image.log`。
- 2026-04-05 测试环境问题: `tests.test_ssl_tls_rgb_image` 在当前机器失败原因为用户目录下 cp312 的 numpy 动态库污染 py39 环境（`_multiarray_umath.cpython-312`），与本次功能改动无直接关系。

- 2026-04-06 MFCP 对齐 MTA 改动: `run_stacking_experiment` 的 MFCP pair 后处理原先硬编码 `class_a=0,class_b=4`，在论文分布对齐后引入 `Cobalt` 的 6 类 MFCP 下会出现索引漂移风险。
- 2026-04-06 MFCP 对齐 MTA 改动: 已改为按类名解析 pair（`Artemis/Ursnif`）并在 `method` 与 `soft_voting` 两条路径统一生效，避免依赖类别顺序。
- 2026-04-06 任务识别鲁棒性: `MFCP` class signature 从单一 5 类扩展为 5/6 类双签名，保证含 `Cobalt` 时仍触发 MFCP 定向后处理。
- 2026-04-06 可观测性补强: `metrics.json` 的 `postprocess` 新增 `mfcp_pair_classes`，用于确认每次 run 实际执行的 pair 类别。
- 2026-04-06 命令文档同步: `README.md` 的 `mfcp_multiclass` stacking 命令已对齐 MTA 当前推荐稳定配置（多 meta learner + focal/class-balance + val_f1 early-stop + reduce scheduler）。

- 2026-04-07 CharBERT 现状复核: `src/CharBERT/src/model.py` 当前是轻量 byte Transformer（Embedding + TransformerEncoder），不含独立 char encoder 与分层 token/char 融合机制。
- 2026-04-07 融合链路兼容性: `src/fusion_common.py` 当前 cross-attention 主体依赖文本序列特征（`Q=image, K/V=text`），文本分支内部升级为 char-aware 后可保持外部接口不变。
- 2026-04-07 设计结论: 在“训练入口不变”约束下，推荐 `legacy/charaware` 双模式兼容升级；默认保留 `legacy`，通过参数显式启用 `charaware`，以降低回归风险。

- 2026-04-07 实现落地: `src/CharBERT/src/model.py` 已新增 `charaware` 模式，包含 token/char 双路径、char-CNN 编码与分层融合（`gated/add/concat`）。
- 2026-04-07 兼容策略: `src/fusion_common.py` 新增 `--charbert_mode` 等参数并透传到文本编码器；默认仍 `legacy`，现有 attention/stacking 命令不改也可跑。
- 2026-04-07 接口兼容: 新模型提供 `encode_tokens`，`CharBERTTextEncoder` 优先走该接口；旧 `legacy` 路径仍可用。
- 2026-04-07 验证结果: 在清理 Python 用户站点污染后（`unset PYTHONPATH PYTHONHOME PYTHONUSERBASE && PYTHONNOUSERSITE=1`），目标测试集 `26 tests` 全部通过。

- 2026-04-07 two-level 改造: `attention_stacking` 新增参数 `--stacking_level/--stacking_calibration/--stacking_threshold_objective/--stacking_minority_lambda/--stacking_oof_folds`，默认开启二层路径（`two_level` + `temp` + `macro_f1_minority_recall`）。
- 2026-04-07 two-level 改造: 新增多分类概率校准链路（temp 为默认，isotonic 可选）与校准指标统计（ECE/Brier），并将校准信息写入每个 method 的 `postprocess`。
- 2026-04-07 two-level 改造: 新增 Level-2 特征构造（多 method 概率 + entropy/margin + 聚合不确定性）、二层 blender OOF/全量训练、per-class threshold 优化，目标显式对齐弱类召回。
- 2026-04-07 two-level 改造: 新增 `requested_level/effective_level` 降级策略；当可用 Level-1 learner 少于 2 个时自动回退 single-layer，保证可用性。
- 2026-04-07 可观测性补强: `metrics.json` 新增 `stacking` 配置块（level/calibration/objective/lambda/folds），并在 `method_results` 增加 `two_level_blender` 条目及阈值/弱类召回前后对比。
- 2026-04-07 设计缺口补齐: Level-2 特征新增 `pairwise KL` 一致性统计，满足 spec 中的“method 间分歧信号”。
- 2026-04-07 设计缺口补齐: Level-2 训练权重新增 `hard_sample_factor`（低置信样本更高权重），并与 `inverse_frequency` 归一化组合。
- 2026-04-07 设计缺口补齐: two-level 新增独立 `oof_test_gap` 字段与大 gap 告警（阈值 `0.12`），与单层路径一致。
- 2026-04-07 严格复核补齐: `build_two_level_postprocess_payload` 调用参数已与签名一致（新增 `threshold_objective/objective_value`），避免运行时参数缺失。
- 2026-04-07 严格复核补齐: `single_layer_baseline` 不再仅是工具函数，已在主流程中落盘到 `method_results`，可直接用于单层/soft_voting/two_level 对照。
