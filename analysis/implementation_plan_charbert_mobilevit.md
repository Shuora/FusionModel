# MVTBA风格预处理重构 + CharBERT-MobileViT 注意力融合 + XGBoost Stacking（先 USTC 后 CIC）

## 摘要
- 保留论文 1.1~1.3（session 切分、清洗、长度统一），把 1.4 改成 RGB：
  - R：论文式 784-byte 灰度分支（28x28）
  - G/B：沿用已有特征通道思路
- 模型分支使用 2.2 CharBERT，去掉论文 2.3 加权融合，改为 attention fusion + XGBoost stacking。
- 执行顺序固定：USTC 先跑通，再跑 CIC，且 CIC 先 4 类后 42 类。
- 代码约束：`old/` 目录仅归档，实施时不复用、不导入、不修改；全部新实现放在 `src/`。
- 目标约束：以更高准确率和可泛化性为优先，不做仅追求表面高分的泄漏式方案。

## 准确率优先原则（强制）
- 数据切分必须按源 pcap 分组，不允许同源 session 同时落入 Train/Test。
- 默认采用 payload-only 输入，避免依赖 IP/MAC 等易泄漏字段。
- CIC-42 默认启用类别不平衡策略：`weighted sampler + focal loss`。
- 早停与模型选择默认使用 `val macro-F1`，而非仅看 `val acc`。
- 必做消融：`R-only` 与 `RGB(R+G+B)`，验证 G/B 通道是否带来真实提升。

## 代码位置约束（强制）
- 禁止在新代码中出现 `from old...` 或 `import old...`。
- 禁止修改 `old/` 下任何文件用于本次实现。
- 所有新增功能必须在 `src/` 新建文件完成。

## src 新文件清单（本次实施）
1. `src/preprocess_splitcap_rgb.py`
- 主预处理入口：SplitCap 调用 + session 后处理 + 784 统一 + RGB 生成 + Train/Test 切分 + 可选时序文件导出。

2. `src/splitcap_runner.py`
- SplitCap 适配层：命令构建、执行、失败重试、日志与临时目录管理。

3. `src/session_postprocess.py`
- 会话级清洗逻辑：空会话过滤、SHA1 去重、长度统一、样本统计导出。

4. `src/rgb_builder.py`
- 图像分支构建：
  - R 通道使用论文 784-byte 灰度映射
  - G/B 通道使用你的语义/行为特征映射

5. `src/fusion_dataset.py`
- 训练数据加载：`image_data` + `pcap_data` 同名配对、类别索引、缓存索引。

6. `src/models_charbert_mobilevit.py`
- 模型定义：CharBERT 编码器、MobileViT 编码器、attention 融合头。
- MobileViT 分支采用论文思路在 `src` 原生实现（小输入 28x28 适配版），不调用外部现成 MobileViT 分类模型。

7. `src/train_attention_fusion.py`
- 单模型训练入口：attention 融合端到端训练与评估。

8. `src/train_attention_stacking.py`
- 集成学习入口：attention 基模型 + XGBoost stacking（元特征与元学习器）。

9. `src/train_common.py`
- 通用训练组件：参数解析、日志、EarlyStopping、损失函数、评估与可视化。

## 公开接口/CLI 变更
1. `src/preprocess_splitcap_rgb.py`
- `--input_root`：原始数据根目录（支持 USTC 文件级、CIC 目录级）
- `--output_root`：输出根目录
- `--splitcap_exe`：SplitCap 可执行文件路径（默认 `tools/SplitCap_2-1/SplitCap.exe`，可手动覆盖）
- `--splitcap_mode {external,auto,python}`：默认 `external`
- `--train_ratio`：默认 `0.8`
- `--seed`：默认 `42`
- `--label_mode {auto,group,family,file_stem}`
  - USTC 用 `file_stem`（Cridex/Geodo...）
  - CIC-4cls 用 `group`
  - CIC-42cls 用 `family`
- `--max_len`：默认 `784`
- `--temporal_formats`：默认 `bin`；可选 `bin,npy,pt`
- `--dedup`：默认开启（SHA1 去重）
- `--save_temp_sessions`：默认关闭（调试时可开）
- `--sanitize_headers`：默认关闭；仅在未来切换到整包字节模式时启用头字段脱敏

2. `src/train_attention_stacking.py`
- `--dataset_root`
- `--dataset_name`
- `--batch_size`
- `--epochs`
- `--lr`
- `--device`
- `--attention_dim`
- `--char_seq_len`（默认 `786`，即 784 bytes + CLS/SEP）
- `--class_balance`
- `--loss_type`
- `--early_stop_metric`（默认 `val_f1`）
- `--output_dir`

## 目标目录规范（与新 DataLoader 兼容）
每个数据集输出都用扁平布局：
- `<dataset_out>/image_data/{Train,Test}/<label>/<stem>.png`
- `<dataset_out>/pcap_data/{Train,Test}/<label>/<stem>.bin`（训练主读取）
- `<dataset_out>/temporal_data/{Train,Test}/<label>/<stem>.npy|.pt`（可选导出，不是训练必需）
- `stem` 同名配对，保证 `src/fusion_dataset.py` 直接匹配

## 预处理实现细节（决策完成）
1. Session 提取（1.1）
- 对每个原始 pcap 调用 SplitCap，按五元组切 session 到临时目录。
- session 顺序按包时间戳拼接 payload。

2. 清洗（1.2）
- 丢弃空 payload session。
- 以 payload SHA1 去重。
- 默认 payload-only 输入，规避 IP/MAC 泄漏。
- 保留可选 `sanitize_headers` 开关，供未来整包输入实验对照。

3. 统一长度（1.3）
- 大于 784 截断，小于 784 尾零填充到 784。

4. RGB 转换（1.4）
- R：直接取统一后的 784 bytes reshape 为 28x28。
- G：用现有语义通道逻辑（握手/明文语义特征 + 填充到 784）。
- B：用现有行为统计通道逻辑（统计特征 + 分散填充到 784）。
- 合成为 `28x28x3` PNG。

5. 后处理
- 按“源 pcap 级”分 Train/Test，避免同源 session 泄漏到两侧。
- 输出统计文件：样本数、去重数、空会话数、各类分布。

6. 784 时序导出与 CharBERT 输入对齐
- 保留论文式 784 bytes 时序导出（默认 `.bin`，可选 `.npy/.pt`）。
- CharBERT 训练时使用这 784 bytes：
  - 先映射为 byte token（0~255）
  - 数据加载阶段补 `CLS` 和 `SEP`，形成 `786` 长度输入
  - 兼容论文时序信息与 CharBERT 编码习惯

## 模型与训练流程（先 USTC 后 CIC）
1. USTC（10 类，多分类）
- 预处理：`label_mode=file_stem`
- 训练：`src/train_attention_stacking.py`（attention + xgboost）

2. CIC 阶段 A（4 类）
- 预处理：`label_mode=group`
- 训练同上，先验证稳定性/超参。

3. CIC 阶段 B（42 类）
- 预处理：`label_mode=family`
- 复用阶段 A 最优超参，小范围再调 `class_balance/loss_type`。

## 资源约束下默认超参（i7-13700 / 16GB / RTX 4060 Laptop 8GB）
- `device=cuda:0`，AMP 开启
- `batch_size=32`（默认起步，OOM 再降到 16）
- `num_workers=4`，`prefetch_factor=2`
- `epochs=30`（USTC 可先 20 做冒烟）
- `lr=1e-3`（CIC 可从 `3e-4` 起）
- `early_stop_metric=val_f1`
- CIC-42 推荐：`class_balance=weighted_sampler_loss`，`loss_type=focal`
- 集成学习默认：XGBoost stacking

## 日志与结果产物规范（强制）
- 预处理输出：
  - `outputs/preprocess/<run_id>/preprocess.log`
  - `outputs/preprocess/<run_id>/preprocess_summary.json`
  - `outputs/preprocess/<run_id>/label_distribution.csv`
- 训练输出：
  - `outputs/train/<run_id>/logs/train.log`
  - `outputs/train/<run_id>/checkpoints/best.pt`
  - `outputs/train/<run_id>/checkpoints/last.pt`
  - `outputs/train/<run_id>/metrics/epoch_metrics.csv`
  - `outputs/train/<run_id>/metrics/final_metrics.json`
  - `outputs/train/<run_id>/reports/classification_report.txt`
  - `outputs/train/<run_id>/reports/confusion_matrix.png`
  - `outputs/train/<run_id>/figures/loss_curve.png`
  - `outputs/train/<run_id>/figures/acc_curve.png`
  - `outputs/train/<run_id>/figures/f1_curve.png`
- stacking 额外输出：
  - `outputs/train/<run_id>/stacking/meta_features_train.npy`
  - `outputs/train/<run_id>/stacking/meta_features_test.npy`
  - `outputs/train/<run_id>/stacking/meta_model_xgboost.json`

## 测试与验收
1. 预处理验收
- 输出目录结构完整，`image_data` 与 `pcap_data` 配对数一致。
- 随机抽检 20 个样本：`bin` 长度恒为 784，图片尺寸 28x28x3。
- 若导出 `.npy/.pt`，抽样比对与 `.bin` 数值一致。
- 去重与空会话统计非零且合理。

2. 训练验收
- USTC 能完整跑完并产出完整日志、checkpoint、指标与图表。
- CIC-4cls 跑通后再执行 CIC-42cls，日志中类别数分别为 4 和 42。
- 消融实验完成并记录：`R-only` vs `RGB`。

3. 架构约束验收
- 新增代码全部位于 `src/`。
- 代码搜索结果中无 `import old`、`from old`。

## 假设与默认
- 仓库已内置 SplitCap（`tools/SplitCap_2-1/SplitCap.exe`），默认直接使用该路径；迁移到其他环境时可通过 `--splitcap_exe` 覆盖。
- CharBERT 采用仓库内 `src/models_charbert_mobilevit.py` 的内置实现（实现思路借鉴 `C:\Repositories\Traffic\CharBERT`）。
- MobileViT 分支采用论文结构的 `src` 原生实现，不调用外部现成分类模型。
- 集成学习固定先用 XGBoost，暂不启用 LightGBM/CatBoost/MLP。
- 当前需求聚焦多分类路线，不做二分类基线。
