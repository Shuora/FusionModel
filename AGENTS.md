# AGENTS.md

## 项目简介

本项目用于加密流量分类，主线是：
1. 原始 pcap 进行 session 级切分与清洗；
2. 统一长度为 784 bytes；
3. 构建双分支输入（RGB 图像分支 + CharBERT 时序分支）；
4. 使用原生 MobileViT + CharBERT 做注意力融合；
5. 使用 XGBoost 做 stacking 集成学习。

## 代码与目录约束（强制）

- `old/` 目录仅归档，不可作为新实现依赖。
- 新功能必须在 `src/` 实现，不允许 `import old` 或 `from old`。
- 默认数据源：
  - `SourceData/USTC-TFC2016`
  - `SourceData/CICAndMal2017`
- 默认 SplitCap：`tools/SplitCap_2-1/SplitCap.exe`

## 核心脚本说明

- `src/preprocess_splitcap_rgb.py`
  - 预处理主入口。
  - 功能：SplitCap 调用、session 后处理、784 统一、RGB 构建、落盘。
- `src/train_attention_fusion.py`
  - 训练基础注意力融合模型。
- `src/train_attention_stacking.py`
  - 在基础模型上构建 XGBoost stacking。
- `src/models_charbert_mobilevit.py`
  - 原生 MobileViT（28x28 适配）+ CharBERT + 注意力融合。

## 数据预处理流程

1. **Session 切分**
   - 默认使用 SplitCap（`--splitcap_mode external`）。
   - 可选自动回退 Python 解析（`--splitcap_mode auto`）。
2. **清洗与去重**
   - 丢弃空 session。
   - 对 payload 做 SHA1 去重。
3. **长度统一**
   - 每条 session 固定为 784 bytes（截断/补零）。
4. **图像构建（28x28x3）**
   - R 通道：论文式 784-byte 映射。
   - G/B 通道：语义与行为特征映射。
5. **输出结构**
   - `image_data/{Train,Test}/{label}/*.png`
   - `pcap_data/{Train,Test}/{label}/*.bin`
   - 可选：`temporal_data/{Train,Test}/{label}/*.(npy|pt)`

## 模型训练流程

1. **输入**
   - 图像分支：RGB `28x28`。
   - 时序分支：`pcap_data/*.bin` 读取后补 `CLS/SEP`，默认序列长度 `786`。
2. **模型结构**
   - 图像编码器：原生 MobileViT（小输入适配）。
   - 时序编码器：CharBERT。
   - 融合：cross-attention。
3. **集成学习**
   - 导出 image/text/fused 概率作为元特征。
   - XGBoost 训练元分类器。

## 推荐执行顺序（必须）

1. 先跑 `USTC`（10 类）验证流程与资源配置。
2. 再跑 `CIC 4 类`（group）。
3. 最后跑 `CIC 42 类`（family）。

## 评估与准确率优先策略

- 必须按源 pcap 切分 Train/Test，避免同源泄漏。
- 默认早停指标使用 `val_f1`（macro-F1）。
- CIC-42 推荐：
  - `--class_balance weighted_sampler_loss`
  - `--loss_type focal`
- 必做消融：`R-only` vs `RGB`。

## 日志与产物规范

### 预处理产物
- `outputs/preprocess/<run_id>/preprocess.log`
- `outputs/preprocess/<run_id>/preprocess_summary.json`
- `outputs/preprocess/<run_id>/label_distribution.csv`

### 训练产物
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

## 快速命令模板

### 依赖安装
```bash
pip install -r requirements.txt
```

### USTC 预处理
```bash
python src/preprocess_splitcap_rgb.py \
  --input_root SourceData/USTC-TFC2016 \
  --output_root dataset/USTC_10 \
  --label_mode file_stem \
  --splitcap_exe tools/SplitCap_2-1/SplitCap.exe \
  --splitcap_mode external \
  --train_ratio 0.8 \
  --temporal_formats bin
```

### USTC 训练（stacking）
```bash
python src/train_attention_stacking.py \
  --dataset_root dataset/USTC_10 \
  --batch_size 32 \
  --epochs 30 \
  --lr 1e-3 \
  --early_stop_metric val_f1 \
  --device auto \
  --output_dir outputs/train
```

## 备注

- 本文档用于约束 agent 与开发协作流程，优先级高于口头约定。
- 若新增流程或脚本，请同步更新本文件。
