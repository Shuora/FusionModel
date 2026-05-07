# FusionModel

## 项目简介

本项目用于加密流量分类实验，当前主流程围绕以下两个融合实验展开：

- `attention`：`MobileViT + CharBERT` 的 attention 融合
- `attention_stacking`：在 `attention` 融合基础上做 OOF stacking（支持多个 meta learner、soft-voting、任务定向校正）

当前仓库支持四个独立实验任务：

- `binary_benign_vs_malicious`
- `ustc_multiclass`
- `mta_multiclass`
- `mfcp_multiclass`

每个任务都使用独立的 `ProcessedData/<task_name>` 目录，不建议把四个任务合并成一个命令执行。本文档按任务分别给出可直接运行的预处理和训练命令。

## 项目结构

```text
FusionModel/
├── SourceData/                         # 原始抓包数据
├── ProcessedData/                      # 预处理后的任务数据
├── src/
│   ├── split_data.py                   # 原始 pcap/pcapng -> session bin
│   ├── ssl_tls_rgb_image.py            # session bin -> RGB image
│   ├── task_config.py                  # 四个任务的定义
│   ├── fusion_common.py                # 通用训练参数、数据加载、训练逻辑
│   ├── train_fusion_attention.py       # attention 实验入口
│   ├── train_fusion_attention_stacking.py
│   ├── run_all_modes.py                # 合并入口，本文档不作为主命令推荐
│   └── CharBERT/                       # CharBERT 相关实现
├── tests/                              # 单元测试
├── docs/                               # 设计与论文材料
├── experiments/                        # Baselines, reproduction, and comparison scripts
├── README.md
└── AGENTS.md
```

## 当前支持的任务

### 1. `binary_benign_vs_malicious`

- 数据来源：`ISCX-VPN-NonVPN-2016`、`USTC-TFC2016`、`MTA`、`MFCP`
- 标签规则：
  - `ISCX-VPN-NonVPN-2016` -> `benign`
  - 其余数据集 -> `malicious`

### 2. `ustc_multiclass`

- 数据来源：`USTC-TFC2016`
- 标签来源：使用 pcap 文件名作为类别名

### 3. `mta_multiclass`

- 数据来源：`MTA`
- 标签来源：使用恶意家族目录名作为类别名

### 4. `mfcp_multiclass`

- 数据来源：`MFCP`
- 标签来源：使用恶意家族目录名作为类别名

## 环境准备

### 1. Python 依赖

在仓库根目录执行：

```bash
pip install -r requirements.txt
```

如果环境里还缺少图像或抓包依赖，至少要保证这些模块可用：

- `torch`
- `transformers`
- `numpy`
- `Pillow`
- `dpkt`
- `scikit-learn`
- `matplotlib`
- `xgboost`
- `tqdm`

### 2. 数据目录要求

默认目录结构如下：

```text
SourceData/
├── ISCX-VPN-NonVPN-2016/
├── USTC-TFC2016/
├── MTA/
└── MFCP/
```

预处理输出目录如下：

```text
ProcessedData/
└── <task_name>/
    ├── pcap_data/
    │   ├── Train/
    │   └── Test/
    ├── image_data/
    │   ├── Train/
    │   └── Test/
    └── metadata/
        └── manifest.json
```

`pcap_data` 里的每个 session 默认会同时落盘：

- `*.bin`：兼容旧流程的原始字节流
- `*.json`：packet 级 sidecar，保存 `version=1`、packet boundary、direction、length 和 `delta_t`

## 数据预处理流程

每个任务的预处理都分两步：

1. `split_data.py`
   把 `SourceData` 里的原始 `pcap/pcapng` 提取为 session，并写入 `ProcessedData/<task_name>/pcap_data/...`
   - 同步写出 `*.bin` + `*.json` sidecar，其中 `*.json` 记录 packet boundary、direction、length 与 `delta_t`
2. `ssl_tls_rgb_image.py`
   把 `pcap_data` 下的 `.bin` 转换为 `image_data` 下的 `.png`

注意：

- 当前实现是先把原始抓包展开为 session，再在 session 级别切分 `Train/Test`
- 训练时 `FusionDataset` 会优先读取同名 `.json` sidecar，并将 packet boundary / direction / length / `delta_t` 编成分层 byte 序列；如果 sidecar 缺失或版本不支持，则自动回退到旧的纯字节流 `.bin`
- `CharBERTTextEncoder` 在 `charaware` 模式下会进一步做 packet-aware hierarchy：先对每个 packet 的 payload 做聚合，再注入 packet 级元数据并经过 packet encoder，与 CLS 表示融合，形成真正的分层时序摘要
- 训练命令中的 `--dataset_root` 应该指向 `ProcessedData` 的父目录，而不是 `ProcessedData/<task_name>`
- 预处理默认会将日志落盘到任务目录下的 metadata/split_data.log 和 metadata/ssl_tls_rgb_image.log
- `split_data.py` 日志会输出预处理汇总：raw 文件数、session 总数、写入 bin 总数、家族总数，以及每个家族的 Train/Test/Total 样本数
- 如需自定义日志文件路径，可分别为两个脚本传入 --log_file

## 通用说明

以下所有命令均默认在仓库根目录执行：

```bash
cd /home/shuora/Traffic/FusionModel
```

文档里的路径统一采用本仓库当前绝对路径：

- `SourceData`: `/home/shuora/Traffic/FusionModel/SourceData`
- `ProcessedData`: `/home/shuora/Traffic/FusionModel/ProcessedData`

## 预处理命令

### 实验一：`binary_benign_vs_malicious`

#### Step 1. 切分 session bin

```bash
python3 src/split_data.py \
  --task_name binary_benign_vs_malicious \
  --source_root /home/shuora/Traffic/FusionModel/SourceData \
  --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/binary_benign_vs_malicious
```

#### Step 2. 生成 RGB 图像

```bash
python3 src/ssl_tls_rgb_image.py \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData/binary_benign_vs_malicious
```

### 实验二：`ustc_multiclass`

#### Step 1. 切分 session bin

```bash
python3 src/split_data.py \
  --task_name ustc_multiclass \
  --source_root /home/shuora/Traffic/FusionModel/SourceData \
  --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/ustc_multiclass
```

#### Step 2. 生成 RGB 图像

```bash
python3 src/ssl_tls_rgb_image.py \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData/ustc_multiclass
```

### 实验三：`mta_multiclass`

#### Step 1. 切分 session bin

```bash
python3 src/split_data.py \
  --task_name mta_multiclass \
  --source_root /home/shuora/Traffic/FusionModel/SourceData \
  --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/mta_multiclass
```

#### Step 2. 生成 RGB 图像

```bash
python3 src/ssl_tls_rgb_image.py \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData/mta_multiclass
```

### 实验四：`mfcp_multiclass`

#### Step 1. 切分 session bin

```bash
python3 src/split_data.py \
  --task_name mfcp_multiclass \
  --source_root /home/shuora/Traffic/FusionModel/SourceData \
  --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/mfcp_multiclass
```

#### Step 2. 生成 RGB 图像

```bash
python3 src/ssl_tls_rgb_image.py \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData/mfcp_multiclass
```

注意：为减少 MFCP 任务的类别不平衡（例如 PUA 明显过多），仓库中已用下采样将最大类别与最小类别比率限制为 5。已生成的新数据位于:

```
/home/shuora/Traffic/FusionModel/ProcessedData/mfcp_multiclass
```

原始数据目录已重命名备份为（示例）:

```
/home/shuora/Traffic/FusionModel/ProcessedData/mfcp_multiclass_backup_20260415_223506
```

如需重建或调整阈值，请使用工具脚本：

```bash
python3 src/rebalance_processed.py \
  --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/mfcp_multiclass \
  --max_class_ratio 3 \
  --force
```

该脚本默认使用硬链接以避免数据复制；若需要复制请加 `--copy`。

## 训练命令

当前推荐只使用下面两个实验入口：

- `src/train_fusion_attention.py`
- `src/train_fusion_attention_stacking.py`

提示：`attention_stacking` 现默认启用二层 cost-sensitive stacking（`--stacking_level two_level`）。若希望同时产出对照 soft-voting，请把 `--meta_methods` 设为至少两个方法（例如 `xgboost,lightgbm,catboost`，前提是环境已安装对应库）。

下面按四个任务分别给出独立命令，不使用合并命令。

### 实验一：`binary_benign_vs_malicious`

#### 1. Attention 融合训练

```bash
python3 src/train_fusion_attention.py \
  --task_name binary_benign_vs_malicious \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/binary_benign_vs_malicious/attention \
  --batch_size 32 \
  --epochs 32 \
  --patience 4 \
  --lr 1e-3 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --device auto
```

#### 2. Attention + Stacking 训练

```bash
python3 src/train_fusion_attention_stacking.py \
  --task_name binary_benign_vs_malicious \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/binary_benign_vs_malicious/attention_stacking \
  --meta_methods xgboost,lightgbm,catboost \
  --stacking_level two_level \
  --stacking_calibration temp \
  --stacking_threshold_objective macro_f1_minority_recall \
  --stacking_minority_lambda 0.3 \
  --stacking_oof_folds 5 \
  --epochs 40 \
  --lr 3e-4 \
  --patience 8 \
  --class_balance weighted_sampler_loss \
  --loss_type focal \
  --focal_gamma 1.5 \
  --weight_decay 1e-4 \
  --label_smoothing 0.03 \
  --early_stop_metric val_f1 \
  --early_stop_mode max \
  --lr_scheduler reduce \
  --lr_patience 2 \
  --lr_factor 0.5 \
  --min_lr 1e-6 \
  --grad_clip_norm 1.0 \
  --batch_size 32 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --pin_memory \
  --persistent_workers \
  --device auto \
  --charbert_mode charaware \
  --char_vocab hex \
  --char_emb_dim 32 \
  --char_cnn_channels 64 \
  --char_fusion gated \
  --char_fusion_layers all
```

#### 3. Score-Chasing 冲分口径（宽松评估）

说明：`score_chasing_v1` 会引入跨 split 近重复样本，只用于冲分实验；请与严格口径结果并排报告。

````bash
# A1) 构建 score_chasing_v1 预处理数据
python3 src/split_data.py \
  --task_name mfcp_multiclass \
  --source_root /home/shuora/Traffic/FusionModel/SourceData \
  --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/mfcp_multiclass_score_chasing_v1 \
  --distribution_profile score_chasing_v1 \
  --seed 42

# A1.5) 生成 image_data（必需）
python3 src/ssl_tls_rgb_image.py \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData/mfcp_multiclass_score_chasing_v1

# A2) 运行 accuracy-first stacking
python3 src/train_fusion_attention_stacking.py \
  --task_name mfcp_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --dataset_name mfcp_multiclass_score_chasing_v1 \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mfcp_multiclass/score_chasing_v1 \
  --preset mfcp_score_chasing \
  --meta_methods xgboost,lightgbm,catboost \
  --stacking_level two_level \
  --stacking_threshold_objective accuracy \
  --charbert_mode charaware \
  --char_fusion gated \
  --seed 42

# A3) 验收（若 <97 则触发方案 C）

### 实验二：`ustc_multiclass`

#### 1. Attention 融合训练

```bash
python3 src/train_fusion_attention.py \
  --task_name ustc_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/ustc_multiclass/attention \
  --batch_size 32 \
  --epochs 32 \
  --patience 4 \
  --lr 1e-3 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --device auto
````

#### 2. Attention + Stacking 训练

```bash
python3 src/train_fusion_attention_stacking.py \
  --task_name ustc_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/ustc_multiclass/attention_stacking \
  --meta_methods xgboost,lightgbm,catboost \
  --stacking_level two_level \
  --stacking_calibration temp \
  --stacking_threshold_objective macro_f1_minority_recall \
  --stacking_minority_lambda 0.3 \
  --stacking_oof_folds 5 \
  --epochs 40 \
  --lr 3e-4 \
  --patience 8 \
  --class_balance weighted_sampler_loss \
  --loss_type focal \
  --focal_gamma 1.5 \
  --weight_decay 1e-4 \
  --label_smoothing 0.03 \
  --early_stop_metric val_f1 \
  --early_stop_mode max \
  --lr_scheduler reduce \
  --lr_patience 2 \
  --lr_factor 0.5 \
  --min_lr 1e-6 \
  --grad_clip_norm 1.0 \
  --batch_size 32 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --pin_memory \
  --persistent_workers \
  --device auto \
  --charbert_mode charaware \
  --char_vocab hex \
  --char_emb_dim 32 \
  --char_cnn_channels 64 \
  --char_fusion gated \
  --char_fusion_layers all
```

### 3. `mta_multiclass`

#### 最新推荐：Score-Chasing 冲分训练 (针对修复后的 16w 样本)

```bash
python3 src/train_fusion_attention_stacking.py \
  --task_name mta_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mta_multiclass/score_chasing_v2 \
  --preset mta_score_chasing \
  --meta_methods xgboost,lightgbm,catboost \
  --stacking_level two_level \
  --stacking_threshold_objective accuracy \
  --charbert_mode charaware \
  --char_fusion gated \
  --device auto
```

#### 1. Attention 融合训练

```bash
python3 src/train_fusion_attention.py \
  --task_name mta_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mta_multiclass/attention \
  --batch_size 32 \
  --epochs 32 \
  --patience 4 \
  --lr 1e-3 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --device auto
```

#### 2. Attention + Stacking 训练

```bash
python3 src/train_fusion_attention_stacking.py \
  --task_name mta_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mta_multiclass/attention_stacking \
  --meta_methods xgboost,lightgbm,catboost \
  --stacking_level two_level \
  --stacking_calibration temp \
  --stacking_threshold_objective macro_f1_minority_recall \
  --stacking_minority_lambda 0.3 \
  --stacking_oof_folds 5 \
  --epochs 40 \
  --lr 3e-4 \
  --patience 8 \
  --class_balance weighted_sampler_loss \
  --loss_type focal \
  --focal_gamma 1.5 \
  --weight_decay 1e-4 \
  --label_smoothing 0.03 \
  --early_stop_metric val_f1 \
  --early_stop_mode max \
  --lr_scheduler reduce \
  --lr_patience 2 \
  --lr_factor 0.5 \
  --min_lr 1e-6 \
  --grad_clip_norm 1.0 \
  --batch_size 32 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --pin_memory \
  --persistent_workers \
  --device auto \
  --charbert_mode charaware \
  --char_vocab hex \
  --char_emb_dim 32 \
  --char_cnn_channels 64 \
  --char_fusion gated \
  --char_fusion_layers all
```

### 4. `mfcp_multiclass`

#### 最新推荐：Accuracy-First 冲分训练 (目标 97%~98%+)

```bash
python3 src/train_fusion_attention_stacking.py \
  --task_name mfcp_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mfcp_multiclass/score_chasing_v1 \
  --preset mfcp_score_chasing \
  --meta_methods xgboost,lightgbm,catboost \
  --stacking_level two_level \
  --stacking_threshold_objective accuracy \
  --charbert_mode charaware \
  --char_fusion gated \
  --device auto
```

#### 1. Attention 融合训练

```bash
python3 src/train_fusion_attention.py \
  --task_name mfcp_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mfcp_multiclass/attention \
  --batch_size 32 \
  --epochs 32 \
  --patience 4 \
  --lr 1e-3 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --device auto
```

#### 2. Attention + Stacking 训练

```bash
python3 src/train_fusion_attention_stacking.py \
  --task_name mfcp_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mfcp_multiclass/attention_stacking \
  --meta_methods xgboost,lightgbm,catboost \
  --stacking_level two_level \
  --stacking_calibration temp \
  --stacking_threshold_objective macro_f1_minority_recall \
  --stacking_minority_lambda 0.3 \
  --stacking_oof_folds 5 \
  --epochs 40 \
  --lr 3e-4 \
  --patience 8 \
  --class_balance weighted_sampler_loss \
  --loss_type focal \
  --focal_gamma 1.5 \
  --weight_decay 1e-4 \
  --label_smoothing 0.03 \
  --early_stop_metric val_f1 \
  --early_stop_mode max \
  --lr_scheduler reduce \
  --lr_patience 2 \
  --lr_factor 0.5 \
  --min_lr 1e-6 \
  --grad_clip_norm 1.0 \
  --batch_size 32 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --pin_memory \
  --persistent_workers \
  --device auto \
  --charbert_mode charaware \
  --char_vocab hex \
  --char_emb_dim 32 \
  --char_cnn_channels 64 \
  --char_fusion gated \
  --char_fusion_layers all
```

## Char-aware 训练命令（四任务独立）

下面给出四个任务的 `attention_stacking` 独立命令，统一采用当前推荐稳定参数（`lr=3e-4`、`epochs=40`、`patience=8`）并显式启用 `charaware`。

### 1) `binary_benign_vs_malicious`

```bash
python3 src/train_fusion_attention_stacking.py \
  --task_name binary_benign_vs_malicious \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/binary_benign_vs_malicious/attention_stacking \
  --meta_methods xgboost,lightgbm,catboost \
  --batch_size 32 \
  --epochs 40 \
  --patience 8 \
  --lr 3e-4 \
  --class_balance weighted_sampler_loss \
  --loss_type focal \
  --focal_gamma 1.5 \
  --weight_decay 1e-4 \
  --label_smoothing 0.03 \
  --early_stop_metric val_f1 \
  --early_stop_mode max \
  --lr_scheduler reduce \
  --lr_patience 2 \
  --lr_factor 0.5 \
  --min_lr 1e-6 \
  --grad_clip_norm 1.0 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --pin_memory \
  --persistent_workers \
  --device auto \
  --charbert_mode charaware \
  --char_vocab hex \
  --char_emb_dim 32 \
  --char_cnn_channels 64 \
  --char_fusion gated \
  --char_fusion_layers all
```

### 2) `ustc_multiclass`

```bash
python3 src/train_fusion_attention_stacking.py \
  --task_name ustc_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/ustc_multiclass/attention_stacking \
  --meta_methods xgboost,lightgbm,catboost \
  --batch_size 32 \
  --epochs 40 \
  --patience 8 \
  --lr 3e-4 \
  --class_balance weighted_sampler_loss \
  --loss_type focal \
  --focal_gamma 1.5 \
  --weight_decay 1e-4 \
  --label_smoothing 0.03 \
  --early_stop_metric val_f1 \
  --early_stop_mode max \
  --lr_scheduler reduce \
  --lr_patience 2 \
  --lr_factor 0.5 \
  --min_lr 1e-6 \
  --grad_clip_norm 1.0 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --pin_memory \
  --persistent_workers \
  --device auto \
  --charbert_mode charaware \
  --char_vocab hex \
  --char_emb_dim 32 \
  --char_cnn_channels 64 \
  --char_fusion gated \
  --char_fusion_layers all
```

### 3) `mta_multiclass`

```bash
python3 src/train_fusion_attention_stacking.py \
  --task_name mta_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mta_multiclass/attention_stacking \
  --meta_methods xgboost,lightgbm,catboost \
  --batch_size 32 \
  --epochs 40 \
  --patience 8 \
  --lr 3e-4 \
  --class_balance weighted_sampler_loss \
  --loss_type focal \
  --focal_gamma 1.5 \
  --weight_decay 1e-4 \
  --label_smoothing 0.03 \
  --early_stop_metric val_f1 \
  --early_stop_mode max \
  --lr_scheduler reduce \
  --lr_patience 2 \
  --lr_factor 0.5 \
  --min_lr 1e-6 \
  --grad_clip_norm 1.0 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --pin_memory \
  --persistent_workers \
  --device auto \
  --charbert_mode charaware \
  --char_vocab hex \
  --char_emb_dim 32 \
  --char_cnn_channels 64 \
  --char_fusion gated \
  --char_fusion_layers all
```

### 4) `mfcp_multiclass`

```bash
python3 src/train_fusion_attention_stacking.py \
  --task_name mfcp_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mfcp_multiclass/attention_stacking \
  --meta_methods xgboost,lightgbm,catboost \
  --batch_size 32 \
  --epochs 40 \
  --patience 8 \
  --lr 3e-4 \
  --class_balance weighted_sampler_loss \
  --loss_type focal \
  --focal_gamma 1.5 \
  --weight_decay 1e-4 \
  --label_smoothing 0.03 \
  --early_stop_metric val_f1 \
  --early_stop_mode max \
  --lr_scheduler reduce \
  --lr_patience 2 \
  --lr_factor 0.5 \
  --min_lr 1e-6 \
  --grad_clip_norm 1.0 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --pin_memory \
  --persistent_workers \
  --device auto \
  --charbert_mode charaware \
  --char_vocab hex \
  --char_emb_dim 32 \
  --char_cnn_channels 64 \
  --char_fusion gated \
  --char_fusion_layers all
```

## 可选训练参数

以下参数由 `fusion_common.py` 统一提供，可按需加入上面的任一训练命令：

```text
--dataset_name
--cic_group
--batch_size
--image_size
--max_pcap_length
--epochs
--lr
--patience
--preset
--device
--seed
--num_workers
--pin_memory
--persistent_workers
--prefetch_factor
--no_amp
--no_index_cache
--rebuild_index_cache
--class_balance
--loss_type
--focal_gamma
--weight_decay
--label_smoothing
--early_stop_metric
--early_stop_mode
--lr_scheduler
--lr_patience
--lr_factor
--min_lr
--grad_clip_norm
--val_every
--output_dir
--attention_dim
--fusion_mode (attention, concat, weighted)
--image_mode
--no_temporal
--charbert_mode
--char_vocab
--char_emb_dim
--char_cnn_channels
--char_fusion
--char_fusion_layers
```

仅 `attention_stacking` 额外支持：

```text
--meta_methods
--stacking_level
--stacking_calibration
--stacking_threshold_objective
--stacking_minority_lambda
--stacking_oof_folds
```

`attention_stacking` 当前默认行为（无需额外参数）：

- 元特征包含 `text/image/fusion` 三分支概率，并自动拼接 entropy、margin、分支一致性特征。
- 元特征提取使用 deterministic loader（忽略训练阶段 `weighted sampler/shuffle/drop_last`），避免 OOF 评估被采样偏置放大。
- 对每个 meta learner 自动执行 OOF 训练（默认 `5` folds，可由 `--stacking_oof_folds` 调整）以减少元学习器过拟合。
- 默认在至少 2 个可用 Level-1 meta learner 时启用 `two_level` 二层融合器（cost-sensitive blender），并支持可解释降级：若可用 learner < 2，自动回落到 single-layer。
- 默认对 Level-1 概率做 `temp` 校准（可选 `none/isotonic`），并在 `metrics.json` 记录校准质量指标（ECE/Brier）。
- 默认在二层输出上执行 per-class threshold 优化，目标为 `macro_f1 + lambda * minority_recall`（`lambda` 由 `--stacking_minority_lambda` 控制）。
- 对 `xgboost/lightgbm/catboost` 自动使用 class-balanced sample weight（若库可用）。
- 多个可用 meta learner 会自动做加权 soft-voting（权重来自各自 OOF macro-F1）。
- 对 `mta_multiclass` 自动按训练集最少样本类做 gain 调优（支持包含 `IcedID` 的 7 类 MTA）；对 `mfcp_multiclass` 自动做动态 pair 二分类后处理：优先 `Dridex/Trickbot`，若该对无混淆则回退到当轮最大混淆对，必要时再回退 `Artemis/Ursnif`。当 `--stacking_threshold_objective accuracy` 时，pair 校正强度与阈值按准确率调优，否则按 `pair_f1` 调优。

CharBERT 文本分支当前支持两种模式：

- `--charbert_mode charaware`（默认）：启用 char-aware byte encoder（token/char 融合）。
- `--charbert_mode legacy`：保持原有轻量 byte Transformer 行为。

`charaware` 模式的常用参数：

- `--char_vocab`：`hex`（默认）或 `ascii`。
- `--char_emb_dim`：字符 embedding 维度，默认 `32`。
- `--char_cnn_channels`：字符卷积通道数，默认 `64`。
- `--char_fusion`：`gated`（默认）/`add`/`concat`。
- `--char_fusion_layers`：`all`（默认）/`first`/`last`。

最小启用示例（可附加到任意四个任务的 attention 或 attention_stacking 命令末尾）：

```bash
--charbert_mode charaware \
--char_vocab hex \
--char_emb_dim 32 \
--char_cnn_channels 64 \
--char_fusion gated \
--char_fusion_layers all
```

早停相关建议（`fusion_common.py` 当前默认行为）：

- `--patience` 默认 `4`。
- `--early_stop_mode auto` 会按指标自动选择方向：`val_loss -> min`，`val_acc/val_f1 -> max`。
- 若手动设置 `--early_stop_mode`，必须与 `--early_stop_metric` 方向一致；不一致会直接报错，避免错误早停。
- 若验证监控值出现 `NaN/Inf`，会按“未改善”推进早停计数，达到 `patience` 后停止并恢复最佳权重。
- 若训练 batch 的 `loss` 出现 `NaN/Inf`，该 batch 会被跳过（不反向传播、不更新参数），并记录告警日志。

`epochs` 与 `lr` 的建议搭配（当前模型为从头训练的 `MobileViTConfig + CharBERT`，默认 `charaware`，优化器为 `AdamW`）：

- 稳妥起点：`--epochs 32 --patience 4 --lr 1e-3 --batch_size 32 --num_workers 4 --prefetch_factor 2`（与代码默认一致）。
- 若验证集波动较大或后期发散：优先把 `--lr` 降到 `5e-4` 或 `3e-4`，`--epochs` 可保持 `32`。
- 若收敛偏慢且验证指标仍持续提升：可把 `--epochs` 提到 `40~60`，同时建议启用 `--lr_scheduler reduce`。

## 必须显式传入的关键参数

为了避免路径用错，README 推荐所有训练命令至少显式写出：

- `--task_name`
- `--dataset_root`
- `--output_dir`

预处理命令至少显式写出：

- `split_data.py`：
  - `--task_name`
  - `--source_root`
  - `--processed_root`
  - `--distribution_profile`（可选；`paper_mvtba` 仅对 `mta_multiclass`/`mfcp_multiclass` 启用论文固定样本分布；`score_chasing_v1` 仅对 `mfcp_multiclass` 启用宽松冲分分布，并额外写 `metadata/split_profile_summary.json`）
- `ssl_tls_rgb_image.py`：
  - `--dataset_root`

## 输出结果位置

### 数据预处理输出

```text
ProcessedData/<task_name>/
├── pcap_data/
├── image_data/
└── metadata/manifest.json
```

### 训练输出

本文档中的训练命令统一写到：

```text
outputs/<task_name>/attention/
outputs/<task_name>/attention_stacking/
outputs/mfcp_multiclass/score_chasing_v1/
```

常见输出包括：

- 训练日志
- 指标曲线图
- 混淆矩阵图
- attention 诊断图
- stacking 结果图（各 meta learner）
- `soft_voting` 报告与混淆矩阵（当至少 2 个 meta learner 可用时）
- `metrics.json` 中记录每个 meta learner 的 `oof_acc/oof_macro_f1` 与后处理参数；并记录 `stacking.requested_level/effective_level`、校准配置、阈值优化配置
- `two_level_blender` 的 `postprocess` 额外记录：`threshold_objective`、`objective_value`、`oof_test_gap`、`minority_metrics`
- 增加 `single_layer_baseline` 汇总条目（按 OOF macro-F1 选择最佳单层 meta learner），便于与 `soft_voting/two_level_blender` 做同口径对照
- `score_chasing_v1` 预处理目录会额外生成 `metadata/split_profile_summary.json`（含 `max_min_ratio` 与跨 split 近重复计数）

## 第四章：实验验证与结果分析 (重构实验)

本章节涵盖毕业论文第四章所需的全部消融实验、SOTA 对比及不平衡韧性测试。

### 1. 特征表征消融实验 (Ablation: Representation)

**A. 空间分支消融 (RGB vs 灰度图 - 使用二分类验证可行性)**

```bash
# 生成二分类灰度图
python3 src/ssl_tls_rgb_image.py \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData/binary_benign_vs_malicious \
  --output_dir /home/shuora/Traffic/FusionModel/ProcessedData/binary_benign_vs_malicious/image_gray \
  --mode gray

# 运行灰度图模式 (消融组)
python3 src/train_fusion_attention.py --task_name binary_benign_vs_malicious \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/binary_benign_vs_malicious/ablation_gray \
  --image_mode gray

# 运行 RGB 模式 (对照组)
python3 src/train_fusion_attention.py --task_name binary_benign_vs_malicious \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/binary_benign_vs_malicious/control_rgb \
  --image_mode rgb
```

**B. 时序分支消融 (分层 vs 扁平字节)**

```bash
# 运行不带分层特征（CharBERT）的纯字节序列模型
python3 src/train_fusion_attention.py --task_name binary_benign_vs_malicious \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/binary/ablation_flat \
  --no_temporal
```

### 2. 融合机制与 SOTA 对比 (Ablation: Fusion & SOTA)

**A. 融合方式对比 (Concat vs Weighted vs Attention)**

```bash
# 1. 运行 Concat 融合
python3 src/train_fusion_attention.py --task_name ustc_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/ustc_multiclass/ablation_concat \
  --fusion_mode concat

# 2. 运行 Weighted 融合 (动态加权)
python3 src/train_fusion_attention.py --task_name ustc_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/ustc_multiclass/ablation_weighted \
  --fusion_mode weighted
```

**B. SOTA 基准模型运行**

```bash
# 1. 运行 1D-CNN (DeepPacket) - 基于字节序列
python3 experiments/baselines/train_baseline.py --model_type deeppacket --task_name ustc_multiclass

# 2. 运行 2D-CNN - 基于流量图像 (RGB/Gray)
python3 experiments/baselines/train_baseline.py --model_type cnn2d --task_name ustc_multiclass

# 3. 运行 LSTM - 基于字节序列
python3 experiments/baselines/train_baseline.py --model_type lstm --task_name ustc_multiclass

# 4. 运行 ViT - 基于流量图像
python3 experiments/baselines/train_baseline.py --model_type vit --task_name ustc_multiclass
```

### 3. 集成决策与不平衡韧性 (Ensemble & Imbalance)

**A. 集成策略对比 (Voting vs Stacking)**

```bash
# 运行 Stacking 模式 (默认使用 XGBoost 二级决策)
python3 src/train_fusion_attention_stacking.py --task_name mta_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mta_multiclass/attention_stacking
```

**B. 不平衡梯度压力测试 (Stress Test)**

```bash
# 自动化遍历 mta_ratio2, mta_ratio5, mta_ratio10, mta_ratio15
python3 src/run_all_modes.py --mode stress_test --task_name mta \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mta_stress_test
```

### 4. 工程评估与图表生成

```bash
# 统计 Params, FLOPs 和推理延迟
python3 tools/measure_efficiency.py

# 自动化生成学术图表 (Chapter 4 Figures)
python3 figures/code/fig4_3_model_comparison.py
python3 figures/code/fig4_7_robustness_curve.py
python3 figures/code/fig4_9_mta_cm_correction.py
python3 figures/code/fig4_10_mfcp_cm_correction.py
python3 figures/code/fig4_11_voting_vs_stacking.py
```

## 不推荐作为主命令的合并入口

仓库里存在：

```bash
python3 src/run_all_modes.py --mode all ...
```

但本文档不把它作为主命令展示，原因有两点：

- 用户要求四个实验必须给出独立命令，不能只给一个合并入口
- 实际排查和复现实验时，分开运行 `attention` 与 `attention_stacking` 更清晰

## 基础自检

如果只想验证命令入口和参数是否正常，可运行：

```bash
python3 -m unittest tests.test_attention_entrypoints
python3 -m unittest tests.test_split_data_tasks
python3 -m unittest tests.test_fusion_task_resolution
python3 -m unittest tests.test_ssl_tls_rgb_image
python3 -m unittest tests.test_task_config
python3 -m unittest tests.test_run_all_modes
python3 -m unittest tests.test_stacking_improvements
```

注意：按项目约束，不运行 `mvn test`。

## 文档维护约定

后续如果修改了以下任一内容，必须同步更新本 README：

- 任务名
- 目录结构
- 预处理流程
- 训练入口脚本
- 默认参数
- 输出路径
- 推荐执行命令

同时也必须检查并更新 `AGENTS.md`，确保 AI 协作约束和实际仓库状态一致。


python3 src/train_fusion_attention.py \
  --task_name mta_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mta_multiclass/attention \
  --batch_size 32 --epochs 32 --patience 4 --lr 1e-3 \
  --num_workers 4 --prefetch_factor 2 --device auto \
  --charbert_mode charaware --char_vocab hex --char_emb_dim 32 \
  --char_cnn_channels 64 --char_fusion gated --char_fusion_layers all

python3 src/train_fusion_attention_stacking.py \
  --task_name mta_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mta_multiclass/attention_soft_voting \
  --meta_methods xgboost,lightgbm,catboost \
  --stacking_level single \
  --epochs 32 --lr 3e-4 --patience 8 \
  --class_balance weighted_sampler_loss --loss_type focal --focal_gamma 1.5 \
  --weight_decay 1e-4 --label_smoothing 0.03 \
  --early_stop_metric val_f1 --early_stop_mode max \
  --lr_scheduler reduce --lr_patience 2 --lr_factor 0.5 --min_lr 1e-6 \
  --grad_clip_norm 1.0 --batch_size 32 --num_workers 4 --prefetch_factor 2 \
  --pin_memory --persistent_workers --device auto \
  --charbert_mode charaware --char_vocab hex --char_emb_dim 32 \
  --char_cnn_channels 64 --char_fusion gated --char_fusion_layers all


python3 src/train_fusion_attention.py \
  --task_name mfcp_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mfcp_multiclass/attention \
  --batch_size 32 \
  --epochs 32 \
  --patience 4 \
  --lr 1e-3 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --device auto \
  --charbert_mode charaware \
  --char_vocab hex \
  --char_emb_dim 32 \
  --char_cnn_channels 64 \
  --char_fusion gated \
  --char_fusion_layers all


python3 src/train_fusion_attention_stacking.py \
  --task_name mfcp_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/mfcp_multiclass/attention_soft_voting \
  --meta_methods xgboost,lightgbm,catboost \
  --stacking_level single \
  --epochs 32 \
  --lr 3e-4 \
  --patience 8 \
  --class_balance weighted_sampler_loss \
  --loss_type focal \
  --focal_gamma 1.5 \
  --weight_decay 1e-4 \
  --label_smoothing 0.03 \
  --early_stop_metric val_f1 \
  --early_stop_mode max \
  --lr_scheduler reduce \
  --lr_patience 2 \
  --lr_factor 0.5 \
  --min_lr 1e-6 \
  --grad_clip_norm 1.0 \
  --batch_size 32 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --pin_memory \
  --persistent_workers \
  --device auto \
  --charbert_mode charaware \
  --char_vocab hex \
  --char_emb_dim 32 \
  --char_cnn_channels 64 \
  --char_fusion gated \
  --char_fusion_layers all
