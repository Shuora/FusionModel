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

## 数据预处理流程

每个任务的预处理都分两步：

1. `split_data.py`
   把 `SourceData` 里的原始 `pcap/pcapng` 提取为 session，并写入 `ProcessedData/<task_name>/pcap_data/...`
2. `ssl_tls_rgb_image.py`
   把 `pcap_data` 下的 `.bin` 转换为 `image_data` 下的 `.png`

注意：

- 当前实现是先把原始抓包展开为 session，再在 session 级别切分 `Train/Test`
- 训练命令中的 `--dataset_root` 应该指向 `ProcessedData` 的父目录，而不是 `ProcessedData/<task_name>`
- 预处理默认会将日志落盘到任务目录下的 metadata/split_data.log 和 metadata/ssl_tls_rgb_image.log
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
  --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/mta_multiclass \
  --distribution_profile paper_mvtba
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
  --processed_root /home/shuora/Traffic/FusionModel/ProcessedData/mfcp_multiclass \
  --distribution_profile paper_mvtba
```

#### Step 2. 生成 RGB 图像

```bash
python3 src/ssl_tls_rgb_image.py \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData/mfcp_multiclass
```

## 训练命令

当前推荐只使用下面两个实验入口：

- `src/train_fusion_attention.py`
- `src/train_fusion_attention_stacking.py`

提示：`attention_stacking` 若希望触发多模型 soft-voting，请把 `--meta_methods` 设为至少两个方法（例如 `xgboost,lightgbm,catboost`，前提是环境已安装对应库）。

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
  --meta_methods xgboost \
  --batch_size 32 \
  --epochs 32 \
  --patience 4 \
  --lr 1e-3 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --device auto
```

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
```

#### 2. Attention + Stacking 训练

```bash
python3 src/train_fusion_attention_stacking.py \
  --task_name ustc_multiclass \
  --dataset_root /home/shuora/Traffic/FusionModel/ProcessedData \
  --output_dir /home/shuora/Traffic/FusionModel/outputs/ustc_multiclass/attention_stacking \
  --meta_methods xgboost \
  --batch_size 32 \
  --epochs 32 \
  --patience 4 \
  --lr 1e-3 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --device auto
```

### 实验三：`mta_multiclass`

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
  --meta_methods xgboost \
  --batch_size 32 \
  --epochs 32 \
  --patience 4 \
  --lr 1e-3 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --device auto
```

### 实验四：`mfcp_multiclass`

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
  --meta_methods xgboost \
  --batch_size 32 \
  --epochs 32 \
  --patience 4 \
  --lr 1e-3 \
  --num_workers 4 \
  --prefetch_factor 2 \
  --device auto
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
```

仅 `attention_stacking` 额外支持：

```text
--meta_methods
```

`attention_stacking` 当前默认行为（无需额外参数）：

- 元特征包含 `text/image/fusion` 三分支概率，并自动拼接 entropy、margin、分支一致性特征。
- 元特征提取使用 deterministic loader（忽略训练阶段 `weighted sampler/shuffle/drop_last`），避免 OOF 评估被采样偏置放大。
- 对每个 meta learner 自动执行 5-fold OOF 训练以减少元学习器过拟合。
- 对 `xgboost/lightgbm/catboost` 自动使用 class-balanced sample weight（若库可用）。
- 多个可用 meta learner 会自动做加权 soft-voting（权重来自各自 OOF macro-F1）。
- 对 `mta_multiclass` 自动按训练集最少样本类做 gain 调优（支持包含 `IcedID` 的 7 类 MTA）；对 `mfcp_multiclass` 自动做 `0/4` 二分类后处理链路：先用 OOF 按 `pair_f1` 选择校正强度 `alpha`（`0~1`），再做 pair 概率温度校准与阈值搜索。

早停相关建议（`fusion_common.py` 当前默认行为）：

- `--patience` 默认 `4`。
- `--early_stop_mode auto` 会按指标自动选择方向：`val_loss -> min`，`val_acc/val_f1 -> max`。
- 若手动设置 `--early_stop_mode`，必须与 `--early_stop_metric` 方向一致；不一致会直接报错，避免错误早停。
- 若验证监控值出现 `NaN/Inf`，会按“未改善”推进早停计数，达到 `patience` 后停止并恢复最佳权重。
- 若训练 batch 的 `loss` 出现 `NaN/Inf`，该 batch 会被跳过（不反向传播、不更新参数），并记录告警日志。

`epochs` 与 `lr` 的建议搭配（当前模型为从头训练的 `MobileViTConfig + CharBERT`，优化器为 `AdamW`）：

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
  - `--distribution_profile`（可选；`paper_mvtba` 仅对 `mta_multiclass`/`mfcp_multiclass` 启用论文固定样本分布。若某类可提取 session 不足，将对该类有放回补齐并写入唯一后缀）
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
```

常见输出包括：

- 训练日志
- 指标曲线图
- 混淆矩阵图
- attention 诊断图
- stacking 结果图（各 meta learner）
- `soft_voting` 报告与混淆矩阵（当至少 2 个 meta learner 可用时）
- `metrics.json` 中记录每个 meta learner 的 `oof_acc/oof_macro_f1` 与后处理参数

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
