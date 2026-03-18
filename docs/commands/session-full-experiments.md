# Session Full 实验命令（MobileViT + ET-BERT Adapter）

本文档按当前仓库代码整理一套可直接执行的分步骤命令，适合你从环境准备、预处理、阶段1、阶段2一路手动跑通。

当前主线说明：

- 预处理：`PCAP -> Session PCAP -> RGB + ET-BERT(input_ids/attention_mask/token_type_ids)`
- 图像主干：`transformers.MobileViTForImageClassification.mobilevit`
- 文本主干：`ETBertBackbone` 兼容适配器
- 实验协议：
  - 阶段1：二分类
  - 阶段2：多分类

说明：当前 ET-BERT 侧不是原始 UER ET-BERT 的完整复刻，而是兼容其 `vocab/config/checkpoint` 形态的工程化 adapter。

## 0. 环境准备

若环境还没创建：

```bash
cd /home/shuora/Traffic/FusionModel
conda env create -f environment.yml
conda activate FusionModel
pip install -r requirements.txt
```

若环境已经存在，只需要：

```bash
cd /home/shuora/Traffic/FusionModel
conda activate FusionModel
pip install -r requirements.txt
```

## 1. 数据目录检查

建议先确认默认目录存在：

```bash
ls SourceData
```

当前主线通常会用到：

- `SourceData/ISCX` 或 `SourceData/ISCX-VPN-NonVPN-2016`
- `SourceData/MFCP`
- `SourceData/MTA`
- `SourceData/USTC-TFC2016`

说明：

- 阶段1默认需要：`ISCX + MFCP + MTA`
- 阶段2默认基础任务包括：`MTA + MFCP + USTC-TFC2016`

## 2. 预处理

### 2.1 全量预处理

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20
```

### 2.2 中断后断点续跑

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20 \
  --resume
```

### 2.3 只处理指定数据集

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --datasets MFCP MTA USTC-TFC2016 \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20
```

### 2.4 调试时保留 session pcap

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --keep-sessions \
  --preview-per-family 20
```

## 3. 阶段1：二分类

标签规则：

- `ISCX = normal (0)`
- `MFCP/MTA = malicious (1)`

### 3.1 只生成 manifest

```bash
python -m src.experiments.stage1_binary \
  --processed-root outputs/processed \
  --policy session_full \
  --output outputs/protocol/stage1_binary_manifest.csv
```

### 3.2 完整执行（train -> evaluate -> report）

```bash
python -m src.experiments.stage1_binary \
  --processed-root outputs/processed \
  --policy session_full \
  --output outputs/protocol/stage1_binary_manifest.csv \
  --execute \
  --run-root runs \
  --run-id stage1-binary \
  --stage fusion \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-3 \
  --seed 42
```

### 3.3 最小 smoke test

想先快速确认流程能通：

```bash
python -m src.experiments.stage1_binary \
  --processed-root outputs/processed \
  --policy session_full \
  --output outputs/protocol/stage1_binary_manifest.csv \
  --execute \
  --run-root runs \
  --run-id stage1-binary-smoke \
  --stage fusion \
  --epochs 1 \
  --batch-size 8 \
  --lr 1e-3 \
  --seed 42
```

约束说明：

- 必须存在 `ISCX`（或 `ISCX-VPN-NonVPN-2016` 别名）、`MFCP`、`MTA` 三个数据集的 `session_full/manifest/session_manifest.*`
- 若缺失，阶段1会直接报错退出

## 4. 阶段2：多分类

### 4.1 只生成任务文件

```bash
python -m src.experiments.stage2_multiclass \
  --output outputs/protocol/stage2_tasks.json
```

说明：

- `stage2_tasks.json` 只写入 3 个基础任务：
  - `MTA`
  - `MFCP`
  - `USTC-TFC2016`

### 4.2 完整执行基础任务 + 默认 USTC 限样任务

```bash
python -m src.experiments.stage2_multiclass \
  --output outputs/protocol/stage2_tasks.json \
  --execute \
  --processed-root outputs/processed \
  --policy session_full \
  --run-root runs \
  --stage fusion \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-3 \
  --seed 42
```

默认基础任务：

- `MTA` 7 类
- `MFCP` 6 类
- `USTC-TFC2016` 10 类

默认执行行为：

- `--execute` 时会额外触发 USTC `4000/3000/2000` 限样实验
- 这些额外 run 不会写进 `stage2_tasks.json`

### 4.3 只跑 3 个基础任务，不跑 USTC 限样

```bash
python -m src.experiments.stage2_multiclass \
  --output outputs/protocol/stage2_tasks.json \
  --execute \
  --processed-root outputs/processed \
  --policy session_full \
  --run-root runs \
  --stage fusion \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-3 \
  --seed 42 \
  --skip-ustc-limited
```

## 5. 单独训练某个数据集

### 5.1 只跑 MFCP

```bash
python -m src.train \
  --processed-root outputs/processed \
  --policy session_full \
  --datasets MFCP \
  --label-mode multiclass \
  --num-classes 6 \
  --stage fusion \
  --run-root runs \
  --run-id mfcp-fusion \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-3 \
  --seed 42
```

### 5.2 只跑 MTA

```bash
python -m src.train \
  --processed-root outputs/processed \
  --policy session_full \
  --datasets MTA \
  --label-mode multiclass \
  --num-classes 7 \
  --stage fusion \
  --run-root runs \
  --run-id mta-fusion \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-3 \
  --seed 42
```

### 5.3 只跑 USTC-TFC2016

```bash
python -m src.train \
  --processed-root outputs/processed \
  --policy session_full \
  --datasets USTC-TFC2016 \
  --label-mode multiclass \
  --num-classes 10 \
  --stage fusion \
  --run-root runs \
  --run-id ustc-fusion \
  --epochs 30 \
  --batch-size 64 \
  --lr 1e-3 \
  --seed 42
```

## 6. 单独评估

```bash
python -m src.evaluate \
  --run-dir runs/stage1-binary \
  --split test \
  --checkpoint best
```

如果请求的 split 不存在，允许自动回退：

```bash
python -m src.evaluate \
  --run-dir runs/stage1-binary \
  --split test \
  --checkpoint best \
  --allow-split-fallback
```

## 7. 单独生成报告

```bash
python -m src.report \
  --run-dir runs/stage1-binary
```

## 8. stacking / moe（可选）

### 8.1 stacking

```bash
python -m src.stacking \
  --run-dir runs/stage1-binary \
  --n-splits 3 \
  --oof-epochs 2 \
  --batch-size 64 \
  --seed 42
```

### 8.2 moe

```bash
python -m src.moe \
  --run-dir runs/stage1-binary \
  --epochs 5 \
  --batch-size 64 \
  --lr 1e-3 \
  --seed 42
```

## 9. 日志、进度条、指标、混淆矩阵

### 9.1 日志

有。

训练会输出结构化日志到终端，同时写入：

- `runs/<run-id>/train.log`

常见事件包括：

- `run_bootstrap`
- `config_summary`
- `dataset_stats`
- `epoch_done`
- `checkpoint_saved`

### 9.2 进度条

有。

- 预处理：`tqdm`
- 训练：`tqdm`
- 验证：`tqdm`

关闭方式：

- 预处理加 `--no-progress`
- 训练加 `--no-progress`

### 9.3 训练指标

训练过程会写入：

- `runs/<run-id>/metrics.csv`

当前会包含的核心列通常有：

- `train_loss`
- `train_acc`
- `train_macro_f1`
- `train_gate_mean`
- `val_loss`
- `val_acc`
- `val_macro_f1`
- `val_gate_mean`
- `epoch_time`

### 9.4 评估指标

评估后会生成：

- `runs/<run-id>/eval_test.json`
- 若回退，也可能是：
  - `eval_val.json`

常见字段：

- `top1`
- `macro_precision`
- `macro_f1`
- `macro_recall`
- `gate_mean`
- `num_samples`
- `requested_split`
- `effective_split`

### 9.5 混淆矩阵

评估后会生成：

- `runs/<run-id>/figures/confusion_matrix_<split>.csv`
- `runs/<run-id>/figures/confusion_matrix_<split>.png`

例如：

- `confusion_matrix_test.csv`
- `confusion_matrix_test.png`

### 9.6 学习曲线和汇总报告

`report` 会生成：

- `runs/<run-id>/report.md`
- `runs/<run-id>/figures/learning_curve.png`

### 9.7 stacking / moe 指标

如果你额外运行：

- `stacking` 会生成：
  - `runs/<run-id>/stacking/meta_metrics.json`
- `moe` 会生成：
  - `runs/<run-id>/moe/moe_metrics.json`
