# Session Full 实验命令（MobileViT + ET-BERT Adapter）

本文档按当前仓库代码整理一套可直接执行的分步骤命令，适合你从环境准备、预处理、阶段1、阶段2一路手动跑通。

实验执行部分按 `train -> evaluate -> report` 拆开写，方便你手动控制每一步。

当前 `train/evaluate` 已支持：

- `--device {auto,cpu,cuda}`
- `--num-workers <int>`（训练）

建议：

- 单卡优先使用 `--device auto`
- 若机器内存较小（如 8GB RAM），优先从 `--num-workers 4` 开始

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

### 3.2 训练

先生成 manifest：

```bash
python -m src.experiments.stage1_binary \
  --processed-root outputs/processed \
  --policy session_full \
  --output outputs/protocol/stage1_binary_manifest.csv
```

再训练：

```bash
python -m src.train \
  --processed-root outputs/processed \
  --policy session_full \
  --datasets ISCX MFCP MTA \
  --session-filter-manifest outputs/protocol/stage1_binary_manifest.csv \
  --label-mode binary \
  --num-classes 2 \
  --run-root runs \
  --run-id stage1-binary \
  --stage fusion \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-3 \
  --seed 42 \
  --device auto \
  --num-workers 4
```

### 3.3 评估

```bash
python -m src.evaluate \
  --run-dir runs/stage1-binary \
  --split test \
  --checkpoint best \
  --device auto
```

如果请求的 split 不存在，允许自动回退：

```bash
python -m src.evaluate \
  --run-dir runs/stage1-binary \
  --split test \
  --checkpoint best \
  --device auto \
  --allow-split-fallback
```

### 3.4 生成报告

```bash
python -m src.report \
  --run-dir runs/stage1-binary
```

### 3.5 最小 smoke test

想先快速确认流程能通：

先生成 manifest：

```bash
python -m src.experiments.stage1_binary \
  --processed-root outputs/processed \
  --policy session_full \
  --output outputs/protocol/stage1_binary_manifest.csv
```

再训练 1 epoch：

```bash
python -m src.train \
  --processed-root outputs/processed \
  --policy session_full \
  --datasets ISCX MFCP MTA \
  --session-filter-manifest outputs/protocol/stage1_binary_manifest.csv \
  --label-mode binary \
  --num-classes 2 \
  --run-root runs \
  --run-id stage1-binary-smoke \
  --stage fusion \
  --epochs 1 \
  --batch-size 8 \
  --lr 1e-3 \
  --seed 42 \
  --device auto \
  --num-workers 4
```

评估：

```bash
python -m src.evaluate \
  --run-dir runs/stage1-binary-smoke \
  --split test \
  --checkpoint best \
  --device auto \
  --allow-split-fallback
```

报告：

```bash
python -m src.report \
  --run-dir runs/stage1-binary-smoke
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

### 4.2 训练基础任务

先生成任务文件：

```bash
python -m src.experiments.stage2_multiclass \
  --output outputs/protocol/stage2_tasks.json
```

#### 4.2.1 训练 MTA

```bash
python -m src.train \
  --processed-root outputs/processed \
  --policy session_full \
  --datasets MTA \
  --label-mode multiclass \
  --num-classes 7 \
  --run-root runs \
  --run-id stage2-mta \
  --stage fusion \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-3 \
  --seed 42 \
  --device auto \
  --num-workers 4
```

#### 4.2.2 训练 MFCP

```bash
python -m src.train \
  --processed-root outputs/processed \
  --policy session_full \
  --datasets MFCP \
  --label-mode multiclass \
  --num-classes 6 \
  --run-root runs \
  --run-id stage2-mfcp \
  --stage fusion \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-3 \
  --seed 42 \
  --device auto \
  --num-workers 4
```

#### 4.2.3 训练 USTC-TFC2016

```bash
python -m src.train \
  --processed-root outputs/processed \
  --policy session_full \
  --datasets USTC-TFC2016 \
  --label-mode multiclass \
  --num-classes 10 \
  --train-max-samples 2000 \
  --run-root runs \
  --run-id stage2-ustc-tfc2016 \
  --stage fusion \
  --epochs 12 \
  --batch-size 16 \
  --lr 1e-3 \
  --seed 42 \
  --device auto \
  --num-workers 4
```

默认基础任务：

- `MTA` 7 类
- `MFCP` 6 类
- `USTC-TFC2016` 10 类

### 4.3 分别评估基础任务

评估 MTA：

```bash
python -m src.evaluate \
  --run-dir runs/stage2-mta \
  --split test \
  --checkpoint best \
  --device auto \
  --allow-split-fallback
```

评估 MFCP：

```bash
python -m src.evaluate \
  --run-dir runs/stage2-mfcp \
  --split test \
  --checkpoint best \
  --device auto \
  --allow-split-fallback
```

评估 USTC-TFC2016：

```bash
python -m src.evaluate \
  --run-dir runs/stage2-ustc-tfc2016 \
  --split test \
  --checkpoint best \
  --device auto \
  --allow-split-fallback
```

### 4.4 分别生成基础任务报告

MTA：

```bash
python -m src.report \
  --run-dir runs/stage2-mta
```

MFCP：

```bash
python -m src.report \
  --run-dir runs/stage2-mfcp
```

USTC-TFC2016：

```bash
python -m src.report \
  --run-dir runs/stage2-ustc-tfc2016
```

### 4.5 可选：USTC 限样训练 / 评估 / 报告

例如训练 `train_max_samples=4000`：

```bash
python -m src.train \
  --processed-root outputs/processed \
  --policy session_full \
  --datasets USTC-TFC2016 \
  --label-mode multiclass \
  --num-classes 10 \
  --train-max-samples 4000 \
  --run-root runs \
  --run-id stage2-ustc-tfc2016-train4000 \
  --stage fusion \
  --epochs 12 \
  --batch-size 16 \
  --lr 1e-3 \
  --seed 42 \
  --device auto \
  --num-workers 4
```

评估：

```bash
python -m src.evaluate \
  --run-dir runs/stage2-ustc-tfc2016-train4000 \
  --split test \
  --checkpoint best \
  --device auto \
  --allow-split-fallback
```

报告：

```bash
python -m src.report \
  --run-dir runs/stage2-ustc-tfc2016-train4000
```

其余限样值同理，把 `4000` 改成 `3000` 或 `2000` 即可。

### 4.6 针对 `RTX 4060 Laptop 8GB + i7-13700 + 8GB RAM` 的推荐值

推荐原则：

- 优先 `--device auto`
- `--num-workers 4` 作为起点，不建议一开始开太高
- 当前代码会把选中的数据一次性读入内存，所以 `8GB RAM` 往往比 `8GB VRAM` 更早成为瓶颈

推荐训练参数：

- `stage1 binary`：
  - `--epochs 12`
  - `--batch-size 16`
  - `--lr 1e-3`
  - `--num-workers 4`
- `stage2 MTA / MFCP`：
  - `--epochs 12`
  - `--batch-size 16`
  - `--lr 1e-3`
  - `--num-workers 4`
- `stage2 USTC-TFC2016`：
  - `--epochs 12`
  - `--batch-size 16`
  - `--lr 1e-3`
  - `--num-workers 4`
  - `--train-max-samples 2000` 起步

如果出现问题：

- CUDA OOM：先把 `--batch-size 16` 降到 `8`
- 系统内存紧张或卡死：先把 `--num-workers 4` 降到 `0`，再降低 `--train-max-samples`
- 只是想确认链路能通：先跑 smoke test

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
  --batch-size 16 \
  --lr 1e-3 \
  --seed 42 \
  --device auto \
  --num-workers 4
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
  --batch-size 16 \
  --lr 1e-3 \
  --seed 42 \
  --device auto \
  --num-workers 4
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
  --epochs 12 \
  --batch-size 16 \
  --lr 1e-3 \
  --seed 42 \
  --device auto \
  --num-workers 4 \
  --train-max-samples 2000
```

## 6. 单独评估

```bash
python -m src.evaluate \
  --run-dir runs/stage1-binary \
  --split test \
  --checkpoint best \
  --device auto
```

如果请求的 split 不存在，允许自动回退：

```bash
python -m src.evaluate \
  --run-dir runs/stage1-binary \
  --split test \
  --checkpoint best \
  --device auto \
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
