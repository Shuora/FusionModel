# FusionModel 运行命令手册

本文档按当前代码实现整理可直接执行的命令，覆盖：

- `session_full` 预处理流程（`PCAP -> Session PCAP -> RGB+时序特征`）
- 阶段协议命令（阶段1二分类清单、阶段2多分类任务清单）
- 三个数据集分开训练命令（`CICAndMal2017`、`MFCP`、`USTC-TFC2016`）

## 0. 环境准备

```bash
cd /home/shuora/Repositories/Traffic/FusionModel
conda activate FusionModel
```

默认数据目录（无需复制）：

- `SourceData/CICAndMal2017`
- `SourceData/MFCP`
- `SourceData/USTC-TFC2016`
- `SourceData/ISCX`（若要跑阶段1二分类必须存在）
- `SourceData/MTA`（若要跑阶段1二分类必须存在）

## 1. 预处理命令（session_full）

说明：

- 默认会在特征提取后清理 `tmp_sessions`
- 默认保留抽检图 `debug/preview_png`（每类上限由 `--preview-per-family` 控制）
- 如需保留切分后的 session pcap，使用 `--keep-sessions`

### 1.1 全数据集一次预处理

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20
```

### 1.2 仅处理指定数据集

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --datasets MFCP USTC-TFC2016 \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20
```

### 1.3 调试模式（保留 session pcap）

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --keep-sessions \
  --preview-per-family 20
```

兼容脚本路径启动：

```bash
python src/data/preprocess_runner.py --help
```

## 2. 阶段协议命令

### 2.1 阶段1（混合二分类）清单生成

标签定义：

- `ISCX -> normal (0)`
- `MFCP/MTA/USTC-TFC2016 -> malicious (1)`

严格要求：缺任一数据集会直接报错退出。

```bash
python -m src.experiments.stage1_binary \
  --processed-root outputs/processed \
  --policy session_full \
  --output outputs/protocol/stage1_binary_manifest.csv
```

### 2.2 阶段2（三任务多分类）清单生成

固定任务：

- `MTA` 7类
- `MFCP` 6类
- `USTC-TFC2016` 10类

```bash
python -m src.experiments.stage2_multiclass \
  --output outputs/protocol/stage2_tasks.json
```

## 3. 三个数据集分开训练命令

训练入口统一为：

- `python -m src.train`（`--stage warmup|fusion|stacking|moe`）
- `python -m src.evaluate`
- `python -m src.report`

注意：`--processed-root` 应指向“包含数据集目录的一层根目录”。

### 3.1 CICAndMal2017

#### 3.1.1 预处理

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed-cicandmal2017 \
  --policies session_full \
  --datasets CICAndMal2017 \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20
```

#### 3.1.2 warmup

```bash
python -m src.train \
  --processed-root outputs/processed-cicandmal2017 \
  --policy session_full \
  --stage warmup \
  --run-root runs/CICAndMal2017 \
  --run-id cicandmal2017-sessionfull-warmup \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 42
```

#### 3.1.3 fusion

```bash
python -m src.train \
  --processed-root outputs/processed-cicandmal2017 \
  --policy session_full \
  --stage fusion \
  --run-root runs/CICAndMal2017 \
  --run-id cicandmal2017-sessionfull-fusion \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 42
```

#### 3.1.4 stacking / moe（可选）

```bash
python -m src.stacking \
  --run-dir runs/CICAndMal2017/cicandmal2017-sessionfull-fusion \
  --n-splits 3 \
  --oof-epochs 2 \
  --batch-size 32 \
  --seed 42
```

```bash
python -m src.moe \
  --run-dir runs/CICAndMal2017/cicandmal2017-sessionfull-fusion \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 42
```

#### 3.1.5 评估与报告

```bash
python -m src.evaluate \
  --run-dir runs/CICAndMal2017/cicandmal2017-sessionfull-fusion \
  --split test \
  --checkpoint best
```

```bash
python -m src.report \
  --run-dir runs/CICAndMal2017/cicandmal2017-sessionfull-fusion
```

### 3.2 MFCP

#### 3.2.1 预处理

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed-mfcp \
  --policies session_full \
  --datasets MFCP \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20
```

#### 3.2.2 warmup

```bash
python -m src.train \
  --processed-root outputs/processed-mfcp \
  --policy session_full \
  --stage warmup \
  --run-root runs/MFCP \
  --run-id mfcp-sessionfull-warmup \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 42
```

#### 3.2.3 fusion

```bash
python -m src.train \
  --processed-root outputs/processed-mfcp \
  --policy session_full \
  --stage fusion \
  --run-root runs/MFCP \
  --run-id mfcp-sessionfull-fusion \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 42
```

#### 3.2.4 stacking / moe（可选）

```bash
python -m src.stacking \
  --run-dir runs/MFCP/mfcp-sessionfull-fusion \
  --n-splits 3 \
  --oof-epochs 2 \
  --batch-size 32 \
  --seed 42
```

```bash
python -m src.moe \
  --run-dir runs/MFCP/mfcp-sessionfull-fusion \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 42
```

#### 3.2.5 评估与报告

```bash
python -m src.evaluate \
  --run-dir runs/MFCP/mfcp-sessionfull-fusion \
  --split test \
  --checkpoint best
```

```bash
python -m src.report \
  --run-dir runs/MFCP/mfcp-sessionfull-fusion
```

### 3.3 USTC-TFC2016

#### 3.3.1 预处理

```bash
python -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed-ustc \
  --policies session_full \
  --datasets USTC-TFC2016 \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20
```

#### 3.3.2 warmup

```bash
python -m src.train \
  --processed-root outputs/processed-ustc \
  --policy session_full \
  --stage warmup \
  --run-root runs/USTC-TFC2016 \
  --run-id ustc-sessionfull-warmup \
  --epochs 10 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 42
```

#### 3.3.3 fusion

```bash
python -m src.train \
  --processed-root outputs/processed-ustc \
  --policy session_full \
  --stage fusion \
  --run-root runs/USTC-TFC2016 \
  --run-id ustc-sessionfull-fusion \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 42
```

#### 3.3.4 stacking / moe（可选）

```bash
python -m src.stacking \
  --run-dir runs/USTC-TFC2016/ustc-sessionfull-fusion \
  --n-splits 3 \
  --oof-epochs 2 \
  --batch-size 32 \
  --seed 42
```

```bash
python -m src.moe \
  --run-dir runs/USTC-TFC2016/ustc-sessionfull-fusion \
  --epochs 5 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 42
```

#### 3.3.5 评估与报告

```bash
python -m src.evaluate \
  --run-dir runs/USTC-TFC2016/ustc-sessionfull-fusion \
  --split test \
  --checkpoint best
```

```bash
python -m src.report \
  --run-dir runs/USTC-TFC2016/ustc-sessionfull-fusion
```

## 4. 消融（可选）

生成消融矩阵：

```bash
python -m src.ablation \
  --mode plan \
  --output runs/ablation/ablation_plan.csv
```

汇总消融结果：

```bash
python -m src.ablation \
  --mode summary \
  --run-root runs \
  --output runs/ablation/ablation_summary.csv
```

