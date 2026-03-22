# Session Full 实验命令（按当前仓库重写）

本文档按当前仓库实现整理，已对照：

- `src.data.preprocess_runner`
- `src.experiments.stage1_binary`
- `src.experiments.stage2_multiclass`
- `src.train`
- `src.evaluate`
- `src.report`
- `docs/commands/stage2-multiclass-e2e.sh`

目标不是保留旧命令，而是给出“现在这份仓库真正能跑”的命令集合。

## 0. 约定

以下命令默认在仓库根目录执行：

```bash
cd /home/shuora/Traffic/FusionModel
conda activate FusionModel

PYTHON_BIN=python
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || PYTHON_BIN=/home/shuora/miniconda3/envs/FusionModel/bin/python
```

当前仓库内常见目录：

- 原始数据：`SourceData/`
- 预处理输出：`outputs/processed/`
- 协议文件：`outputs/protocol/`
- 训练产物：`runs/`

当前 `SourceData/` 下可见数据集：

- `CICAndMal2017`
- `ISCX-VPN-NonVPN-2016`
- `MFCP`
- `MTA`
- `USTC-TFC2016`

## 1. 预处理

### 1.1 全量 `session_full` 预处理

```bash
"${PYTHON_BIN}" -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20
```

### 1.2 中断后续跑

```bash
"${PYTHON_BIN}" -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20 \
  --resume
```

### 1.3 只处理指定数据集

```bash
"${PYTHON_BIN}" -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --datasets MFCP MTA USTC-TFC2016 \
  --seed 42 \
  --cleanup-sessions \
  --preview-per-family 20
```

### 1.4 调试时保留切分后的 session 文件

```bash
"${PYTHON_BIN}" -m src.data.preprocess_runner \
  --source-root SourceData \
  --output-root outputs/processed \
  --policies session_full \
  --keep-sessions \
  --preview-per-family 20
```

说明：

- `preprocess_runner` 当前支持的参数只有：
  - `--source-root`
  - `--output-root`
  - `--policies`
  - `--datasets`
  - `--seed`
  - `--cleanup-sessions / --keep-sessions`
  - `--preview-per-family`
  - `--resume / --no-resume`
  - `--no-progress`
- 当前默认 policy 映射里，主线实验应使用 `session_full`。

## 2. Stage1 二分类

标签口径：

- `ISCX` -> `normal (0)`
- `MFCP + MTA` -> `malicious (1)`

当前 `stage1_binary` 的固定依赖数据集是：

- `ISCX`
- `MFCP`
- `MTA`

### 2.1 只生成 manifest

```bash
"${PYTHON_BIN}" -m src.experiments.stage1_binary \
  --processed-root outputs/processed \
  --policy session_full \
  --protocol-mode paper_balanced \
  --output outputs/protocol/stage1_binary_manifest.csv
```

当前支持两种协议模式：

- `paper_balanced`
  - 默认值
  - 保留论文类别集合
  - 对超大组裁到论文配额的 120%
  - 对不足组全保留
- `paper_strict`
  - 严格按论文 train/test 配额取样
  - 任一组不足会直接失败

### 2.2 推荐：手动执行完整流程

推荐手动拆成 `manifest -> train -> evaluate -> report`。原因很简单：

- `stage1_binary --execute` 只会透传：
  - `--device`
  - `--num-workers`
  - `--best-metric`
- 它不会透传 `src.train` 里这些可调参数：
  - `--hidden-dim`
  - `--num-heads`
  - `--alpha`
  - `--beta`
  - `--val-fraction`
  - `--train-max-samples`

先生成 manifest：

```bash
"${PYTHON_BIN}" -m src.experiments.stage1_binary \
  --processed-root outputs/processed \
  --policy session_full \
  --protocol-mode paper_balanced \
  --output outputs/protocol/stage1_binary_manifest.csv
```

定义本次 run：

```bash
export RUN_DATE=$(date +%F)
export RUN_ID="stage1-binary-$(date +%H%M%S)"
```

训练：

```bash
"${PYTHON_BIN}" -m src.train \
  --processed-root outputs/processed \
  --policy session_full \
  --datasets ISCX MFCP MTA \
  --session-filter-manifest outputs/protocol/stage1_binary_manifest.csv \
  --label-mode binary \
  --num-classes 2 \
  --run-root "runs/${RUN_DATE}" \
  --run-id "${RUN_ID}" \
  --stage fusion \
  --epochs 30 \
  --batch-size 24 \
  --lr 1e-3 \
  --seed 42 \
  --hidden-dim 192 \
  --num-heads 6 \
  --best-metric val_acc \
  --device auto \
  --num-workers 4
```

评估：

```bash
"${PYTHON_BIN}" -m src.evaluate \
  --run-dir "runs/${RUN_DATE}/${RUN_ID}" \
  --split test \
  --checkpoint best \
  --device auto
```

报告：

```bash
"${PYTHON_BIN}" -m src.report \
  --run-dir "runs/${RUN_DATE}/${RUN_ID}"
```

当前实现下的重要说明：

- `src.train` 当前支持：
  - `--hidden-dim`
  - `--num-heads`
  - `--alpha`
  - `--beta`
  - `--val-fraction`
  - `--train-max-samples`
  - `--best-metric {val_macro_f1,val_acc}`
- 当前已经没有 `early-stopping` 相关参数，旧文档里的 `--early-stopping-patience` / `--early-stopping-min-delta` 已过期。
- 当 manifest 没有显式 `val` split 时，`src.train` 会从 `train` 中按 `--val-fraction`（默认 `0.1`）派生验证集。
- 二分类下，训练会为验证集搜索 `decision_threshold`，`src.evaluate` 会自动复用 `best.ckpt` 或 `config.yaml` 中保存的该阈值。
- 如果你把 `--best-metric` 设为 `val_acc`，`best.ckpt` 会按 `val_acc` 保存；但当前 `src.report` 的 `Best Validation` 段仍按 `val_macro_f1` 排序展示，这一点和 checkpoint 选择逻辑不是完全一致。

### 2.3 一键执行版本

如果你只想快速跑通而不是精调参数，可以直接用：

```bash
"${PYTHON_BIN}" -m src.experiments.stage1_binary \
  --processed-root outputs/processed \
  --policy session_full \
  --protocol-mode paper_balanced \
  --output outputs/protocol/stage1_binary_manifest.csv \
  --execute \
  --run-root runs \
  --run-id stage1-binary \
  --stage fusion \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3 \
  --seed 42 \
  --best-metric val_acc \
  --device auto \
  --num-workers 4
```

说明：

- `--execute` 在 `stage in {warmup, fusion}` 时会自动调用：
  - `train`
  - `evaluate --split test`
  - `report`
- 若 `stage` 是 `stacking` 或 `moe`，会跳过 `evaluate`，然后直接生成报告。

## 3. Stage2 多分类

当前 `stage2_multiclass` 的基础任务固定为：

- `MTA`，`num_classes=7`
- `MFCP`，`num_classes=6`
- `USTC-TFC2016`，`num_classes=10`

### 3.1 只生成任务文件

```bash
"${PYTHON_BIN}" -m src.experiments.stage2_multiclass \
  --output outputs/protocol/stage2_tasks.json
```

注意：

- `stage2_tasks.json` 只会写入上面 3 个基础任务。
- `USTC 4000/3000/2000` 限样任务不会写入这个 JSON。

### 3.2 一键执行全部基础任务

```bash
"${PYTHON_BIN}" -m src.experiments.stage2_multiclass \
  --output outputs/protocol/stage2_tasks.json \
  --execute \
  --processed-root outputs/processed \
  --policy session_full \
  --run-root runs \
  --stage fusion \
  --epochs 12 \
  --batch-size 16 \
  --lr 1e-3 \
  --seed 42 \
  --device auto \
  --num-workers 4
```

当前行为：

- 会先跑：
  - `stage2-mta`
  - `stage2-mfcp`
  - `stage2-ustc-tfc2016`
- 默认还会继续跑额外的 USTC 限样任务：
  - `stage2-ustc-tfc2016-train4000`
  - `stage2-ustc-tfc2016-train3000`
  - `stage2-ustc-tfc2016-train2000`
- 汇总结果会写到：
  - `runs/stage2_execution_summary.json`

如果你只想跑 3 个基础任务，不跑额外 USTC 限样任务：

```bash
"${PYTHON_BIN}" -m src.experiments.stage2_multiclass \
  --output outputs/protocol/stage2_tasks.json \
  --execute \
  --processed-root outputs/processed \
  --policy session_full \
  --run-root runs \
  --stage fusion \
  --epochs 12 \
  --batch-size 16 \
  --lr 1e-3 \
  --seed 42 \
  --device auto \
  --num-workers 4 \
  --skip-ustc-limited
```

如果想自定义 USTC 限样规模：

```bash
"${PYTHON_BIN}" -m src.experiments.stage2_multiclass \
  --output outputs/protocol/stage2_tasks.json \
  --execute \
  --processed-root outputs/processed \
  --policy session_full \
  --run-root runs \
  --stage fusion \
  --epochs 12 \
  --batch-size 16 \
  --lr 1e-3 \
  --seed 42 \
  --device auto \
  --num-workers 4 \
  --ustc-train-limits 5000 2500 1000
```

### 3.3 按数据集分别跑

仓库里已经有可直接复用的脚本：

```bash
bash docs/commands/stage2-multiclass-e2e.sh
```

支持：

- `bash docs/commands/stage2-multiclass-e2e.sh`
- `bash docs/commands/stage2-multiclass-e2e.sh mta`
- `bash docs/commands/stage2-multiclass-e2e.sh mfcp`
- `bash docs/commands/stage2-multiclass-e2e.sh ustc`

该脚本当前与代码实现保持一致：

- 先按数据集做 `session_full` 预处理
- 再执行 `train`
- 然后执行 `evaluate --allow-split-fallback`
- 最后执行 `report`

## 4. 通用评估与报告

### 4.1 单独评估

```bash
"${PYTHON_BIN}" -m src.evaluate \
  --run-dir runs/stage2-mta \
  --split test \
  --checkpoint best \
  --device auto
```

当目标 split 不存在时，可以允许回退：

```bash
"${PYTHON_BIN}" -m src.evaluate \
  --run-dir runs/stage2-mta \
  --split test \
  --checkpoint best \
  --device auto \
  --allow-split-fallback
```

回退顺序是：

- 先尝试请求的 split
- 再尝试 `val`
- 最后尝试 `all`

### 4.2 单独生成报告

```bash
"${PYTHON_BIN}" -m src.report \
  --run-dir runs/stage2-mta
```

当前 `evaluate` 会产出：

- `eval_<split>.json`
- `figures/confusion_matrix_<split>.csv`
- `figures/confusion_matrix_<split>.png`
- `figures/classification_report_<split>.csv`
- `figures/classification_report_<split>.json`

当前 `report` 会：

- 生成 `figures/learning_curve.png`
- 汇总 `metrics.csv` 与评估 JSON
- 在 `report.md` 中直接渲染：
  - `Confusion Matrix`
  - `Classification Report`
  - `Paper-Compatible Metrics`

## 5. 当前文档替换掉的旧内容

下面这些旧说法已经不再适用，因此本次已整体删除：

- 训练支持 `early-stopping` 参数
- `stage1_binary --execute` 能透传 attention-fusion 的所有训练参数
- 文档里默认直接使用系统 `python` 一定可行
- Stage2 只会跑 3 个基础任务，不会额外生成 USTC 限样 run

如果后续代码再改，优先以这几个入口文件为准，而不是继续在旧文档上打补丁：

- `src/data/preprocess_runner.py`
- `src/experiments/stage1_binary.py`
- `src/experiments/stage2_multiclass.py`
- `src/train.py`
- `src/evaluate.py`
- `src/report.py`
