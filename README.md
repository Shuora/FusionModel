# FusionModel Experiment Workflow

This repository now contains the non-training pipeline you need before launching experiments:

- raw capture discovery
- optional `SplitCap` session slicing
- anonymization and duplicate filtering
- RGB image cache generation
- token text / tokenizer id generation
- manifest + train/val/test split export
- real model builder interfaces
- evaluation CLI and result export

Training entry scripts are still lightweight wrappers. You said you will handle the actual training run yourself.

## Setup

```bash
conda create -n FusionModel python=3.9 || true
conda activate FusionModel

pip install -r requirements.txt
```

If you want to use the real MobileViT builder, `timm` must be installed in `FusionModel`.

## Model Defaults

You said you do not have a local ET-BERT checkpoint yet.

So the current stable defaults are:

- image backbone: `mobilevit_s`
- text backbone: `distilbert-base-uncased`
- tokenizer: `distilbert-base-uncased`

These defaults are only a fallback to keep the pipeline runnable.
Once you obtain a real ET-BERT model/tokenizer, replace:

- `text_model`
- `tokenizer_model`

in the YAML config or override them from the CLI.

## Prepare the Dataset

```bash
cd /home/shuora/Traffic/FusionModel/.worktrees/codex-fusion-malicious-pipeline
bash scripts/run_prepare_binary.sh
```

By default the shell wrappers use `--skip-splitcap` for stability.
That means each discovered capture file is treated as a sample source directly.

If you want real session-level slicing with `SplitCap`, run:

```bash
USE_SPLITCAP=1 bash scripts/run_prepare_binary.sh
```

On Linux/WSL, the script now assumes `SplitCap` should be launched as:

```bash
mono Tools/SplitCap.exe
```

If your launcher differs, override it:

```bash
USE_SPLITCAP=1 SPLITCAP_LAUNCHER="mono" bash scripts/run_prepare_binary.sh
```

If your raw file is `.pcapng`, the pipeline will first do:

```bash
editcap -F pcap input.pcapng output.pcap
```

before handing the converted file to `SplitCap`.
So on Linux/WSL you should make sure `editcap` is installed and visible in `PATH`.

Multiclass preparation:

```bash
bash scripts/run_prepare_multiclass.sh mta
bash scripts/run_prepare_multiclass.sh mfcp
bash scripts/run_prepare_multiclass.sh ustc
```

Useful overrides:

```bash
TOKENIZER_MODEL=/path/to/your/etbert-tokenizer \
OUTPUT_ROOT=/home/shuora/Traffic/FusionModel/dataset \
SOURCE_ROOT=/home/shuora/Traffic/FusionModel/SourceData \
SPLITCAP_LAUNCHER="mono" \
EDITCAP_BIN="editcap" \
bash scripts/run_prepare_binary.sh
```

After preparation, the generated files will live under:

```text
dataset/<task>/
  manifest.csv
  train.csv
  val.csv
  test.csv
  cache/
  sessions_clean/
  sessions_raw/   # only when SplitCap is used
```

## Training

```bash
python scripts/train_binary.py

python scripts/train_multiclass.py --config configs/mta.yaml
```

These scripts currently only:

- load config
- require CUDA
- create the dated run directory
- print the resolved task/config/device

They do not yet execute the full training loop.

## Evaluation

```bash
bash scripts/run_evaluate.sh mta /path/to/checkpoint.pt
```

The evaluation CLI assumes you already have a checkpoint from your own training run.

It will:

- load the configured manifest/split CSV
- rebuild the model from config / CLI options
- load checkpoint weights
- run inference
- write metrics, classification report, and confusion matrix under `runs/<date>/<task>/outputs/`

You can still call the Python script directly when needed:

```bash
python scripts/evaluate.py \
  --config configs/mta.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --manifest dataset/mta/test.csv \
  --image-model mobilevit_s \
  --text-model /path/to/your/etbert-model
```

## Current Completion State

Completed:

- non-training data pipeline
- real backbone builder API
- evaluation helper + CLI
- config / requirements / README

Not completed:

- full training orchestration inside `train_binary.py` / `train_multiclass.py`
- real experiment execution on your checkpoint/data by me
- final git commit for the newest non-training updates
