# FusionModel Experiment CLI

This repository hosts a lightweight CLI surface covering the binary/multiclass experiments described in Task 9. The scripts rely on the `FusionModel` conda environment and the YAML slices located in `configs/`.

## Setup

```bash
# create or activate the shared environment
conda create -n FusionModel python=3.9 || true
conda activate FusionModel

# install runtime dependencies
pip install -r requirements.txt
```

## Prepare the Dataset

```bash
cd /home/shuora/Traffic/FusionModel/.worktrees/codex-fusion-malicious-pipeline
python scripts/prepare_dataset.py
```

## Training

```bash
# Binary experiment with the default YAML
python scripts/train_binary.py

# Multiclass experiment with an explicit config
python scripts/train_multiclass.py --config configs/mta.yaml
```

## Evaluation

```bash
python scripts/evaluate.py --config configs/mta.yaml --checkpoint runs/mta_7cls/<latest>/checkpoints/model.pt
```

Adjust the dataset splits under `configs/` and the training/evaluation scripts as the project matures.
