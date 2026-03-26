# FusionModel Experiment CLI

This repository hosts a lightweight CLI surface covering the binary/multiclass experiments described in Task 9. The scripts rely on the `FusionModel` conda environment and the YAML slices located in `configs/`.

**Note:** These Task 9 scripts are configuration routings and CLI scaffolding only. They parse YAML, request CUDA, and prepare the dated run directories, but they do not yet execute the complete training pipeline or emit checkpoints.

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
python scripts/evaluate.py --config configs/mta.yaml --checkpoint <path-to-preexisting-checkpoint>.pt
```

The evaluation script assumes you already have a checkpoint (e.g., produced by the integrated training pipeline that will run once Task 9 evolves); it validates the provided file rather than creating one.

Adjust the dataset splits under `configs/` and the training/evaluation scripts as the project matures.
