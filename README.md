# FusionModel (CharBERT + Native MobileViT)

This repository builds an encrypted-traffic classification pipeline with:
- MVTBA-style preprocessing (session split, cleaning, 784-byte unification),
- RGB image branch (`R` paper-style + custom `G/B` channels),
- CharBERT temporal branch,
- Attention fusion,
- XGBoost stacking.

`old/` is archived only. All active implementation is in `src/`.

## Repository Layout

- `src/preprocess_splitcap_rgb.py`: end-to-end preprocessing entrypoint
- `src/splitcap_runner.py`: SplitCap command runner
- `src/session_postprocess.py`: session extraction and cleanup helpers
- `src/rgb_builder.py`: RGB channel construction
- `src/fusion_dataset.py`: dataset loading and byte-token conversion
- `src/models_charbert_mobilevit.py`: native MobileViT + CharBERT + attention fusion model
- `src/train_common.py`: shared training loops, metrics, plots, checkpoints
- `src/train_attention_fusion.py`: base attention-fusion training
- `src/train_attention_stacking.py`: attention-fusion + XGBoost stacking

Data source folders (already present in this repo):
- `SourceData/USTC-TFC2016`
- `SourceData/CICAndMal2017`

Bundled SplitCap:
- `tools/SplitCap_2-1/SplitCap.exe`

## Environment

Python 3.9+ recommended.

Install dependencies:

```bash
pip install numpy pillow tqdm dpkt torch torchvision scikit-learn matplotlib xgboost
```

## 1) Preprocessing

All commands below are executed from repo root (`C:\Repositories\Traffic\FusionModel`).

### 1.1 USTC (10-class by file stem)

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

### 1.2 CIC 4-class (group labels)

```bash
python src/preprocess_splitcap_rgb.py \
  --input_root SourceData/CICAndMal2017 \
  --output_root dataset/CIC_4 \
  --label_mode group \
  --splitcap_exe tools/SplitCap_2-1/SplitCap.exe \
  --splitcap_mode external \
  --train_ratio 0.8 \
  --temporal_formats bin
```

### 1.3 CIC 42-class (family labels)

```bash
python src/preprocess_splitcap_rgb.py \
  --input_root SourceData/CICAndMal2017 \
  --output_root dataset/CIC_42 \
  --label_mode family \
  --splitcap_exe tools/SplitCap_2-1/SplitCap.exe \
  --splitcap_mode external \
  --train_ratio 0.8 \
  --temporal_formats bin
```

Optional temporal export for analysis:

```bash
--temporal_formats bin,npy,pt
```

## 2) Training

### 2.1 Base attention fusion

```bash
python src/train_attention_fusion.py \
  --dataset_root dataset/USTC_10 \
  --batch_size 32 \
  --epochs 30 \
  --lr 1e-3 \
  --device auto \
  --output_dir outputs/train
```

### 2.2 Attention + XGBoost stacking (recommended main pipeline)

USTC first:

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

Then CIC 4-class:

```bash
python src/train_attention_stacking.py \
  --dataset_root dataset/CIC_4 \
  --batch_size 32 \
  --epochs 30 \
  --lr 3e-4 \
  --class_balance weighted_sampler_loss \
  --loss_type focal \
  --early_stop_metric val_f1 \
  --device auto \
  --output_dir outputs/train
```

Finally CIC 42-class:

```bash
python src/train_attention_stacking.py \
  --dataset_root dataset/CIC_42 \
  --batch_size 16 \
  --epochs 30 \
  --lr 3e-4 \
  --class_balance weighted_sampler_loss \
  --loss_type focal \
  --early_stop_metric val_f1 \
  --device auto \
  --output_dir outputs/train
```

## 3) Ablation (R-only)

```bash
python src/train_attention_stacking.py \
  --dataset_root dataset/CIC_42 \
  --ablation r_only \
  --output_dir outputs/train
```

Compare this run against default RGB (`--ablation none`) to verify G/B benefit.

## Outputs

### Preprocessing outputs
- Dataset files: `<output_root>/image_data/...` and `<output_root>/pcap_data/...`
- Optional temporal files: `<output_root>/temporal_data/...`
- Logs and summaries: `outputs/preprocess/<run_id>/`
  - `preprocess.log`
  - `preprocess_summary.json`
  - `label_distribution.csv`

### Training outputs
`outputs/train/<run_id>/`
- `logs/train.log`
- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `metrics/epoch_metrics.csv`
- `metrics/final_metrics.json`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`
- `reports/stacking_classification_report.txt` (stacking mode)
- `reports/stacking_confusion_matrix.png` (stacking mode)
- `figures/loss_curve.png`
- `figures/acc_curve.png`
- `figures/f1_curve.png`
- `stacking/meta_features_train.npy` (stacking mode)
- `stacking/meta_features_test.npy` (stacking mode)
- `stacking/meta_model_xgboost.json` (stacking mode)

## Notes

- Default SplitCap path is `tools/SplitCap_2-1/SplitCap.exe`.
- Training reads `.bin` from `pcap_data` and converts to byte tokens with `CLS/SEP` internally.
- Current setup is accuracy-first: source-pcap-level split, macro-F1 early stopping, and class-imbalance handling for CIC-42.
