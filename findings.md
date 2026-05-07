# Findings - Chapter 4 Experiment Reconstruction

## Codebase Status
- `src/split_data.py`: Supports `score_chasing_mta_v2` which already implements a 10:1 ratio for MTA.
- `tools/downsample_processed.py`: New tool added to downsample existing processed datasets using hard links.
- `ProcessedData/binary_benign_vs_malicious`: Reduced from 32G to 2.5G (10% samples) for faster training iteration. Original backed up as `_backup_32G`.
- `src/ssl_tls_rgb_image.py`: Currently only generates RGB images (R: head, G: handshake, B: flow stats). No grayscale support yet.
- `src/fusion_common.py`:
    - `AttentionFusionModel`: Cross-attention between MobileViT and CharBERT.
    - `FusionDataset`: Handles temporal sidecar (`.json`) and flat byte loading.
    - `run_stacking_experiment`: Supports multiple meta-learners and soft-voting.
- `experiments/baselines/`: Implemented four SOTA baselines:
    - `deeppacket.py`: 1D-CNN (packet bytes).
    - `cnn2d_baseline.py`: 2D-CNN (traffic images).
    - `lstm_baseline.py`: Bi-LSTM with embedding.
    - `vit_baseline.py`: Pure MobileViT (image only).
    - `train_baseline.py`: Unified trainer for all four models.

## Visuals
- **Thesis Figure Generation**:
    - Created a dedicated `figures/code/` directory for thesis-ready plots.
    - Implemented scripts for Figure 4.3, 4.7, 4.9, 4.10, and 4.11.
    - **Encoding Fix**: Successfully resolved Chinese "乱码" (character encoding) issues by implementing a font fallback list (SimHei, Microsoft YaHei, WenQuanYi, etc.) and explicitly setting the 'Agg' backend for headless execution.
    - Data Integration: The scripts now use actual metrics from the latest experiment runs in the `outputs/` directory.

## Debugging: Baseline Training Script
- **Default Path Issue**: `src/fusion_common.py` defaulted `--dataset_root` to `src/dataset`, which was incorrect. Changed to `PROJECT_ROOT / "ProcessedData"`.
- **Datetime Error**: `train_baseline.py` incorrectly tried to use `torch.utils.data.datetime`. Fixed to use standard `datetime`.
- **Input Mismatch**: `evaluate_full` (common code) expects models to take two inputs (image, pcap). Baseline models (e.g. DeepPacket) only take one. Resolved by wrapping baseline models in `BaselineWrapper`.
- **Plotting Error**: `plot_training_curves` failed if `train_acc` or `train_f1` were missing from `history`. Updated `train_baseline.py` to calculate training accuracy.

## Imbalance Gradient
- The plan requires 2:1, 5:1, 10:1, 15:1. Currently, `split_data.py` only has fixed profiles. Need a more dynamic way to generate these or multiple new profiles.
