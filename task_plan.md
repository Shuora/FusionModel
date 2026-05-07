# Task Plan - Chapter 4 Experiment Reconstruction

According to the reconstruction plan in `docs/superpowers/plans/2026-04-28-chapter4-reconstruction.md`, we need to implement several missing experiments and baselines to validate the proposed model.

## Phase 1: Data Preparation & Imbalance Gradient
- [ ] Create `tools/generate_imbalance_datasets.py` to automate 10:1 and 2:1, 5:1, 15:1 splits.
- [x] Create `tools/downsample_processed.py` to reduce large datasets for fast iteration.
- [ ] Update `src/split_data.py` if necessary to support custom ratios via CLI.
- [x] Prepare standard 10:1 datasets for MTA and MFCP (MTA/MFCP downsampled/rebalanced).

## Phase 2: Representation Layer Ablation (Exp 4.2)
- [ ] **Sub-item A (Space branch)**: Modify `src/ssl_tls_rgb_image.py` to add `--mode {rgb,gray}`.
- [ ] **Sub-item B (Time branch)**: Add a CLI flag to training scripts to disable temporal sidecar (`.json`) and fallback to flat bytes.
- [x] **Sub-item C (Single branch)**: Implement training for single branch models (MobileViT-only, CharBERT-only).

## Phase 3: Fusion Layer Ablation & SOTA (Exp 4.3)
- [ ] **Sub-item A (Fusion mode)**: Implement `Concat` fusion in `src/fusion_common.py`.
- [x] **Sub-item B (Baselines)**: Implement `DeepPacket` (1D-CNN), `CNN2D` (2D-CNN), `LSTM`, and `ViT` in `experiments/baselines/`.
- [ ] **Sub-item C (Interpretability)**: Add cross-modal attention heatmap visualization to `src/fusion_common.py`.

## Phase 4: Decision Layer & Robustness (Exp 4.4)
- [ ] **Sub-item A (Stacking vs Voting)**: Ensure systematic comparison results are saved.
- [ ] **Sub-item B (Minority Gain)**: Implement logic to extract and compare minority recall before/after Stacking.
- [ ] **Sub-item C (Stress Test)**: Script to run models across the 2:1 -> 15:1 gradient.
- [ ] **Sub-item D (Engineering)**: Implement `tools/measure_efficiency.py` for Params, FLOPs, and Latency.

## Phase 5: Execution & Report Generation
- [ ] Run all experiments.
- [ ] Organize outputs into `outputs/chapter4_experiments/`.
- [ ] Generate summary tables/charts for the thesis.

## Decisions & Assumptions
- Use `fvcore` or `thop` for FLOPs calculation if available.
- "ViT" baseline will be implemented as a standard Vision Transformer or a pure MobileViT to contrast with our fusion model.
- "DeepPacket" will follow the 1D-CNN architecture described in the original paper.
