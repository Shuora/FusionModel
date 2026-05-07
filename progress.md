# Progress - Chapter 4 Experiment Reconstruction

## 2026-05-02
- [x] Created `figures/code/` directory for thesis figures.
- [x] Implemented `fig4_3_model_comparison.py`, `fig4_7_robustness_curve.py`, `fig4_9_mta_cm_correction.py`, `fig4_10_mfcp_cm_correction.py`, and `fig4_11_voting_vs_stacking.py`.
- [x] Fixed Chinese character rendering ("乱码") in all figure scripts using standard font fallback list.
- [x] Populated figure scripts with actual experiment data from `outputs/`.

- [x] Research existing codebase and compare with `docs/superpowers/plans/2026-04-28-chapter4-reconstruction.md`.
- [x] Create implementation plan (`docs/superpowers/plans/2026-04-28-chapter4-reconstruction-implementation.md`).
- [x] Downsample `binary_benign_vs_malicious` dataset by 10x (32G -> 2.5G) to speed up experiments.
- [x] Task 1: Add support for grayscale, flat sequence, and concat fusion ablation.
- [x] Task 2: Implement SOTA baselines (DeepPacket, LSTM, ViT).
- [x] Task 3: Implement efficiency measurement and thesis figure generation tools.
- [x] Task 4: Automation of stress tests for class imbalance.
- [x] Finalize infrastructure for Chapter 4 experiments.
- [x] Implemented `mobilevit_ablation.py` and `charbert_ablation.py` for single-branch study.
- [x] Updated `train_baseline.py` and `README.md` to support new ablation models.
