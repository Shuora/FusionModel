# AGENTS.md

## Communication

- Prefer Chinese for communication.
- English may be used for technical terms, code, commands, and identifiers.

## Environment

- Before running project commands, activate the conda environment:

```bash
conda activate FusionModel
```

- When writing documentation or command examples for this repository, use `python3` in commands.
- Run commands from the repository root: `/home/shuora/Traffic/FusionModel`.

## Documentation Sync

- If you modify README, usage instructions, environment instructions, workflow documents, or other repository-facing guidance, you must also review and update `AGENTS.md` when needed so agent instructions stay consistent with the docs.
- If you modify project behavior, command flow, environment assumptions, output paths, or task conventions, sync those changes to both `README.md` and `AGENTS.md`.

## Project Notes

- This repository currently documents and runs the V4 `MobileViT + CharBERT` attention fusion workflow.
- Supported `task_name` values are:
  - `binary_benign_vs_malicious`
  - `ustc_multiclass`
  - `mta_multiclass`
  - `mfcp_multiclass`
- Standard data flow:
  - `SourceData/<dataset>`
  - `ProcessedData/<task>/pcap_data/{Train,Test}`
  - `ProcessedData/<task>/image_data/{Train,Test}`
  - training via `src/train_fusion_attention.py`, `src/train_fusion_attention_stacking.py`, or `src/run_all_modes.py`
- Default training runtime parameters should stay aligned across code and docs:
  - `batch_size=32`
  - `num_workers=4`
  - `prefetch_factor=2`

## Constraints

- Do not run `mvn test`.
- If historical helper scripts under `tools/` contain hardcoded paths, do not present them as standard cross-platform commands without updating them first.
