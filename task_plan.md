# Task Plan

## Goal

Refactor preprocessing and training around the new `SourceData` layout, keep only attention-based fusion plus attention stacking, and support four independent training tasks:

- `binary_benign_vs_malicious`
- `ustc_multiclass`
- `mta_multiclass`
- `mfcp_multiclass`

## Planned Steps

1. Audit the current repository state and document the new raw-data layouts and current training/fusion boundaries.
2. Write and review the design spec for config-driven preprocessing and task-driven training.
3. Create an isolated git worktree before implementation.
4. Refactor preprocessing to build standardized processed datasets from `SourceData`.
5. Refactor image generation to operate on the new processed dataset roots.
6. Refactor training code to remove non-attention fusion paths while preserving attention stacking.
7. Introduce config-driven task definitions for binary and per-dataset multiclass training.
8. Update run entrypoints and related documentation.
9. Run targeted verification without `mvn test`.

## Constraints

- Do not revert unrelated user changes in the dirty worktree.
- Do not run `mvn test`.
- Use the new `SourceData` tree as the raw source of truth.
- Second-stage multiclass training is independent from first-stage binary training.
