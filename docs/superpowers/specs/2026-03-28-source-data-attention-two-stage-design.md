# SourceData Attention Two-Stage Design

## Summary

This design replaces the legacy dataset assumptions with a config-driven pipeline rooted in `SourceData`, removes non-attention fusion modes from the neural model path, retains attention stacking, and introduces four explicit training tasks:

- `binary_benign_vs_malicious`
- `ustc_multiclass`
- `mta_multiclass`
- `mfcp_multiclass`

The binary task and the multiclass tasks are independent. The multiclass stage does not consume predictions from the binary stage.

## Objectives

- Support the new raw-data layout under `SourceData`.
- Produce standardized processed datasets for training.
- Preserve the paired-modality training contract: binary session bytes plus derived RGB image.
- Restrict the neural fusion model path to attention-based fusion only.
- Preserve stacking, but only for the attention base model.
- Encode task definitions in one place so preprocessing and training follow the same source of truth.

## Non-Goals

- No attempt to make training read raw `SourceData` directly.
- No support for concat or weighted neural fusion after the refactor.
- No coupling between binary classification and the per-dataset multiclass tasks.
- No inclusion of `ISCX-VPN-NonVPN-2016` in the multiclass tasks.

## Raw Data Assumptions

`SourceData` currently contains heterogeneous layouts:

- `USTC-TFC2016/*.pcap`
- `MTA/<family>/*.pcap`
- `MFCP/<family>/*.pcap`
- `ISCX-VPN-NonVPN-2016/<group>/*.pcap|*.pcapng`

The pipeline must tolerate `.pcap` and `.pcapng`, and it must resolve labels based on task-specific rules instead of fixed path depth assumptions.

## Processed Data Contract

Each task writes to a dedicated processed root:

```text
ProcessedData/<task_name>/
├── pcap_data/
│   ├── Train/<label>/*.bin
│   └── Test/<label>/*.bin
├── image_data/
│   ├── Train/<label>/*.png
│   └── Test/<label>/*.png
└── metadata/
    ├── manifest.json
    └── label_map.json
```

This keeps training independent from the raw-data source layout and allows `fusion_common.py` to remain focused on standardized inputs.

## Task Definitions

Task definitions are the source of truth for both preprocessing and training.

Each task definition declares:

- `task_name`
- `source datasets`
- `label resolution rules`
- `allowed raw file patterns`
- `train/test split ratio`
- `output root`

The initial task set is:

### `binary_benign_vs_malicious`

- Benign source:
  - `ISCX-VPN-NonVPN-2016`
- Malicious sources:
  - `USTC-TFC2016`
  - `MTA`
  - `MFCP`
- Labels:
  - `benign`
  - `malicious`

### `ustc_multiclass`

- Source:
  - `USTC-TFC2016`
- Label resolver:
  - filename stem

### `mta_multiclass`

- Source:
  - `MTA`
- Label resolver:
  - first-level subdirectory name

### `mfcp_multiclass`

- Source:
  - `MFCP`
- Label resolver:
  - first-level subdirectory name

## Preprocessing Design

### Responsibilities

The preprocessing stage has two steps:

1. Convert raw packets into per-session `.bin` files under `pcap_data`.
2. Convert each `.bin` file into an RGB image under `image_data`.

### Session Extraction

`split_data.py` becomes a task-driven preprocessing entrypoint.

Required changes:

- Replace the hard-coded `CICAndMal2017` root with a configurable raw root, defaulting to `SourceData`.
- Support `.pcap` and `.pcapng`.
- Replace the current fixed two-level family scan with recursive discovery plus task-specific label resolution.
- Keep the current session extraction concept based on transport five-tuples and payload concatenation.
- Write metadata describing the originating raw file, resolved label, split, and generated session artifact names.

### Split Policy

Each task applies its own train/test split after collecting raw files for that task.

- Default ratio remains `0.8 / 0.2`.
- Splitting happens at the raw-file level before session extraction so sessions from one raw capture do not leak across train and test.
- Randomization remains seed-driven for reproducibility.

### `.pcapng` Support

The current code path depends on `dpkt.pcap.Reader`, which is insufficient for `.pcapng`. The refactor must route `.pcapng` through a compatible reader. This is a hard requirement because `ISCX-VPN-NonVPN-2016` contains `.pcapng` files.

## Image Generation Design

`ssl_tls_rgb_image.py` becomes a processed-root transformer.

Required changes:

- Accept a processed dataset root, such as `ProcessedData/binary_benign_vs_malicious`.
- Read from `<root>/pcap_data`.
- Write to `<root>/image_data`.
- Preserve relative paths so `.bin` and `.png` remain paired by split, label, and sample stem.
- Use task-aware logging instead of the hard-coded `CICAndMal2017` log filename.

The current RGB channel logic remains intact unless implementation reveals a correctness issue. This refactor is about pathing and orchestration, not feature redesign.

## Training Design

### Supported Model Paths

Neural fusion support is reduced to:

- attention fusion
- attention stacking

Removed paths:

- concat fusion
- weighted fusion
- concat stacking
- weighted stacking
- concat all-ensemble entrypoints

### Shared Training Contract

Training always consumes a standardized processed dataset root and a task name. It does not inspect raw `SourceData`.

The training stack must infer:

- train image directory
- train pcap directory
- test image directory
- test pcap directory
- label set

from `ProcessedData/<task_name>`.

### Task-Oriented Entrypoints

The runtime interface should shift from “all fusion modes” to “selected task plus selected attention runner”.

Expected supported entrypoints:

- train attention for a task
- train attention stacking for a task
- optionally run all supported attention runners for a task

The former `run_all_modes.py` should be narrowed accordingly.

## Code Organization

The implementation should converge on these boundaries:

- preprocessing configuration and task definitions
- raw-file discovery and label resolution
- session extraction and manifest writing
- image generation for processed datasets
- attention model definition and training utilities
- task-oriented training entrypoints

If a new module is needed for task configuration, both preprocessing and training must import the same definition source instead of duplicating rules.

## Metrics And Outputs

Each training task should continue to produce:

- model checkpoint
- training history
- evaluation metrics
- confusion matrix
- attention diagnostics when applicable

Outputs must remain scoped by task name so binary and multiclass runs cannot overwrite one another.

## Error Handling

- Missing raw roots or task definitions must fail fast with explicit messages.
- Unsupported raw file layouts should be reported with the originating path.
- Empty class directories in processed datasets should fail before training starts.
- `.pcapng` parse failures should identify the exact file and continue only when the failure is isolated and recoverable.

## Testing Strategy

Verification should cover:

- raw-file discovery for all four supported source layouts
- task label resolution
- train/test split isolation at raw-file granularity
- processed dataset directory creation
- `.bin` to `.png` path preservation
- task-based dataset resolution in training
- attention-only runner selection

No `mvn test` should be used.

## Migration Plan

1. Introduce task definitions and processed-root conventions.
2. Refactor preprocessing to generate standardized processed datasets from `SourceData`.
3. Refactor image generation to operate on processed roots.
4. Refactor training to consume task roots and remove non-attention fusion modes.
5. Update entrypoints and docs.
6. Run targeted verification on representative tasks.

## Open Implementation Questions

- Which library path will be used for `.pcapng` decoding in the current environment.
- Whether any deleted in-progress files from the current dirty worktree should be preserved or superseded during the implementation phase.

## Decision Check

This design explicitly encodes the user-approved decisions:

- config-driven refactor
- `SourceData` as the raw root
- attention-only neural fusion
- stacking retained
- binary task is benign vs malicious
- multiclass tasks are independent and dataset-specific
