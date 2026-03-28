# Findings

## Repository State

- The worktree is already dirty, with many deleted files and a newly introduced `src/` training layout.
- Existing planning files from a previous iteration were deleted in the current worktree state.

## Data Layout

- Raw data now lives under `SourceData/`.
- `SourceData` contains heterogeneous dataset layouts:
  - flat dataset files: `USTC-TFC2016/*.pcap`
  - family directories: `MTA/<family>/*.pcap`, `MFCP/<family>/*.pcap`
  - grouped files with `.pcap` and `.pcapng`: `ISCX-VPN-NonVPN-2016/<group>/*`

## Current Code Limits

- `src/split_data.py` is hard-coded for the old `CICAndMal2017` layout and only scans `.pcap`.
- `src/ssl_tls_rgb_image.py` is hard-coded to fixed `pcap_data` and `image_data` roots under the old dataset path.
- `src/fusion_common.py` still carries `concat`, `weighted`, and `attention` fusion paths plus multiple ensemble entrypoints.
- Current training code expects already-processed datasets, not raw `SourceData`.

## Confirmed Product Decisions

- Keep only attention fusion for the neural fusion model.
- Keep stacking support, but only on top of the attention base model.
- Binary task is `benign` vs `malicious`.
- Multiclass stage is independent from binary stage.
- Multiclass tasks are dataset-specific and malicious-only:
  - `USTC-TFC2016`
  - `MTA`
  - `MFCP`
