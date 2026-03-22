#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash docs/commands/stage2-multiclass-e2e.sh            # run all datasets
#   bash docs/commands/stage2-multiclass-e2e.sh mta        # run MTA only
#   bash docs/commands/stage2-multiclass-e2e.sh mfcp       # run MFCP only
#   bash docs/commands/stage2-multiclass-e2e.sh ustc       # run USTC-TFC2016 only

DATASET="${1:-all}"

run_mta() {
  python -m src.data.preprocess_runner \
    --source-root SourceData \
    --output-root outputs/processed \
    --policies session_full \
    --datasets MTA \
    --seed 42 \
    --cleanup-sessions \
    --preview-per-family 20

  python -m src.train \
    --processed-root outputs/processed \
    --policy session_full \
    --datasets MTA \
    --label-mode multiclass \
    --num-classes 7 \
    --run-root runs \
    --run-id stage2-mta \
    --stage fusion \
    --epochs 12 \
    --batch-size 16 \
    --lr 1e-3 \
    --seed 42 \
    --device auto \
    --num-workers 4

  python -m src.evaluate \
    --run-dir runs/stage2-mta \
    --split test \
    --checkpoint best \
    --device auto \
    --allow-split-fallback

  python -m src.report \
    --run-dir runs/stage2-mta
}

run_mfcp() {
  python -m src.data.preprocess_runner \
    --source-root SourceData \
    --output-root outputs/processed \
    --policies session_full \
    --datasets MFCP \
    --seed 42 \
    --cleanup-sessions \
    --preview-per-family 20

  python -m src.train \
    --processed-root outputs/processed \
    --policy session_full \
    --datasets MFCP \
    --label-mode multiclass \
    --num-classes 6 \
    --run-root runs \
    --run-id stage2-mfcp \
    --stage fusion \
    --epochs 12 \
    --batch-size 16 \
    --lr 1e-3 \
    --seed 42 \
    --device auto \
    --num-workers 4

  python -m src.evaluate \
    --run-dir runs/stage2-mfcp \
    --split test \
    --checkpoint best \
    --device auto \
    --allow-split-fallback

  python -m src.report \
    --run-dir runs/stage2-mfcp
}

run_ustc() {
  python -m src.data.preprocess_runner \
    --source-root SourceData \
    --output-root outputs/processed \
    --policies session_full \
    --datasets USTC-TFC2016 \
    --seed 42 \
    --cleanup-sessions \
    --preview-per-family 20

  python -m src.train \
    --processed-root outputs/processed \
    --policy session_full \
    --datasets USTC-TFC2016 \
    --label-mode multiclass \
    --num-classes 10 \
    --train-max-samples 2000 \
    --run-root runs \
    --run-id stage2-ustc-tfc2016 \
    --stage fusion \
    --epochs 12 \
    --batch-size 16 \
    --lr 1e-3 \
    --seed 42 \
    --device auto \
    --num-workers 4

  python -m src.evaluate \
    --run-dir runs/stage2-ustc-tfc2016 \
    --split test \
    --checkpoint best \
    --device auto \
    --allow-split-fallback

  python -m src.report \
    --run-dir runs/stage2-ustc-tfc2016
}

case "${DATASET}" in
  all)
    run_mta
    run_mfcp
    run_ustc
    ;;
  mta)
    run_mta
    ;;
  mfcp)
    run_mfcp
    ;;
  ustc)
    run_ustc
    ;;
  *)
    echo "Unknown dataset: ${DATASET}"
    echo "Use one of: all, mta, mfcp, ustc"
    exit 1
    ;;
esac
