#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <binary|mta|mfcp|ustc> <checkpoint> [extra evaluate args...]"
  exit 1
fi

TASK="$1"
CHECKPOINT="$2"
shift 2

case "$TASK" in
  binary) CONFIG="configs/binary.yaml" ;;
  mta) CONFIG="configs/mta.yaml" ;;
  mfcp) CONFIG="configs/mfcp.yaml" ;;
  ustc) CONFIG="configs/ustc.yaml" ;;
  *)
    echo "Unsupported task: $TASK"
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /home/shuora/miniconda3/etc/profile.d/conda.sh
conda activate FusionModel
unset PYTHONPATH PYTHONUSERBASE
export MPLCONFIGDIR="${TMPDIR:-/tmp}/fusionmodel-mpl"
mkdir -p "$MPLCONFIGDIR"

python "$REPO_ROOT/scripts/evaluate.py" \
  --config "$REPO_ROOT/$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  "$@"
