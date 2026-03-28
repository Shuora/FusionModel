#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <mta|mfcp|ustc> [extra prepare_dataset args...]"
  exit 1
fi

TASK="$1"
shift

case "$TASK" in
  mta|mfcp|ustc) ;;
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

TOKENIZER_MODEL="${TOKENIZER_MODEL:-distilbert-base-uncased}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/dataset}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/SourceData}"

ARGS=(
  --task "$TASK"
  --source-root "$SOURCE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --tokenizer-model "$TOKENIZER_MODEL"
  --splitcap-launcher "${SPLITCAP_LAUNCHER:-mono}"
  --editcap-path "${EDITCAP_BIN:-editcap}"
)

if [[ "${USE_SPLITCAP:-0}" != "1" ]]; then
  ARGS+=(--skip-splitcap)
fi

python "$REPO_ROOT/scripts/prepare_dataset.py" "${ARGS[@]}" "$@"
