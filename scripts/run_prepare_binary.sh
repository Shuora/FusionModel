#!/usr/bin/env bash
set -euo pipefail

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
  --task binary
  --source-root "$SOURCE_ROOT"
  --output-root "$OUTPUT_ROOT"
  --tokenizer-model "$TOKENIZER_MODEL"
  --splitcap-launcher "${SPLITCAP_LAUNCHER:-mono}"
  --editcap-path "${EDITCAP_BIN:-editcap}"
)

if [[ "${USE_SPLITCAP:-0}" != "1" ]]; then
  ARGS+=(--skip-splitcap)
else
  if [[ "${RESUME_SPLITCAP:-1}" == "0" ]]; then
    ARGS+=(--no-resume-splitcap)
  else
    ARGS+=(--resume-splitcap)
  fi
fi

python "$REPO_ROOT/scripts/prepare_dataset.py" "${ARGS[@]}" "$@"
