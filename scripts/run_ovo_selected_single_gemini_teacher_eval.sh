#!/usr/bin/env bash
set -euo pipefail

# Run selected OVO samples with the existing sequential eval runner.
# Gemini credentials must be provided through GOOGLE_APPLICATION_CREDENTIALS.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python"
CONFIG="${CONFIG:-configs/debug_ovo_gemini_teacher_eval_selected_single.yaml}"
MODEL="${1:-${MODEL:-gemini-2.5-pro}}"
OUT_DIR="${OUT_DIR:-outputs/ovo_gemini_teacher_eval_selected_single}"
OUTPUT="${OUTPUT:-$OUT_DIR/results.jsonl}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: required simple python not found: $PYTHON" >&2
  exit 1
fi

if [[ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  echo "ERROR: GOOGLE_APPLICATION_CREDENTIALS is not set." >&2
  echo "Example:" >&2
  echo "  export GOOGLE_APPLICATION_CREDENTIALS=/path/to/config.json" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[eval] runner=evaluation/runner.py"
echo "[eval] config=$CONFIG"
echo "[eval] backend=gemini"
echo "[eval] model=$MODEL"
echo "[eval] prompt=teacher_eval"
echo "[eval] output=$OUTPUT"

"$PYTHON" evaluation/runner.py \
  --config "$CONFIG" \
  --benchmark ovo \
  --backend gemini \
  --model "$MODEL" \
  --output "$OUTPUT"
