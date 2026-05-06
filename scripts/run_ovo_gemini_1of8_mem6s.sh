#!/usr/bin/env bash
set -euo pipefail

# Run OVO-Bench 1/8 with Gemini teacher_eval and a 6s memory image window.
# Gemini credentials must be provided through GOOGLE_APPLICATION_CREDENTIALS.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python"
CONFIG="${CONFIG:-configs/batch_ovo_gemini_1of8_mem6s.yaml}"
MODEL="${1:-${MODEL:-gemini-2.5-pro}}"
WORKERS="${WORKERS:-32}"

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

echo "[eval] runner=evaluation/eval_batch.py"
echo "[eval] config=$CONFIG"
echo "[eval] backend=gemini"
echo "[eval] model=$MODEL"
echo "[eval] prompt=teacher_eval"
echo "[eval] memory_window_seconds=6.0"
echo "[eval] workers=$WORKERS"
echo "[eval] output=outputs/ovo_gemini_1of8_mem6s/results.jsonl"
echo "[eval] traces=outputs/ovo_gemini_1of8_mem6s/traces"

"$PYTHON" evaluation/eval_batch.py \
  --config "$CONFIG" \
  --benchmark ovo \
  --backend gemini \
  --model "$MODEL" \
  --workers "$WORKERS"
