#!/usr/bin/env bash
set -euo pipefail

# Run OVO-Bench 1/8 with Gemini teacher_eval on the current <state> + timestamp-only note protocol.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON="${PYTHON:-/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python}"
CONFIG="${CONFIG:-configs/batch_ovo_gemini_1of8_state_note_t.yaml}"
MODEL="${1:-${MODEL:-gemini-2.5-pro}}"
WORKERS="${WORKERS:-32}"
OUT_DIR="outputs/ovo_gemini_1of8_state_note_t"
OUTPUT="$OUT_DIR/results.jsonl"
WORKER_LOG_DIR="$OUT_DIR/worker_logs"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/mmu_ssd3/group_lisize/hetu/xujia10/joint_tags/scripts/gemini_client/config.json}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: required python not found or not executable: $PYTHON" >&2
  exit 1
fi

if [[ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
  echo "ERROR: GOOGLE_APPLICATION_CREDENTIALS points to a missing file: $GOOGLE_APPLICATION_CREDENTIALS" >&2
  echo "Set it to the real Gemini/Vertex service account JSON before running this eval." >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$WORKER_LOG_DIR"

echo "[eval] runner=evaluation/eval_batch.py"
echo "[eval] config=$CONFIG"
echo "[eval] backend=gemini"
echo "[eval] model=$MODEL"
echo "[eval] credentials=$GOOGLE_APPLICATION_CREDENTIALS"
echo "[eval] prompt=teacher_eval"
echo "[eval] postprocess=eval_repair"
echo "[eval] protocol=state_note_t"
echo "[eval] memory_window_seconds=120.0"
echo "[eval] workers=$WORKERS"
echo "[eval] output=$OUTPUT"
echo "[eval] traces=$OUT_DIR/traces"
echo "[eval] worker_logs=$WORKER_LOG_DIR"

"$PYTHON" evaluation/eval_batch.py \
  --config "$CONFIG" \
  --benchmark ovo \
  --backend gemini \
  --model "$MODEL" \
  --workers "$WORKERS" \
  --output "$OUTPUT" \
  --worker-log-dir "$WORKER_LOG_DIR"
