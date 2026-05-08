#!/usr/bin/env bash
set -euo pipefail

# Sequential OVO debug run for selected SFT-regression samples.
# Uses evaluation/runner.py, not evaluation/eval_batch.py.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${1:-/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm}"
GPU="${GPU:-0}"
PORT="${PORT:-8000}"
PYTHON="/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python"
VLLM="/mmu_mllm_hdd/zhouhanshu/conda/envs/vllm/bin/vllm"
CONFIG="configs/debug_ovo_qwen3vl8b_finetuned_selected_single.yaml"
OUT_DIR="outputs/ovo_qwen3vl8b_finetuned_selected_single"
ALLOW_EXISTING_SERVER="${ALLOW_EXISTING_SERVER:-0}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: required simple python not found or not executable: $PYTHON" >&2
  exit 1
fi
[[ -x "$VLLM" ]] || VLLM="vllm"

mkdir -p "$OUT_DIR/vllm_logs" "$OUT_DIR/vllm_pids"

started_pid=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "$started_pid" ]]; then
    echo "[cleanup] stopping vLLM server pid=$started_pid"
    kill -- "-$started_pid" >/dev/null 2>&1 || kill "$started_pid" >/dev/null 2>&1 || true
    sleep 2
    if kill -0 "$started_pid" >/dev/null 2>&1; then
      echo "[cleanup] force killing vLLM process group pid=$started_pid"
      kill -9 -- "-$started_pid" >/dev/null 2>&1 || kill -9 "$started_pid" >/dev/null 2>&1 || true
    fi
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

check_endpoint() {
  "$PYTHON" - "$1/models" >/dev/null 2>&1 <<'PY'
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=2) as resp:
    if resp.status >= 400:
        raise SystemExit(1)
PY
}

wait_for_endpoint() {
  local endpoint="$1"
  local deadline=$((SECONDS + 600))
  until check_endpoint "$endpoint"; do
    if (( SECONDS >= deadline )); then
      echo "ERROR: endpoint did not become ready: $endpoint" >&2
      return 1
    fi
    sleep 3
  done
}

endpoint="http://127.0.0.1:${PORT}/v1"
log_path="$OUT_DIR/vllm_logs/vllm_${PORT}.log"

if check_endpoint "$endpoint"; then
  if [[ "$ALLOW_EXISTING_SERVER" != "1" ]]; then
    echo "ERROR: endpoint already has a running server: $endpoint" >&2
    echo "Set ALLOW_EXISTING_SERVER=1 to reuse it intentionally." >&2
    exit 1
  fi
  echo "[server] ready already: endpoint=$endpoint"
else
  echo "[server] starting gpu=$GPU port=$PORT log=$log_path"
  CUDA_VISIBLE_DEVICES="$GPU" setsid "$VLLM" serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 1 \
    >"$log_path" 2>&1 &
  started_pid=$!
  echo "$started_pid" > "$OUT_DIR/vllm_pids/vllm_${PORT}.pid"
  wait_for_endpoint "$endpoint"
fi

echo "[eval] sequential runner: evaluation/runner.py"
echo "[eval] config=$CONFIG"
echo "[eval] endpoint=$endpoint"
echo "[eval] output=$OUT_DIR/results.jsonl"
echo "[eval] traces=$OUT_DIR/traces"
"$PYTHON" evaluation/runner.py \
  --config "$CONFIG" \
  --benchmark ovo \
  --backend vllm \
  --model "$MODEL" \
  --endpoint "$endpoint" \
  --output "$OUT_DIR/results.jsonl"
