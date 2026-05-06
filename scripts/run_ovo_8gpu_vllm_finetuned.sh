#!/usr/bin/env bash
set -euo pipefail

# V4 OVO 1/8 eval on 8 local vLLM replicas.
# Usage:
#   ./scripts/run_ovo_8gpu_vllm_finetuned.sh
#   ./scripts/run_ovo_8gpu_vllm_finetuned.sh /path/to/vllm-compatible-finetuned-model

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${1:-/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_v2_3077_vllm}"
PYTHON="/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python"
VLLM="/mmu_mllm_hdd/zhouhanshu/conda/envs/vllm/bin/vllm"
CONFIG="configs/batch_ovo_qwen3vl8b_finetuned_8gpu.yaml"
LIMIT=""  # Set to e.g. "8" for smoke; keep empty for the configured annotation file.
ALLOW_EXISTING_SERVERS="${ALLOW_EXISTING_SERVERS:-0}"

[[ -x "$PYTHON" ]] || PYTHON="python"
[[ -x "$VLLM" ]] || VLLM="vllm"

mkdir -p \
  outputs/ovo_qwen3vl8b_finetuned_1of8/vllm_logs \
  outputs/ovo_qwen3vl8b_finetuned_1of8/worker_logs \
  outputs/ovo_qwen3vl8b_finetuned_1of8/vllm_pids

started_pids=()

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ "${#started_pids[@]}" -gt 0 ]]; then
    echo "[cleanup] stopping ${#started_pids[@]} vLLM server(s)"
    for pid in "${started_pids[@]}"; do
      kill -- "-$pid" >/dev/null 2>&1 || kill "$pid" >/dev/null 2>&1 || true
    done
    sleep 2
    for pid in "${started_pids[@]}"; do
      kill -0 "$pid" >/dev/null 2>&1 || continue
      echo "[cleanup] force killing vLLM process group pid=$pid"
      kill -9 -- "-$pid" >/dev/null 2>&1 || kill -9 "$pid" >/dev/null 2>&1 || true
    done
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

for gpu in 0 1 2 3 4 5 6 7; do
  port=$((8000 + gpu))
  endpoint="http://127.0.0.1:${port}/v1"
  log_path="outputs/ovo_qwen3vl8b_finetuned_1of8/vllm_logs/vllm_${port}.log"

  if check_endpoint "$endpoint"; then
    if [[ "$ALLOW_EXISTING_SERVERS" != "1" ]]; then
      echo "ERROR: endpoint already has a running server: $endpoint" >&2
      echo "Stop existing vLLM servers first, or set ALLOW_EXISTING_SERVERS=1 to reuse them intentionally." >&2
      exit 1
    fi
    echo "[server] ready already: gpu=$gpu endpoint=$endpoint"
    continue
  fi

  echo "[server] starting gpu=$gpu port=$port log=$log_path"
  CUDA_VISIBLE_DEVICES="$gpu" setsid "$VLLM" serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$port" \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 16 \
    >"$log_path" 2>&1 &

  pid=$!
  started_pids+=("$pid")
  echo "$pid" > "outputs/ovo_qwen3vl8b_finetuned_1of8/vllm_pids/vllm_${port}.pid"
done

for port in 8000 8001 8002 8003 8004 8005 8006 8007; do
  endpoint="http://127.0.0.1:${port}/v1"
  echo "[server] waiting for $endpoint"
  wait_for_endpoint "$endpoint"
done

endpoints="http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1,http://127.0.0.1:8003/v1,http://127.0.0.1:8004/v1,http://127.0.0.1:8005/v1,http://127.0.0.1:8006/v1,http://127.0.0.1:8007/v1"

eval_cmd=(
  "$PYTHON" evaluation/eval_batch.py
  --config "$CONFIG"
  --benchmark ovo
  --backend vllm
  --model "$MODEL"
  --endpoints "$endpoints"
  --workers 16
  --output outputs/ovo_qwen3vl8b_finetuned_1of8/results.jsonl
  --worker-log-dir outputs/ovo_qwen3vl8b_finetuned_1of8/worker_logs
)

if [[ -n "$LIMIT" ]]; then
  eval_cmd+=(--limit "$LIMIT")
fi

eval_profile="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
prompt = (cfg.get("prompt") or {}).get("profile", "")
postprocess = (cfg.get("postprocess") or {}).get("mode", "")
print(f"{prompt or '<unset>'} + {postprocess or '<unset>'}")
PY
)"
echo "[eval] path: $eval_profile"
echo "[eval] endpoints=$endpoints"
echo "[eval] output=outputs/ovo_qwen3vl8b_finetuned_1of8/results.jsonl"
"${eval_cmd[@]}"
