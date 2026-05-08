#!/usr/bin/env bash
set -euo pipefail

# V4 OVO 1/8 eval on 8 local vLLM replicas.
# Usage:
#   ./scripts/run_ovo_8gpu_vllm_finetuned.sh
#   ./scripts/run_ovo_8gpu_vllm_finetuned.sh /path/to/vllm-compatible-finetuned-model

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${1:-/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm}"
PYTHON="/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python"
VLLM="/mmu_mllm_hdd/zhouhanshu/conda/envs/vllm/bin/vllm"
BASE_CONFIG="configs/batch_ovo_qwen3vl8b_finetuned_8gpu.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ovo_qwen3vl8b_finetuned_1of8}"
CONFIG="${OUTPUT_DIR}/run_config.yaml"
LIMIT=""  # Set to e.g. "8" for smoke; keep empty for the configured annotation file.
ALLOW_EXISTING_SERVERS="${ALLOW_EXISTING_SERVERS:-0}"

[[ -x "$PYTHON" ]] || PYTHON="python"
[[ -x "$VLLM" ]] || VLLM="vllm"

mkdir -p "$OUTPUT_DIR"
"$PYTHON" - "$BASE_CONFIG" "$CONFIG" "$OUTPUT_DIR" "$MODEL" <<'PY'
import sys
from pathlib import Path

import yaml

base_config, output_config, output_dir, model = sys.argv[1:5]
with open(base_config, encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}

cfg["result_output"] = f"{output_dir}/results.jsonl"
cfg.setdefault("trace", {})["output_root"] = f"{output_dir}/traces"
cfg.setdefault("trace", {})["experiment_name"] = ""
cfg.setdefault("batch", {})["output"] = f"{output_dir}/results.jsonl"
cfg.setdefault("batch", {})["worker_log_dir"] = f"{output_dir}/worker_logs"
cfg.setdefault("backend", {})["model"] = model

Path(output_config).parent.mkdir(parents=True, exist_ok=True)
with open(output_config, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, allow_unicode=True, sort_keys=False)
PY

mkdir -p \
  "$OUTPUT_DIR/vllm_logs" \
  "$OUTPUT_DIR/worker_logs" \
  "$OUTPUT_DIR/vllm_pids"

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
  log_path="$OUTPUT_DIR/vllm_logs/vllm_${port}.log"

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
  echo "$pid" > "$OUTPUT_DIR/vllm_pids/vllm_${port}.pid"
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
  --output "$OUTPUT_DIR/results.jsonl"
  --worker-log-dir "$OUTPUT_DIR/worker_logs"
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
echo "[eval] output=$OUTPUT_DIR/results.jsonl"
"${eval_cmd[@]}"
