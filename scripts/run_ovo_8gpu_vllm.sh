#!/usr/bin/env bash
set -euo pipefail

# V4 OVO eval on configurable local vLLM replicas.
# Usage:
#   ./scripts/run_ovo_8gpu_vllm.sh
#   ./scripts/run_ovo_8gpu_vllm.sh /path/to/Qwen3-VL-8B-Instruct
#   RESUME=1 ./scripts/run_ovo_8gpu_vllm.sh /path/to/Qwen3-VL-8B-Instruct
#   GPUS="2 3 4 5 6 7" RESUME=1 ./scripts/run_ovo_8gpu_vllm.sh /path/to/Qwen3-VL-8B-Instruct

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${1:-/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct}"
PYTHON="/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python"
VLLM="/mmu_mllm_hdd/zhouhanshu/conda/envs/vllm/bin/vllm"
BASE_CONFIG="${BASE_CONFIG:-configs/batch_ovo_qwen3vl8b_8gpu_full.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ovo_qwen3vl8b_base_full}"
ANNO_PATH="${ANNO_PATH:-}"
PROMPT_PROFILE="${PROMPT_PROFILE:-}"
RUNTIME_RESOLUTION="${RUNTIME_RESOLUTION:-}"
MEMORY_WINDOW_SECONDS="${MEMORY_WINDOW_SECONDS:-}"
CONFIG="${OUTPUT_DIR}/run_config.yaml"
LIMIT=""  # Set to e.g. "8" for smoke; keep empty for the configured annotation file.
ALLOW_EXISTING_SERVERS="${ALLOW_EXISTING_SERVERS:-0}"
RESUME="${RESUME:-0}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
PORT_BASE="${PORT_BASE:-8000}"

[[ -x "$PYTHON" ]] || PYTHON="python"
[[ -x "$VLLM" ]] || VLLM="vllm"

read -r -a GPU_ARRAY <<< "${GPUS//,/ }"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "ERROR: GPUS must contain at least one GPU id." >&2
  exit 1
fi
if ! [[ "$PORT_BASE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: PORT_BASE must be a non-negative integer, got: $PORT_BASE" >&2
  exit 1
fi
WORKERS="${WORKERS:-$((${#GPU_ARRAY[@]} * 2))}"
if ! [[ "$WORKERS" =~ ^[0-9]+$ ]] || [[ "$WORKERS" -lt 1 ]]; then
  echo "ERROR: WORKERS must be a positive integer, got: $WORKERS" >&2
  exit 1
fi

ENDPOINT_ARRAY=()
for gpu in "${GPU_ARRAY[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU id must be a non-negative integer, got: $gpu" >&2
    exit 1
  fi
  port=$((PORT_BASE + gpu))
  ENDPOINT_ARRAY+=("http://127.0.0.1:${port}/v1")
done
endpoints="$(IFS=,; echo "${ENDPOINT_ARRAY[*]}")"

CONFIG_SOURCE="$BASE_CONFIG"
if [[ "$RESUME" == "1" && -f "$CONFIG" ]]; then
  CONFIG_SOURCE="$CONFIG"
  echo "[config] resume using existing config: $CONFIG"
fi

mkdir -p "$OUTPUT_DIR"
"$PYTHON" - "$CONFIG_SOURCE" "$CONFIG" "$OUTPUT_DIR" "$MODEL" "$ANNO_PATH" "$endpoints" "$WORKERS" "$PROMPT_PROFILE" "$RUNTIME_RESOLUTION" "$MEMORY_WINDOW_SECONDS" <<'PY'
import sys
from pathlib import Path

import yaml

config_source, output_config, output_dir, model, anno_path, endpoints_csv, workers, prompt_profile, runtime_resolution, memory_window_seconds = sys.argv[1:11]
with open(config_source, encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}

endpoints = [item for item in endpoints_csv.split(",") if item]
cfg["result_output"] = f"{output_dir}/results.jsonl"
cfg.setdefault("trace", {})["output_root"] = f"{output_dir}/traces"
cfg.setdefault("trace", {})["experiment_name"] = ""
cfg.setdefault("batch", {})["output"] = f"{output_dir}/results.jsonl"
cfg.setdefault("batch", {})["worker_log_dir"] = f"{output_dir}/worker_logs"
cfg["batch"]["endpoints"] = endpoints
cfg["batch"]["workers"] = int(workers)
cfg.setdefault("backend", {})["model"] = model
if endpoints:
    cfg["backend"]["base_url"] = endpoints[0]
if anno_path:
    cfg.setdefault("benchmark_args", {})["anno_path"] = anno_path
if prompt_profile:
    cfg.setdefault("prompt", {})["profile"] = prompt_profile
if runtime_resolution:
    cfg.setdefault("runtime", {})["resolution"] = int(runtime_resolution)
if memory_window_seconds:
    cfg.setdefault("memory", {})["window_seconds"] = float(memory_window_seconds)

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

for gpu in "${GPU_ARRAY[@]}"; do
  port=$((PORT_BASE + gpu))
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

for endpoint in "${ENDPOINT_ARRAY[@]}"; do
  echo "[server] waiting for $endpoint"
  wait_for_endpoint "$endpoint"
done

eval_cmd=(
  "$PYTHON" evaluation/eval_batch.py
  --config "$CONFIG"
  --benchmark ovo
  --backend vllm
  --model "$MODEL"
  --endpoints "$endpoints"
  --workers "$WORKERS"
  --output "$OUTPUT_DIR/results.jsonl"
  --worker-log-dir "$OUTPUT_DIR/worker_logs"
)

if [[ -n "$LIMIT" ]]; then
  eval_cmd+=(--limit "$LIMIT")
fi
if [[ "$RESUME" == "1" ]]; then
  eval_cmd+=(--resume)
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
echo "[eval] gpus=${GPU_ARRAY[*]}"
echo "[eval] workers=$WORKERS"
echo "[eval] endpoints=$endpoints"
echo "[eval] output=$OUTPUT_DIR/results.jsonl"
echo "[eval] resume=$RESUME"
"${eval_cmd[@]}"
