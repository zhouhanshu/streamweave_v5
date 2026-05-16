#!/usr/bin/env bash
set -euo pipefail

# VideoHallucer eval on N local vLLM replicas (one per GPU).
# Mirrors scripts/run_ovo_8gpu_vllm.sh but targets dataset3/eval_videohallucer_batch.py.
#
# Usage:
#   ./dataset3/scripts/run_videohallucer_8gpu.sh <model_path> <policy>
#     policy: anchor_delta | delta_only
#
# Examples:
#   ./dataset3/scripts/run_videohallucer_8gpu.sh /mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct anchor_delta
#   ./dataset3/scripts/run_videohallucer_8gpu.sh /mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct delta_only
#
# Env overrides:
#   GPUS="0 1 2 3 4 5 6 7"     which GPUs to use (default 0..7)
#   PORT_BASE=8000              first port; port = PORT_BASE + gpu_id
#   WORKERS=<N>                 worker count (default = 2 * GPU count)
#   OUTPUT_DIR=...              where to write results (default dataset3/outputs/<run_name>)
#   RUN_NAME=...                override run name (default vh_<policy>_<model_basename>_<date>)
#   RESUME=1                    reuse existing shard files; only run pending entries
#   ALLOW_EXISTING_SERVERS=1    don't fail when ports are already serving
#   LIMIT=<N>                   debug: cap entry count
#   TASK=<subset>               debug: only run a single subset
#   VLLM=<path>                 override vllm binary
#   PYTHON=<path>               override python binary

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${1:?model path required, e.g. /mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct}"
POLICY="${2:?policy required: anchor_delta | delta_only}"

case "$POLICY" in
  anchor_delta|delta_only) ;;
  *) echo "ERROR: policy must be anchor_delta or delta_only, got: $POLICY" >&2; exit 1 ;;
esac

BASE_CONFIG="dataset3/configs/eval_videohallucer_${POLICY}.yaml"
if [[ ! -f "$BASE_CONFIG" ]]; then
  echo "ERROR: config not found: $BASE_CONFIG" >&2
  exit 1
fi

PYTHON="${PYTHON:-/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python}"
VLLM="${VLLM:-/mmu_mllm_hdd/zhouhanshu/conda/envs/vllm/bin/vllm}"
[[ -x "$PYTHON" ]] || PYTHON="python"
[[ -x "$VLLM" ]] || VLLM="vllm"

GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
PORT_BASE="${PORT_BASE:-8000}"
RESUME="${RESUME:-0}"
ALLOW_EXISTING_SERVERS="${ALLOW_EXISTING_SERVERS:-0}"

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

ENDPOINT_ARRAY=()
for gpu in "${GPU_ARRAY[@]}"; do
  port=$((PORT_BASE + gpu))
  ENDPOINT_ARRAY+=("http://127.0.0.1:${port}/v1")
done

model_basename="$(basename "$MODEL")"
date_tag="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${RUN_NAME:-vh_${POLICY}_${model_basename}_${date_tag}}"
OUTPUT_DIR="${OUTPUT_DIR:-dataset3/outputs/${RUN_NAME}}"

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
      kill -9 -- "-$pid" >/dev/null 2>&1 || kill -9 "$pid" >/dev/null 2>&1 || true
    done
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

check_endpoint() {
  "$PYTHON" - "$1/models" >/dev/null 2>&1 <<'PY'
import sys, urllib.request
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
      echo "ERROR: endpoint already serving: $endpoint" >&2
      echo "Stop existing vLLM servers first, or set ALLOW_EXISTING_SERVERS=1." >&2
      exit 1
    fi
    echo "[server] reuse: gpu=$gpu endpoint=$endpoint"
    continue
  fi

  echo "[server] start gpu=$gpu port=$port log=$log_path"
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
  echo "[server] waiting: $endpoint"
  wait_for_endpoint "$endpoint"
done

eval_cmd=(
  "$PYTHON" dataset3/eval_videohallucer_batch.py
  --config "$BASE_CONFIG"
  --backend vllm
  --model "$MODEL"
  --endpoints "${ENDPOINT_ARRAY[@]}"
  --workers "$WORKERS"
  --output-name "$RUN_NAME"
  --worker-log-dir "$OUTPUT_DIR/worker_logs"
)

if [[ -n "${LIMIT:-}" ]]; then
  eval_cmd+=(--limit "$LIMIT")
fi
if [[ -n "${TASK:-}" ]]; then
  eval_cmd+=(--task "$TASK")
fi
if [[ "$RESUME" == "1" ]]; then
  eval_cmd+=(--resume)
fi

echo "[eval] policy=$POLICY"
echo "[eval] gpus=${GPU_ARRAY[*]}"
echo "[eval] workers=$WORKERS"
echo "[eval] endpoints=${ENDPOINT_ARRAY[*]}"
echo "[eval] output=$OUTPUT_DIR"
echo "[eval] resume=$RESUME"
"${eval_cmd[@]}"
