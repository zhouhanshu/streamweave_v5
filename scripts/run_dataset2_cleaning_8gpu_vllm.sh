#!/usr/bin/env bash
set -euo pipefail

# Run dataset2 3x cleaning on local vLLM replicas.
# Usage:
#   bash scripts/run_dataset2_cleaning_8gpu_vllm.sh dataset2/NeXTVideo
#   bash scripts/run_dataset2_cleaning_8gpu_vllm.sh dataset2/NeXTVideo/by_video.jsonl
#   bash scripts/run_dataset2_cleaning_8gpu_vllm.sh dataset2/LLaVA-Video-178K-PerceptionTest /path/to/vllm-model
#   RESUME=1 bash scripts/run_dataset2_cleaning_8gpu_vllm.sh dataset2/NeXTVideo
#   GPUS="0 1 2 3" WORKERS=8 PORT_BASE=8100 bash scripts/run_dataset2_cleaning_8gpu_vllm.sh dataset2/NeXTVideo

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ "${1:-}" == "" ]]; then
  echo "Usage: bash scripts/run_dataset2_cleaning_8gpu_vllm.sh <dataset2/dataset_name> [model]" >&2
  exit 1
fi

DATASET_PATH="${1%/}"
DATASET_BASENAME="$(basename "$DATASET_PATH")"
if [[ -f "$DATASET_PATH" ]]; then
  DATASET_NAME="$(basename "$(dirname "$DATASET_PATH")")"
else
  DATASET_NAME="${DATASET_BASENAME%.jsonl}"
fi
MODEL="${2:-/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_sft_anchor_delta_step200_vllm}"
PYTHON="/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python"
VLLM="/mmu_mllm_hdd/zhouhanshu/conda/envs/vllm/bin/vllm"
BASE_CONFIG="configs/data_cleaning_dataset2.yaml"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/data_cleaning_0510/${DATASET_NAME}}"
CONFIG="${OUTPUT_DIR}/run_config.yaml"
ALLOW_EXISTING_SERVERS="${ALLOW_EXISTING_SERVERS:-0}"
RESUME="${RESUME:-0}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
PORT_BASE="${PORT_BASE:-8000}"
REPEATS="${REPEATS:-3}"
TEMPERATURE="${TEMPERATURE:-0.3}"
TOP_P="${TOP_P:-0.95}"
LIMIT="${LIMIT:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

[[ -x "$PYTHON" ]] || PYTHON="python"
[[ -x "$VLLM" ]] || VLLM="vllm"

if [[ ! -d "$DATASET_PATH" && ! -f "$DATASET_PATH" ]]; then
  echo "ERROR: dataset path does not exist: $DATASET_PATH" >&2
  exit 1
fi
if [[ -d "$DATASET_PATH" && ! -f "$DATASET_PATH/annotations.jsonl" ]]; then
  echo "ERROR: missing annotations.jsonl under: $DATASET_PATH" >&2
  exit 1
fi
if [[ -f "$DATASET_PATH" && "${DATASET_PATH##*.}" != "jsonl" ]]; then
  echo "ERROR: dataset file must be a .jsonl file: $DATASET_PATH" >&2
  exit 1
fi
if ! [[ "$PORT_BASE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: PORT_BASE must be a non-negative integer, got: $PORT_BASE" >&2
  exit 1
fi
if ! [[ "$REPEATS" =~ ^[0-9]+$ ]] || [[ "$REPEATS" -lt 1 ]]; then
  echo "ERROR: REPEATS must be a positive integer, got: $REPEATS" >&2
  exit 1
fi

read -r -a GPU_ARRAY <<< "${GPUS//,/ }"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
  echo "ERROR: GPUS must contain at least one GPU id." >&2
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
"$PYTHON" - "$CONFIG_SOURCE" "$CONFIG" "$OUTPUT_DIR" "$MODEL" "$DATASET_PATH" "$DATASET_NAME" "$endpoints" "$WORKERS" "$REPEATS" "$TEMPERATURE" "$TOP_P" <<'PY'
import sys
from pathlib import Path

import yaml

(
    config_source,
    output_config,
    output_dir,
    model,
    dataset_path,
    dataset_name,
    endpoints_csv,
    workers,
    repeats,
    temperature,
    top_p,
) = sys.argv[1:12]

with open(config_source, encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}

endpoints = [item for item in endpoints_csv.split(",") if item]
dataset_path_obj = Path(dataset_path)
if dataset_path_obj.suffix == ".jsonl":
    dataset_root = dataset_path_obj.parent.parent
    dataset_config_name = dataset_path_obj.parent.name
else:
    dataset_root = dataset_path_obj.parent
    dataset_config_name = dataset_name
cfg["benchmark"] = "dataset2"
cfg["result_output"] = f"{output_dir}/annotations_with_pass_count.jsonl"
cfg.setdefault("trace", {})["output_root"] = f"{output_dir}/traces"
cfg.setdefault("trace", {})["experiment_name"] = ""
cfg.setdefault("batch", {})["output"] = f"{output_dir}/annotations_with_pass_count.jsonl"
cfg["batch"]["worker_log_dir"] = f"{output_dir}/worker_logs"
cfg["batch"]["endpoints"] = endpoints
cfg["batch"]["workers"] = int(workers)
cfg.setdefault("backend", {})["model"] = model
cfg["backend"]["temperature"] = float(temperature)
cfg["backend"]["top_p"] = float(top_p)
if endpoints:
    cfg["backend"]["base_url"] = endpoints[0]
cfg.setdefault("dataset", {})["dataset_root"] = str(dataset_root)
cfg["dataset"]["dataset_name"] = dataset_config_name
cfg["dataset"]["video_root"] = ""
cfg["dataset"]["frame_id_base"] = 0
cfg["dataset"]["image_ext"] = "jpg"
cfg.setdefault("benchmark_args", {})["dataset_path"] = dataset_path
cfg["benchmark_args"]["require_options"] = True
cfg["benchmark_args"]["limit"] = 0

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
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    >"$log_path" 2>&1 &

  pid=$!
  started_pids+=("$pid")
  echo "$pid" > "$OUTPUT_DIR/vllm_pids/vllm_${port}.pid"
done

for endpoint in "${ENDPOINT_ARRAY[@]}"; do
  echo "[server] waiting for $endpoint"
  wait_for_endpoint "$endpoint"
done

clean_cmd=(
  "$PYTHON" evaluation/run_data_cleaning_3x.py
  --config "$CONFIG"
  --benchmark dataset2
  --dataset-path "$DATASET_PATH"
  --backend vllm
  --model "$MODEL"
  --endpoints "$endpoints"
  --workers "$WORKERS"
  --repeats "$REPEATS"
  --temperature "$TEMPERATURE"
  --top-p "$TOP_P"
  --output "$OUTPUT_DIR/annotations_with_pass_count.jsonl"
  --worker-log-dir "$OUTPUT_DIR/worker_logs"
)

if [[ -n "$LIMIT" ]]; then
  clean_cmd+=(--limit "$LIMIT")
fi
if [[ "$RESUME" == "1" ]]; then
  clean_cmd+=(--resume)
fi

echo "[clean] dataset=$DATASET_PATH"
echo "[clean] model=$MODEL"
echo "[clean] gpus=${GPU_ARRAY[*]}"
echo "[clean] workers=$WORKERS repeats=$REPEATS temperature=$TEMPERATURE top_p=$TOP_P"
echo "[clean] endpoints=$endpoints"
echo "[clean] output=$OUTPUT_DIR/annotations_with_pass_count.jsonl"
echo "[clean] traces=$OUTPUT_DIR/traces"
echo "[clean] resume=$RESUME"
"${clean_cmd[@]}"
