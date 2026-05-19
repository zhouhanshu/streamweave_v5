#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${MODEL:-/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-outputs/streamtext/streamingbench_qwen3vl8b_base_448}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
WORKERS="${WORKERS:-32}"
RESUME="${RESUME:-0}"
PORT_BASE="${PORT_BASE:-8000}"
ALLOW_EXISTING_SERVERS="${ALLOW_EXISTING_SERVERS:-0}"
LIMIT="${LIMIT:-}"
RUNTIME_RESOLUTION="${RUNTIME_RESOLUTION:-448}"
MEMORY_WINDOW_SECONDS="${MEMORY_WINDOW_SECONDS:-180}"
OMNI_TASK_FILTER="${OMNI_TASK_FILTER:-Misleading Context Understanding,Anomaly Context Understanding}"

run_split() {
  local split="$1"
  local output_dir="$2"
  local task_filter="${3:-}"

  echo "[streamtext-streamingbench] split=$split output=$output_dir"
  SPLIT="$split" \
  OUTPUT_DIR="$output_dir" \
  TASK_FILTER="$task_filter" \
  GPUS="$GPUS" \
  WORKERS="$WORKERS" \
  RESUME="$RESUME" \
  PORT_BASE="$PORT_BASE" \
  ALLOW_EXISTING_SERVERS="$ALLOW_EXISTING_SERVERS" \
  LIMIT="$LIMIT" \
  RUNTIME_RESOLUTION="$RUNTIME_RESOLUTION" \
  MEMORY_WINDOW_SECONDS="$MEMORY_WINDOW_SECONDS" \
  bash streamtext/run_streamingbench_8gpu_vllm.sh "$MODEL"
}

echo "[streamtext-streamingbench] model=$MODEL"
echo "[streamtext-streamingbench] prompt=text_memory_eval resolution=$RUNTIME_RESOLUTION workers=$WORKERS gpus=$GPUS"

run_split real "${OUTPUT_PREFIX}_real"
run_split sqa "${OUTPUT_PREFIX}_sqa"
run_split omni "${OUTPUT_PREFIX}_omni_mislead_anomaly" "$OMNI_TASK_FILTER"
run_split proactive "${OUTPUT_PREFIX}_proactive"
