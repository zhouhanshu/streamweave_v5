#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="models/qwen3vl_rl_exp10_step20"
OUTPUT_PREFIX="outputs/streamingbench_qwen3vl_rl_exp10_step20_768"
GPUS="0 1 2 3 4 5 6 7"
WORKERS="32"
RESUME="${RESUME:-0}"
PORT_BASE="8000"
ALLOW_EXISTING_SERVERS="0"
LIMIT=""
PROMPT_PROFILE="eval"
RUNTIME_RESOLUTION="768"
OMNI_TASK_FILTER="Misleading Context Understanding,Anomaly Context Understanding"

run_split() {
  local split="$1"
  local output_dir="$2"
  local task_filter="${3:-}"

  echo "[streamingbench] split=$split output=$output_dir"
  SPLIT="$split" \
  OUTPUT_DIR="$output_dir" \
  TASK_FILTER="$task_filter" \
  GPUS="$GPUS" \
  WORKERS="$WORKERS" \
  RESUME="$RESUME" \
  PORT_BASE="$PORT_BASE" \
  ALLOW_EXISTING_SERVERS="$ALLOW_EXISTING_SERVERS" \
  LIMIT="$LIMIT" \
  PROMPT_PROFILE="$PROMPT_PROFILE" \
  RUNTIME_RESOLUTION="$RUNTIME_RESOLUTION" \
  bash scripts/run_streamingbench_8gpu_vllm.sh "$MODEL"
}

echo "[streamingbench] model=$MODEL"
echo "[streamingbench] prompt=$PROMPT_PROFILE resolution=$RUNTIME_RESOLUTION workers=$WORKERS gpus=$GPUS"

run_split real "${OUTPUT_PREFIX}_real"
run_split sqa "${OUTPUT_PREFIX}_sqa"
run_split omni "${OUTPUT_PREFIX}_omni_mislead_anomaly" "$OMNI_TASK_FILTER"
run_split proactive "${OUTPUT_PREFIX}_proactive"
