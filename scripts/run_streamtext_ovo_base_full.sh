#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${MODEL:-/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/streamtext/ovo_qwen3vl8b_base_448_full}"
ANNO_PATH="${ANNO_PATH:-/mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_new.json}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
WORKERS="${WORKERS:-32}"
RESUME="${RESUME:-0}"
RUNTIME_RESOLUTION="${RUNTIME_RESOLUTION:-448}"
MEMORY_WINDOW_SECONDS="${MEMORY_WINDOW_SECONDS:-180}"

echo "[streamtext-ovo] model=$MODEL"
echo "[streamtext-ovo] output=$OUTPUT_DIR resolution=$RUNTIME_RESOLUTION memory_window_seconds=$MEMORY_WINDOW_SECONDS workers=$WORKERS gpus=$GPUS"

OUTPUT_DIR="$OUTPUT_DIR" \
ANNO_PATH="$ANNO_PATH" \
GPUS="$GPUS" \
WORKERS="$WORKERS" \
RESUME="$RESUME" \
RUNTIME_RESOLUTION="$RUNTIME_RESOLUTION" \
MEMORY_WINDOW_SECONDS="$MEMORY_WINDOW_SECONDS" \
bash streamtext/run_ovo_8gpu_vllm.sh "$MODEL"
