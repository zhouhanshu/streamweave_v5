#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODEL="${MODEL:-/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct}"
FULL_ANNO_PATH="${ANNO_PATH:-/mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_new.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ovo_qwen3vl8b_base_full_eval_no_abstain_hint}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
WORKERS="${WORKERS:-32}"
RESUME="${RESUME:-0}"
PROMPT_PROFILE="${PROMPT_PROFILE:-eval_no_abstain_hint}"
RUNTIME_RESOLUTION="${RUNTIME_RESOLUTION:-448}"

OUTPUT_DIR="$OUTPUT_DIR" \
ANNO_PATH="$FULL_ANNO_PATH" \
GPUS="$GPUS" \
WORKERS="$WORKERS" \
RESUME="$RESUME" \
PROMPT_PROFILE="$PROMPT_PROFILE" \
RUNTIME_RESOLUTION="$RUNTIME_RESOLUTION" \
bash scripts/run_ovo_8gpu_vllm.sh "$MODEL"
