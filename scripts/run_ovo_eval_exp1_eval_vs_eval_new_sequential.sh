#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

FULL_ANNO_PATH="${ANNO_PATH:-/mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_new.json}"

ANNO_PATH="$FULL_ANNO_PATH" \
OUTPUT_DIR=outputs/ovo_exp1_rl0511_step60_full_eval \
PROMPT_PROFILE=eval \
  bash scripts/run_ovo_8gpu_vllm_finetuned.sh models/exp1_rl0511_step60

ANNO_PATH="$FULL_ANNO_PATH" \
OUTPUT_DIR=outputs/ovo_exp1_rl0511_step60_full_eval_new \
PROMPT_PROFILE=eval_new \
  bash scripts/run_ovo_8gpu_vllm_finetuned.sh models/exp1_rl0511_step60
