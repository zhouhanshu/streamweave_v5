#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

FULL_ANNO_PATH="${ANNO_PATH:-/mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_new.json}"

ANNO_PATH="$FULL_ANNO_PATH" \
OUTPUT_DIR=outputs/ovo_qwen_sft_0513_full_eval \
  bash scripts/run_ovo_8gpu_vllm_finetuned.sh models/qwen_sft_0513

ANNO_PATH="$FULL_ANNO_PATH" \
OUTPUT_DIR=outputs/ovo_qwen3vl_sft_0516_step100_full_eval \
  bash scripts/run_ovo_8gpu_vllm_finetuned.sh models/qwen3vl_sft_0516_step100
