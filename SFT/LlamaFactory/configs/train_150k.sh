#!/usr/bin/env bash
set -euo pipefail

LF_ROOT=/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory
CONDA_ENV=/mmu_mllm_hdd/zhouhanshu/conda/envs/llama_0425
CONFIG=configs/qwen3vl_8b_full_sft_streamweave_0511_150k_6to4.yaml
OUTPUT_DIR=saves/qwen3-vl-8b/full/streamweave_sft_0511_150k_6to4

cd "$LF_ROOT"

if [ -d "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
  echo "Refusing to train: output dir already exists and is not empty: $LF_ROOT/$OUTPUT_DIR" >&2
  exit 1
fi

export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$LF_ROOT/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FORCE_TORCHRUN=1

exec "$CONDA_ENV/bin/llamafactory-cli" train "$CONFIG"
