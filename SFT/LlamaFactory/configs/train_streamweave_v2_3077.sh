#!/usr/bin/env bash
set -euo pipefail

LF_ROOT=/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory
CONDA_ENV=/mmu_mllm_hdd/zhouhanshu/conda/envs/llama_0425
CONFIG=configs/qwen3vl_8b_lora_sft_streamweave_v2_3077.yaml

cd "$LF_ROOT"

export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$LF_ROOT/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FORCE_TORCHRUN=1

exec "$CONDA_ENV/bin/llamafactory-cli" train "$CONFIG"
