#!/usr/bin/env bash
set -euo pipefail

LF_ROOT=/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory
CONDA_ENV=/mmu_mllm_hdd/zhouhanshu/conda/envs/llama_0425
CONFIG=configs/qwen3vl_8b_full_sft_streamweave_0511_note.yaml
OUTPUT_DIR=saves/qwen3-vl-8b/full/streamweave_sft_0511_note
RESUME=${RESUME:-0}

cd "$LF_ROOT"

if [ "$RESUME" = "1" ]; then
  LATEST=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
  if [ -z "$LATEST" ]; then
    echo "RESUME=1 but no checkpoint-* found under $LF_ROOT/$OUTPUT_DIR" >&2
    exit 1
  fi
  TMP_CONFIG=$(mktemp --suffix=.yaml)
  trap 'rm -f "$TMP_CONFIG"' EXIT
  cp "$CONFIG" "$TMP_CONFIG"
  # Overwrite the resume_from_checkpoint line in the temp copy
  sed -i "s|^resume_from_checkpoint:.*|resume_from_checkpoint: $LATEST|" "$TMP_CONFIG"
  # Belt-and-suspenders: do NOT auto-rebuild cache on resume, even if yaml said true
  sed -i "s|^overwrite_cache:.*|overwrite_cache: false|" "$TMP_CONFIG"
  CONFIG="$TMP_CONFIG"
  echo "RESUME=1: resuming from $LATEST"
else
  if [ -d "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "Refusing to train: output dir already exists and is not empty: $LF_ROOT/$OUTPUT_DIR" >&2
    echo "Use 'RESUME=1 bash $0' to resume from the latest checkpoint." >&2
    exit 1
  fi
fi

export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$LF_ROOT/src"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FORCE_TORCHRUN=1

"$CONDA_ENV/bin/llamafactory-cli" train "$CONFIG"
