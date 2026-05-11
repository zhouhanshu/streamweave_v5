#!/usr/bin/env bash
set -euo pipefail

cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5

/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python data_engine/sft/run_parallel_pipeline.py \
  --source dataset2 \
  --input dataset2/NeXTVideo \
  --raw-data-root dataset2 \
  --output-dir data_engine/sft/outputs/dataset2_nextvideo \
  --backend gemini \
  --model gemini-2.5-flash \
  --num-workers 64 \
  --frames-per-step 5 \
  --answer-step-rollouts 2

/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python data_engine/sft/run_parallel_pipeline.py \
  --source dataset2 \
  --input dataset2/LLaVA-Video-178K-PerceptionTest \
  --raw-data-root dataset2 \
  --output-dir data_engine/sft/outputs/dataset2_perceptiontest \
  --backend gemini \
  --model gemini-2.5-flash \
  --num-workers 64 \
  --frames-per-step 5 \
  --answer-step-rollouts 2

/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python data_engine/sft/run_parallel_pipeline.py \
  --source dataset2 \
  --input dataset2/LLaVA-Video-178K-YouTube \
  --raw-data-root dataset2 \
  --output-dir data_engine/sft/outputs/dataset2_youtube \
  --backend gemini \
  --model gemini-2.5-flash \
  --num-workers 64 \
  --frames-per-step 5 \
  --answer-step-rollouts 2

/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python data_engine/sft/run_parallel_pipeline.py \
  --source dataset2 \
  --input dataset2/TimeChat-Online-139K \
  --raw-data-root dataset2 \
  --output-dir data_engine/sft/outputs/dataset2_timechat \
  --backend gemini \
  --model gemini-2.5-flash \
  --num-workers 64 \
  --frames-per-step 5 \
  --answer-step-rollouts 2
