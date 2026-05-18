#!/usr/bin/env bash

# PhoStream
python /mmu_mllm_hdd/zhouhanshu/test/download_repo.py \
  --source hf --repo-type dataset \
  --repo-id lucky-lance/PhoStream \
  --output-dir dataset3/raw/phostream

# LVBench
python /mmu_mllm_hdd/zhouhanshu/test/download_repo.py \
  --source hf --repo-type dataset \
  --repo-id THUDM/LVBench \
  --output-dir dataset3/raw/lvbench

# Video-MME
python /mmu_mllm_hdd/zhouhanshu/test/download_repo.py \
  --source hf --repo-type dataset \
  --repo-id lmms-lab/Video-MME \
  --output-dir dataset3/raw/video_mme
