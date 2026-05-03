# 2026-04-20 下一步计划摘要

## 背景

当前仓库已有一条已验证可运行的路线：

- 仓库：`/mmu_mllm_hdd/zhouhanshu/test2/test1/verl`
- 记录：`../2026-04-20-verl-vllm-bringup-summary.md`
- 已跑通环境：`/mmu_mllm_hdd/zhouhanshu/conda/envs/verl312_vllm0110_ray2492`
- 该路线特征：`vllm 0.11.0` + `sdpa`，不使用 `flash-attn`

接下来准备并行保留两条线：

1. 先把当前已跑通环境备份一份，避免后续试验污染
2. 新开一条基于 `verl_1126` 的环境线，在新的 `verl` 代码库中尝试 `Qwen3` + `flash-attn`

## 计划中的环境操作

### 1. 备份当前已跑通环境

源环境：

- `/mmu_mllm_hdd/zhouhanshu/conda/envs/verl312_vllm0110_ray2492`

目标环境名：

- `verl_qwen3vl_bak`

目标环境路径：

- `/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_qwen3vl_bak`

用途：

- 保存当前 `vllm + sdpa` 可运行基线

### 2. 从 `verl_1126` 克隆新的试验环境

源环境：

- `/share/zhangzejian/miniconda3/envs/verl_1126`

目标环境名：

- `verl_qwen3_bak`

目标环境路径：

- `/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_qwen3_bak`

用途：

- 作为新的 `verl` 代码库实验底座
- 目标方向是尝试跑 `Qwen3`
- 本轮重点是尝试启用 `flash-attn`

## 操作原则

- 两个环境都只做克隆备份，不在原环境上直接改
- 新实验尽量在新的 `verl` 代码库里进行，不污染当前已跑通仓库
- `flash-attn` 相关尝试单独放在 `verl_qwen3_bak` 这条线上
