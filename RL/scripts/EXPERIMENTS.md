# StreamWeave RL Experiment Record

This file tracks RL launch scripts, run names, and the main config deltas for each
experiment. Keep one launch script per experiment so old runs remain reproducible.

Before starting a new run:

1. Create a new launch script under `RL/scripts/`.
2. Set a unique `RUN_NAME`.
3. Confirm `trainer.default_local_dir` points to `RL/outputs/runs/${RUN_NAME}/checkpoints`.
4. Add an entry to the table below before launch.
5. After launch, fill in the PID, start time, and any observed issues.

## Runs

| Date UTC | Status | PID | Run name | Launch script | Base config | Dataset | Source model | Algorithm / key deltas | Judge | Output dir | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-12 | running-old-config | 3927597 | Experiment 1: `grpo_rl0511_8gpu_judge` | `RL/scripts/train_grpo.sh` | `RL/configs/streamweave_stepwise.yaml` | `dataset2/rl_0511.jsonl` | `models/qwen3vl8b_streamweave_sft_answered_full_anchor_delta_init_anchor_step200_vllm` | `streamweave_stepwise_traj_grpo`; `rollout.n=16`; DAPO filter enabled; `gen_batch_size=16` in the live command | Gemini judge enabled | `RL/outputs/runs/grpo_rl0511_8gpu_judge` | Demo experiment. Existing process uses an old config. Curves: <https://swanlab.cn/@zhs/streamweave_rl/runs/gmtjhm9u05m58326zmpe6/chart>. Do not stop or reuse this run name for new experiments. |
| 2026-05-12 | ready | TBD | Experiment 2: `exp2_rlmlr` | `RL/scripts/train_exp2_rlmlr.sh` | `RL/configs/streamweave_stepwise.yaml` | `dataset2/rl_0512_train.jsonl`; val `dataset2/rl_0512_val.jsonl` | `models/qwen3vl8b_streamweave_sft_answered_full_anchor_delta_init_anchor_step200_vllm` | `streamweave_stepwise_rlmlr`; `train_batch_size=16`; `gen_batch_size=16`; `rollout.n=8`; `outcome=success_score`; `state=1.0*step_score+0.2*format_score`; DAPO filter disabled; validation every 20 steps | Gemini judge enabled | `RL/outputs/runs/exp2_rlmlr` | Uses `rl_0512` deterministic video-level split: train 1420 rows, val 80 rows. Fixed output dir; do not start the same script concurrently on multiple machines sharing this disk. |
| 2026-05-14 | stopped-after-step20 | manual stop | Experiment 3: `exp3` | `RL/scripts/train_exp3.sh` | `RL/configs/streamweave_stepwise.yaml` | `dataset2/rl_exp3.jsonl`; val same file | `models/qwen3vl8b_streamweave_sft_answered_full_anchor_delta_init_anchor_step200_vllm` | `streamweave_stepwise_grppo`; `train_batch_size=16`; `gen_batch_size=16`; `rollout.n=8`; `answer_decay=0.4`; final advantage `1.0*step_adv + 0.3*answer_adv`; `grppo_min_std=0.07`; step filter enabled with `filter_min_std=0.04`; `process_weight=1.0`; `format_weight=0.1`; validation every 30 steps | Gemini judge enabled, `streamweave_grppo_judge_v1` | `RL/outputs/runs/exp3` | Stopped manually after `global_step_20`; checkpoint available. Curves were close to exp4/exp5, so no immediate evaluation. |
| 2026-05-14 | stopped-evaluating-step20 | manual stop | Experiment 4: `exp4` | `RL/scripts/train_exp4.sh` | `RL/configs/streamweave_stepwise.yaml` | `dataset2/rl_exp3.jsonl`; val same file | `models/qwen3vl8b_streamweave_sft_answered_full_anchor_delta_init_anchor_step200_vllm` | Same as exp3 except final advantage `1.0*step_adv + 0.5*answer_adv`; `filter_min_std=0.04` | Gemini judge enabled, `streamweave_grppo_judge_v1` | `RL/outputs/runs/exp4` | Stopped manually after `global_step_20`; exported to `models/exp4_grppo_aw05_step20`; OVO 1/8 evaluation directory created at `outputs/ovo_exp4_grppo_aw05_step20_1of8`. |
| 2026-05-14 | running | TBD | Experiment 5: `exp5` | `RL/scripts/train_exp5.sh` | `RL/configs/streamweave_stepwise.yaml` | `dataset2/rl_exp3.jsonl`; val same file | `models/qwen3vl8b_streamweave_sft_answered_full_anchor_delta_init_anchor_step200_vllm` | Same as exp3/4 except final advantage `1.0*step_adv + 0.7*answer_adv`; `filter_min_std=0.04` | Gemini judge enabled, `streamweave_grppo_judge_v1` | `RL/outputs/runs/exp5` | Continue running while exp4 step20 is evaluated; latest checkpoint observed at `global_step_20`. |
| 2026-05-12 | planned | TBD | `TBD` | `RL/scripts/TBD.sh` | `RL/configs/streamweave_stepwise.yaml` | `TBD` | `TBD` | `TBD` | `TBD` | `RL/outputs/runs/TBD` | Fill this row when the next launch script is created. |

## Experiment 3-5: GRPPO Answer-Weight Sweep

These runs share the same dataset, model initialization, judge prompt, and GRPPO estimator. They differ
only in the final answer-advantage weight:

| Run | Script | Output dir | Final advantage | Step reward | Answer credit | Filter |
| --- | --- | --- | --- | --- | --- | --- |
| `exp3` | `RL/scripts/train_exp3.sh` | `RL/outputs/runs/exp3` | `1.0*step_adv + 0.3*answer_adv` | `(1.0*judge_step + 0.1*format_score)/1.1` | `answer_credit_t = answer_reward_t + 0.4*answer_credit_{t+1}` | step-level GRPPO filter enabled, `filter_min_std=0.04` |
| `exp4` | `RL/scripts/train_exp4.sh` | `RL/outputs/runs/exp4` | `1.0*step_adv + 0.5*answer_adv` | same as exp3 | same as exp3 | same as exp3 |
| `exp5` | `RL/scripts/train_exp5.sh` | `RL/outputs/runs/exp5` | `1.0*step_adv + 0.7*answer_adv` | same as exp3 | same as exp3 | same as exp3 |

Common launch config for exp3/exp4/exp5:

Data and runtime overrides:

- Base config: `--config-name=streamweave_stepwise`.
- Dataset root: `data.streamweave.dataset.dataset_root=dataset2`.
- Dataset name: `data.streamweave.dataset_name=mixed_rl_exp3` and
  `data.streamweave.dataset.dataset_name=mixed_rl_exp3`.
- Train file: `data.train_files=dataset2/rl_exp3.jsonl`.
- Validation file: `data.val_files=dataset2/rl_exp3.jsonl`.
- Batch sizes: `data.train_batch_size=16`, `+data.gen_batch_size=16`,
  `data.val_batch_size=16`.
- Lengths: `data.max_prompt_length=6144`, `data.max_response_length=2048`.
- Stream runtime override: `data.streamweave.runtime.max_steps=0`; other stream runtime settings
  come from `RL/configs/streamweave_stepwise.yaml`.

Model initialization:

- Source model: `models/qwen3vl8b_streamweave_sft_answered_full_anchor_delta_init_anchor_step200_vllm`.
- Prepared model config: `RL/outputs/runs/{exp3,exp4,exp5}/model_config`.
- Actor model path override: `actor_rollout_ref.model.path=${RUN_DIR}/model_config`.

Actor, ref, and rollout:

- Actor optimizer: `actor_rollout_ref.actor.optim.lr=1e-5`.
- PPO batch: `actor_rollout_ref.actor.ppo_mini_batch_size=16`,
  `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8`.
- Actor token cap: `actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768`.
- PPO clip: `actor_rollout_ref.actor.clip_ratio_low=0.2`,
  `actor_rollout_ref.actor.clip_ratio_high=0.28`.
- Ref logprob: `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8`,
  `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768`.
- Rollout sampling: `actor_rollout_ref.rollout.n=8`, `temperature=1.0`, `top_p=0.95`.
- Rollout logprob: `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8`,
  `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768`.
- vLLM limits: `actor_rollout_ref.rollout.gpu_memory_utilization=0.7`,
  `actor_rollout_ref.rollout.max_model_len=8192`,
  `actor_rollout_ref.rollout.max_num_batched_tokens=65536`,
  `actor_rollout_ref.rollout.max_num_seqs=2048`.
- Agent workers: `actor_rollout_ref.rollout.agent.num_workers=32`.
- Validation generation: `actor_rollout_ref.rollout.val_kwargs.n=1`,
  `do_sample=false`, `temperature=0`.

GRPPO, reward, and judge:

- Estimator: `algorithm.adv_estimator=streamweave_stepwise_grppo`.
- KL/reward path: `algorithm.use_kl_in_reward=false`; `critic.enable=false`.
- Legacy DAPO filter disabled: `algorithm.filter_groups.enable=false`.
- GRPPO answer credit: `+algorithm.grppo_answer_decay=0.4`.
- Final advantage weights: exp3 uses `+algorithm.grppo_answer_weight=0.3`, exp4 uses `0.5`,
  exp5 uses `0.7`; all use `+algorithm.grppo_step_weight=1.0`.
- Component normalization: `+algorithm.grppo_norm_by_std=true`,
  `+algorithm.grppo_min_std=0.07`.
- Step-level GRPPO filter: `+algorithm.grppo_filter_groups.enable=true`,
  `+algorithm.grppo_filter_groups.min_std=0.04`.
- GRPPO step reward mix: `+data.streamweave.reward.grppo_process_weight=1.0`,
  `+data.streamweave.reward.grppo_format_weight=0.1`, so
  `grppo_step_reward=(1.0*grppo_judge_step_reward + 0.1*grppo_format_score)/1.1`.
- Judge: `data.streamweave.reward.judge.enable=true`,
  `data.streamweave.reward.judge.backend=gemini`,
  `+data.streamweave.reward.judge.prompt_version=streamweave_grppo_judge_v1`.
- Default Gemini credentials path in scripts:
  `/mmu_ssd3/group_lisize/hetu/xujia10/joint_tags/scripts/gemini_client/config.json`,
  unless `GOOGLE_APPLICATION_CREDENTIALS` is already set.

Trainer and Ray:

- Trainer worker path: `trainer.use_legacy_worker_impl=enable`, `trainer.critic_warmup=0`.
- Logging: `trainer.logger=["console","swanlab"]`,
  `trainer.project_name=streamweave_rl`, `trainer.experiment_name={exp3,exp4,exp5}`.
- Checkpoint dir: `trainer.default_local_dir=RL/outputs/runs/{exp3,exp4,exp5}/checkpoints`.
- Resume: `trainer.resume_mode=auto`.
- Hardware: `trainer.n_gpus_per_node=8`, `trainer.nnodes=1`.
- Ray: `ray_kwargs.ray_init.num_cpus=64`,
  `+ray_kwargs.ray_init.object_store_memory=40000000000`,
  `+ray_kwargs.ray_init.include_dashboard=false`.
- Ray temp dirs: exp3 uses `/tmp/swray_$$`; exp4 uses `/tmp/swray_exp4_$$`;
  exp5 uses `/tmp/swray_exp5_$$`.
- Schedule: `trainer.save_freq=20`, `trainer.test_freq=30`, `trainer.total_epochs=2`.

Script environment and artifacts:

- `CUDA_VISIBLE_DEVICES` defaults to `0,1,2,3,4,5,6,7`.
- `STREAMWEAVE_CONSOLE_METRICS=compact`, `TOKENIZERS_PARALLELISM=false`,
  `RAY_DEDUP_LOGS=0`, `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0`.
- `SWANLAB_LOG_DIR=RL/outputs/runs/{exp3,exp4,exp5}/swanlab`.
- Trace defaults: `STREAMWEAVE_TRACE_FIRST_ROLLOUT=1`,
  `STREAMWEAVE_TRACE_SAMPLE_EVERY=64`.
- Each script stops Ray before launch unless `STREAMWEAVE_ALLOW_EXISTING_RL=1` is set.
- On exit, each script records `exit_code.txt`, `git_status.txt`, `python_version.txt`,
  `pip_list.txt`, and Ray logs under the run directory when available.

### 2026-05-14 Evaluation Decision

Initial SwanLab curves showed exp3, exp4, and exp5 are close. exp3 and exp4 were stopped manually;
exp5 remains running. Start evaluation from exp4 because it is the middle answer-weight setting
(`answer_weight=0.5`) and already has checkpoint `global_step_20`.

Progress snapshot:

- `exp3`: manually stopped after `global_step_20`; checkpoint exists at
  `RL/outputs/runs/exp3/checkpoints/global_step_20/actor`.
- `exp4`: manually stopped after `global_step_20`; checkpoint exists at
  `RL/outputs/runs/exp4/checkpoints/global_step_20/actor`.
- `exp5`: continues running; latest observed checkpoint is
  `RL/outputs/runs/exp5/checkpoints/global_step_20/actor`.
- Exp4 step20 export completed: `models/exp4_grppo_aw05_step20` contains sharded safetensors.
- Exp4 OVO 1/8 evaluation directory exists at `outputs/ovo_exp4_grppo_aw05_step20_1of8`;
  `results.jsonl` had not been observed yet when this note was written.

Export exp4 step 20 FSDP actor checkpoint to a vLLM/HF-loadable model:

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
PYTHONPATH=RL/verl:RL:. /mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/python \
  -m verl.model_merger merge \
  --backend fsdp \
  --local_dir RL/outputs/runs/exp4/checkpoints/global_step_20/actor \
  --target_dir models/exp4_grppo_aw05_step20 \
  --trust-remote-code
```

Run OVO-Bench 1/8 sanity evaluation:

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
OUTPUT_DIR=outputs/ovo_exp4_grppo_aw05_step20_1of8 \
  bash scripts/run_ovo_8gpu_vllm_finetuned.sh models/exp4_grppo_aw05_step20
```

Run OVO-Bench full evaluation:

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
ANNO_PATH=/mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_new.json \
OUTPUT_DIR=outputs/ovo_exp4_grppo_aw05_step20_full \
  bash scripts/run_ovo_8gpu_vllm_finetuned.sh models/exp4_grppo_aw05_step20
```

Do not launch the OVO vLLM evaluation on the same machine as a live exp5 run unless the GPUs are
intentionally reserved for evaluation; the script starts eight local vLLM servers on ports 8000-8007.

## Experiment 1: `grpo_rl0511_8gpu_judge`

第一次完整跑通的 GRPO 训练；step 60 已导出并完成 OVO 1/8 + full 回评。Full 总分 `59.87`，HLD 单项崩盘到 `24.73`，**该 run 已定性为负诊断样本**，不作为后续训练 base。当前 V5/Qwen full SOTA 仍是 GRPO0509 `63.90`。

### 元信息

- SwanLab 在线曲线：<https://swanlab.cn/@zhs/streamweave_rl/runs/gmtjhm9u05m58326zmpe6/chart>
- 训练记录时间：2026-05-12 UTC；full 回评与 trace 分析：2026-05-13。
- 记录时进程状态：running，PID `3927597`。
- 配置来源：在线进程命令行 + `RL/configs/streamweave_stepwise.yaml` 默认值合并而成。
- 注意：在线命令是 `+data.gen_batch_size=16`，磁盘 `RL/scripts/train_grpo.sh` 已漂移到 `+data.gen_batch_size=8`，复现请以这里记录的为准。

### 训练配置

数据与初始化：

- Base config：`--config-name=streamweave_stepwise`
- Train/val files：`data.train_files=dataset2/rl_0511.jsonl`，`data.val_files=dataset2/rl_0511.jsonl`，`data.streamweave.dataset_name=mixed_rl_0511`，`data.streamweave.dataset.dataset_root=dataset2`
- Source model：`models/qwen3vl8b_streamweave_sft_answered_full_anchor_delta_init_anchor_step200_vllm`
- 训练用 model_config：`RL/outputs/runs/grpo_rl0511_8gpu_judge/model_config`

Batch 与长度：

- `data.train_batch_size=32`，`+data.gen_batch_size=16`，`data.val_batch_size=4`
- `data.max_prompt_length=6144`，`data.max_response_length=2048`
- Runtime：`sample_fps=1.0`，`frames_per_step=5`，`max_frames=0`，`max_steps=0`，`resolution=336`

Actor / Ref / Rollout：

- Actor：`lr=1e-5`，`ppo_mini_batch_size=32`，`ppo_micro_batch_size_per_gpu=8`，`ppo_max_token_len_per_gpu=32768`，`clip_ratio_low=0.2`，`clip_ratio_high=0.28`
- Ref logprob：`ref.log_prob_micro_batch_size_per_gpu=8`，`ref.log_prob_max_token_len_per_gpu=32768`
- Rollout：`rollout.n=16`，`temperature=1.0`，`top_p=0.95`，`gpu_memory_utilization=0.7`，`max_model_len=8192`，`max_num_batched_tokens=65536`，`max_num_seqs=2048`，`agent.num_workers=32`

Reward / Judge / Algorithm：

- Reward 权重：`w_format=0.1`，`w_step=0.1`，`w_success=0.8`，`format_mode=valid`，`success_mode=dataset`，note-frequency reward 启用
- Judge：`data.streamweave.reward.judge.enable=true`，Gemini `gemini-2.5-flash`，`judge_weight=1.0`
- Algorithm：`algorithm.adv_estimator=streamweave_stepwise_traj_grpo`，`algorithm.use_kl_in_reward=false`，`algorithm.filter_groups.enable=true`，`actor_rollout_ref.actor.use_kl_loss=false`
- Critic：`critic.enable=false`

Trainer / Ray / 输出：

- Trainer：`trainer.logger=["console","swanlab"]`，`trainer.project_name=streamweave_rl`，`trainer.experiment_name=grpo_rl0511_8gpu_judge`，`trainer.resume_mode=auto`，`trainer.n_gpus_per_node=8`，`trainer.nnodes=1`，`trainer.save_freq=20`，`trainer.test_freq=-1`，`trainer.total_epochs=2`
- Ray：`num_cpus=64`，`object_store_memory=40000000000`，`include_dashboard=false`，`_temp_dir=/tmp/swray_3927346`
- Output dir：`RL/outputs/runs/grpo_rl0511_8gpu_judge`

### 产物路径

- Checkpoints：`RL/outputs/runs/grpo_rl0511_8gpu_judge/checkpoints/global_step_{20,40,60}`
- Latest 标记：`RL/outputs/runs/grpo_rl0511_8gpu_judge/checkpoints/latest_checkpointed_iteration.txt`
- 训练日志：`RL/outputs/runs/grpo_rl0511_8gpu_judge/train.log`
- SwanLab 本地：`RL/outputs/runs/grpo_rl0511_8gpu_judge/swanlab/run-20260511_170803-gmtjhm9u05m58326zmpe6`
- 本地解析曲线脚本与图：`RL/scripts/log/plot_traj_metrics.py`、`RL/scripts/log/experiment1_traj_score_success.png`
- HF 导出模型：`models/exp1_rl0511_step60`（从 `global_step_60/actor` 导出）
- OVO 1/8 评测：`outputs/ovo_exp1_rl0511_step60_1of8`
- OVO full 评测：`outputs/ovo_exp1_rl0511_step60_full`

### 评测复现命令

step 60 导出：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
PYTHONPATH=RL/verl:RL:. /mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/python \
  -m verl.model_merger merge \
  --backend fsdp \
  --local_dir RL/outputs/runs/grpo_rl0511_8gpu_judge/checkpoints/global_step_60/actor \
  --target_dir models/exp1_rl0511_step60 \
  --trust-remote-code
```

OVO 1/8 回评：

```bash
OUTPUT_DIR=outputs/ovo_exp1_rl0511_step60_1of8 \
  bash scripts/run_ovo_8gpu_vllm_finetuned.sh models/exp1_rl0511_step60
```

OVO full 回评：

```bash
ANNO_PATH=/mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_new.json \
OUTPUT_DIR=outputs/ovo_exp1_rl0511_step60_full \
  bash scripts/run_ovo_8gpu_vllm_finetuned.sh models/exp1_rl0511_step60
```

### 训练阶段观察

- 训练初期 reward 上升缓慢、噪声大，但整体在升。
- 每步约 30 分钟，主要瓶颈在 `update_actor` 和 `old_log_prob`。
- 当时怀疑慢 + 噪声大是 reward attribution 太稀疏导致；full 评测之后定性根因是 reward 的**结构性偏差**（详见根因诊断），不是稀疏度。
- 下一版 run 必须降 `rollout.n`、降 `w_success`、启用 KL term。

### OVO Full 回评（2026-05-13）

主结果：

| Category | Task | Accuracy |
| --- | --- | ---: |
| Backward | EPM | 58.92 |
| Backward | ASI | 59.46 |
| Backward | HLD | 24.73 |
| Backward | AVG | 47.70 |
| Realtime | AVG | 77.72 |
| Forward | REC | 37.11 |
| Forward | SSR | 63.75 |
| Forward | CRR | 61.67 |
| Forward | AVG | 54.17 |
| Total | AVG | 59.87 |

与现有 full 基线对比：

- 当前 V5/Qwen full SOTA：`ovo_qwen3vl8b_grpo_0509_full_state_note_t`，Total `63.90` / Backward `60.88` / Realtime `76.08` / Forward `54.74` / HLD `60.22`。
- AURA full 目标：Total `65.30` / Backward `60.40` / Realtime `79.80` / Forward `55.80` / HLD `58.60`。
- 反事实推演：若 HLD 维持 base 的 `65.05`，Total 推到 `64.34`，可挤掉 GRPO0509；若仅追平 AURA HLD `58.60`，Total `63.63`，仍低于 GRPO0509 `0.27`。

同 id 对比 GRPO0509 full（3035 条）：

| Task | RL0511 acc | GRPO0509 acc | RL 赢 | RL 输 | 净增 |
| --- | ---: | ---: | ---: | ---: | ---: |
| EPM | 58.92 | 59.60 | 25 | 27 | −2 |
| ASI | 59.46 | 62.84 | 17 | 22 | −5 |
| HLD | 24.73 | 60.22 | 3 | 69 | **−66** |
| OCR | 89.26 | 87.92 | 8 | 6 | +2 |
| ACR | 84.40 | 80.73 | 8 | 4 | +4 |
| ATR | 75.86 | 79.31 | 5 | 9 | −4 |
| STU | 65.73 | 62.36 | 17 | 11 | +6 |
| FPD | 72.28 | 67.33 | 11 | 6 | +5 |
| OJR | 78.80 | 78.80 | 11 | 11 | 0 |
| REC | 37.11 | 35.67 | 101 | 91 | +10 |
| SSR | 63.75 | 64.39 | 87 | 91 | −4 |
| CRR | 61.67 | 64.17 | 30 | 36 | −6 |

剔除 HLD 后非 HLD 类相对 GRPO0509 净增仅 `+6 / 2849`，本质未带来广泛收益。Realtime 和 REC 的小幅领先被 EPM / ASI / ATR / SSR / CRR 抵消。

### Memory 密度与 delta 长度

Note 占比口径：`note / unique video frames`，分母是 trace 中去重后的实际视频帧，不是 `num_steps`。

| Run | Notes | Unique frames | Note/frame |
| --- | ---: | ---: | ---: |
| RL0511 step60 full | 89828 | 641137 | 14.01% |
| GRPO0509 full | 66406 | 641137 | 10.36% |
| SFT full | 66188 | 641137 | 10.32% |
| Base full | 118904 | 641137 | 18.55% |

RL0511 的 note/frame 比 GRPO0509 / SFT 略高约 `+3.65pp`，但低于 base 的过密状态。

RL0511 分大类 note/frame：

| Category | Notes | Unique frames | Note/frame |
| --- | ---: | ---: | ---: |
| Backward | 16637 | 133773 | 12.44% |
| Realtime | 30900 | 212056 | 14.57% |
| Forward | 42291 | 295308 | 14.32% |

最终 memory 的 delta / bridge 文本长度，按 `memory.txt` 中最终保留下来的 `delta` / `bridge` 行统计字符数：

| Run | Final delta count | Final delta chars | Chars/delta | Chars/sample |
| --- | ---: | ---: | ---: | ---: |
| RL0511 step60 full | 89010 | 11251241 | 126.40 | 3707.16 |
| GRPO0509 full | 65760 | 8812440 | 134.01 | 2903.60 |
| SFT full | 65481 | 8662928 | 132.30 | 2854.34 |
| Base full | 112919 | 40741594 | 360.80 | 13423.92 |

结论：RL0511 的单条 delta 不比 GRPO0509 更长，反而略短；总 delta 字符数高约 `+27.7%`，主要来自写入/保留的 delta 数量更多。

RL0511 评测过程中实际 accepted step outputs 里 emitted delta / bridge 总量：`209655` 条、`24642560` 字符，平均 `117.54` chars/delta、`8119.46` chars/sample。这个 emitted count 与 rollout summary 的 `num_bridges=209675` 基本一致，差异来自 repair/drop。

### 失败模式 trace 分析

最初的猜测是"HLD 崩盘 + 其他 task 微退是同一个 question-conditioned memory pollution 机制"。**程序化扫 522 条 QA step 的结论是这个猜测太窄**：

| 模型 | QA-step 新 delta 含 question 关键词的比例 | leak n | leak 命中率 | no-leak n | no-leak 命中率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 34.56% | 531 | 71.19% | 924 | 66.67% |
| SFT | 21.50% | 313 | **75.40%** | 1146 | 68.41% |
| GRPO0509 | 23.75% | 312 | 72.76% | 1148 | 68.21% |
| RL0511 | 21.46% | 319 | 71.16% | 1142 | 63.49% |

观察：

- **leak 在所有模型上都比 no-leak 命中率高**。把"question 关键词出现在新 delta 里"等同于"作弊污染 memory"是错的；多数 leak 只是模型在当前帧里如实抄录与 question 相关的实体，反而帮助答题。
- RL0511 的 leak 命中率仅比 SFT 低 `4.24pp`、比 GRPO0509 低 `1.60pp`。leak 端的退化是次要项。
- RL0511 的 no-leak 命中率比 SFT 低 `4.92pp`、比 GRPO0509 低 `4.72pp`。**真正的退化主要发生在 no-leak 区**，即 question 关键词没进新 delta、但模型依然偏向具体答案的样本。

含义：失败不是单一机制，而是 RL 把整体答题分布往"敢答"方向推了一档，**在不同 task 上呈现成至少 5 种不同的失败模式**。逐一拆解：

**A. HLD-style 拒答先验消失（−66 净损，最严重）**

`186 / 186` 全 `Unable to answer` 的 HLD task，base `121` 对 / SFT `111` / GRPO0509 `112` / RL0511 仅 `46`。同 id：base 对但 RL 错 `78` 条，反向只有 `3` 条。机制是 `w_success=0.8` 训了 60 步把任何"答 Unable / 空答"的概率压低。这是单维度先验被推平。

**B. Counting drift（REC tail blowup，少数样本造成大失分）**

REC 预测偏差均值：base `-1.679`，SFT `-0.513`，GRPO0509 `-0.298`，**RL0511 `+0.563`**。整条 RL 训练链路在做单调的"敢数"calibration push。问题在于这个 push 在长视频上会放大成 tail，从 trace 里抓的最大过数案例：

| 样本 | GT | RL 预测 | bias | 模型 final state 摘要 |
| --- | ---: | ---: | ---: | --- |
| `1599_31` | 9 | **186** | +177 | "Based on the memory, the count has reached 186." |
| `1599_32` | 9 | 43 | +34 | 同视频下一 QA step，count 没复位 |
| `1599_29` | 9 | 34 | +25 | 同视频前一 QA step |
| `1633_4` | 2 | 25 | +23 | repetitive shots 误当独立事件 |
| `1633_6` | 2 | 24 | +22 | 同视频 |
| `1602_17` | 8 | **0** | −8 | 反向 tail：直接漏数 |
| `1602_18` | 8 | 0 | −8 |  |

机制：**长视频的 step-by-step memory 在 step level 累加 count 时没有清零或验证机制**。一旦某一步把 count 写成大数，后续 step 在生成 state 时把这条 delta 当事实复用。这是 memory 的 monotonic drift，跟 question 没关系。RL 训练把"敢给具体数字"的概率提上去，进一步放大 tail。

**C. QA-conditioned memory write（EPM 20 / CRR 1471_4 这种小数量但典型的样本）**

模型在 QA step 同时干 memory write 和 read，且把 question 关键词倒灌回 delta，并捏造伪时间戳支撑答案。属于 leak 中的 confabulation 子集。

- EPM `20`，question `What did I pick from the utensil holder?`，GT `C. Dying pan`。RL0511 final state 引用 `t=261.0 picked up a spoon`，但 memory `t=261` 实际是 `takes a pan with ice from the drawer`。同 step 把伪事件写入 `delta 301.0-304.0: ... picks up a spoon from the utensil holder`。
- CRR `1471_4`，Yes/No GT `Yes`。RL0511 把 `t=366-370` 同时间窗 delta 写成 `paper and hat handed to the man`，GRPO0509 同时间窗写成 `paper and money handed to the woman`。两侧都自信，分歧在 memory 写入侧。
- 这个机制在所有模型上都有 base case，但 RL0511 在 EPM/HLD/CRR 上的发生 + 答错频率高于 SFT/GRPO，所以表面上看像"独有"机制，实际是其他机制的次要贡献。

**D. State disambiguation 被 question framing 牵引（ASI 491 这类）**

memory 里同时存在多个语义接近的候选动作，state 选了符合 question framing 的那个。

- ASI `491`，question `What does the person do after wipe screen again?`，GT `A. remove the label`。RL0511 delta 是对的（`peel off the orange instruction card`），但 state 把焦点偏到`placed white card on the edge`，答 `C. place label`。GRPO0509 state 显式锚定 `t=85 peeled off small white tab → option A`。
- ASI `63`，question 问倒进碗里什么。RL state 自己写 `oil is poured into a pan, not a bowl`，仍答 `oil`。这是 state 端的逻辑断裂，不是 memory 端。

**E. Forward / Realtime 边界混淆（SSR / CRR 时机题）**

模型不区分 `准备做 / 正要做 / 正在做 / 刚做完 / 转入下一动作` 五个临界状态。

- SSR `1517_5` GT `Yes`（pour egg into pot），RL state 说 `bowl is over the pan, preparing to pour`，答 `No`。GRPO/SFT 答 `Yes`。
- SSR `1517_8` GT `No`（fry eggs），RL 把刚倒进热锅当 `frying`，答 `Yes`。GRPO/SFT 区分 pouring 和 frying，答 `No`。
- CRR `1476_0` GT `No`，RL 看到 man 起身说话就答 `Yes`；CRR `1478_2` GT `Yes`，RL 看到 woman leave the room 却说 `next action not visible` 答 `No`。
- 同一模型在 forward 题里同时出现"过早 Yes"和"过晚 No"两种相反偏差，说明它没有学到 onset 的时间锚。

**F. ATR 视觉属性识别本身不强（base 上限，不是 RL 退化）**

- `1008` GT `Purple`，RL 关注红橙火把答 `Red`；`1046` GT `Smooth and liquidy`，RL 答 `Thick and creamy`；`1055` 表情题。
- SFT / GRPO / RL 在 ATR 上的差距小，更像 8B Qwen-VL 在 attribute-grained 视觉 task 的上限。这类样本不应进入 RL fix-list。

**结构性遗留：base 对但 SFT/GRPO/RL 都错的样本**（共 105 条）

按 task 分布：`SSR 25 / REC 21 / HLD 17 / EPM 10 / STU 9 / OJR 6 / ASI 4 / FPD 4 / CRR 4 / ATR 2 / OCR 2 / ACR 1`。这些是 SFT 阶段就丢掉、后续 RL 没再恢复的题。HLD 17 条说明 SFT 已经把拒答先验削弱过一次，RL 只是继续推。说明**修拒答能力不能只盯 RL，SFT 数据本身也需要 Unable-as-gold 注入**。

### 根因诊断

根因从单一机制扩展到结构 + 动力学两层，按可干预度排序：

**第一层：训练目标和反馈结构**

1. **success reward 让"敢答"系统性优于"拒答"**：`compute_success_score` 是 `exact_or_contains` 命中给 `score_scale=2.0`、否则 0。瞎猜任一具体选项期望 `(1/k)·2.0`，拒答恒 0。`w_success=0.8` 让该偏差主导 trajectory_score。所有失败模式 A/B/C/D/E 都共享这个梯度方向。
2. **reward 不看 memory 写入质量**：`compute_step_score` 是 anchor frequency + 可选 process judge。judge 维度是 `keyframe_selection / bridge_quality / semantic_alignment / state_factuality`，**都不**校验 state 引用的时间戳能否在 prior delta 中解析、也不校验当前 delta 是否被 question 条件化、也不约束 long-video 上 count 的单调性。这导致失败 B/C 在 reward 端完全无信号。
3. **没有 KL 约束**：`use_kl_loss=false`、`kl_loss_coef=0`、`use_kl_in_reward=false`。模型可以自由偏离 SFT / GRPO0509 的答题分布，60 步训练把整体答案分布向"敢答"推一档，所有 task 同向漂移。

**第二层：训练数据和分布**

4. **数据单一**：`dataset2/rl_0511.jsonl` `3444` 条全部 `task=backward`，`0` 条 gold 是 Unable，`0` 条 forward/realtime 边界负样本，`0` 条 counting 题。RL 训练等于在一个 1-D 投影上 push 答案分布，但 OVO 评测是多维度 calibration。push 在 backward 单维度上的副作用在 HLD/REC/SSR/CRR 这些维度上是反向的。
5. **没有 task-conditioned 边界提示**：prompt 通用模板。模型不知道 EPM/ASI 需要守住关系约束、HLD 需要拒答、REC 需要严格计数、SSR 需要区分 onset、CRR 需要确认 next action 已可见。这些边界在 SFT 数据里都没有，全靠模型自己拼。

**第三层：rollout 协议结构**

6. **memory write 和 answer 在同一个 forward 里**：QA step 同时输入 question + 当前帧 + 写新 delta + 写 state + 给 answer。这条 forward 是 differentiable 的，question 的梯度可以倒灌到 delta 的 token logits。即使整段训练 reward 设计完美，结构本身保留了 shortcut。失败模式 C 是这种结构最敏感的情况。
7. **长视频上 step 之间没有 state 复位机制**：REC tail 的 1599_31 出现 `count = 186` 是因为每一步 state 都引用前一步 state 写的 count，错误一旦写入就被后续 step 当事实复用。这是 monotonic drift，跟 reward 设计无关，是 rollout / memory 协议的缺陷。

**第四层：训练动力学**

8. **校准在 base / SFT / GRPO / RL 这条链上单调右推**：REC bias `-1.68 / -0.51 / -0.30 / +0.56`，HLD 拒答率 `65 / 60 / 60 / 25`，Note/frame `18.55% / 10.32% / 10.36% / 14.01%`。每一步训练都让模型更敢说、更敢写。这意味着即使下一版 RL 修了 reward，如果 step budget 仍然是 60+ 步，最终也会被推过头。**step budget 应当作 hyperparameter，配合 early-stop on HLD acc 来收**。

### 下一步建议

下一步建议沿 6 个独立维度展开。每条都标注**可干预度 / 成本 / 期望影响 task**，便于挑组合。

**维度 1：评测端（零成本，立即可做，影响 HLD / EPM 子集）**

零重训 postprocess：扫 RL0511 现有 `outputs/ovo_exp1_rl0511_step60_full/results.jsonl` 和 `traces/`，把 state 含 `likely / appears / suggests / assume / no mention / not visible / cannot determine / no evidence / not specified` 且选项含 `Unable` 的样本，answer 强制改成 Unable 的 letter；把 state 写出"具体观察到的答案"但答案不在选项里的样本，answer 也强制改成 Unable。重新跑 OVO scorer，预期 HLD 拉回到 `45-55`，Total 拉回到 `61-62`。

零重训 trace replay：抽 30-50 条 EPM/ASI/HLD 错样本，重跑 rollout 但 per-step prompt 屏蔽 QA section（只在最终一次 forward 暴露），逐字符比较新旧 delta：
- 若新 delta 不再含 question 关键词，证明失败 C 是纯结构问题，下一版必须改 rollout pipeline。
- 若新 delta 仍 confabulate，证明模型本身的视觉 grounding 在某些场景下已经退化，必须回 SFT 数据修。

**维度 2：训练数据（一周成本，影响所有 task）**

数据按 task-family 分层 + 配对 hard negative，建议 `rl_0513_guardrail` 先做 `1.5k-3k` 条，比例：

- `backward_relation` 15%：EPM/ASI/HLD 历史关系题，配 wrong container / wrong location / wrong object alias / wrong action relation / observed answer not in options 五类 paired hard negative。
- `forward_boundary` 20%：SSR/CRR 时机题，同一事件切 5 个时间锚（`start-2s / start-1s / start / start+1s / end+1s`），分别标 `before_onset_preparation / near_onset_not_started / onset / in_progress / after_completion`。
- `realtime_answerability` 10%：当前帧是否足够回答；目标事件未可见或被遮挡时输出空 answer / Unable。
- `counting_boundary` 5%：REC 0/1/2/3 边界，覆盖 `touching ≠ completed action`、`looks at ≠ moves` 等区分。
- `false_premise_unable` 25%：gold 必须是 Unable / Cannot answer。同一段视频成对构造，让 hard negative 紧贴 positive。
- 当前 RL v2 hard/veryhard 25%：保留原口径。

数据文件新增可审计字段：`task_family / ovo_subtask / boundary_type / query_time / evidence_intervals / distractor_intervals / answerable_at / gold_is_unable / rationale_short`。训练前必须出统计报告：`task_family` 分布、`gold_is_unable` 占比、Yes/No/MCQ 分布、`last4_pass` 分布、`train/val` 是否视频级去重。

**维度 3：Reward / 算法（一两周成本，影响 HLD / SSR / EPM）**

- `compute_success_score` 改成 abstention-aware：`gt=Unable` 时答 Unable +`score_scale`、答具体 `0`；`gt=具体` 时答 Unable `0`、答对 `score_scale`、答错具体 `-0.5·score_scale`。这条直接对应失败 A。
- 新增 `compute_groundedness_score`：对 QA step 新 delta 用 small LM / 规则核对其中提到的物体、动作、关系能否在当前帧 + prior delta 找到 backing，找不到惩罚。起始权重 `w_groundedness=0.2`，`w_success=0.8 → 0.6`。这条直接对应失败 C。
- 新增 `compute_consistency_score`：state 引用的 `t=X` 时间戳必须能在 prior delta range 里命中，否则惩罚；state 的 final claim 必须和当前 delta 的 final 状态一致。直接对应失败 D。
- 启用 KL：`use_kl_loss=true`、`kl_loss_coef=0.001~0.01`，ref policy = answered-full SFT 或 GRPO0509，限制单步漂移幅度。
- 把 trajectory-level outcome reward 改成 step-level process reward：每步给一个分（基于该步 delta 的 groundedness + answer-readiness），不再等到 trajectory 结尾。这能解决 long-horizon credit assignment 稀疏问题。

**维度 4：算法 / Rollout 结构（结构性改动，影响所有 task）**

- **Memory-write 阶段和 answer 阶段分离**：QA step 拆成两个 forward。第一个 forward 只看当前帧 + memory，不看 QA，产出 delta；提交 memory 后，第二个 forward 只看 frozen memory + QA，产出 answer。这能从源头堵失败 C 的 shortcut，即使 reward 不变也有效。
- **每步 state 不再持久化**：当前协议把 state 写进 memory 中。改成每步独立计算 state，memory 只保留 delta / note。这能阻断失败 B 的 monotonic drift（错的 count 写入 state 后不会被下一步当事实复用）。
- **N-sample + verifier 投票（test time）**：QA step 采 `n=5` 个候选 answer + 候选 state，用一个 verifier（小 LM 或规则）按 evidence-grounding 打分，取分数最高的；如果 top-1 与 top-2 接近且涉及 Unable，强制 Unable。这条不需要重训。
- **PRM（process reward model）替换 outcome-only reward**：训练一个独立 judge 在每个 step 给分，按 OVO sub-task 类型条件化。比当前的 anchor-frequency step reward 信号密得多，能直接给 forward boundary 类失败提供梯度。

**维度 5：推理 / Prompt（一两天成本，影响 HLD / SSR / CRR）**

- Task-aware prompt 注入：QA step 注入前检测 question 类型，prepend 一段 task-specific 边界提示。例如检到 HLD 模板就强调"如果问题指向的对象、地点、关系在 memory 里没出现就选 Unable，不要类比"；检到 SSR 模板就强调"区分 about to / starting / doing / completed，只有 starting 之后才能答 Yes"。这种 routing 不改模型权重，但可以快速试错。
- Confidence verbalize：让模型在 state 末尾输出一个 `confidence: 0.X` 数字，answer 选择时用 `confidence < threshold → Unable` 硬规则。配合维度 3 的 calibrated reward 可以训出可用的 verbalized confidence。

**维度 6：度量 / 接受门槛（提早预警，影响实验决策本身）**

每个 checkpoint 在做 full 评测前先过下面一组 quick guardrail（成本约 1-2 GPU·h）：

- `HLD on 1/8` ≥ 50（拒答先验未崩）
- `Unable rate over HLD 1/8` ≥ 0.5（行为层校验）
- `REC bias mean` ∈ `[-0.5, +0.3]`、`REC max over` ≤ 10（无 tail blowup）
- `same-id non-HLD net vs GRPO0509` ≥ -5（没把别的 task 带崩）
- `Note/frame` ∈ `[8%, 13%]`（避免过密 / 过稀）
- `SSR / CRR FP / FN imbalance` < `0.15`（forward 边界）

任何一条不过就不开 full 评测，避免每次浪费 8 小时 + 8 卡。

**优先组合（一周内能落地的最小集）**

立即（今天/明天）：维度 1 全做，确认机制归因。

下一版 RL run（建议命名 `rl_0513_guardrail`）：维度 2 做 false-premise + forward boundary + backward relation 三项；维度 3 做 abstention-aware success + KL；维度 4 做 memory-write / answer 分离；维度 5 做 task-aware prompt；维度 6 全做并作为 trainer.test 触发条件。

不建议在第一版同时上 groundedness reward 和 PRM，先用 abstention-aware + KL + 数据修复看能否把 Total 推过 GRPO0509 `63.90`；如果还是过不去再加 groundedness / PRM。

### 下一步建议

可立即做的零重训验证：

- 抽 30-50 条 RL0511 错的 EPM / ASI / HLD id，重跑 rollout，但每个 per-step prompt 屏蔽 QA section（QA 只在最终单独一次 forward 时暴露）。逐字符比较新旧 delta：
  - 若新 delta 不再出现 question 关键词（"utensil holder"、"belt"、"spoon at t=..."），shortcut 是纯 question-conditioned，结构修复就是把 rollout 拆成 memory-update 阶段（永不看 QA）+ answer 阶段（只读冻结 memory）。
  - 若新 delta 仍 confabulate，那是模型本身的视觉 grounding 退化，必须回 SFT 数据或 visual encoder 阶段修。

下一版 RL 的硬性约束：

- **base 不再用 RL0511 step60**，改回 GRPO0509 或 answered-full SFT；RL0511 作为负诊断保留。
- 数据：当前 RL0511 / RL0512 的结构性缺口必须先修，否则继续调算法只会把“更敢猜”和“QA-conditioned memory write”放大。实测当前 RL 数据仍是 `100% backward`，gold 里没有 `Unable to answer`；模型训练时只见过“看完历史后从具体选项里选一个”，没有学过“当前证据不够时该拒答 / 该答 No / 该等待”。下一版数据要按下面口径重做。

  **1. 显式注入 `Unable to answer` / false-premise MCQ。**

  这类样本不是给选项里随便加一个 Unable，而是 gold 本身必须是 Unable / Cannot answer。目标占比先设为 `20%-30%`，并且要和正样本成对出现，最好来自同一个视频、同一个时间段、同一批候选物体。需要覆盖以下负样本类型：

  - `wrong_container`：视频显示把 `water` 倒进 `bowl`，问题问“倒进 pan 里的是什么”；模型必须知道“同一个物体出现在视频里，但关系/容器错了”不能答具体物体。
  - `wrong_location`：视频显示从 drawer / cutting board 拿东西，问题问“从 utensil holder 拿了什么”；日志里的 EPM `20` 就是这类错误，RL0511 为了迎合问题把 `spoon at t=261` 写进 memory。
  - `wrong_object_alias`：选项里有相似物体或别名，例如 pan / pot / lid / bowl / plate；只有真实被操作的对象可以作为答案，不能因为语义接近就答。
  - `wrong_action_relation`：视频里确实出现 oil / egg / label，但发生的是 `pour into pan`、`peel off label`、`place card` 等不同关系；日志里的 ASI `491` 和 EPM/ASI `63` 都属于关系被 question framing 拉偏。
  - `observed_answer_not_in_options`：视频里能看出答案，但所有选项都不对，gold 必须是 Unable，而不是选最接近的错项。

  **2. 按 OVO sub-task 分层抽样，不再单一 backward。**

  backward 题训练的是“读历史 memory 后回答过去发生了什么”；realtime / forward 题训练的是“当前是否已经有足够证据、是否应该等待、是否应该答 No/Unable”。这两个能力不等价。当前 RL 数据全是 backward，会把策略推成“只要问题出现，就从选项里猜一个具体答案”。下一版数据至少要分层记录 `task_family` / `ovo_subtask`，并保证 train 和 guardrail validation 都有样本：

  - `backward_relation`：EPM / ASI / HLD 这类历史关系题。重点不是多加 easy QA，而是加 wrong container / wrong location / wrong object / wrong relation 的 hard negative，让模型守住“历史里出现过 ≠ 符合问题关系”。
  - `realtime_answerability`：当前帧是否足够回答。若目标事件尚未可见，或关键对象被遮挡，应该输出空 answer / Unable，而不是提前猜。它用于压住 RL0511 训练后出现的 confidence 偏移。
  - `forward_boundary`：SSR / CRR 这类未来或下一动作判断。模型必须区分“准备做”“刚开始做”“正在做”“已经做完”“只是相似动作”，否则会出现 `about to pour` 被当成 `pouring`、`pouring` 被当成 `frying`。
  - `counting_boundary`：REC 这类计数题。当前日志显示 RL0511 从欠数偏到多数，说明需要专门覆盖 `0/1/2/3` 和“触碰但未完成一次动作”的边界。

  推荐第一版 `rl_0513_guardrail` 先做 `1.5k-3k` 条，比例可用：`25%` 当前 RL v2 hard/veryhard answerable、`25%` false-premise Unable、`20%` forward boundary、`15%` backward relation hard negative、`10%` realtime answerability、`5%` REC/counting。等小步验证过 HLD / Unable / SSR-CRR 后再扩大到 `4k-6k`。

  **3. forward 边界样本要围绕同一个事件切时间点。**

  “边界”不是泛泛地说长视频难，而是同一个动作在时间线上有几个临界状态，标签不同。以 `pour egg into pot` 为例，应从同一段视频构造一组 paired samples：

  `No` 和 `Unable` 也要分开标。`No` 表示当前证据已经足够判定命题为假，例如问题问“现在是否正在倒蛋”，画面清楚显示人只是拿着碗准备靠近锅；`Unable` 表示缺少判定所需证据，或真实答案不在选项里，例如关键动作被遮挡、事件发生在还没看到的未来、问题预设的视频关系不存在。这个区别必须进数据，否则模型会把“没看到”“还没发生”“明确没发生”混成一种拒答策略。

  ```text
  query_time = start - 2s: 人拿着碗靠近锅，但还没有蛋液流下
  label: No / Unable
  boundary_type: before_onset_preparation
  explanation: preparing to pour is not pouring

  query_time = start - 1s: 碗悬在锅上方，动作即将发生
  label: No / Unable
  boundary_type: near_onset_not_started
  explanation: about to pour is still not evidence of pouring

  query_time = start: 第一帧出现蛋液从碗进入锅
  label: Yes
  boundary_type: onset
  explanation: the action becomes visible here

  query_time = start + 1s: 蛋液正在进入锅
  label: Yes
  boundary_type: in_progress

  query_time = end + 1s: 蛋已经进锅，但当前正在 stirring / frying / moving bowl
  label: depends on question wording
  boundary_type: after_completion_or_new_action
  explanation: poured into pot can be true historically, but "is pouring now" is false
  ```

  这组数据要迫使模型学习“证据刚刚足够”的位置。日志里的 SSR `1517_5` / `1517_8` 就说明模型没有学会这个边界：一边把准备倒误判成还不能答，另一边又把刚倒进热锅误当成 frying。CRR 也类似：人起身、转身、说话只是下一动作的前兆，不等于目标动作已经发生；如果当前帧还没显示目标动作，应该答 No/Unable，而不是提前猜 Yes。

  **4. backward 关系边界样本要成对构造。**

  关系边界是“视频里有这个物体/动作，但问题问的关系不成立”。这比普通 hard QA 更重要，因为 RL0511 的失败不是完全没看到对象，而是为了匹配 question 把对象放进错误关系里。构造时每个正样本旁边都要有 1-3 个 counterfactual 负样本：

  ```text
  positive:
    question: What did the person put into the bowl?
    evidence: water is poured into the bowl at t=63-68
    gt: water

  negative_wrong_container:
    question: What did the person put into the pan?
    evidence: the pan is visible, but water was poured into the bowl, not the pan
    gt: Unable / none of the options

  negative_wrong_action:
    question: What did the person fry in the pan?
    evidence: water was poured; no frying action occurred
    gt: Unable / No
  ```

  这种 paired construction 可以防止模型只记住对象词频。它必须同时满足 `object`、`action`、`container/location`、`time order`，缺一项就不能答具体选项。

  **5. 数据文件需要新增可审计字段。**

  如果只保留 `question/options/answer/gt`，后面很难判断模型为什么错，也没法写 groundedness reward。新数据至少加这些字段：

  ```text
  task_family: backward_relation | realtime_answerability | forward_boundary | counting_boundary
  ovo_subtask: EPM | ASI | HLD | SSR | CRR | REC | ...
  boundary_type: wrong_container | wrong_location | before_onset_preparation | onset | in_progress | after_completion | observed_answer_not_in_options | ...
  query_time: 问题注入时间
  evidence_intervals: 支撑 gold 的时间段
  distractor_intervals: 容易误导模型的相似事件时间段
  answerable_at: 第一次足够回答的时间；如果永远不可答则为 null
  gold_is_unable: true/false
  rationale_short: 一句话解释为什么是这个标签，供抽查和 judge 使用
  ```

  训练前必须跑数据统计，报告至少包含：task_family 分布、ovo_subtask 分布、gold_is_unable 占比、Yes/No/MCQ 分布、dataset/video 分布、last4_pass 分布、每类 boundary_type 数量、train/val 是否视频级去重。
- Reward：
  - `compute_success_score` 改成 abstention-aware：GT=Unable 时答 Unable +`score_scale`、答具体 `0`；GT=具体时答 Unable `0`、答对 `score_scale`、答错具体 `-0.5·score_scale`。
  - 加 memory-write-vs-frame 一致性 reward：对 QA step 新生成的 delta，用 small LM 或规则核对 delta 提到的物体 / 事件能否在当前 frames 和 prior delta 中找到 backing；找不到惩罚。起始权重 `w_groundedness=0.2`，同时 `w_success` 降到 `0.6`。
  - 启用 KL：`use_kl_loss=true`、`kl_loss_coef=0.001~0.01`，ref policy = SFT 或 GRPO0509。
- 结构：rollout pipeline 拆成 memory-update + answer 两阶段，per-step prompt 永远不包含 future QA。
- 接受门槛：每个新 checkpoint 在做 full 评测前必须先过 HLD full 或 HLD 1/8 + Unable rate + same-id non-HLD net vs GRPO0509 + REC 计数偏差 + SSR/CRR FP/FN + notes/step + repair/step 这套快速指标。
- Preference / DPO 锚点样本：`1517_5`、`1517_8`、`1520_3`、`1567_0`、`63`、`491`、`20`、`341`、`390`、`1471_4`（positive trace = GRPO0509 / SFT 正确轨迹，negative trace = RL0511 错误轨迹）。

## Known Launch Scripts

| Script | Intended use | Current default run name | Notes |
| --- | --- | --- | --- |
| `RL/scripts/train_grpo.sh` | GRPO-style stepwise trajectory training | `grpo_rl0511_8gpu_judge` | This script matches the currently running old-config process. Copy it before changing major experiment settings. |
| `RL/scripts/train_exp2_rlmlr.sh` | Experiment 2 RLMLR training on `rl_0512` split | `exp2_rlmlr` | Uses `train_batch_size=16`, `gen_batch_size=16`, `rollout.n=8`, validation every 20 steps. |
| `RL/scripts/train_exp3.sh` | Experiment 3 GRPPO baseline on `rl_exp3` | `exp3` | Final advantage `1.0*step_adv + 0.3*answer_adv`; step filter min std `0.04`. |
| `RL/scripts/train_exp4.sh` | Experiment 4 GRPPO answer-weight sweep | `exp4` | Same as exp3, but final answer-advantage weight is `0.5`. |
| `RL/scripts/train_exp5.sh` | Experiment 5 GRPPO answer-weight sweep | `exp5` | Same as exp3, but final answer-advantage weight is `0.7`. |
| `RL/scripts/train_ppo.sh` | PPO/GAE with critic enabled | `ppo_test4rl_8gpu` | Uses `streamweave_stepwise_ppo_gae`, `rollout.n=1`, critic enabled, and judge weight forced to `0.0` by default. |
| `RL/scripts/train_rlmlr.sh` | RLMLR stepwise/outcome mixed advantage | `rlmlr_rl0511_8gpu` | Uses `streamweave_stepwise_rlmlr`; DAPO filter is off by default in the script. |
| `RL/scripts/run_smoke.sh` | Smoke test | N/A | Use for quick integration checks before long runs. |

## Entry Template

Copy this row when adding a new experiment:

| Date UTC | Status | PID | Run name | Launch script | Base config | Dataset | Source model | Algorithm / key deltas | Judge | Output dir | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YYYY-MM-DD | planned | TBD | `run_name` | `RL/scripts/script_name.sh` | `RL/configs/streamweave_stepwise.yaml` | `dataset2/file.jsonl` | `models/model_name` | `adv_estimator=...`; `rollout.n=...`; important CLI overrides | enabled/disabled, backend/model | `RL/outputs/runs/run_name` | Hypothesis, expected change, or reason for this run. |

## Minimum Fields To Preserve

- Exact launch script path.
- Unique `RUN_NAME`.
- Dataset file and dataset root.
- Source model path and prepared model-config output path.
- Advantage estimator and group-filter/DAPO setting.
- Reward weights and judge setting.
- Prompt length, response length, rollout `n`, worker count, and batch sizes.
- Git status snapshot from the run output directory.
- Final status: running, completed, failed, killed, or superseded.
