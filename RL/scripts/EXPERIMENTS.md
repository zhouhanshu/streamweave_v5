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
| 2026-05-12 | planned | TBD | `TBD` | `RL/scripts/TBD.sh` | `RL/configs/streamweave_stepwise.yaml` | `TBD` | `TBD` | `TBD` | `TBD` | `RL/outputs/runs/TBD` | Fill this row when the next launch script is created. |

## Experiment 1: `grpo_rl0511_8gpu_judge`

- SwanLab curves/logs: <https://swanlab.cn/@zhs/streamweave_rl/runs/gmtjhm9u05m58326zmpe6/chart>
- Record time: 2026-05-12 UTC.
- Status at record time: running, PID `3927597`.
- Config source: live process command line plus defaults from `RL/configs/streamweave_stepwise.yaml`.
- Note: this is the old-config run. The live command has `+data.gen_batch_size=16`; the current `RL/scripts/train_grpo.sh` on disk has drifted to `+data.gen_batch_size=8`.
- Base config: `--config-name=streamweave_stepwise`.
- Dataset: `data.train_files=dataset2/rl_0511.jsonl`, `data.val_files=dataset2/rl_0511.jsonl`, `data.streamweave.dataset_name=mixed_rl_0511`, `data.streamweave.dataset.dataset_root=dataset2`.
- Data/batch: `data.train_batch_size=32`, `+data.gen_batch_size=16`, `data.val_batch_size=4`.
- Lengths: `data.max_prompt_length=6144`, `data.max_response_length=2048`.
- Runtime defaults from base config: `sample_fps=1.0`, `frames_per_step=5`, `max_frames=0`, `max_steps=0`, `resolution=336`.
- Source model: `models/qwen3vl8b_streamweave_sft_answered_full_anchor_delta_init_anchor_step200_vllm`.
- Training model path: `RL/outputs/runs/grpo_rl0511_8gpu_judge/model_config`.
- Actor: `lr=1e-5`, `ppo_mini_batch_size=32`, `ppo_micro_batch_size_per_gpu=8`, `ppo_max_token_len_per_gpu=32768`, `clip_ratio_low=0.2`, `clip_ratio_high=0.28`.
- Reference/logprob: `ref.log_prob_micro_batch_size_per_gpu=8`, `ref.log_prob_max_token_len_per_gpu=32768`.
- Rollout: `rollout.n=16`, `temperature=1.0`, `top_p=0.95`, `gpu_memory_utilization=0.7`, `max_model_len=8192`, `max_num_batched_tokens=65536`, `max_num_seqs=2048`, `agent.num_workers=32`.
- Reward defaults from base config: `w_format=0.1`, `w_step=0.1`, `w_success=0.8`, `format_mode=valid`, `success_mode=dataset`, note-frequency reward enabled.
- Judge: `data.streamweave.reward.judge.enable=true`; base config uses Gemini backend with `model=gemini-2.5-flash`, `judge_weight=1.0`.
- Algorithm: `algorithm.adv_estimator=streamweave_stepwise_traj_grpo`, `algorithm.use_kl_in_reward=false`, `algorithm.filter_groups.enable=true`.
- Critic: `critic.enable=false`.
- Trainer: `trainer.logger=["console","swanlab"]`, `trainer.project_name=streamweave_rl`, `trainer.experiment_name=grpo_rl0511_8gpu_judge`, `trainer.resume_mode=auto`, `trainer.n_gpus_per_node=8`, `trainer.nnodes=1`, `trainer.save_freq=20`, `trainer.test_freq=-1`, `trainer.total_epochs=2`.
- Ray: `num_cpus=64`, `object_store_memory=40000000000`, `include_dashboard=false`, `_temp_dir=/tmp/swray_3927346`.
- Output dir: `RL/outputs/runs/grpo_rl0511_8gpu_judge`.
- Model/checkpoint storage:
  - Prepared model config used by training: `RL/outputs/runs/grpo_rl0511_8gpu_judge/model_config`.
  - Training checkpoints root: `RL/outputs/runs/grpo_rl0511_8gpu_judge/checkpoints`.
  - Observed checkpoints at record time: `RL/outputs/runs/grpo_rl0511_8gpu_judge/checkpoints/global_step_20`, `RL/outputs/runs/grpo_rl0511_8gpu_judge/checkpoints/global_step_40`.
  - Latest-checkpoint marker: `RL/outputs/runs/grpo_rl0511_8gpu_judge/checkpoints/latest_checkpointed_iteration.txt`.
- Logs:
  - Main training log: `RL/outputs/runs/grpo_rl0511_8gpu_judge/train.log`.
  - Local SwanLab log dir: `RL/outputs/runs/grpo_rl0511_8gpu_judge/swanlab/run-20260511_170803-gmtjhm9u05m58326zmpe6`.
  - Online SwanLab chart: <https://swanlab.cn/@zhs/streamweave_rl/runs/gmtjhm9u05m58326zmpe6/chart>.
  - Local parsed curve script: `RL/scripts/log/plot_traj_metrics.py`.
  - Local parsed curve image: `RL/scripts/log/experiment1_traj_score_success.png`.
- Post-training export/eval plan:
  - Expected checkpoint for first OVO eval: `RL/outputs/runs/grpo_rl0511_8gpu_judge/checkpoints/global_step_60/actor`.
  - HuggingFace export target: `models/exp1_rl0511_step60`.
  - OVO 1/8 eval output: `outputs/ovo_exp1_rl0511_step60_1of8`.
  - Export command:
    ```bash
    cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
    PYTHONPATH=RL/verl:RL:. /mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/python -m verl.model_merger merge --backend fsdp --local_dir RL/outputs/runs/grpo_rl0511_8gpu_judge/checkpoints/global_step_60/actor --target_dir models/exp1_rl0511_step60 --trust-remote-code
    ```
  - OVO 1/8 eval command:
    ```bash
    OUTPUT_DIR=outputs/ovo_exp1_rl0511_step60_1of8 bash scripts/run_ovo_8gpu_vllm_finetuned.sh models/exp1_rl0511_step60
    ```

Observations:

- This is a demo experiment using the `rl_0511` training dataset.
- During training, the reward rises very slowly at the beginning and has high noise.
- Training is very slow, roughly 30 minutes per step.
- Despite the slow and noisy curve, the reward is increasing.
- After the run reaches a more meaningful training stage, run OVO-Bench evaluation to check whether the reward increase transfers to benchmark performance.
- Current hypothesis: training is too slow partly because reward attribution is weak. Future runs should improve reward attribution.
- Future runs should also reduce `rollout.n`.

## Known Launch Scripts

| Script | Intended use | Current default run name | Notes |
| --- | --- | --- | --- |
| `RL/scripts/train_grpo.sh` | GRPO-style stepwise trajectory training | `grpo_rl0511_8gpu_judge` | This script matches the currently running old-config process. Copy it before changing major experiment settings. |
| `RL/scripts/train_exp2_rlmlr.sh` | Experiment 2 RLMLR training on `rl_0512` split | `exp2_rlmlr` | Uses `train_batch_size=16`, `gen_batch_size=16`, `rollout.n=8`, validation every 20 steps. |
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
