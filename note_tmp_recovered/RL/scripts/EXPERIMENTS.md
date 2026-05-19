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
| 2026-05-12 | running-old-config | 3927597 | Experiment 1: `grpo_rl0511_8gpu_judge` | `RL/scripts/train_grpo.sh` | `RL/configs/streamweave_stepwise.yaml` | `dataset2/rl_0511.jsonl` | `models/qwen3vl8b_sft_anchor_delta_step200_vllm` | `streamweave_stepwise_traj_grpo`; `rollout.n=16`; DAPO filter enabled; `gen_batch_size=16` in the live command | Gemini judge enabled | `RL/outputs/runs/grpo_rl0511_8gpu_judge` | Demo experiment. Existing process uses an old config. Curves: <https://swanlab.cn/@zhs/streamweave_rl/runs/gmtjhm9u05m58326zmpe6/chart>. Do not stop or reuse this run name for new experiments. |
| 2026-05-12 | ready | TBD | Experiment 2: `exp2_rlmlr` | `RL/scripts/train_exp2_rlmlr.sh` | `RL/configs/streamweave_stepwise.yaml` | `dataset2/rl_0512_train.jsonl`; val `dataset2/rl_0512_val.jsonl` | `models/qwen3vl8b_sft_anchor_delta_step200_vllm` | `streamweave_stepwise_rlmlr`; `train_batch_size=16`; `gen_batch_size=16`; `rollout.n=8`; `outcome=success_score`; `state=1.0*step_score+0.2*format_score`; DAPO filter disabled; validation every 20 steps | Gemini judge enabled | `RL/outputs/runs/exp2_rlmlr` | Uses `rl_0512` deterministic video-level split: train 1420 rows, val 80 rows. Fixed output dir; do not start the same script concurrently on multiple machines sharing this disk. |
| 2026-05-14 | stopped-after-step20 | manual stop | Experiment 3: `exp3` | `RL/scripts/train_exp3.sh` | `RL/configs/streamweave_stepwise.yaml` | `dataset2/rl_exp3.jsonl`; val same file | `models/qwen3vl8b_sft_anchor_delta_step200_vllm` | `streamweave_stepwise_grppo`; `train_batch_size=16`; `gen_batch_size=16`; `rollout.n=8`; `answer_decay=0.4`; final advantage `1.0*step_adv + 0.3*answer_adv`; `grppo_min_std=0.04`; step filter enabled with `filter_min_std=0.04`; `process_weight=1.0`; `format_weight=0.1`; validation every 30 steps | Gemini judge enabled, `streamweave_grppo_judge_v1` | `RL/outputs/runs/exp3` | Stopped manually after `global_step_20`; checkpoint available. Curves were close to exp4/exp5, so no immediate evaluation. |
| 2026-05-14 | stopped-evaluating-step20 | manual stop | Experiment 4: `exp4` | `RL/scripts/train_exp4.sh` | `RL/configs/streamweave_stepwise.yaml` | `dataset2/rl_exp3.jsonl`; val same file | `models/qwen3vl8b_sft_anchor_delta_step200_vllm` | Same as exp3 except final advantage `1.0*step_adv + 0.5*answer_adv`; `filter_min_std=0.04` | Gemini judge enabled, `streamweave_grppo_judge_v1` | `RL/outputs/runs/exp4` | Stopped manually after `global_step_20`; exported to `models/exp4_grppo_aw05_step20`; OVO 1/8 evaluation directory created at `outputs/ovo_exp4_grppo_aw05_step20_1of8`. |
| 2026-05-14 | stopped-in-validation-step30 | manual/external stop | Experiment 5: `exp5` | `RL/scripts/train_exp5.sh` | `RL/configs/streamweave_stepwise.yaml` | `dataset2/rl_exp3.jsonl`; val same file, capped by `data.val_max_samples=200` for future resumes | `models/qwen3vl8b_sft_anchor_delta_step200_vllm` | Same as exp3/4 except final advantage `1.0*step_adv + 0.7*answer_adv`; `filter_min_std=0.04` | Gemini judge enabled, `streamweave_grppo_judge_v1` | `RL/outputs/runs/exp5` | Reached step 29, then entered step-30 validation over the previous full 2000-row validation set and did not finish cleanly; latest checkpoint observed at `global_step_20`. Script now caps future validation to 200 rows. |
| 2026-05-14 | ready | TBD | Experiment 6: `exp6` | `RL/scripts/train_exp6.sh` | `RL/configs/streamweave_stepwise.yaml` | `dataset2/rl_exp3.jsonl`; val same file, capped by `data.val_max_samples=200` | `models/qwen3vl8b_sft_anchor_delta_step200_vllm` | Exp5 follow-up: final advantage `1.0*step_adv + 0.5*answer_adv`; `answer_decay=0.7`; no std division; GRPPO step reward uses `0.7*judge + 0.25*format + 0.25*note_frequency`; timeline answer supervision; silence scalar `0.1`; actor KL loss enabled with coef `0.001`; validation score key `grppo_target_trajectory_score` | Gemini judge enabled, `streamweave_grppo_judge_v1`; answer label uses natural-language answer text and includes MCQ options | `RL/outputs/runs/exp6` | Prepared to test LLM-judged answer/silence supervision without trainer rule overwrite. Validation is capped to 200 rows to avoid step-30 full-set stalls. |
| 2026-05-15 | prepared | TBD | Experiment 7: `exp7` | `RL/scripts/train_exp7.sh` | `RL/configs/streamweave_stepwise.yaml` | train `dataset2/rl_0515_train.jsonl`; val `dataset2/rl_0515_val.jsonl`, capped by `data.val_max_samples=200` | `models/qwen_sft_0513` | Initial copy of exp6 for the next parameter experiment, with source model switched to `qwen_sft_0513`: timeline answer supervision, final advantage `1.0*step_adv + 0.5*answer_adv`, `answer_decay=0.7`, no std division, GRPPO step reward `0.7*judge + 0.25*format + 0.25*note_frequency`, silence scalar `0.2`, actor KL loss coef `0.001`, both `grppo_min_std` and `grppo_filter_groups.min_std` set to `0.05` | Gemini judge enabled, `streamweave_grppo_judge_v1` | `RL/outputs/runs/exp7` | Dedicated script/output directory so exp6 remains reproducible. Uses the 2026-05-15 canonical train/val split. |
| 2026-05-15 | prepared | TBD | Experiment 7 smoke: `exp7_smoke` | `RL/scripts/train_exp7_smoke.sh` | `RL/configs/streamweave_stepwise.yaml` | train `dataset2/rl_0515_train.jsonl`; val `dataset2/rl_0515_val.jsonl`, capped by `data.val_max_samples=16` | `models/qwen_sft_0513` | Two-step debug run for exp7: `train_batch_size=2`, `gen_batch_size=2`, `rollout.n=4`, real batch `8`, `total_training_steps=2`; same reward/advantage settings as exp7, including silence scalar `0.2` and both std thresholds `0.05`; GRPPO debug dumps enabled for 2 groups x 4 trajectories | Gemini judge enabled, `streamweave_grppo_judge_v1` | `RL/outputs/runs/exp7_smoke` | Use this before full exp7 to inspect reward scale, cohort stds, kept ratio, and per-turn advantages. Console-only logger, no checkpoint save, no validation during training. |
| 2026-05-15 | prepared | TBD | Experiment 8: `exp8` | `RL/scripts/train_exp8.sh` | `RL/configs/streamweave_stepwise.yaml` | train `dataset2/rl_0515_train.jsonl`; val `dataset2/rl_0515_val.jsonl`, capped by `data.val_max_samples=200` | `models/qwen_sft_0513` | Exp7 algorithm on a 2-node Ray cluster: `trainer.nnodes=2`, `trainer.n_gpus_per_node=8`, `train/gen/val_batch_size=16`, `rollout.n=8`, real batch `128`, `agent.num_workers=64`, actor KL enabled, timeline answer supervision, silence scalar `0.2`, both std thresholds `0.05` | Gemini judge enabled, `streamweave_grppo_judge_v1` | `RL/outputs/runs/exp8` | Multi-node script with explicit `head`, `worker`, and `driver` modes. Driver connects to the existing Ray cluster and does not stop Ray before launch. |
| 2026-05-17 | prepared | TBD | Experiment 9 24-GPU: `exp9_24` | `RL/scripts/train_exp9_24.sh` | `RL/configs/streamweave_stepwise.yaml` | train `dataset2/rl_0516_filter.jsonl`; val `dataset2/rl_0515_val.jsonl`, capped by `data.val_max_samples=200` | `models/qwen3vl_sft_0516_step50` | Self-contained 24-GPU exp9 script: `trainer.nnodes=3`, `trainer.n_gpus_per_node=8`, `train/gen/val_batch_size=24`, `rollout.n=8`, real batch `192`, actor lr `5e-6`, `agent.num_workers=96`, rollout `max_num_seqs=3072`, Ray object store `40GB`; uses exp9 context budget, KL, reward mix, answer decay, silence scalar, and filtering settings | Gemini judge enabled, `streamweave_grppo_judge_v1` | `RL/outputs/runs/exp9_24` | Use when three 8-GPU nodes are available. Supports `EXP9_24_RAY_ROLE=head|worker|driver` as an alias for `EXP9_RAY_ROLE`. |
| 2026-05-14 | prepared | TBD | Experiment 9: `exp9_localjudge` | `RL/scripts/train_exp9_localjudge.sh` | `RL/configs/streamweave_stepwise.yaml` | `dataset2/rl_exp3.jsonl`; val same file, capped by `data.val_max_samples=200` | `models/qwen3vl8b_sft_anchor_delta_step200_vllm` | Same single-node algorithm as exp7, but judge backend defaults to local OpenAI-compatible/vLLM endpoint: `JUDGE_BACKEND=vllm`, `JUDGE_BASE_URL=http://127.0.0.1:9000/v1`, `JUDGE_MODEL=qwen3vl-32b-judge` | Local Qwen3VL-32B judge assumed already deployed; `streamweave_grppo_judge_v1` | `RL/outputs/runs/exp9_localjudge` | Use this to switch scoring from Gemini to a local judge without changing exp6/exp7/exp8 scripts. |
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
- Validation subset: exp5 future resumes use `data.val_max_samples=200`; exp3/exp4 ran with
  the full validation file.
- Lengths: `data.max_prompt_length=6144`, `data.max_response_length=2048`.
- Stream runtime override: `data.streamweave.runtime.max_steps=0`; other stream runtime settings
  come from `RL/configs/streamweave_stepwise.yaml`.

Model initialization:

- Source model: `models/qwen3vl8b_sft_anchor_delta_step200_vllm`.
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
  `+algorithm.grppo_min_std=0.04`.
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

## Experiment 6: Exp5 Follow-Up With Timeline Answer Supervision

`exp6` keeps the exp5 GRPPO training chain and changes only the answer supervision policy and weights:

- Launch script: `RL/scripts/train_exp6.sh`.
- Run name: `exp6`; output dir: `RL/outputs/runs/exp6`.
- Dataset: train and validation both use `dataset2/rl_exp3.jsonl`; validation is capped with
  `data.val_max_samples=200`.
- Batch/runtime/model settings: same as exp5 (`train_batch_size=16`, `gen_batch_size=16`,
  `val_batch_size=16`, `val_max_samples=200`, `rollout.n=8`, `max_steps=0`, same source model).
- GRPPO estimator: `algorithm.adv_estimator=streamweave_stepwise_grppo`.
- Final advantage: `1.0*grppo_step_advantage + 0.5*grppo_answer_advantage`.
- Answer credit decay: `+algorithm.grppo_answer_decay=0.7`.
- Component normalization: `+algorithm.grppo_norm_by_std=false`, `+algorithm.grppo_min_std=0.03`.
- Step-level GRPPO filter: enabled with `+algorithm.grppo_filter_groups.min_std=0.03`.
- Step reward mix: `+data.streamweave.reward.grppo_process_weight=0.7`,
  `+data.streamweave.reward.grppo_format_weight=0.25`,
  `+data.streamweave.reward.grppo_note_frequency_weight=0.25`.
- Answer trigger policy: `+data.streamweave.reward.grppo_answer_event_mode=timeline`.
  The env derives `none/silence/answer` from the query/answer-target timeline before judging the
  step. `silence` and `answer` both use the answer-aware LLM judge prompt.
- Silence reward: `+data.streamweave.reward.grppo_silence_reward=true`,
  `+data.streamweave.reward.grppo_silence_reward_value=0.1`. In timeline mode the final silence
  scalar is `0.1 * binarize(LLM answer_reward)`.
- Forced-answer/silence cohort postprocess: `+algorithm.grppo_forced_answer_postprocess_enable=false`;
  exp6 does not overwrite LLM answer rewards in the trainer.
- Target trajectory metric: `grppo_target_trajectory_score =
  1.0*grppo_target_answer_reward + 0.0*grppo_target_format_reward`; this is for validation/debug
  observation only and does not enter GRPPO training advantages.
- Validation score key: `+algorithm.stepwise_validation_score_key=grppo_target_trajectory_score`.
- KL: actor-side KL loss is enabled with
  `actor_rollout_ref.actor.use_kl_loss=true`,
  `actor_rollout_ref.actor.kl_loss_coef=0.001`,
  `actor_rollout_ref.actor.kl_loss_type=low_var_kl`.
  `algorithm.use_kl_in_reward` remains false because the GRPPO estimator uses its own step/answer
  advantages rather than token-level reward sums.
- Judge label fix: answer-target labels prefer natural-language `answer`/`content` over MCQ
  letter-only `gt`, and MCQ options are rendered in the answer label section. For example,
  `gt="C", answer="blue sponge"` becomes `Reference answer: blue sponge` plus the A/B/C
  option list.
- Runtime compatibility fix: legacy async FSDP/Megatron ref workers dispatch `compute_ref_log_prob`
  on the actor mesh, so KL-enabled runs use the actor dispatch role for ref logprob padding.

### 2026-05-14 Bug Fix Notes

**Bug A: `grppo_filter_groups.min_std` was a dead knob.**

- Symptom: the trainer logged `grppo/filter_min_std`, but keep/drop actually used
  `grppo_step_signal_valid` and `grppo_answer_signal_valid`, which were computed earlier from
  `algorithm.grppo_min_std`.
- Impact: changing only `algorithm.grppo_filter_groups.min_std` did not change step-level GRPPO
  filtering. The exp3/exp4/exp5 scripts had `GRPPO_MIN_STD=0.07` and
  `GRPPO_FILTER_MIN_STD=0.04`, so the effective threshold was still `0.07`.
- Fix: `_maybe_apply_streamweave_grppo_filter` now computes cohort std directly from
  `grppo_step_reward` and `grppo_answer_credit`, then compares those std values with
  `algorithm.grppo_filter_groups.min_std`. It also logs `grppo/advantage_min_std` separately so
  the advantage-zeroing threshold and filter threshold are visible independently.
- Script alignment: exp3/exp4/exp5 now set both `GRPPO_MIN_STD=0.04` and
  `GRPPO_FILTER_MIN_STD=0.04`. Exp6 already sets both to `0.03`.
- Verification after fix: `py_compile` for `ray_trainer.py`, `bash -n` for exp3/exp4/exp5/exp6
  scripts, and `RL/streamweave_rl/smoke_test.py` all pass.

## Experiment 7: Parameter-Tuning Fork From Exp6

`exp7` is a separate launch script/output directory forked from exp6 so the next parameter changes do
not affect exp6 reproducibility.

- Launch script: `RL/scripts/train_exp7.sh`.
- Run name: `exp7`; output dir: `RL/outputs/runs/exp7`; Ray temp dir: `/tmp/swray_exp7_$$`.
- Current initial config is identical to exp6:
  `train_batch_size=16`, `gen_batch_size=16`, `val_batch_size=16`, `val_max_samples=200`,
  `rollout.n=8`, `answer_decay=0.7`, `answer_weight=0.5`, `norm_by_std=false`,
  `grppo_min_std=0.03`, `grppo_filter_groups.min_std=0.03`, timeline answer supervision,
  `silence_reward_value=0.1`, forced answer postprocess disabled, target validation score
  `grppo_target_trajectory_score`, and actor KL loss enabled with coef `0.001`.
- Pending: record the actual parameter deltas before launch.

## Experiment 8: 2-Node Multi-Node GRPPO

`exp8` is the two-node version of the current exp7 GRPPO chain. It follows the VERL
multi-node launch pattern: start a Ray head, attach worker nodes, confirm the cluster with
`ray status`, then run `verl.trainer.main_ppo` from the driver with `trainer.nnodes` and
`trainer.n_gpus_per_node` set to the actual cluster size.

- Launch script: `RL/scripts/train_exp8.sh`.
- Run name: `exp8`; output dir: `RL/outputs/runs/exp8`; Ray temp dir: `/tmp/swray_exp8_$$`.
- Script modes:
  - `EXP8_RAY_ROLE=head`: starts the Ray head and exits.
  - `EXP8_RAY_ROLE=worker`: starts a Ray worker connected to `RAY_HEAD_IP:RAY_PORT` and exits.
  - `EXP8_RAY_ROLE=driver`: connects to the existing Ray cluster and launches training.
- Default cluster shape: `NNODES=2`, `N_GPUS_PER_NODE=8`, so `trainer.nnodes=2` and
  `trainer.n_gpus_per_node=8`.
- Batch sizes stay at exp7 `16`: with `rollout.n=8`, the real train batch is `16*8=128`,
  which is divisible by the 16 total GPUs.
- Agent rollout workers are raised from single-node exp7 `32` to `64`.
- Ray object store memory for head/worker startup defaults to `28000000000` bytes to avoid the
  40 GB `/dev/shm` mismatch seen on some pods; override with `RAY_OBJECT_STORE_MEMORY=...` if the
  target node has more shared memory.
- Driver mode passes `+ray_kwargs.ray_init.address=${RAY_HEAD_IP}:${RAY_PORT}` or `$RAY_ADDRESS`.
  It intentionally does not pass `num_cpus`, `object_store_memory`, or `_temp_dir` into
  `ray.init(address=...)` because those resources belong to the already-started Ray cluster.
- Driver mode also does not run `ray stop --force`; head/worker modes stop only their local Ray
  process before starting the requested role.
- Head/worker Ray daemons are started with the same critical environment as the driver
  (`PYTHONPATH`, `STREAMWEAVE_RL_DIR`, `GOOGLE_APPLICATION_CREDENTIALS`, trace settings, and Ray
  log/runtime flags). This matters because multi-node Ray is started before the VERL driver, so the
  worker-side Python processes cannot rely on inheriting the driver's shell environment.
- Algorithm parameters currently match exp7 except for multi-node scale:
  `answer_decay=0.7`, `answer_weight=0.5`, `norm_by_std=false`, `grppo_min_std=0.05`,
  `grppo_filter_groups.min_std=0.05`, timeline answer supervision, silence scalar `0.2`,
  forced answer postprocess disabled, target validation score `grppo_target_trajectory_score`,
  and actor KL loss coef `0.001`.

Launch commands:

Current 2-node assignment for the planned exp8 run:

- Head node: `aiplatform-wlf2-ge42-28`, `RAY_HEAD_IP=10.82.121.78`.
- Worker node: `aiplatform-wlf2-ge26-34`, local IP `10.82.122.215`.
- Always pass the head IP, `10.82.121.78`, as `RAY_HEAD_IP` on both machines and in the
  driver command.

Execution order:

1. Use `aiplatform-wlf2-ge42-28` as the Ray head.
2. Run the head command on that machine.
3. Run the worker command on `aiplatform-wlf2-ge26-34`.
4. Check `ray status --address=10.82.121.78:6379` from the head machine until it shows
   two nodes and 16 GPUs.
5. Run the driver command on the head machine. Only this final driver command launches VERL
   training; the head/worker commands only start the Ray cluster.

Head machine:

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
RAY_HEAD_IP=10.82.121.78 EXP8_RAY_ROLE=head bash RL/scripts/train_exp8.sh
```

Worker machine:

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
RAY_HEAD_IP=10.82.121.78 EXP8_RAY_ROLE=worker bash RL/scripts/train_exp8.sh
```

Check cluster status on the head machine:

```bash
/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/ray status --address=10.82.121.78:6379
```

After the status output shows two nodes and 16 GPUs, launch the driver on the head machine:

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
RAY_HEAD_IP=10.82.121.78 EXP8_RAY_ROLE=driver bash RL/scripts/train_exp8.sh
```

The script assumes both machines can see the same repo, dataset, frames, source model, and
output directory paths under `/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5`.

### 2026-05-15 OVO 1/8 Step-40 Error Analysis

Exported model under evaluation:

- Checkpoint: `RL/outputs/runs/exp8/checkpoints/global_step_40/actor`.
- HF export: `models/qwen3_rl_exp8_step40`.
- OVO 1/8 output: `outputs/ovo_qwen3_rl_exp8_step40_1of8_6gpu`.
- Current trace source: `outputs/ovo_qwen3_rl_exp8_step40_1of8_6gpu/traces/{sample_id}/trace.jsonl`.
- Gemini comparison source: `outputs/ovo_gemini_1of8_state_note_t/traces/{sample_id}/trace.jsonl`.
- Final Gemini-retry loss report: `RL/scripts/EXP8_STEP40_GEMINI_RETRY_ERROR_REPORT.md`.

The first focused set is the early same-id cases where Gemini answered correctly but
`qwen3_rl_exp8_step40` did not: `20`, `28`, `36`, `100`, `164`, `188`.

Five-way same-id snapshot at `2026-05-15 14:46:33` while the exp8 step40 OVO run was still
in progress:

| Model / run | Same-id score |
| --- | ---: |
| `exp8_step40` | `106/158 = 67.09%` |
| `gemini_state` (`outputs/ovo_gemini_1of8_state_note_t`) | `104/158 = 65.82%` |
| `gemini_retry` (`outputs/ovo_gemini_full_retry`) | `117/158 = 74.05%` |
| `base` (`outputs/ovo_qwen3vl8b_base_full_state_note_t`) | `104/158 = 65.82%` |
| `sft_0513` (`outputs/ovo_qwen_sft_0513_1of8`) | `98/158 = 62.03%` |

Relative to `exp8_step40` at this snapshot:

- vs `gemini_state`: exp8 wins `23`, loses `21`, net `+2`.
- vs `gemini_retry`: exp8 wins `13`, loses `24`, net `-11`.
- vs `base`: exp8 wins `18`, loses `16`, net `+2`.
- vs `sft_0513`: exp8 wins `17`, loses `9`, net `+8`.

Current `gemini_retry` correct but `exp8_step40` wrong cases to inspect:

```text
28, 92, 100, 148, 164, 301, 438, 462, 541, 557, 572, 580,
619, 683, 780, 804, 845, 957, 1030, 1143, 1151, 1159, 1206, 1276
```

Answer-level snapshot for these cases:

| sample_id | exp8 step40 | gemini_retry | GT |
| --- | --- | --- | --- |
| `28` | empty | `A` | `A` |
| `92` | `D` | `C` | `C` |
| `100` | `A. yes` | `B` | `B` |
| `148` | `B` | `A` | `A` |
| `164` | `A` | `B` | `B` |
| `301` | `B` | `A` | `A` |
| `438` | `C` | `B` | `B` |
| `462` | `E` | `B` | `B` |
| `541` | `D` | `C` | `C` |
| `557` | `B` | `C` | `C` |
| `572` | empty | `A` | `A` |
| `580` | empty | `C` | `C` |
| `619` | `D` | `C` | `C` |
| `683` | `D` | `B` | `B` |
| `780` | `D` | `A` | `A` |
| `804` | `A` | `C` | `C` |
| `845` | `C` | `A` | `A` |
| `957` | `A. A metal clasp.` | `C` | `C` |
| `1030` | `A. There is a green traffic light illuminated.` | `B` | `B` |
| `1143` | empty | `B` | `B` |
| `1151` | empty | `B` | `B` |
| `1159` | `B` | `C` | `C` |
| `1206` | empty | `A` | `A` |
| `1276` | `B` | `A` | `A` |

#### Gemini-retry loss queue at ~205/364 progress

At the `2026-05-15 15:08:04` live snapshot, `exp8_step40` was `137/205 = 66.83%`,
while `gemini_retry` was `152/205 = 74.15%`. The monitor showed `28` cases where
`gemini_retry` was correct and `exp8_step40` was wrong. A direct read of the live
`.results_parts` immediately afterwards had advanced one more case, adding `1520_1`;
the current queue to inspect is therefore `29` cases:

```text
28, 92, 100, 148, 164, 301, 438, 462, 541, 557, 572, 580,
619, 683, 780, 804, 845, 957, 1030, 1143, 1151, 1159, 1206, 1276,
1472_3, 1480_2, 1496_4, 1520_0, 1520_1
```

| sample_id | exp8 step40 | GT | gemini_retry | base score | sft_0513 score |
| --- | --- | --- | --- | ---: | ---: |
| `28` | empty | `A` | `A` | `1` | `1` |
| `92` | `D` | `C` | `C` | `0` | `0` |
| `100` | `A. yes` | `B` | `B` | `0` | `0` |
| `148` | `B` | `A` | `A` | `0` | `0` |
| `164` | `A` | `B` | `B` | `0` | `0` |
| `301` | `B` | `A` | `A` | `0` | `0` |
| `438` | `C` | `B` | `B` | `0` | `0` |
| `462` | `E` | `B` | `B` | `1` | `0` |
| `541` | `D` | `C` | `C` | `0` | `0` |
| `557` | `B` | `C` | `C` | `1` | `0` |
| `572` | empty | `A` | `A` | `1` | `1` |
| `580` | empty | `C` | `C` | `1` | `1` |
| `619` | `D` | `C` | `C` | `1` | `0` |
| `683` | `D` | `B` | `B` | `0` | `0` |
| `780` | `D` | `A` | `A` | `0` | `0` |
| `804` | `A` | `C` | `C` | `1` | `0` |
| `845` | `C` | `A` | `A` | `0` | `1` |
| `957` | `A. A metal clasp.` | `C` | `C` | `1` | `0` |
| `1030` | `A. There is a green traffic light illuminated.` | `B` | `B` | `0` | `0` |
| `1143` | empty | `B` | `B` | `1` | `1` |
| `1151` | empty | `B` | `B` | `1` | `1` |
| `1159` | `B` | `C` | `C` | `0` | `0` |
| `1206` | empty | `A` | `A` | `1` | `1` |
| `1276` | `B` | `A` | `A` | `1` | `0` |
| `1472_3` | `No` | `1` | correct long-form yes answer | `0` | `0` |
| `1480_2` | `No` | `1` | correct long-form yes answer | `0` | `0` |
| `1496_4` | `No` | `1` | correct long-form yes answer | `0` | `0` |
| `1520_0` | `No` | `1` | `Yes` | `1` | `1` |
| `1520_1` | `No` | `1` | `Yes` | `0` | `1` |

Priority buckets for manual trace analysis:

- Empty-answer / abstention despite QA: `28`, `572`, `580`, `1143`, `1151`, `1206`.
- Multiple-choice wrong-letter choices: `92`, `100`, `148`, `164`, `301`, `438`, `462`,
  `541`, `557`, `619`, `683`, `780`, `804`, `845`, `957`, `1030`, `1159`, `1276`.
- Yes/no temporal or causal misses: `1472_3`, `1480_2`, `1496_4`, `1520_0`, `1520_1`.

Main failure taxonomy from the final-step traces:

| Failure type | Cases | Trace diagnosis |
| --- | --- | --- |
| QA present but model abstains or says no question | `1143`, `1151`, `1206`, `1520_0`, `1520_1` | Final prompts contain QA and images, but exp8 raw output says "no question" or leaves `<answer>` empty while describing the answer-relevant scene. This is model-side answer gating / prompt-following failure. |
| Over-conservative "not enough evidence" | `28`, `572`, `580`, `1472_3`, `1480_2`, `1496_4` | The model often sees partial evidence but refuses to decide. For recipe/procedure and CRR questions it waits for a future or completed action even when the target answer is inferable from the current/latest evidence or memory. |
| Past-memory / object-location tracking failure | `28`, `92`, `148`, `164`, `619` | The model either loses the object in memory (`28`), answers from the current visible support instead of the earlier queried location (`92`, `148`, `619`), or substitutes a likely object class from context instead of the actual inserted object (`164`). |
| Procedure order confusion | `541`, `557`, `572`, `580` | The model confuses previous and next cooking steps. It answers an earlier ingredient step for "after noodles" (`557`), chooses tomato puree instead of cover pan after chicken/spices (`541`), or abstains when the recipe sequence is enough (`572`, `580`). |
| Overconfident answer where GT is unable | `301`, `438`, `462` | These HLD cases have `Unable to answer` as GT. Exp8 hallucinates a specific answer from weak visual evidence: rice cooker location, closet vs bedroom door, chandelier target. |
| Fine visual detail error | `683`, `780`, `804`, `845`, `957`, `1030`, `1276` | These are mostly visual discrimination failures: fridge side compartment, facing direction, court depth/audience side, counting three dogs, zip tie vs metal clasp, traffic light color, and standing still vs dancing/sipping. |
| Option granularity / semantic mismatch | `100`, `1159`, `1206` | The model's observation is plausible but maps to the wrong option granularity. `100` is borderline because the current frame visually looks open while GT/Gemini say closed from memory; keep it as an annotation/timestamp ambiguity candidate. `1159` picks dishwashing while GT asks broader cleaning. `1206` has a possible label/scene mismatch. |

Case-level short diagnoses:

| sample_id | Diagnosis |
| --- | --- |
| `28` | Book/notebook was not retained in memory; exp8 says no book visible and abstains, while base/SFT/Gemini answer table. |
| `92` | Soy sauce location is in a box; exp8 latches onto the later/current bottle/sink area. |
| `100` | Borderline visual/timestamp ambiguity: exp8 sees drawer open at `395-396s`, but Gemini memory says it was closed and GT is no. |
| `148` | Asked source location before picking the litter bin; exp8 answers physical support "on the floor" instead of room-level "stock room". |
| `164` | Washing-machine object identity failure: exp8 infers clothes from context, but GT is detergent. |
| `301` | GT is Unable; exp8 overconfidently claims top cabinet based on weak/old evidence. |
| `438` | Bedroom door question; exp8 answers about closed closet doors, not the bedroom door. |
| `462` | Chandelier target not determined; exp8 over-interprets background and chooses living room. |
| `541` | Recipe order error: after chicken/spices/mix should cover pan; exp8 chooses earlier tomato puree step. |
| `557` | Recipe order error: after noodles/stir should add sauce; exp8 chooses vegetables/chicken step before noodles. |
| `572` | State already contains drain/rinse beans but exp8 abstains because sauteing garlic has not happened yet. |
| `580` | Exp8 treats the spring-roll frying follow-up as unanswerable future; Gemini uses recipe sequence and answers remove to paper towels. |
| `619` | Box action verb error: trace shows putting box down before sitting; exp8 says opened. |
| `683` | Fine fridge-location error: chooses door middle holder instead of second/fourth side compartments. |
| `780` | Facing-direction error: girl is facing left; exp8 says forward. |
| `804` | Court-depth/perspective error: white-shirt players are closer to audience; exp8 says red. |
| `845` | Counting miss: exp8 answers two dogs although a third dog enters. |
| `957` | Attachment object miss: zip tie mistaken for metal clasp. |
| `1030` | Traffic light color/stale-memory error: exp8 answers green, GT/Gemini yellow. |
| `1143` | Drawer/cabinet naming mismatch triggers abstention despite clear option `B`. |
| `1151` | Sink/faucet action anticipation failure; exp8 describes the faucet-adjacent action but says no question. |
| `1159` | Option granularity: exp8 picks washing dishes; GT is broader "clean with sponge". |
| `1206` | Possible noisy-label case: trace is workbench/tool scene but option says napkin/table; exp8 still wrongly abstains despite QA. |
| `1276` | Action classification error: woman is standing and drinking; exp8 over-interprets as dancing while sipping. |
| `1472_3` | CRR future-intent miss: exp8 sees a person near a tree but says insufficient; Gemini uses latest context to answer yes. |
| `1480_2` | CRR completion miss: exp8 only records holding books, missing that the man places them on the table. |
| `1496_4` | Causal memory miss: reason for prisoners' excitement is guards spraying water; exp8 looks only at later distressed conversation. |
| `1520_0` | SSR step is happening: adding beetroot/raw material to processor; exp8 describes it but says no question. |
| `1520_1` | Same SSR failure extended: exp8 describes adding raw materials but outputs empty/no. |

##### Live degradation after ~205 samples

At a live `.results_parts` read with `353` completed samples, exp8 step40 was
`180/353 = 50.99%`. The drop is not a change in the model; the later evaluation slice is
dominated by task types where exp8 is weak:

| Slice | exp8 | gemini_state | gemini_retry | base | sft_0513 | Main contents |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| first `205` | `138/205 = 67.32%` | `136/205 = 66.34%` | `152/205 = 74.15%` | `130/205 = 63.41%` | `130/205 = 63.41%` | mixed OVO tasks |
| `205-270` | `29/65 = 44.62%` | `47/65 = 72.31%` | `45/65 = 69.23%` | `33/65 = 50.77%` | `40/65 = 61.54%` | mostly CRR + SSR |
| `270-current` | `13/83 = 15.66%` | `35/83 = 42.17%` | `25/83 = 30.12%` | `25/83 = 30.12%` | `41/83 = 49.40%` | mostly REC counting |
| `205-current` | `42/148 = 28.38%` | `82/148 = 55.41%` | `70/148 = 47.30%` | `58/148 = 39.19%` | `81/148 = 54.73%` | SSR/REC-heavy tail |

By task type in the degrading tail:

- `SSR/forward`: `33/69 = 47.8%`. The model frequently outputs `No` or empty even when the
  prompted step is visibly happening, e.g. `1520_0`, `1520_1`, `1528_0`, `1545_1`.
- `REC/forward`: `6/70 = 8.6%`. The model badly undercounts repeated actions or fails to
  answer numeric-count questions. Examples: `1562_0-1562_4`, `1578_0-1578_4`,
  `1603_14-1603_15`, `1635_0-1635_6`.
- `CRR/forward`: `3/9 = 33.3%`. Some CRR questions ask only whether enough evidence exists,
  but exp8 outputs content answers or stale object labels instead of `Yes/No`, e.g. `1504_2`,
  `1512_2`.

Representative trace evidence:

- `1520_0`: QA asks if the current tutorial step is "add raw materials". Exp8 state says the
  woman is pouring the yellow bowl into the food processor, but then says "There is no
  question to answer" and outputs empty/`No`. This is the same answer-gating failure seen
  earlier.
- `1528_0`: QA asks "remove pumpkin pedicle". Exp8 state says the man begins cutting around
  the pumpkin with a knife, then again says no QA and outputs empty/`No`.
- `1562_0`: REC asks how many times someone shows something to the camera. Exp8 state says a
  person is holding up a jersey and displaying it to the camera, but outputs empty instead of
  `1`.
- `1603_15`: REC asks total pole vault count. Exp8 answers `5` while GT is `7`; the memory
  stores generic "multiple vaults" instead of a robust running count.
- `1635_6`: REC asks total cliff diving count. Exp8 state says one person jumped twice and a
  second person is mid-air, but outputs empty/`0` while GT is `5`; repeated-event counting is
  not preserved in memory.

##### Final 364/364 task-level comparison

Final live monitor snapshot at `2026-05-15 15:28:36`: exp8 step40 is
`180/364 = 49.45%`, Gemini state is `221/364 = 60.71%`, Gemini retry is
`225/364 = 61.81%`, and base is `189/364 = 51.92%`. Same-id task-level comparison
including SFT:

| Task | n | exp8 | gemini_state | gemini_retry | base | sft_0513 | exp8 vs base | exp8 vs sft |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ACR | 14 | `12/14 = 85.7%` | `13/14 = 92.9%` | `13/14 = 92.9%` | `12/14 = 85.7%` | `11/14 = 78.6%` | `+0.0` | `+7.1` |
| ASI | 19 | `9/19 = 47.4%` | `10/19 = 52.6%` | `14/19 = 73.7%` | `13/19 = 68.4%` | `11/19 = 57.9%` | `-21.1` | `-10.5` |
| ATR | 14 | `11/14 = 78.6%` | `9/14 = 64.3%` | `12/14 = 85.7%` | `10/14 = 71.4%` | `12/14 = 85.7%` | `+7.1` | `-7.1` |
| CRR | 30 | `12/30 = 40.0%` | `14/30 = 46.7%` | `16/30 = 53.3%` | `8/30 = 26.7%` | `12/30 = 40.0%` | `+13.3` | `+0.0` |
| EPM | 37 | `22/37 = 59.5%` | `23/37 = 62.2%` | `24/37 = 64.9%` | `19/37 = 51.4%` | `20/37 = 54.1%` | `+8.1` | `+5.4` |
| FPD | 13 | `7/13 = 53.8%` | `8/13 = 61.5%` | `11/13 = 84.6%` | `9/13 = 69.2%` | `9/13 = 69.2%` | `-15.4` | `-15.4` |
| HLD | 23 | `17/23 = 73.9%` | `16/23 = 69.6%` | `13/23 = 56.5%` | `13/23 = 56.5%` | `10/23 = 43.5%` | `+17.4` | `+30.4` |
| OCR | 19 | `18/19 = 94.7%` | `18/19 = 94.7%` | `18/19 = 94.7%` | `16/19 = 84.2%` | `17/19 = 89.5%` | `+10.5` | `+5.3` |
| OJR | 23 | `18/23 = 78.3%` | `15/23 = 65.2%` | `19/23 = 82.6%` | `19/23 = 82.6%` | `18/23 = 78.3%` | `-4.3` | `+0.0` |
| REC | 81 | `6/81 = 7.4%` | `30/81 = 37.0%` | `20/81 = 24.7%` | `20/81 = 24.7%` | `31/81 = 38.3%` | `-17.3` | `-30.9` |
| SSR | 69 | `33/69 = 47.8%` | `51/69 = 73.9%` | `49/69 = 71.0%` | `36/69 = 52.2%` | `47/69 = 68.1%` | `-4.3` | `-20.3` |
| STU | 22 | `15/22 = 68.2%` | `14/22 = 63.6%` | `16/22 = 72.7%` | `14/22 = 63.6%` | `13/22 = 59.1%` | `+4.5` | `+9.1` |

Takeaways:

- Clear gains vs base/SFT: `HLD`, `OCR`, `EPM`, `STU`, and partly `CRR`/`ATR`.
- Clear regressions: `REC`, `SSR`, `ASI`, `FPD`.
- `REC` is the largest single blocker: only `6/81 = 7.4%`, losing `25` net cases vs
  SFT and `14` net cases vs base/Gemini retry. This task needs explicit running-count
  memory rather than generic event summaries.
- `SSR` remains a prompt-following / answer-gating issue: the model often describes the
  queried step but still answers `No` or empty.

##### Detailed trace investigation: ASI / CRR / REC / FPD

Raw final and intermediate trace outputs were inspected for all wrong cases in these four
tasks:

- `ASI`: `10` wrong out of `19`.
- `CRR`: `18` wrong out of `30`.
- `REC`: `75` wrong out of `81`.
- `FPD`: `6` wrong out of `13`.

###### ASI: procedure-order reasoning is not step-indexed

Wrong cases: `494`, `502`, `510`, `518`, `541`, `549`, `557`, `572`, `580`, `619`.

Important split:

- Not exp8-specific / likely ambiguous or hard labels: `494`, `502`, `510`, `518`.
  Gemini/base/SFT are also wrong on these. Example `494`: the prompt asks what happens
  after loading a new phone battery, but the current trace is still removing the old battery;
  exp8 raw state says "the new battery has not been loaded yet" and answers `D` ("take down
  the old battery"), while GT is `B`.
- Exp8-specific or actionable failures: `541`, `557`, `572`, `580`, `619`.

Representative raw evidence:

- `541`: prompt asks what happens after adding chicken/salt/coriander/garam masala and mixing.
  Exp8 state reviews earlier spice/tomato steps and answers `D` ("add tomato puree"), but
  Gemini bridge reaches the later action: "cover the pan". The model is anchoring to an
  earlier ingredient step instead of the target step boundary.
- `557`: prompt asks after adding noodles and stirring. Exp8 answers `B` ("add carrots green
  onion and chicken") although Gemini bridge shows those were before noodles; after noodles is
  sauce. This is a before/after alignment error.
- `572`: exp8 state itself says the woman drains/rinses cannellini beans, but because sauteing
  garlic has not happened yet it outputs empty. The question is answerable from the recipe
  sequence/options, so this is over-conservative gating.
- `580`: exp8 sees the oil/wok stage but says the step after golden-brown rolls is future and
  unanswerable. Gemini chooses the recipe-consistent next step `C`.
- `619`: exp8 raw bridge says the person places the box on the floor, but answer is `D`
  ("opened"). The correct option is `C` ("put down"). This is verb/action mapping failure.

ASI improvement plan:

- Add a procedure memory format, not just free-text deltas. Example fields:
  `step_id`, `time`, `action`, `objects`, `status=observed|inferred|future_recipe`.
- For ASI prompts containing `before`/`after`, force a two-stage decision:
  identify the anchor step in memory/options, then choose the adjacent step.
- Do not let "future step not shown" cause empty answers when the options are procedural
  recipe/tutorial steps and one option is the natural next/previous step.
- Add a repair path for multiple-choice ASI when QA exists but `<answer>` is empty.

###### CRR: format confusion plus late-evidence misses

Wrong cases: `1472_2`, `1472_3`, `1472_4`, `1480_2`, `1480_3`, `1480_4`,
`1488_2`, `1488_3`, `1488_4`, `1496_2`, `1496_3`, `1496_4`, `1504_2`,
`1504_3`, `1504_4`, `1512_2`, `1512_3`, `1512_4`.

Raw-output summary over the `18` wrong CRR cases:

- `8` cases output `No` while GT is `Yes`.
- `10` cases output a content answer instead of the required `Yes/No`.
- All `18` wrong traces contain many states with "no question" language; later QA is present,
  but the model often keeps treating the task as normal content QA or silence.

Representative raw evidence:

- `1504_2`: prompt explicitly says "Decide whether ... enough information ... Answer only
  Yes or No." Exp8 result is "The woman finds a snake on the floor in her room." This content
  proves the answer should be `Yes`, but the output format is wrong.
- `1512_2`: asks whether there is enough evidence to know what is in the blue box. Exp8
  says "The blue box contains a keychain" instead of `Yes`, and the object is also wrong
  compared with Gemini's necklace/pendant trace.
- `1488_3`: asks whether the video has enough information about what the man does next with
  the yellow life float. Exp8 outputs a descriptive sentence about pulling the float, not
  `Yes/No`.
- `1480_2` and `1496_4`: exp8 answers `No` even though later memory contains the books being
  placed/handed over or the prisoners being sprayed with water. These are late-evidence misses.

CRR improvement plan:

- Make CRR a separate prompt template: "This is a sufficiency-classification task, not a
  content-answer task. Output `Yes` if the content answer is knowable, otherwise `No`."
- Add few-shot examples where a content answer is available but the required output is `Yes`.
- Add a repair/postprocess rule for CRR: if the answer is neither `Yes` nor `No`, retry with
  only the question and the relevant memory; as a weaker heuristic, non-empty content answers
  usually imply `Yes`.
- Keep an active QA flag in memory so "no question" boilerplate cannot dominate after a CRR
  question appears.

###### REC: no reliable running counter

Wrong cases: `75/81`. The full sequences show systematic undercounting:

| video prefix | action | GT sequence | exp8 sequence |
| --- | --- | --- | --- |
| `1562` | showing something to the camera | `1,2,3,4,4` | empty, empty, `2,2,2` |
| `1570` | hitting something against/with something | `0,1,2,3,4,5,5` | `0,0,0,0,0,0,0` |
| `1578` | showing something to the camera | `1,2,3,4,4` | empty, `0,0,0,0` |
| `1586` | showing something to the camera | `0,1,2,3,4,4` | all `0` |
| `1594` | opening something | `1,1,1,1` | empty, `0,0,0` |
| `1603` | pole vault | `0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,7` | `0,0,1,1,1,2,2,3,2,4,4,4,4,4,5,5` |
| `1611` | tennis swing | `0,0,1,1,2,3,3,3` | all `0` |
| `1619` | clean and jerk | `1,1,1` | all `0` |
| `1627` | shotput | `1,2,3,3,4,5,5,6,6,7,7` | `0,1,1,1,1,0,1,1,3,5,1` |
| `1635` | cliff diving | `1,1,2,3,4,4,5,5,6,6,7,8,8,8,8,8` | `0,0,0,1,1,2,0,4,1,0,1,5,1,1,3,7` |

Raw-output summary over the `75` wrong REC cases:

- `71/75` traces have states containing "no question" or "no question to answer" even though
  the REC QA is present from the start.
- `4/75` are empty answers; most others are numeric but stale or undercounted.
- The model usually describes the visible action, but the memory stores generic text such as
  "continues" or "multiple instances" instead of a countable event ledger.

Representative raw evidence:

- `1562_0`: prompt asks count of "showing something to the camera"; exp8 state says the
  person is holding up a jersey and displaying it to the camera, then says no question and
  outputs empty instead of `1`.
- `1603_15`: pole vault GT is `7`; exp8 answers `5`. Its state says "multiple instances"
  rather than listing completed vaults, so the count cannot be recovered.
- `1635_6`: cliff diving GT is `5`; exp8 state says one person jumped twice and a second
  person is mid-air, then outputs empty/`0`. Count state is not preserved.

REC improvement plan:

- Add a dedicated counter memory for REC:
  `counter_name`, `count`, `last_counted_event_time`, `pending_event`, `evidence`.
- Increment only on completed action cycles, with debouncing so one long action does not count
  multiple times.
- Output the current count at every REC QA step; never output "no question" when a counting QA
  is active.
- Train/reward intermediate count correctness, not only final trace quality. The current delta
  rewards do not force count preservation.

###### FPD: anticipation plus option mapping

Wrong cases: `1120`, `1143`, `1151`, `1159`, `1198`, `1206`.

Important split:

- Strong exp8 failures: `1143`, `1151`, `1159`.
- Likely ambiguous / label-boundary cases: `1120`, `1198`, `1206`.

Representative raw evidence:

- `1143`: exp8 says the person put clothing in a drawer and opens a cabinet, then outputs
  empty because it thinks there is no drawer question. This is drawer/cabinet synonym mismatch
  plus answer gating.
- `1151`: exp8 describes sink/faucet-adjacent action but says "There is no question to
  answer"; Gemini maps it to opening the faucet.
- `1159`: exp8 selects `B` ("wash dishes") because a sponge is being used on dishes; GT is
  broader `C` ("clean with sponge"). This is option granularity.
- `1198`: exp8 and Gemini both answer power drill, but GT is saw. Raw trace shows the person
  reaching for/picking up a power drill, so this looks like annotation/timestamp ambiguity.
- `1206`: trace scene is workbench/tools while option says napkin/table; likely label/scene
  mismatch, though exp8 still incorrectly abstains.

FPD improvement plan:

- Add a prompt rule for "about to / preparing to": choose the most likely option from affordance
  and immediate hand/object trajectory; do not require the future action to complete.
- Add synonym/affordance normalization for furniture and tools (`drawer`/`cabinet`,
  `cleaning`/`washing dishes`, `faucet`/`sink`).
- Add empty-answer repair for multiple-choice FPD.
- Maintain a small label-noise list for cases like `1198` and `1206` so they do not dominate
  training conclusions.

##### Refined conclusions for next iteration

1. `ASI` in this dataset is best understood as action-sequence inference over instructional
   videos. The question names a procedure step and asks what happens before/after it. The
   model must recover the ordered recipe/tutorial steps, not merely describe the latest
   visible frame. Delta oversimplification hurts this because it removes the exact step
   boundaries and action-object pairs needed for before/after alignment.

2. `572` and `580` are not mainly "missing memory" cases. In both traces, exp8 has enough
   evidence in raw state/memory but applies the wrong answering policy:

   - `572`: dataset row is YouCook2, question at `221s`, answer is `A` ("drain and rinse
     cannellini beans and set aside"). Exp8 timeline records holding cans of cannellini beans
     at `125-150s`, pouring/rinsing one can at `155-180s`, and pouring the second can at
     `215-221s`. Final raw state says the woman has been preparing ingredients including
     rinsing beans, but then refuses because "sauteing minced garlic" has not happened yet.
     The error is over-conservative procedural gating: it treats the named anchor step as
     needing to be observed before answering, although the before-step is already observed.
   - `580`: dataset row is YouCook2, question at `458s`, answer is `C` ("remove brown rolls
     and place them on paper towels"). Exp8 timeline records filling preparation, wrapper
     filling/rolling, several finished rolls on a baking sheet at `445-450s`, and oil/wok at
     `450-458s`. Final raw state says this is a future step and refuses. The error is again
     not missing images but a mismatch between live evidence policy and recipe/procedure
     semantics. For ASI, the model should use the procedure context and options to answer
     adjacent steps even if the named future anchor is not yet visually completed.

3. `CRR` data has a built-in format trap. The dataset row stores a content question and answer
   (e.g. `1504`: question asks what frightened the woman; answer is "A snake"), plus
   `test_info` with `type=0/1` at different timestamps. The evaluation prompt asks whether
   enough visual evidence is available and requires `Yes/No`. Exp8 often answers the content
   question instead of the sufficiency question. Future prompts/training should separate
   content-answer QA from sufficiency-classification QA and include explicit few-shot examples
   where the content answer is known but the required output is only `Yes`.

4. `FPD` is not only entity recognition. The failures mix several causes:

   - object naming/synonym mismatch: `1143` uses `cabinet` vs question `drawer`.
   - action affordance / anticipation: `1151` sees sink/faucet-adjacent movement but does not
     map it to "open faucet".
   - option granularity: `1159` picks "wash dishes" while GT uses broader "clean with sponge".
   - possible label/timestamp noise: `1198` raw trace supports power drill while GT says saw;
     `1206` trace is workbench/tools while option says napkin/table.

5. For the next iteration, the highest-impact changes should be:

   - Add task-aware memory fields: procedure step ledger for `ASI`, running counter for `REC`,
     active sufficiency flag for `CRR`, and affordance/object-normalization hints for `FPD`.
   - Penalize or repair `QA present + empty answer` and `QA present + state says no question`.
   - Add a CRR-specific output validator: only `Yes/No` is valid; non-empty content answers
     should be retried.
   - Add REC-specific reward on the numeric count at every step, not only generic delta quality.

#### Empty-answer group: 28 / 572 / 580 / 1143 / 1151 / 1206

Checked the raw final-step traces against `gemini_retry` for these six cases. The final
prompts all contain the QA block, all contain nonzero prompt images, and all include the
current frame(s). The current model's raw outputs literally contain empty answers such as
`<answer></answer>`, so this is not an answer parser bug. The current model also describes
the visible scene in the generated state, so the evidence points to model-side answer gating
or visual/action reasoning failures rather than missing image input.

| sample_id | Final current frames | prompt images | exp8 raw answer | gemini_retry | Main diagnosis |
| --- | --- | ---: | --- | --- | --- |
| `28` | `210-214s` | `21` | empty | `A` | Object/memory tracking failure. The prompt asks where the carried book ended up; exp8 says no book is visible and abstains, while Gemini uses the bridge that the notebook/book was placed on the table. |
| `572` | `220-221s` | `19` | empty | `A` | Process-order reasoning failure. Exp8's own state says the person drained/rinsed cannellini beans, but it refuses because sauteing garlic has not happened yet. |
| `580` | `455-458s` | `18` | empty | `C` | Future/procedure reasoning failure. Exp8 sees the oil/wok stage but treats "after cooking until golden brown" as unanswerable, while Gemini picks the recipe-consistent next step. |
| `1143` | `15-16s` | `3` | empty | `B` | Prompt-following/object-name mismatch. Exp8 sees the person moving clothing into a drawer/cabinet area but says there is no drawer question, despite QA being present. |
| `1151` | `235-240s` | `22` | empty | `B` | Prompt-following/action anticipation failure. Exp8 describes sink/faucet-adjacent activity but still says there is no question to answer; Gemini maps the motion to opening the faucet. |
| `1206` | `25-28s` | `7` | empty | `A` | Prompt-following plus possible dataset label/noisy-scene issue. The trace scene is workbench/tools, while the option says napkin/table; exp8 still saw the image and abstained instead of choosing. |

Actionable hypothesis: these are mostly "answer required but model abstains" cases. A useful
debug metric is `QA present + empty <answer>` rate. For evaluation, a repair pass could retry
only these cases with a hard instruction to choose one option. For RL/prompting, these should
be penalized separately from wrong-letter choices because the failure mode is different.

##### Sample 1143 detail

Question appears at `t=15.1`:

```text
Question: What is this person doing with the drawer?
Options:
A. The person is testing the drawer's mechanisms for maintenance.
B. The person is opening the drawer to retrieve or place an item inside.
C. The person is cleaning out the drawer for cleanliness.
D. The person is inspecting the contents for safety.
```

The final exp8 prompt contains this QA block plus memory:

- `1.0-9.0`: person handles clothing and puts it into a drawer.
- `10.0-15.0`: person picks up another piece of clothing and moves towards a cabinet.
- current frame `15.0-16.0`: `dataset/ovo/video/1143/000015.jpg`.

Exp8 final raw output:

```xml
<state>The person was manipulating a piece of clothing and putting it into a drawer, then picked up another piece of clothing and moved towards a cabinet. The current frames show the person opening the cabinet. There is no question about the drawer, so I do not need to answer.</state>
<answer></answer>
<delta t="10.0-16.0">The person picks up another piece of clothing and moves towards a cabinet, then opens the cabinet.</delta>
```

Gemini retry final raw output:

```xml
<eta>18.0</eta>
<answer>B</answer>
<note t="15.0-16.0" frame="1"/>
<bridge t="10.0-16.0">The person puts the folded clothing away, picks up a blue item, inspects it, moves towards a blue cabinet, opens the cabinet door, and prepares to place the item inside.</bridge>
```

Diagnosis: the model has enough visual and memory evidence to choose `B`; it even says
"putting it into a drawer" and "opening the cabinet". The failure is that it treats the
current object as a cabinet rather than the drawer mentioned by the question, then incorrectly
uses that mismatch to abstain. This is a model-side prompt-following / object-name mismatch
failure, not a missing-QA or missing-image program bug.

#### Sample 20: Utensil Holder / Frying Pan

Full question at `t=304.0`:

```text
Question: What did I pick from the utensil holder?
Options:
A. Knife; B. Spoon; C. Dying pan; D. Fork;

Respond only with the letter corresponding to your chosen option.
Do not include any additional text or explanation in your response.
```

Result:

- Ground truth: `C`.
- `qwen3_rl_exp8_step40`: `B` (`Spoon`), incorrect.
- Gemini teacher trace: `C`, correct.

Current model timeline:

- `0-10s`: kitchen counter, sink area, lower drawer/cabinet opened; metal bowl and colander
  retrieved. The model already records generic kitchen tools but no later-use object identity.
- `10-30s`: colander placed in the sink; potatoes taken out of packaging and washed.
- `30-55s`: potatoes washed; first potato peeled with a knife over the cutting board.
- `55-105s`: first potato peeled/cut; cut pieces placed into a metal bowl and rinsed.
- `105-194s`: second potato peeled and cut. The model repeatedly summarizes this as
  "continues to peel the second potato", which is correct but very repetitive and consumes most
  of the memory with low-information deltas.
- `195-224s`: cut potatoes rinsed; plate taken out; potatoes transferred to the plate.
- `225-234s`: plastic wrap picked up; plate wrapped and put in the microwave.
- `235-259s`: potato peels, cutting board, and knife cleaned up at the sink.
- `260-295s`: current model says the operator picks up a pot from the stove, pours water into
  the sink, washes/scrubs the pot, holds it over the sink, and dries it with a towel.
- `295-300s`: current model says the operator moves toward the stove area.
- `300-304s`: current model answers `B`, with the final bridge:

  ```text
  The operator holds the pot over the sink, dries it with a towel, moves towards the stove area,
  and then reaches for a spoon from the utensil holder near the sink.
  ```

Gemini comparison timeline around the divergence:

- `248-255s`: after microwaving potatoes, Gemini records washing the cutting board and knife,
  then washing hands.
- `257-265s`: Gemini records a missing action that the current model did not capture:

  ```text
  The person turns away from the sink, opens the drawer under the counter, and takes out a small
  frying pan. The person places the small frying pan on the stove next to the boiling pot.
  ```

- `263-285s`: Gemini tracks the pot/eggs sequence: the person empties/rinses the pot, places two
  eggs in it, and fills it with water.
- `295-304s`: Gemini says the pot with eggs/water is placed on the stove, then the person turns
  back toward the counter. Its final state says the person is picking up a frying pan from the
  dish rack / holder area and therefore answers `C`.

Error diagnosis:

- The current model loses the `small frying pan` event around `257-265s`. Instead, from `260s`
  onward it collapses the scene into "picked up a pot from the stove / washed a pot".
- By the final question, the model no longer has a reliable representation of the pan. It then
  hallucinates a visually plausible kitchen utensil, `spoon`, from the same area.
- This is a long-horizon memory error plus a fine-grained object recognition error. The model's
  format reward is perfect on this sample, but the memory content is not discriminative enough:
  repetitive potato-peeling summaries dominate earlier context, while the key small-pan event is
  not preserved.
- Gemini succeeds because it names and preserves the object as `small frying pan` at the moment it
  appears, then carries that object identity forward to the final backward question.

## Experiment 9: Local Qwen3VL Judge

`exp9_localjudge` is the single-node exp7/exp6 training chain with the LLM judge switched from
Gemini to a local OpenAI-compatible endpoint, intended for a deployed Qwen3VL-32B judge service.
It does not modify the old Gemini scripts.

- Launch script: `RL/scripts/train_exp9_localjudge.sh`.
- Run name: default `exp9_localjudge`; override with `RUN_NAME=...` if needed.
- Default judge backend: `JUDGE_BACKEND=vllm`.
- Default judge model id: `JUDGE_MODEL=qwen3vl-32b-judge`. This must match the served model name
  exposed by the local vLLM/OpenAI-compatible server.
- Default judge endpoint: `JUDGE_BASE_URL=http://127.0.0.1:9000/v1`.
- Default judge API key: `JUDGE_API_KEY=EMPTY`.
- Endpoint readiness check: enabled by default through `CHECK_JUDGE_ENDPOINT=1`; it checks
  `${JUDGE_BASE_URL}/models` before starting training.
- Judge timeout: `JUDGE_TIMEOUT_SECONDS=300`; max response tokens: `JUDGE_MAX_TOKENS=2048`;
  judge image side: `JUDGE_MAX_IMAGE_SIDE=512`.
- Training parameters otherwise match exp7: `train_batch_size=16`, `gen_batch_size=16`,
  `val_batch_size=16`, `val_max_samples=200`, `rollout.n=8`, timeline answer supervision,
  `answer_decay=0.7`, `answer_weight=0.5`, no std division, actor KL coef `0.001`.

Assuming the local judge service is already running:

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
JUDGE_BASE_URL=http://127.0.0.1:9000/v1 \
JUDGE_MODEL=qwen3vl-32b-judge \
bash RL/scripts/train_exp9_localjudge.sh
```

If the served model name is the model path instead of `qwen3vl-32b-judge`, pass that exact value:

```bash
JUDGE_MODEL=/path/to/Qwen3-VL-32B-Instruct \
JUDGE_BASE_URL=http://127.0.0.1:9000/v1 \
bash RL/scripts/train_exp9_localjudge.sh
```

For a judge service deployed on another machine:

```bash
JUDGE_BASE_URL=http://<judge_host>:9000/v1 \
JUDGE_MODEL=qwen3vl-32b-judge \
bash RL/scripts/train_exp9_localjudge.sh
```

If the endpoint does not implement `/models` but `/v1/chat/completions` works, bypass only the
startup readiness check:

```bash
CHECK_JUDGE_ENDPOINT=0 \
JUDGE_BASE_URL=http://<judge_host>:9000/v1 \
JUDGE_MODEL=qwen3vl-32b-judge \
bash RL/scripts/train_exp9_localjudge.sh
```

### 2026-05-14 Evaluation Decision

Initial SwanLab curves showed exp3, exp4, and exp5 are close. exp3 and exp4 were stopped manually.
Exp5 later reached step 29, entered the step-30 full-set validation, and did not finish cleanly.
Start evaluation from exp4 because it is the middle answer-weight setting (`answer_weight=0.5`)
and already has checkpoint `global_step_20`.

Progress snapshot:

- `exp3`: manually stopped after `global_step_20`; checkpoint exists at
  `RL/outputs/runs/exp3/checkpoints/global_step_20/actor`.
- `exp4`: manually stopped after `global_step_20`; checkpoint exists at
  `RL/outputs/runs/exp4/checkpoints/global_step_20/actor`.
- `exp5`: stopped during the previous full 2000-row step-30 validation; latest observed
  checkpoint is `RL/outputs/runs/exp5/checkpoints/global_step_20/actor`.
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
- Source model：`models/qwen3vl8b_sft_anchor_delta_step200_vllm`
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

## Evaluation Results Snapshot

Last updated: 2026-05-15. Scores below are read from `outputs/**/results_summary.json`
or `outputs/**/ovo_results_summary.json` with bounded-depth scan. OVO scores use
`total_avg` from the summary file; StreamingBench scores use exact MCQ accuracy.

### StreamingBench

| Output dir | Split | Model / note | Samples | Score | Details |
| --- | --- | --- | ---: | ---: | --- |
| `outputs/streamingbench_real_exp8_step40_grouped` | `real` | `models/qwen3_rl_exp8_step40`; grouped by video on GPUs 0-6 | 2500 | 82.68% | 2067/2500; grouped actual calls 52,311 vs logical 132,742, saving 2.54x |
| `outputs/streamingbench_sqa_exp8_step40` | `sqa` | `models/qwen3_rl_exp8_step40`; sequential QA history enabled | 250 | 41.20% | 103/250; no grouped video merge |
| `outputs/streamingbench_proactive_exp8_step40` | `proactive` | `models/qwen3_rl_exp8_step40`; proactive trigger scoring, GPUs 0-7 | 235 | 60.43% | 142/235 final answer correct; time correct 146/235 = 62.13%; skipped 15 samples from missing videos `sample_45_proactive.mp4`, `sample_48_proactive.mp4`, `sample_50_proactive.mp4` |
| `outputs/streamingbench_omni_mislead_anomaly_exp8_step40` | `omni` visual subset | `models/qwen3_rl_exp8_step40`; grouped by video; task filter `Misleading Context Understanding,Anomaly Context Understanding` | 500 | 61.20% | 306/500; Misleading 150/250 = 60.00%; Anomaly 156/250 = 62.40%; grouped actual call saving 2.75x |

StreamingBench real by task for `outputs/streamingbench_real_exp8_step40_grouped`:

| Task | Correct / Total | Accuracy |
| --- | ---: | ---: |
| Clips Summarize | 294/317 | 92.74% |
| Text-Rich Understanding | 292/321 | 90.97% |
| Attribute Recognition | 260/306 | 84.97% |
| Prospective Reasoning | 90/108 | 83.33% |
| Object Recognition | 305/367 | 83.11% |
| Action Recognition | 289/353 | 81.87% |
| Causal Reasoning | 103/128 | 80.47% |
| Event Understanding | 126/161 | 78.26% |
| Spatial Understanding | 192/246 | 78.05% |
| Counting | 116/193 | 60.10% |

### OVO Full Evaluations

Model/backend values below are read from each output `run_config.yaml` when it exists; Gemini
rows are matched to the corresponding checked-in config or launch script.

| Output dir | Model / backend | Samples | OVO total_avg | Category scores |
| --- | --- | ---: | ---: | --- |
| `outputs/ovo_gemini_full_retry` | Gemini `gemini-2.5-pro` | n/a | 66.75% | backward 65.02%; realtime 85.54%; forward 49.68% |
| `outputs/ovo_qwen3vl8b_grpo_0509_full_state_note_t` | vLLM `/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl_8b_grpo_0509` | 3035 | 63.90% | backward 60.88%; realtime 76.08%; forward 54.74% |
| `outputs/ovo_qwen3vl8b_finetuned_full_state_note_t` | vLLM `/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm` | 3035 | 62.71% | backward 59.92%; realtime 78.15%; forward 50.07% |
| `outputs/streamtext/ovo_gemini_flash_full` | StreamText + Gemini `gemini-2.5-flash` | 3035 | 62.22% | backward 68.50%; realtime 76.09%; forward 42.06% |
| `outputs/ovo_exp1_rl0511_step60_full` | vLLM `models/exp1_rl0511_step60` | 3035 | 59.87% | backward 47.70%; realtime 77.72%; forward 54.17% |
| `outputs/ovo_qwen3vl8b_base_full_state_note_t` | vLLM `/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct` | 3035 | 57.52% | backward 60.59%; realtime 75.10%; forward 36.87% |

### OVO 1-of-8 / Subset Evaluations

| Output dir | Model / backend | Samples | OVO total_avg | Category scores |
| --- | --- | ---: | ---: | --- |
| `outputs/ovo_exp1_rl0511_step60_1of8` | vLLM `models/exp1_rl0511_step60` | 364 | 64.70% | backward 52.27%; realtime 81.54%; forward 60.30% |
| `outputs/ovo_qwen3vl8b_sft_anchor_delta_step200_1of8` | vLLM `/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_sft_anchor_delta_step200_vllm` | 364 | 63.10% | backward 52.71%; realtime 77.19%; forward 59.40% |
| `outputs/ovo_gemini_1of8_mem6s` | Gemini `gemini-2.5-pro` | 364 | 62.85% | backward 56.51%; realtime 83.10%; forward 48.94% |
| `outputs/ovo_gemini_1of8_state_note_t` | Gemini `gemini-2.5-pro` | 364 | 62.57% | backward 61.45%; realtime 73.71%; forward 52.54% |
| `outputs/streamtext/ovo_gemini_flash_1of8` | StreamText + Gemini `gemini-2.5-flash` | 364 | 61.92% | backward 63.21%; realtime 77.37%; forward 45.19% |
| `outputs/ovo_qwen3vl8b_finetuned_1of8_rerun` | vLLM `/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm` | 364 | 61.54% | backward 57.41%; realtime 76.50%; forward 50.72% |
| `outputs/ovo_qwen3vl8b_finetuned_1of8` | vLLM `/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm` | 364 | 61.28% | backward 55.37%; realtime 79.49%; forward 48.99% |
| `outputs/ovo_gemini_1of8_state` | Gemini `gemini-2.5-pro` | 364 | 61.13% | backward 55.56%; realtime 79.14%; forward 48.70% |
| `outputs/ovo_qwen_sft_0513_1of8` | vLLM `/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen_sft_0513` | 364 | 59.11% | backward 51.81%; realtime 76.72%; forward 48.80% |
| `outputs/ovo_qwen3vl8b_8gpu_1of8_eval` | vLLM `/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct` | 364 | 57.89% | backward 58.46%; realtime 77.54%; forward 37.68% |
| `outputs/ovo_qwen3_rl_exp8_step40_1of8_6gpu` | vLLM `models/qwen3_rl_exp8_step40` | 364 | 56.18% | backward 60.25%; realtime 76.55%; forward 31.74% |
| `outputs/ovo_exp4_grppo_aw05_step20_1of8` | vLLM `models/exp4_grppo_aw05_step20` | 364 | 45.79% | backward 44.78%; realtime 61.91%; forward 30.69% |

### OVO Debug / Small Subsets

| Output dir | Model / backend | Samples | OVO total_avg | Notes |
| --- | --- | ---: | ---: | --- |
| `outputs/ovo_gemini_teacher_eval_selected_single` | Gemini `gemini-2.5-pro` | 61 | 71.35% | selected subset; backward 83.33%; realtime 73.33%; forward 57.38% |
| `outputs/ovo_gemini_one_eval` | Gemini `gemini-2.5-pro` | 1 | 100.00% | one-sample smoke |
| `outputs/ovo_gemini_one_state_eval` | Gemini `gemini-2.5-pro` | 1 | 100.00% | one-sample smoke |
| `outputs/streamtext/ovo_gemini_one_test` | StreamText + Gemini `gemini-2.5-pro` | 1 | 100.00% | streamtext one-sample smoke |
| `outputs/streamtext/ovo_gemini_one_eval` | StreamText + Gemini backend; exact model not persisted in output dir | 1 | 0.00% | streamtext one-sample debug |
| `outputs/ovo_qwen3vl32b_one` | OpenAI-compatible `Qwen/Qwen3-VL-32B-Instruct` | n/a | 0.00% | one-sample/debug summary; not comparable |

Current best confirmed completed results:

- StreamingBench real: `qwen3_rl_exp8_step40`, 82.68%.
- StreamingBench SQA: `qwen3_rl_exp8_step40`, 41.20%; this is low and needs follow-up error analysis before using as a claim.
- StreamingBench proactive: `qwen3_rl_exp8_step40`, 60.43% on 235/250 available samples; 15 samples were skipped because three proactive videos were missing locally.
- StreamingBench omni visual subset: `qwen3_rl_exp8_step40`, 61.20% on Misleading + Anomaly only; this is not full omni because audio-dependent tasks are excluded.
- OVO full: Gemini retry baseline, 66.75%; best local full model summary is `ovo_qwen3vl8b_grpo_0509_full_state_note_t`, 63.90%.
- OVO 1-of-8: `ovo_exp1_rl0511_step60_1of8`, 64.70%, but its full run is 59.87%, so treat the 1-of-8 number as a quick-screen metric.

## Known Launch Scripts

## GRPPO Observation Metric Notes

- 2026-05-15:
  - Added SwanLab/console metrics for step-level answer behavior:
    `grppo/answer_required_rate`, `grppo/answer_required_missing_rate`,
    `grppo/silence_required_rate`, and `grppo/silence_violation_rate`.
    These are computed on the post-filter training rows; pre-filter raw rollout
    versions are also logged under `grppo/prefilter/*`.
  - Changed `traj/target_answer` / `streamweave/grppo_target_answer_reward`
    from "final answer vs latest answer target" to the mean rule correctness
    across all required `answer_target` steps in the trajectory. This makes
    multi-query / multi-answer samples observable at trajectory level.
  - Added `traj/target_answer_count` / `streamweave/grppo_target_answer_count`
    to show how many required answer targets contributed to the trajectory
    target-answer mean.
- 2026-05-16:
  - Changed the GRPPO process judge rubric from coarse 0/0.4/0.7/1.0 score
    anchors to binary per-dimension checklists. The parser now computes each
    process score as the mean of the listed checks while preserving backward
    compatibility with old `score`-only judge responses.
  - Added two conservative actor-prompt sentences only: one asks `<state>` to be
    brief but evidence-bearing, and one asks `<delta>` to preserve action/event
    changes while staying concise.
  - Exp9 32-GPU run was restarted with `train/gen/val_batch_size=32`.
    It reached `global_step_25` after saving `global_step_20`, then failed in
    `old_log_prob` with `Image features and image tokens do not match`, shortly
    after a `prompt too long: 5536 tokens > 5120` abort. The attempted visual-token
    guard fix was reverted; current diagnosis is simple overlong-input truncation.
  - Updated GRPPO score aggregation for the next exp9 run: `grppo_judge_step_reward`
    is now `2.0 * mean(all process checklist values)`, `grppo_step_reward`
    is the unnormalized weighted sum `0.7*judge + 0.15*format + 0.15*note_frequency`,
    and the dimension-only fallback path was moved from the old 0-1 average to the
    same 0-2 judge scale.
  - Removed `null` from the GRPPO process checklist protocol. Checks now use numeric
    `0/0.5/1`; for anchor-specific checks, no `<anchor>` gives `0.5` on
    `anchor_time_and_body_valid` and `anchor_representative_if_present` rather than
    being skipped.

| Script | Intended use | Current default run name | Notes |
| --- | --- | --- | --- |
| `RL/scripts/train_grpo.sh` | GRPO-style stepwise trajectory training | `grpo_rl0511_8gpu_judge` | This script matches the currently running old-config process. Copy it before changing major experiment settings. |
| `RL/scripts/train_exp2_rlmlr.sh` | Experiment 2 RLMLR training on `rl_0512` split | `exp2_rlmlr` | Uses `train_batch_size=16`, `gen_batch_size=16`, `rollout.n=8`, validation every 20 steps. |
| `RL/scripts/train_exp3.sh` | Experiment 3 GRPPO baseline on `rl_exp3` | `exp3` | Final advantage `1.0*step_adv + 0.3*answer_adv`; advantage/filter min std both `0.04`. |
| `RL/scripts/train_exp4.sh` | Experiment 4 GRPPO answer-weight sweep | `exp4` | Same as exp3, but final answer-advantage weight is `0.5`; advantage/filter min std both `0.04`. |
| `RL/scripts/train_exp5.sh` | Experiment 5 GRPPO answer-weight sweep | `exp5` | Same as exp3, but final answer-advantage weight is `0.7`; advantage/filter min std both `0.04`. |
| `RL/scripts/train_exp6.sh` | Experiment 6 GRPPO timeline answer supervision | `exp6` | Timeline `none/silence/answer`, no std division, answer decay `0.7`, answer weight `0.5`, silence scalar `0.1`, actor KL enabled. |
| `RL/scripts/train_exp7.sh` | Experiment 7 parameter-tuning fork from exp6 | `exp7` | Initially identical to exp6; use this script for the next parameter changes without touching exp6. |
| `RL/scripts/train_exp7_smoke.sh` | Experiment 7 two-step debug smoke | `exp7_smoke` | Uses exp7 data/model/reward settings with `train/gen batch=2`, `rollout.n=4`, `total_training_steps=2`, and GRPPO debug dumps enabled. |
| `RL/scripts/train_exp8.sh` | Experiment 8 multi-node GRPPO | `exp8` | Two-node Ray launch helper; default `nnodes=2`, `n_gpus_per_node=8`, `train/gen/val_batch_size=16`, `agent.num_workers=64`. |
| `RL/scripts/train_exp9.sh` | Experiment 9 four-node 32-GPU GRPPO | `exp9` | Four-node Ray launch helper forked from exp8; default `nnodes=4`, `n_gpus_per_node=8`, source model `models/qwen3_rl_exp8_step100`, train data `rl_0516_train`, validation default unchanged at `rl_0515_val`, prompt/response length `6144/3072`, rollout `max_model_len=9216`, Ray object store `40GB`, default `train/gen/val_batch_size=32`, `rollout.n=8`, micro batch per GPU `8`, `agent.num_workers=128`; GRPPO step mix is unnormalized `0.7*judge_0to2 + 0.15*format + 0.15*note_frequency`; workers can set `RAY_NODE_IP` when a pod has multiple NICs. |
| `RL/scripts/train_exp9_24.sh` | Experiment 9 24-GPU GRPPO | `exp9_24` | Self-contained exp9 script; defaults to source model `models/qwen3vl_sft_0516_step50`, train data `rl_0516_filter`, actor lr `5e-6`, `nnodes=3`, `n_gpus_per_node=8`, `train/gen/val_batch_size=24`, `agent.num_workers=96`, rollout `max_num_seqs=3072`, and keeps exp9 reward/KL/filter settings. |
| `RL/scripts/train_exp9_smoke.sh` | Experiment 9 single-node debug smoke | `exp9_smoke` | Self-contained exp9 smoke script with `models/qwen3vl_sft_0516_step50`, actor lr `5e-6`, and train data `rl_0516_train`; defaults to `total_training_steps=8`, `train/gen/val_batch_size=2`, `rollout.n=4`, debug `2` groups x `4` trajectories; uses exp9 context budget (`max_prompt_length=6144`, `max_response_length=3072`, rollout `max_model_len=9216`) and reward mix `0.7/0.15/0.15`; validation is disabled with `test_freq=-1`; dumps GRPPO debug JSONL under `outputs/runs/exp9_smoke/grppo_debug`. |
| `RL/scripts/train_exp9_localjudge.sh` | Experiment 9 local judge GRPPO | `exp9_localjudge` | Single-node exp7 config with local OpenAI-compatible/vLLM judge; defaults to `JUDGE_BASE_URL=http://127.0.0.1:9000/v1`, `JUDGE_MODEL=qwen3vl-32b-judge`, and the same exp9 GRPPO step mix `0.7/0.15/0.15`. |
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
