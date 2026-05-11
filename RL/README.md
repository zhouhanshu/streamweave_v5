# StreamWeave RL

This folder contains the StreamWeave V5 reinforcement-learning adapter and a
local copy of verl.

The rollout protocol is stepwise: each video window is generated and trained as
its own prompt-response row, while `group_idx`, `traj_idx`, and `turn_idx`
preserve the full trajectory structure for PPO and GRPO advantage computation.

`configs/streamweave_stepwise.yaml` is the shared StreamWeave stepwise base
config. It owns the stable StreamWeave defaults: dataset adapter, runtime,
memory, reward weights, judge defaults, DAPO defaults, and stable agent-loop
registration. Launch scripts own training-scale and resource knobs such as
batch sizes, prompt/response length, vLLM capacity, GPU count, save frequency,
and the concrete dataset/model/run path.

The RL environment is read-only with respect to video frames. It expects frames
to already exist under `dataset_root/dataset_name/video/<video_id>/`.
Each rollout turn renders the full StreamWeave instruction, memory, QA history,
and current frame window into one multimodal user message.
Query events follow the current StreamWeave frame alignment logic.
If `runtime.max_steps` truncates away every query event, the trajectory is
aborted with zero reward instead of silently training on format reward only.
Rollout also aborts with zero reward when the rendered prompt exceeds
`actor_rollout_ref.rollout.max_model_len - max_response_length`.

Rewards:

- format score per step, computed from the raw XML output before repair
- dense step score, currently composed from anchor frequency and optional LLM-as-judge scoring
- final task success score, computed from the trajectory final answer
- optional DAPO-style group filtering over trajectory scores

The anchor frequency reward penalizes more than one anchor in a window and penalizes
three or more consecutive windows without an anchor. The LLM judge is configured
under `data.streamweave.reward.judge`. The shared config sets the current reward
defaults (`format/step/success = 0.1/0.2/0.7`, `note/judge = 0.3/0.7`), and the
8GPU GRPO launcher enables Gemini Flash judge so judge scores affect
`step_score`.

For stepwise GRPO, the trainer logs group-level metrics before actor update:
`traj/score_mean`, `traj/score_std`, and `traj/valid_group_ratio`. When
`algorithm.filter_groups.enable=true`, groups whose rollout trajectories all
have the same selected metric are filtered before old-log-prob and actor update.
The default launcher enables this StreamWeave DAPO-style filtering over
`trajectory_score`, uses token-mean policy loss, and sets DAPO-style separated
PPO clip ratios to `0.2/0.28`.

Most training defaults now live in `configs/streamweave_stepwise.yaml`. The
main environment variables left for normal use are external run controls:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
STREAMWEAVE_RUN_NAME=grpo_test4rl_8gpu_judge
STREAMWEAVE_MODEL_PATH=/path/to/source/model
STREAMWEAVE_JUDGE_ENABLE=true
GOOGLE_APPLICATION_CREDENTIALS=/mmu_ssd3/group_lisize/hetu/xujia10/joint_tags/scripts/gemini_client/config.json
STREAMWEAVE_TRACE_FIRST_ROLLOUT=1
STREAMWEAVE_TRACE_SAMPLE_EVERY=64
```

For calibration runs, pass Hydra overrides at the end, for example:

```bash
./scripts/train_grpo.sh data.streamweave.reward.judge_weight=0.0
```

Entrypoints:

```bash
./scripts/run_smoke.sh
./scripts/train_grpo.sh
./scripts/train_ppo.sh data.train_files=/path/to/train.parquet data.val_files=/path/to/val.parquet
```
