# StreamWeave RL

This folder contains the StreamWeave V5 reinforcement-learning adapter and a
local copy of verl.

The rollout protocol is stepwise: each video window is generated and trained as
its own prompt-response row, while `group_idx`, `traj_idx`, and `turn_idx`
preserve the full trajectory structure for PPO and GRPO advantage computation.

The RL environment is read-only with respect to video frames. It expects frames
to already exist under `dataset_root/dataset_name/video/<video_id>/`.
Each rollout turn renders the full StreamWeave instruction, memory, QA history,
and current frame window into one multimodal user message.
Query events follow the V4 StreamWeave frame alignment logic.
If `runtime.max_steps` truncates away every query event, the trajectory is
aborted with zero reward instead of silently training on format reward only.
Rollout also aborts with zero reward when the rendered prompt exceeds
`actor_rollout_ref.rollout.max_model_len - max_response_length`.

Initial rewards:

- format score per step
- final task success score
- reserved step score hook for later dense rewards

Entrypoints:

```bash
./scripts/run_smoke.sh
./scripts/train_grpo.sh data.train_files=/path/to/train.parquet data.val_files=/path/to/val.parquet
./scripts/train_ppo.sh data.train_files=/path/to/train.parquet data.val_files=/path/to/val.parquet
```
