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
Query events follow the current StreamWeave frame alignment logic.
If `runtime.max_steps` truncates away every query event, the trajectory is
aborted with zero reward instead of silently training on format reward only.
Rollout also aborts with zero reward when the rendered prompt exceeds
`actor_rollout_ref.rollout.max_model_len - max_response_length`.

Rewards:

- format score per step, computed from the raw XML output before repair
- dense step score, currently composed from note frequency and optional LLM-as-judge scoring
- final task success score, computed from the trajectory final answer

The note frequency reward penalizes more than one note in a window and penalizes
three or more consecutive windows without a note. The LLM judge is configured
under `data.streamweave.reward.judge`. The current 8GPU GRPO launcher enables
Gemini judge debug by default with `STREAMWEAVE_REWARD_JUDGE_WEIGHT=0.0`, so it
logs judge scores without changing `step_score`; set `judge_weight > 0` for it
to affect training.

The GRPO scripts expose the judge settings through environment variables:

```bash
STREAMWEAVE_REWARD_JUDGE_ENABLE=true
STREAMWEAVE_REWARD_JUDGE_WEIGHT=1.0
STREAMWEAVE_JUDGE_BACKEND=gemini
STREAMWEAVE_JUDGE_MODEL=gemini-2.5-flash
GOOGLE_APPLICATION_CREDENTIALS=/mmu_ssd3/group_lisize/hetu/xujia10/joint_tags/scripts/gemini_client/config.json
STREAMWEAVE_JUDGE_MAX_TOKENS=768
STREAMWEAVE_JUDGE_TIMEOUT_SECONDS=180
STREAMWEAVE_JUDGE_IMAGE_RESOLUTION=512
STREAMWEAVE_JUDGE_MAX_RETRIES=2
STREAMWEAVE_JUDGE_RETRY_BACKOFF_SECONDS=5
```

For calibration runs, keep `STREAMWEAVE_REWARD_JUDGE_WEIGHT=0.0`; the judge
will run and log `judge_score`, but it will not change `step_score`.

Entrypoints:

```bash
./scripts/run_smoke.sh
./scripts/train_grpo_ovo_8gpu.sh
./scripts/train_ppo.sh data.train_files=/path/to/train.parquet data.val_files=/path/to/val.parquet
```
