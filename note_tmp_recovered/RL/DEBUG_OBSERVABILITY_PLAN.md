# StreamWeave RL Observability Plan

This plan is for debugging StreamWeave RL under verl/Ray, where interactive
debuggers are usually unreliable. The default path should stay silent and cheap;
debug output should be explicitly enabled per run.

## Goals

- Make one trajectory easy to reconstruct from dataset row to reward.
- Keep logs usable when Ray workers run concurrently.
- Avoid writing image bytes or huge prompt blobs by default.
- Keep debug behavior disabled unless requested.

## Proposed Controls

- `STREAMWEAVE_RL_DEBUG=0|summary|steps|full`
  - `0`: no debug output.
  - `summary`: one line per trajectory plus terminal reward.
  - `steps`: one JSON record per step.
  - `full`: step JSON plus truncated prompt/response snapshots.
- `STREAMWEAVE_RL_DEBUG_DIR=/path/to/debug`
  - Worker-safe output root for JSONL trace files.
- `STREAMWEAVE_RL_DEBUG_SAMPLE_IDS=id1,id2`
  - Optional allowlist to keep logs small.
- `STREAMWEAVE_RL_DEBUG_MAX_TEXT=4000`
  - Maximum stored characters for prompt, response, memory, and final answer.

## Output Shape

Human-readable text traces are the primary debug output. JSONL should be an
optional machine-readable backup, not the main thing a developer has to read.

Prefer one text file per trajectory:

- `debug_root/run_id/sample_<sample_id>_traj_<traj_idx>.trace.txt`

The text trace should be sectioned and aligned so it can be opened directly in
less/vim:

```text
================================================================================
STREAMWEAVE RL TRACE sample=42_0 traj=1 dataset=ovo task=REC
video_id=42_0 gt=3 scorer=ovo
frames=100 groups=20 fps=1.0 frames_per_step=5 max_steps=20
query_by_frame: 0 -> "How many times did they ...?"
================================================================================

[TURN 001/020] frames=0,1,2,3,4 time=0.0-5.0
prompt_images:
  - /.../video/42_0/000000.jpg
  - /.../video/42_0/000001.jpg
qa_history:
  q@0.0: How many times did they ...?
memory_before: chars=0
response:
  <eta></eta><answer></answer>...
quality: parser_ok=1 valid=1 issues=[]
repair: []
reward: format=1.000 step=0.000 turn=0.010

[TURN 020/020] frames=95,96,97,98,99 time=95.0-100.0
response:
  <eta>100</eta><answer>3</answer>...
quality: parser_ok=1 valid=1 issues=[]
reward: format=1.000 success=1.000 trajectory=1.000

[FINAL]
final_answer: 3
format_mean=1.000 success_score=1.000 trajectory_score=1.000
```

Also emit compact one-line stderr summaries for Ray log search:

```text
[SW_RL][summary] sample=42_0 traj=1 steps=20 success=1.0 format=1.0 reward=1.0
[SW_RL][abort] sample=17 traj=0 turn=3 reason=empty_response
```

Optional JSONL can still be written when needed. Prefer one file per Ray worker
or per trajectory:

- `debug_root/run_id/worker_<rank>.jsonl`, or
- `debug_root/run_id/sample_<sample_id>_traj_<traj_idx>.jsonl`.

Each JSON object should include:

- `event`: `dataset_row`, `reset`, `step_prompt`, `step_result`, `trajectory`, or `abort`.
- `sample_id`, `video_id`, `dataset`, `task`, `traj_idx`, `turn_idx`.
- `ray_worker`: pid, hostname, rank when available.
- `timestamp`: wall-clock time.

## Required Events

### dataset_row

Emit from `StreamWeaveAgentDataset.__getitem__`.

Fields:

- resolved `sample_id`, `video_id`, `data_source`, `task`.
- `query_timestamp`, `ground_truth`, `scorer`.
- frame dataset config: `dataset_name`, `frame_id_base`, `sample_fps`, `max_frames`.
- source annotation keys, not the full raw row unless `full`.

### reset

Emit from `StreamWeaveRLEnv.reset`.

Fields:

- `frame_count`, `first_frame`, `last_frame`.
- `group_count`, `frames_per_step`, `max_steps`.
- `query_by_frame` mapping after main rollout alignment.
- frame directory path.

### step_prompt

Emit from `StreamWeaveRLEnv._prepare_current_turn`.

Fields:

- `turn_idx`, global frame ids in the group.
- prompt frame ids and image paths.
- memory length and optional truncated memory.
- prompt text hash; include truncated prompt only at `full`.
- active QA history count.

### step_result

Emit from `StreamWeaveRLEnv.step`.

Fields:

- raw response hash and truncated response.
- `quality_valid`, `parser_ok`, issue codes.
- applied repair types.
- `format_score`, `step_score`, `turn_reward`.
- answer text if produced.

### trajectory

Emit on the final turn.

Fields:

- final answer, truncated by `STREAMWEAVE_RL_DEBUG_MAX_TEXT`.
- `format_mean`, `step_mean`, `success_score`, `trajectory_score`.
- reward weights and scorer mode.

### abort

Emit in `StreamWeaveAgentLoop.run` before returning an empty trajectory or marking
rollout failure.

Fields:

- abort reason.
- prompt length, response length, attention-mask position details if relevant.
- sample identity and last completed turn.

## Printing Policy

Use human-readable `.trace.txt` for detailed data. Print only compact one-line
summaries to stderr:

```text
[SW_RL][summary] sample=... traj=... steps=... success=... format=... reward=...
[SW_RL][abort] sample=... traj=... reason=...
```

Ray aggregates stderr reasonably well, while the text trace keeps the real
debugging payload readable. JSONL is optional for scripts and aggregation.

## Hook Points

- `streamweave_rl/dataset.py`: dataset row resolution.
- `streamweave_rl/env.py`: reset, prompt construction, step result, trajectory.
- `streamweave_rl/agent_loop_stepwise.py`: empty response, prompt overflow, and
  rollout aborts.
- `verl/verl/trainer/ppo/ray_trainer.py`: validation dedup metrics and batch
  reward summaries.

## Implementation Notes

- Add a small `streamweave_rl/debug.py` helper later.
- The helper should be safe if called from multiple Ray workers.
- It should never raise into training. On write failure, print a single warning
  and disable itself in that process.
- Keep text truncation deterministic and include SHA256 hashes for full matching.
- Default to text trace output. Add JSONL only behind a separate option such as
  `STREAMWEAVE_RL_DEBUG_JSONL=1`.
