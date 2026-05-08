# StreamWeave V5 Agent Codebase Guide

Last read-through: 2026-05-06.

This file is for future agents who need to understand `streamweave_v5` quickly. It focuses on the three operational paths that matter most: inference/evaluation, SFT data generation/export, and RL stepwise training.

## One-Screen Mental Model

StreamWeave is a stateful streaming video agent. Each step sees:

```text
Memory + QA History + Current frame window -> model XML -> validate/repair -> commit to Memory
```

The model speaks a small XML protocol:

```xml
<state>...</state>
<answer>...</answer>
<note t="..."></note>
<bridge t="...">...</bridge>
```

- `state`: current-step summary used before answering. It is not committed to Memory.
- `note`: visual anchor. It stores one current frame as evidence for future steps.
- `bridge`: text summary between notes, or across a note/window gap.
- `answer`: emitted only when the active QA should be answered or updated.

The core runtime package is `streamweave/`. Evaluation, SFT, and RL all reuse the same prompt/parser/quality/memory logic, but differ in how model outputs are generated and accepted.

## Top-Level Layout

```text
streamweave/                 core runtime: env, rollout loop, prompts, parser, quality, memory
backend/                     model backends: mock, Gemini, OpenAI-compatible/vLLM, endpoint pool
evaluation/                  OVO/StreamingBench loaders, scoring, single/batch eval entrypoints
configs/                     eval configs
scripts/                     convenience eval/data scripts

data_engine/synthesize/      optional QA annotation synthesis pipeline
data_engine/sft/             step-level SFT data synthesis and ShareGPT export

RL/streamweave_rl/           StreamWeave adapter for verl agent-loop RL
RL/configs/                  Hydra configs for PPO/GRPO stepwise RL
RL/scripts/                  RL launch scripts
RL/verl/                     vendored verl with local StreamWeave plumbing

dataset/                     local annotations / frame datasets used by eval or RL
models/                      local model/config artifacts
note/                        experiment notes; start with note/README.md
outputs/                     eval outputs
```

Generated files and caches are common in this repo (`__pycache__`, outputs, extracted frames). For code changes, start from the source folders above.

## Core Runtime

Important files:

- `streamweave/schemas.py`: shared dataclasses (`FrameRef`, `BenchmarkSample`, `ModelAction`, `QualityReport`, `Transition`, `RolloutTrace`).
- `streamweave/config.py`: config dataclasses and YAML/JSON loading.
- `streamweave/prompts.py`: `teacher_synthesis`, `teacher_eval`, and `production/eval/final` prompt builders.
- `streamweave/parser.py`: strict XML parser and lenient repair parser.
- `streamweave/quality.py`: format/timing/open-tail/gap validation and reward feature bits.
- `streamweave/postprocess.py`: deterministic repair for eval/RL and retry feedback for synthesis.
- `streamweave/memory.py`: note/bridge/QA memory, eviction, open-tail detection, memory rendering.
- `streamweave/env.py`: state machine around memory, prompt building, validation/repair, and commit.
- `streamweave/rollout.py`: single-sample inference loop.
- `streamweave/frame_store.py`: extracted-frame loading/extraction and frame manifest validation.

The runtime state machine is `StreamWeaveEnv`:

1. `build_prompt(frames, retry_feedback="", extra_context="")`
   - normalizes current frames into step-local `FrameRef` objects;
   - renders memory, QA history, and timestamp-only current frames;
   - returns multimodal `ContentItem`s plus text/image views.
2. `evaluate_attempt(raw_output, frames, reward_config, repair=True)`
   - parses and scores the raw XML;
   - either repairs for execution or applies raw valid action;
   - drops answers when no question exists.
3. `commit(applied)`
   - replaces open-tail bridge when needed;
   - appends bridges, notes, and answers to memory.

Memory policies live in `streamweave/policies.py`:

- `streamweave/full/note_bridge`: read notes and bridges, use open-tail.
- `note_only/keyframe_only`: do not read bridges, no open-tail.
- `bridge_only`: read bridges but not note images.
- `recent_frames`: ignore memory and show recent frames instead.
- `no_memory/none`: no memory read, but still commit outputs.

## Inference And Evaluation Path

Main single-process entrypoints:

```bash
python evaluation/eval_ovo.py --config configs/eval_ovo.yaml --limit 1
python evaluation/eval_streamingbench.py --config configs/eval_streamingbench.yaml --limit 1
```

Batch/multi-process entrypoint:

```bash
python evaluation/eval_batch.py --config configs/batch_ovo_qwen3vl8b_8gpu_full.yaml
```

Call chain:

```text
evaluation/eval_ovo.py or eval_streamingbench.py
  -> evaluation.runner.run_cli()
  -> load_eval_config()
  -> evaluation.runner.run_eval()
     -> load_samples()
        -> evaluation.ovo_adapter.load_samples()
        -> evaluation.streamingbench_adapter.load_samples()
     -> backend.factory.create_backend()
     -> FrameStore(cfg.dataset)
     -> RolloutRunner(...)
     -> runner.run_sample(sample)
     -> adapter.score_trace(trace)
     -> write summary
```

`RolloutRunner.run_sample()` is the inference main loop:

1. `FrameStore.ensure_frames()` loads or extracts frames under:
   `dataset_root/dataset_name/video/<video_id>/`.
2. Creates `StreamWeaveEnv`.
3. Maps each `QueryEvent.timestamp` to a frame id.
4. Splits frames with `runtime.frames_per_step`.
5. For each step:
   - injects questions whose timestamp falls into current frames;
   - evicts old note images by `memory.window_seconds`;
   - builds prompt;
   - calls backend once;
   - validates/repairs output;
   - commits memory;
   - writes trace files and appends a `Transition`.
6. Returns `RolloutTrace`; final answer is the last non-empty committed answer.

Inference postprocess modes in `streamweave/rollout.py`:

- `eval_repair` / `rollout_repair`: one backend call, repair invalid XML for execution, never retry.
- `synthesis_raw_retry`: retry until raw output is valid; used by teacher data generation more than eval.

Backends:

- `backend/base.py`: `BaseBackend`, `MockBackend`.
- `backend/openai.py`: OpenAI-compatible chat completions, used for local vLLM and hosted compatible APIs.
- `backend/gemini.py`: VertexAI Gemini via `google-genai`.
- `backend/pool.py`: process-local round-robin over endpoints.
- `backend/factory.py`: picks backend from config.

Evaluation adapters:

- `evaluation/ovo_adapter.py`
  - expands OVO annotations into `BenchmarkSample`.
  - scores MCQ, counting, and yes/no tasks.
  - adds rollout metrics and writes OVO summary tables.
- `evaluation/streamingbench_adapter.py`
  - loads StreamingBench JSON by split.
  - scores option-letter tasks; proactive output has no numeric score.
- `evaluation/rollout_metrics.py`
  - counts notes, bridges, repairs, backend attempts, retry errors, reward means.

## Parser, Quality, And Repair Rules

Strict validation (`streamweave/parser.py`) requires:

- exactly one `<state>` and one `<answer>`;
- output starts with `<state>` then `<answer>`;
- all later tags are `<note>` or `<bridge>`;
- notes use paired tags with only a `t` attribute, not self-closing tags;
- no text outside allowed XML tags;
- at least one observation tag.

Quality checks (`streamweave/quality.py`) add:

- note time matches exactly one current frame;
- bridge text is non-empty;
- bridge time is inside current step unless it is the first open-tail bridge;
- events are chronological and non-overlapping;
- bridge gaps exactly match required gaps between note/window boundaries;
- open-tail memory requires the first current event to be a bridge inheriting the original start time.

Repair (`streamweave/postprocess.py`) is intentionally deterministic:

- extracts usable tags with lenient parser;
- matches note times to current frames and normalizes to the exact frame interval;
- clamps bridges into current step;
- fixes open-tail bridge start;
- drops empty/invalid bridges and out-of-window notes;
- can recover some malformed note tags.

SFT uses raw-valid output; eval/RL usually run with repair enabled.

## SFT Data Generation Path

The SFT directory does not train a model directly. It synthesizes step-level XML targets and exports LLaMAFactory-compatible ShareGPT.

Recommended large-scale entrypoint:

```bash
python data_engine/sft/run_parallel_pipeline.py --output-dir data_engine/sft/outputs/<run>
```

Single-process entrypoint:

```bash
python data_engine/sft/run_pipeline.py --stage all --output-dir data_engine/sft/outputs/<run>
```

Core flow:

```text
annotation JSON/JSONL or OVO annotations
  -> data_engine.sft.sample_sources.load_sample_source()
  -> SamplePlan
  -> data_engine.sft.rollout_sft.iter_sft_sample_records()
  -> samples/<task_index>_<sample_id>.json
  -> sample_manifest.jsonl
  -> sft_steps.jsonl
  -> export_llamafactory.export_sharegpt()
  -> llamafactory_sharegpt.jsonl
  -> dataset_info_streamweave_sft.json
```

Source adapters:

- `data_engine/sft/frame_dataset.py`: legacy extracted-frame annotations.
- `data_engine/sft/ovo_dataset.py`: OVO-Bench via `evaluation/ovo_adapter.py`, plus `FrameStore.ensure_frames()`.
- `data_engine/sft/sample_sources.py`: dispatches `source=frames` or `source=ovo`.
- `data_engine/sft/schemas.py`: `SamplePlan`, SFT-local `FrameRef`, `QueryPlan`.

SFT rollout chain:

```text
run_parallel_pipeline.py
  -> SQLite queue: pending/running/accepted/failed/error
  -> worker_loop()
     -> make_backend()
     -> make_synthesis_config()
     -> iter_sft_sample_records([sample], backend, config)
        -> _run_sft_sample()
           -> for each frame group:
              -> inject query into StreamWeaveEnv
              -> _prepare_step_context()
              -> _run_teacher_attempts()
                 -> env.build_prompt(... retry_feedback ...)
                 -> backend.generate()
                 -> env.evaluate_attempt(... repair=False)
                 -> apply SFT-only constraints
                 -> accept valid attempt or synthesize retry feedback
              -> env.commit(accepted.applied)
              -> _build_success_row()
        -> sample-level answer check
```

SFT-only constraints in `data_engine/sft/constraints.py` and `data_engine/sft/rollout_sft.py`:

- `apply_note_count_constraint()` limits each step to `max_notes_per_step` notes, default 1.
- `note_reminder_context()` can add a soft reminder when no recent note exists.
- `apply_qa_answer_constraints()` enforces answer scheduling:
  - no question: empty answer;
  - already answered: empty answer;
  - realtime/backward: answer when the question is active;
  - forward: keep answer empty before the clue window, answer when due.
- `check_sample_answer()` accepts a sample only when every emitted answer matches GT.

Current source does not implement annotated key-frame hard constraints. There is no `_key_frame_context()` or `_apply_key_frame_quality_constraints()` in the V5 SFT path.

Acceptance rule:

```text
usable_for_sft = all steps valid AND sample answer correct
```

Only usable samples contribute rows to `sft_steps.jsonl` and `llamafactory_sharegpt.jsonl`.

Important output files:

```text
samples/*.json                       per-sample full debug record
sample_manifest.jsonl                one row per sample
sft_steps.jsonl                      accepted step rows only
llamafactory_sharegpt.jsonl          final training JSONL
dataset_info_streamweave_sft.json    LLaMAFactory dataset_info snippet
summary.json                         counts and config summary
sft_jobs.sqlite                      parallel queue state
```

ShareGPT export (`data_engine/sft/export_llamafactory.py`):

- default `train_prompt_type=production`;
- target is `target_xml`;
- renders each image as a `<image>` placeholder plus `images` list;
- `recorded` prompt mode keeps the teacher prompt stored in intermediate rows, but this can include teacher-only constraints and retry feedback.

The project notes explicitly warn that normal SFT export should use production prompt, not teacher/retry prompts.

## Optional Annotation Synthesis Path

`data_engine/synthesize/` builds QA annotations before SFT. It is separate from step-level SFT synthesis.

Sequential pipeline:

```text
run_pipeline.py
  -> windows       build_windows.py
  -> captions      gen_captions.py
  -> qa            gen_qa.py
  -> filter        filter_qa.py + filter_prompts.py
  -> export        export_annotations.py
```

Parallel pipeline:

```bash
python data_engine/synthesize/run_pipeline_parallel.py --workers 2 --overwrite
```

It runs one video per worker thread and appends:

```text
windows.jsonl
captions.jsonl
qa_candidates.jsonl
qa_filtered.jsonl
annotations_qa.jsonl
progress.jsonl
summary.json
```

VLM provider wrapper is `data_engine/synthesize/vlm_client.py`, supporting OpenAI-compatible local endpoints and Gemini.

## RL Stepwise Training Path

RL is under `RL/`. The local copy of `RL/verl/` is vendored and includes StreamWeave-specific hooks. The self-owned adapter code is in `RL/streamweave_rl/`.

Basic smoke:

```bash
RL/scripts/run_smoke.sh
```

Retained launchers:

```bash
RL/scripts/train_grpo_ovo_8gpu.sh
RL/scripts/train_ppo.sh data.train_files=/path/to/train.parquet data.val_files=/path/to/val.parquet
```

Deleted historical GRPO launchers:

```text
legacy baseline GRPO launcher
legacy OVO GRPO launcher
legacy long-name 8GPU GRPO launcher
```

High-level RL flow:

```text
RL/scripts/train_grpo*.sh
  -> export STREAMWEAVE_RL_DIR and PYTHONPATH
  -> python -m verl.trainer.main_ppo --config-name=grpo_stepwise
  -> RL/verl/verl/trainer/main_ppo.py
     -> import streamweave_rl when StreamWeave config is detected
     -> RayPPOTrainer
  -> data.custom_cls = StreamWeaveAgentDataset
  -> actor_rollout_ref.rollout.agent.default_agent_loop = streamweave_agent
  -> StreamWeaveAgentLoop.run()
  -> StreamWeaveRLEnv.reset()/step()
  -> reward and stepwise advantage
```

RL config files:

- `RL/configs/grpo_stepwise.yaml`
  - custom dataset class;
  - `algorithm.adv_estimator=streamweave_stepwise_traj_grpo`;
  - `critic.enable=false`;
  - rollout `n=8`;
  - multi-turn agent loop enabled.
- `RL/configs/ppo_stepwise.yaml`
  - `algorithm.adv_estimator=streamweave_stepwise_gae`;
  - critic enabled;
  - rollout `n=1`.
- `RL/configs/streamweave_agent_stepwise.yaml`
  - registers `streamweave_agent` to `streamweave_rl.agent_loop_stepwise.StreamWeaveAgentLoop`.

### RL Dataset

`RL/streamweave_rl/dataset.py` returns metadata, not real prompts. Real multimodal prompts are rendered inside the agent loop because every turn depends on evolving StreamWeave memory.

Each item includes:

```text
dummy_tensor
agent_name = streamweave_agent
sample_id, video_id, video_path
question, query_timestamp, ground_truth
sample_metadata
streamweave_config
reward_model.ground_truth
raw_prompt placeholder
```

Supported input formats: parquet, jsonl, json, csv.

OVO-specific handling:

- backward/realtime tasks are direct rows;
- forward tasks with `test_info` are expanded into one row per test item;
- MCQ questions are formatted consistently with eval adapter;
- ground truth is normalized for OVO scoring.

### RL Environment

`RL/streamweave_rl/env.py` wraps the same `StreamWeaveEnv`.

Important difference from eval/SFT: RL expects frames to already be extracted. `reset()` calls `FrameStore.load_frames()`, not `ensure_frames()`. Missing frames under:

```text
dataset_root/dataset_name/video/<video_id>/
```

raise an error and abort the trajectory with zero reward in the agent loop.

`StreamWeaveRLEnv.reset()`:

1. loads pre-extracted frames;
2. groups them by `runtime.frames_per_step`;
3. applies `runtime.max_steps`;
4. checks at least one query survives truncation;
5. creates `StreamWeaveEnv`;
6. prepares first prompt turn.

`StreamWeaveRLEnv.step(action_str)`:

1. validates/repairs XML with `env.evaluate_attempt(... repair=True)`;
2. commits memory;
3. computes per-step reward;
4. on final step computes final answer and trajectory reward;
5. returns next observation or done.

The observation is:

```text
messages: one multimodal user message
images: PIL images for vLLM/SGLang processor
prompt_text / prompt_images: debug mirrors
```

### RL Agent Loop

`RL/streamweave_rl/agent_loop_stepwise.py` registers:

```python
@register("streamweave_agent")
class StreamWeaveAgentLoop(AgentLoopBase)
```

For each trajectory:

1. builds `StreamWeaveRLEnv` from dataset kwargs;
2. `env.reset()`;
3. loop:
   - apply chat template to current multimodal observation;
   - guard prompt length against `max_model_len - response_length` and `prompt_length`;
   - call async rollout server;
   - decode response;
   - `env.step(response_text)`;
   - emit one `AgentLoopOutput` per StreamWeave step.
4. on done, backfills final `trajectory_score`, `success_score`, and `final_answer` into every step output.
5. on prompt too long, empty response, missing frames, or other error, returns a zero-reward abort output.

The stepwise identity fields are critical:

```text
group_idx: original sample uid
traj_idx: rollout sample index for GRPO
turn_idx: StreamWeave step index
last_turn: whether this is final turn
trajectory_score / success_score / format_score / step_score / turn_reward
```

### RL Rewards

`RL/streamweave_rl/rewards.py`:

- per-step reward = `(w_format * format_score + w_step * step_score) / total_steps`;
- final turn adds `w_success * success_score`;
- trajectory score = `w_format * format_mean + w_step * step_mean + w_success * success_score`.

Default config:

```text
w_format=0.3
w_success=0.4
w_step=0.3
score_scale=2.0
format_mode=valid
success_mode=dataset
success_scorer=auto
```

`RL/streamweave_rl/scorers.py` implements:

- default exact/contains modes;
- StreamWeave MCQ scoring;
- OVO task scoring for MCQ, counting, yes/no.

### RL Advantages

`RL/streamweave_rl/advantage.py` registers two custom estimators into verl:

- `streamweave_stepwise_gae`
  - groups by `(group_idx, traj_idx, turn_idx)`;
  - runs GAE over turn-level rewards within each trajectory;
  - writes returns only at final valid token of each response;
  - uses `ignore_value` elsewhere for value masking.
- `streamweave_stepwise_traj_grpo`
  - computes trajectory-level advantage across sampled trajectories for each `group_idx`;
  - assigns the same normalized trajectory advantage to every response token in that trajectory.

### Vendored verl Hooks

Most `RL/verl/` is upstream framework code. StreamWeave-relevant hooks are:

- `RL/verl/verl/trainer/main_ppo.py`
  - `_register_optional_streamweave_integrations()` imports `streamweave_rl` when selected by config.
- `RL/verl/verl/trainer/ppo/core_algos.py`
  - generic `register_adv_est()` / `get_adv_estimator_fn()` registry supports custom string estimators.
- `RL/verl/verl/trainer/ppo/ray_trainer.py`
  - imports `streamweave_rl.advantage`;
  - `_assign_stepwise_indices()` creates `group_idx` and `traj_idx`;
  - `_align_stepwise_batch()` expands original batch rows to match multiple step outputs;
  - `_add_stepwise_value_mask()` masks critic value loss to valid return positions;
  - `_compute_streamweave_stepwise_metrics()` logs StreamWeave scores;
  - validation uses `final_answer` and `trajectory_score` when stepwise mode is on.
- `RL/verl/verl/utils/dataset/rl_dataset.py`
  - `data.custom_cls` dynamically loads `StreamWeaveAgentDataset`.

Avoid editing vendored verl unless the bug is in stepwise trainer plumbing. For prompt/memory/reward behavior, edit `streamweave/` or `RL/streamweave_rl/`.

## Data And Frame Expectations

Eval/SFT can extract frames via `FrameStore.ensure_frames()` when a source video path exists.

RL intentionally does not extract frames during rollout. Prepare frames in advance under:

```text
dataset/<dataset_name>/video/<video_id>/000000.jpg
dataset/<dataset_name>/video/<video_id>/000001.jpg
...
dataset/<dataset_name>/video/<video_id>/manifest.json
```

`FrameStore` expects numeric contiguous names relative to `frame_id_base`. The manifest records source path, fps, max frames, image extension, frame count, extractor version, and source file metadata.

## Common Change Targets

- Prompt wording: `streamweave/prompts.py`.
- XML grammar/strictness: `streamweave/parser.py`.
- note/bridge timing/open-tail reward: `streamweave/quality.py`.
- eval/RL repair behavior: `streamweave/postprocess.py`.
- inference loop behavior: `streamweave/rollout.py`.
- OVO loading/scoring: `evaluation/ovo_adapter.py`.
- SFT note-count/QA constraints: `data_engine/sft/constraints.py` and `data_engine/sft/rollout_sft.py`.
- SFT export prompt or image placeholders: `data_engine/sft/export_llamafactory.py`.
- RL dataset row normalization: `RL/streamweave_rl/dataset.py`.
- RL environment reward/step behavior: `RL/streamweave_rl/env.py` and `RL/streamweave_rl/rewards.py`.
- RL trajectory grouping/advantage: `RL/streamweave_rl/advantage.py`.
- RL launch hyperparameters: `RL/configs/*.yaml` and `RL/scripts/*.sh`.

## Useful Smoke Commands

Parser/runtime package import:

```bash
python - <<'PY'
from streamweave.parser import strict_validate_raw_output
raw = '<state>x</state><answer></answer><bridge t="0.0-1.0">x</bridge>'
print(strict_validate_raw_output(raw).parser_ok)
PY
```

SFT mock pipeline, tiny local run:

```bash
python data_engine/sft/run_pipeline.py \
  --backend mock \
  --limit 1 \
  --output-dir /tmp/streamweave_sft_smoke \
  --overwrite
```

RL adapter smoke:

```bash
RL/scripts/run_smoke.sh
```

Eval with mock backend:

```bash
python evaluation/eval_ovo.py --config configs/eval_ovo.yaml --backend mock --limit 1
```

The mock eval may still need configured annotation/video paths unless the config points at existing local data.

## Current Experiment Notes

For current status and historical conclusions, read:

```text
note/README.md
note/experiment-log.md
note/00-overview.md
note/04-sft-training.md
note/05-rl-training.md
note/07-key-points.md
note/08-commands-and-tools.md
```

As of the notes read during this guide:

- first SFT evaluation was negative relative to base instruct;
- current main line is V5 GRPO stepwise RL;
- recent GRPO run reached partway through training but ended without checkpoint;
- the next operational priority in notes is lowering checkpoint save frequency before long reruns.

Treat these notes as experiment state, not source of truth for code behavior.
