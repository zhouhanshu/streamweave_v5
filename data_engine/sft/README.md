# StreamWeave SFT Data Pipeline

This directory generates step-level SFT rows from StreamWeave rollouts and exports accepted rows to ShareGPT/LLaMAFactory format.

## Current Protocol

The SFT target follows the same state protocol as the main evaluation path:

```xml
<state>...</state>
<answer>...</answer>
<delta t="0.0-4.0">...</delta>
<anchor t="4.0-5.0"></anchor>
<delta t="5.0-10.0">...</delta>
```

Rules:

- `<state>` is required once and appears before `<answer>`.
- `<state>` is a current-step reasoning summary only. It is not committed to Memory.
- `<answer>` is required once. It is empty when the current QA History should not be answered.
- `<anchor>` uses only `t`; do not emit frame ids or local ids.
- `<anchor>` must use paired tags: `<anchor t="12.0-13.0"></anchor>`.
- `<delta>` records observable progression between anchor/window boundaries.
- The eta tag is not part of the protocol.

Current frames in prompts are rendered as:

```xml
<frame t="12.0-13.0"><image></frame>
```

The model should select anchors by copying the frame time range into the anchor `t` attribute.

## SFT Rollout

Main module:

```text
data_engine/sft/rollout_sft.py
```

Important flow:

```text
SamplePlan
  -> group frames
  -> build StreamWeave prompt
  -> call teacher backend
  -> strict validate raw XML
  -> apply SFT-only constraints
  -> commit accepted action to Memory
  -> write step row
  -> sample-level final answer check
```

Input sources:

- `--source frames`: legacy one-row-one-QA frame annotations.
- `--source dataset2`: one dataset2 `sft.jsonl` file or one dataset directory containing `sft.jsonl`. Single-QA rows follow the legacy flow. Rows with `qa_list` are treated as one video with multiple QA branches: prefix steps build memory once, then the final step is copied per QA and only answer-correct branches are exported.
- `--source dataset2` intentionally does not recursively collect every child dataset. Export each dataset to its own output directory, then merge ShareGPT JSONL explicitly.

SFT-only constraints:

- Each step can be limited to at most `max_notes_per_step` anchors.
- A soft anchor reminder can be added when memory has gone too long without a recent anchor.
- QA scheduling only checks whether `<answer>` should be empty or non-empty. There is no eta target.
- `target_timestamp` or `answer_time` truncates the sample frames so the teacher does not see after the target boundary.
- Open-ended final answer checks use conservative text similarity heuristics; MCQ answers still use option-letter/option-text matching.

Current V5 source does not implement annotated key-frame hard constraints. If future experiments need required key-frame anchors, that constraint must be added explicitly.

## Output Rows

Accepted rows contain:

- `memory_before`: structured memory before the step.
- `qa_history`: QA records before the step.
- `current_frames`: current frame time ranges and image paths.
- `target_xml`: accepted teacher XML.
- `metadata.raw_action`: parsed raw action.
- `metadata.applied`: committed anchors, deltas, and answer.

`target_xml` is the assistant-side SFT target.

## ShareGPT Export

Exporter:

```text
data_engine/sft/export_llamafactory.py
```

Default export rebuilds the training prompt from the row using the current prompt builder, so it inherits the state protocol and timestamp-only anchors. Use `train_prompt_type=recorded` only when you intentionally want the prompt exactly as captured during synthesis.

## Smoke Checks

Compile the SFT path:

```bash
/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python -m py_compile data_engine/sft/rollout_sft.py data_engine/sft/export_llamafactory.py
```

Validate a minimal XML target:

```bash
/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python - <<'PY'
from streamweave.parser import strict_validate_raw_output
raw = '<state>x</state><answer></answer><anchor t="0.0-1.0"></anchor>'
print(strict_validate_raw_output(raw).parser_ok)
PY
```
