# StreamWeave SFT Data Pipeline

This directory generates step-level SFT rows from StreamWeave rollouts and exports accepted rows to ShareGPT/LLaMAFactory format.

## Current Protocol

The SFT target follows the same state protocol as the main evaluation path:

```xml
<state>...</state>
<answer>...</answer>
<bridge t="0.0-4.0">...</bridge>
<note t="4.0-5.0"></note>
<bridge t="5.0-10.0">...</bridge>
```

Rules:

- `<state>` is required once and appears before `<answer>`.
- `<state>` is a current-step reasoning summary only. It is not committed to Memory.
- `<answer>` is required once. It is empty when the current QA History should not be answered.
- `<note>` uses only `t`; do not emit frame ids or local ids.
- `<note>` must use paired tags: `<note t="12.0-13.0"></note>`.
- `<bridge>` records observable progression between note/window boundaries.
- The eta tag is not part of the protocol.

Current frames in prompts are rendered as:

```xml
<frame t="12.0-13.0"><image></frame>
```

The model should select notes by copying the frame time range into the note `t` attribute.

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

SFT-only constraints:

- Annotated key frames are converted to required note time ranges.
- If a required annotated note is present in the current window, the output must include exactly one matching `<note t="..."></note>`.
- If no annotated key frame is present in the current window, no note should be emitted for that key-frame constraint.
- QA scheduling only checks whether `<answer>` should be empty or non-empty. There is no eta target.
- `target_timestamp` or `answer_time` truncates the sample frames so the teacher does not see after the target boundary.

## Output Rows

Accepted rows contain:

- `memory_before`: structured memory before the step.
- `qa_history`: QA records before the step.
- `current_frames`: current frame time ranges and image paths.
- `target_xml`: accepted teacher XML.
- `metadata.raw_action`: parsed raw action.
- `metadata.applied`: committed notes, bridges, and answer.

`target_xml` is the assistant-side SFT target.

## ShareGPT Export

Exporter:

```text
data_engine/sft/export_llamafactory.py
```

Default export rebuilds the training prompt from the row using the current prompt builder, so it inherits the state protocol and timestamp-only notes. Use `train_prompt_type=recorded` only when you intentionally want the prompt exactly as captured during synthesis.

## Smoke Checks

Compile the SFT path:

```bash
/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python -m py_compile data_engine/sft/rollout_sft.py data_engine/sft/export_llamafactory.py
```

Validate a minimal XML target:

```bash
/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python - <<'PY'
from streamweave.parser import strict_validate_raw_output
raw = '<state>x</state><answer></answer><note t="0.0-1.0"></note>'
print(strict_validate_raw_output(raw).parser_ok)
PY
```
