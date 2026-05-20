# case_493_asi_inner_tube_leak

修补内胎场景。RL 识别到 polish 前先找漏点；base/text 都误判为 apply glue。

## Task

- Benchmark: OVO-Bench
- sample_id: `493`
- task: `ASI`
- realtime/query time: `53`
- question: What does the person do before use sandpaper/metal to polish rubber near leak
- answer text: look for leaks
- ground truth option: `B`

Options:
- A. put inner tube back
- B. look for leaks
- C. apply glue
- D. paste patch

## Model Comparison

| Model | Response | GT | Score | Steps | Repairs | Trace | Step outputs |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| RL qwen3vl_rl_exp10_step40 | `B` | `B` | 1 | 11 | 6 | [trace.txt](models/rl_exp10_step40/trace.txt) / [trace.jsonl](models/rl_exp10_step40/trace.jsonl) | [step_outputs.md](models/rl_exp10_step40/step_outputs.md) |
| Base qwen3vl_8b_base_0516 | `C` | `B` | 0 | 11 | 25 | [trace.txt](models/base_qwen3vl_8b_0516/trace.txt) / [trace.jsonl](models/base_qwen3vl_8b_0516/trace.jsonl) | [step_outputs.md](models/base_qwen3vl_8b_0516/step_outputs.md) |
| StreamText Gemini Flash | `C` | `B` | 0 | 11 | 0 | [trace.txt](models/streamtext_gemini_flash/trace.txt) / [trace.jsonl](models/streamtext_gemini_flash/trace.jsonl) | [step_outputs.md](models/streamtext_gemini_flash/step_outputs.md) |

## Files

- Complete copied frames: [images/](images/) (53 image frames, plus manifest if present)
- All-frame visual index: [frames.md](frames.md)
- Machine-readable manifest: [manifest.json](manifest.json)
- Each model directory contains `result.json`, `trace.jsonl`, `trace.txt`, `memory.txt`, and `step_outputs.md`.

## Selection Criterion

RL scored `1`; base and streamtext both scored `0` on the same OVO sample.
