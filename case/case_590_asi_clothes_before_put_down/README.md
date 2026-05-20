# case_590_asi_clothes_before_put_down

短 backward/ASI 样例。RL 答对前序动作；base 和纯文本都选错。

## Task

- Benchmark: OVO-Bench
- sample_id: `590`
- task: `ASI`
- realtime/query time: `18`
- question: What happened before the person put down the clothes?
- answer text: Tidied up the blanket.
- ground truth option: `B`

Options:
- A. Put down the clothes.
- B. Tidied up the blanket.
- C. Took the blanket.
- D. Washed the cup/glass/bottle.

## Model Comparison

| Model | Response | GT | Score | Steps | Repairs | Trace | Step outputs |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| RL qwen3vl_rl_exp10_step40 | `B` | `B` | 1 | 4 | 2 | [trace.txt](models/rl_exp10_step40/trace.txt) / [trace.jsonl](models/rl_exp10_step40/trace.jsonl) | [step_outputs.md](models/rl_exp10_step40/step_outputs.md) |
| Base qwen3vl_8b_base_0516 | `C` | `B` | 0 | 4 | 11 | [trace.txt](models/base_qwen3vl_8b_0516/trace.txt) / [trace.jsonl](models/base_qwen3vl_8b_0516/trace.jsonl) | [step_outputs.md](models/base_qwen3vl_8b_0516/step_outputs.md) |
| StreamText Gemini Flash | `A` | `B` | 0 | 4 | 0 | [trace.txt](models/streamtext_gemini_flash/trace.txt) / [trace.jsonl](models/streamtext_gemini_flash/trace.jsonl) | [step_outputs.md](models/streamtext_gemini_flash/step_outputs.md) |

## Files

- Complete copied frames: [images/](images/) (18 image frames, plus manifest if present)
- All-frame visual index: [frames.md](frames.md)
- Machine-readable manifest: [manifest.json](manifest.json)
- Each model directory contains `result.json`, `trace.jsonl`, `trace.txt`, `memory.txt`, and `step_outputs.md`.

## Selection Criterion

RL scored `1`; base and streamtext both scored `0` on the same OVO sample.
