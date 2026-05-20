# case_1266_acr_white_outfit_snowdrift

较长 realtime/ACR 样例。RL 答对人在雪堆后喝水；base/text 误判其他动作。

## Task

- Benchmark: OVO-Bench
- sample_id: `1266`
- task: `ACR`
- realtime/query time: `193.85`
- question: What is the person in the white outfit doing?
- answer text: He is having a drink.
- ground truth option: `C`

Options:
- A. Lying down on a frozen lake, adjusting the sights on their rifle.
- B. Standing on top of a snowy hill, peering through binoculars towards the horizon.
- C. Hiding behind a snowdrift,having a drink.
- D. Hiding behind a snowdrift, setting up a camera on a tripod.

## Model Comparison

| Model | Response | GT | Score | Steps | Repairs | Trace | Step outputs |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| RL qwen3vl_rl_exp10_step40 | `C` | `C` | 1 | 39 | 29 | [trace.txt](models/rl_exp10_step40/trace.txt) / [trace.jsonl](models/rl_exp10_step40/trace.jsonl) | [step_outputs.md](models/rl_exp10_step40/step_outputs.md) |
| Base qwen3vl_8b_base_0516 | `D` | `C` | 0 | 39 | 85 | [trace.txt](models/base_qwen3vl_8b_0516/trace.txt) / [trace.jsonl](models/base_qwen3vl_8b_0516/trace.jsonl) | [step_outputs.md](models/base_qwen3vl_8b_0516/step_outputs.md) |
| StreamText Gemini Flash | `A` | `C` | 0 | 39 | 0 | [trace.txt](models/streamtext_gemini_flash/trace.txt) / [trace.jsonl](models/streamtext_gemini_flash/trace.jsonl) | [step_outputs.md](models/streamtext_gemini_flash/step_outputs.md) |

## Files

- Complete copied frames: [images/](images/) (194 image frames, plus manifest if present)
- All-frame visual index: [frames.md](frames.md)
- Machine-readable manifest: [manifest.json](manifest.json)
- Each model directory contains `result.json`, `trace.jsonl`, `trace.txt`, `memory.txt`, and `step_outputs.md`.

## Selection Criterion

RL scored `1`; base and streamtext both scored `0` on the same OVO sample.
