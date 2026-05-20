# case_591_asi_sandwich_before_throw_clothes

纯文本失败明显：streamtext 输出了非选项解释；base 直接 cannot answer；RL 选对。

## Task

- Benchmark: OVO-Bench
- sample_id: `591`
- task: `ASI`
- realtime/query time: `46`
- question: Which object did the person eat before they threw the clothes?
- answer text: The sandwich.
- ground truth option: `A`

Options:
- A. The sandwich.
- B. The medicine.
- C. The refrigerator.
- D. The towel.

## Model Comparison

| Model | Response | GT | Score | Steps | Repairs | Trace | Step outputs |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| RL qwen3vl_rl_exp10_step40 | `A` | `A` | 1 | 10 | 4 | [trace.txt](models/rl_exp10_step40/trace.txt) / [trace.jsonl](models/rl_exp10_step40/trace.jsonl) | [step_outputs.md](models/rl_exp10_step40/step_outputs.md) |
| Base qwen3vl_8b_base_0516 | `cannot answer` | `A` | 0 | 10 | 21 | [trace.txt](models/base_qwen3vl_8b_0516/trace.txt) / [trace.jsonl](models/base_qwen3vl_8b_0516/trace.jsonl) | [step_outputs.md](models/base_qwen3vl_8b_0516/step_outputs.md) |
| StreamText Gemini Flash | `None of the above options are correct. The person was eating chips.` | `A` | 0 | 10 | 0 | [trace.txt](models/streamtext_gemini_flash/trace.txt) / [trace.jsonl](models/streamtext_gemini_flash/trace.jsonl) | [step_outputs.md](models/streamtext_gemini_flash/step_outputs.md) |

## Files

- Complete copied frames: [images/](images/) (46 image frames, plus manifest if present)
- All-frame visual index: [frames.md](frames.md)
- Machine-readable manifest: [manifest.json](manifest.json)
- Each model directory contains `result.json`, `trace.jsonl`, `trace.txt`, `memory.txt`, and `step_outputs.md`.

## Selection Criterion

RL scored `1`; base and streamtext both scored `0` on the same OVO sample.
