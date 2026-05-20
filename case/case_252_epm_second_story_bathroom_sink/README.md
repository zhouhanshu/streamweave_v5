# case_252_epm_second_story_bathroom_sink

新增 EPM 样例。RL 和 streamtext 都答对二楼洗手位置；base 输出 cannot answer。

## Task

- Benchmark: OVO-Bench
- sample_id: `252`
- task: `EPM`
- realtime/query time: `234`
- question: Where can I wash my hands on the second story of the house?
- answer text: The bathroom sink
- ground truth option: `A`

Options:
- A. The bathroom sink
- B. The table in the dining room
- C. The shelf in the hallway
- D. The shelf in the closet

## Model Comparison

| Model | Response | GT | Score | Steps | Repairs | Trace | Step outputs |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| RL qwen3vl_rl_exp10_step40 | `A` | `A` | 1 | 47 | 35 | [trace.txt](models/rl_exp10_step40/trace.txt) / [trace.jsonl](models/rl_exp10_step40/trace.jsonl) | [step_outputs.md](models/rl_exp10_step40/step_outputs.md) |
| Base qwen3vl_8b_base_0516 | `cannot answer` | `A` | 0 | 47 | 80 | [trace.txt](models/base_qwen3vl_8b_0516/trace.txt) / [trace.jsonl](models/base_qwen3vl_8b_0516/trace.jsonl) | [step_outputs.md](models/base_qwen3vl_8b_0516/step_outputs.md) |
| StreamText Gemini Flash | `A` | `A` | 1 | 47 | 0 | [trace.txt](models/streamtext_gemini_flash/trace.txt) / [trace.jsonl](models/streamtext_gemini_flash/trace.jsonl) | [step_outputs.md](models/streamtext_gemini_flash/step_outputs.md) |

## Files

- Complete copied frames: [images/](images/) (234 image frames, plus manifest if present)
- All-frame visual index: [frames.md](frames.md)
- Machine-readable manifest: [manifest.json](manifest.json)
- Each model directory contains `result.json`, `trace.jsonl`, `trace.txt`, `memory.txt`, and `step_outputs.md`.

## Selection Criterion

RL scored `1`; base scored `0`; streamtext also scored `1`, so this is primarily an RL/base contrast case.
