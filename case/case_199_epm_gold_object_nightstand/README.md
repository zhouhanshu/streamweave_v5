# case_199_epm_gold_object_nightstand

新增 hard EPM 样例。正确答案是 nightlamp；RL 误选 vase，base/text 都未能回答。

## Task

- Benchmark: OVO-Bench
- sample_id: `199`
- task: `EPM`
- realtime/query time: `110`
- question: What is the gold object on the nightstand?
- answer text: A nightlamp
- ground truth option: `C`

Options:
- A. A painting
- B. A mirror
- C. A nightlamp
- D. A vase

## Model Comparison

| Model | Response | GT | Score | Steps | Repairs | Trace | Step outputs |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| RL qwen3vl_rl_exp10_step40 | `D` | `C` | 0 | 22 | 11 | [trace.txt](models/rl_exp10_step40/trace.txt) / [trace.jsonl](models/rl_exp10_step40/trace.jsonl) | [step_outputs.md](models/rl_exp10_step40/step_outputs.md) |
| Base qwen3vl_8b_base_0516 | `cannot answer` | `C` | 0 | 22 | 37 | [trace.txt](models/base_qwen3vl_8b_0516/trace.txt) / [trace.jsonl](models/base_qwen3vl_8b_0516/trace.jsonl) | [step_outputs.md](models/base_qwen3vl_8b_0516/step_outputs.md) |
| StreamText Gemini Flash | `The question cannot be answered from the provided video frames.` | `C` | 0 | 22 | 0 | [trace.txt](models/streamtext_gemini_flash/trace.txt) / [trace.jsonl](models/streamtext_gemini_flash/trace.jsonl) | [step_outputs.md](models/streamtext_gemini_flash/step_outputs.md) |

## Files

- Complete copied frames: [images/](images/) (110 image frames, plus manifest if present)
- All-frame visual index: [frames.md](frames.md)
- Machine-readable manifest: [manifest.json](manifest.json)
- Each model directory contains `result.json`, `trace.jsonl`, `trace.txt`, `memory.txt`, and `step_outputs.md`.

## Selection Criterion

All three compared models scored `0`; this is included as a hard failure example rather than an RL-success case.
