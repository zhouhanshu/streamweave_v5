# OVO vLLM Evaluation Notes

This note records the exact StreamWeave v5 OVO evaluation path and the common pitfalls hit during the step-200 SFT evaluation.

## Step-200 Model

The exported step-200 model for the anchor/delta + initial-anchor SFT run is:

```bash
/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_anchor_delta_init_anchor_step200_vllm
```

It was exported from:

```bash
SFT/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_answered_full_anchor_delta_init_anchor/checkpoint-200
```

This is a full fine-tuned HF model, not a LoRA adapter. It does not need merge.

The inference model directory should contain only the HF inference files, not DeepSpeed optimizer state:

```text
chat_template.jinja
chat_template.json
config.json
generation_config.json
merges.txt
model.safetensors
preprocessor_config.json
processor_config.json
tokenizer.json
tokenizer_config.json
video_preprocessor_config.json
vocab.json
```

## Tokenizer Pitfall

LlamaFactory checkpoint export may write this into `tokenizer_config.json`:

```json
"extra_special_tokens": [
  "<|im_start|>",
  "<|im_end|>",
  ...
]
```

The local vLLM/Transformers stack can crash on this with:

```text
AttributeError: 'list' object has no attribute 'keys'
```

Fix:

1. Remove the `extra_special_tokens` field from the exported model's `tokenizer_config.json`.
2. Copy the missing tokenizer/processor sidecar files from the base model:

```bash
MODEL_DIR=/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_anchor_delta_init_anchor_step200_vllm
BASE=/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct

cp -a \
  "$BASE/chat_template.json" \
  "$BASE/merges.txt" \
  "$BASE/preprocessor_config.json" \
  "$BASE/video_preprocessor_config.json" \
  "$BASE/vocab.json" \
  "$MODEL_DIR/"
```

Verify tokenizer/processor load:

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5

/mmu_mllm_hdd/zhouhanshu/conda/envs/vllm/bin/python - <<'PY'
from transformers import AutoProcessor, AutoTokenizer

p = "models/qwen3vl8b_streamweave_sft_answered_full_anchor_delta_init_anchor_step200_vllm"
tok = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
print("tokenizer_ok", type(tok).__name__, tok.eos_token, tok.pad_token)
proc = AutoProcessor.from_pretrained(p, trust_remote_code=True)
print("processor_ok", type(proc).__name__)
PY
```

Expected:

```text
tokenizer_ok Qwen2TokenizerFast <|im_end|> <|endoftext|>
processor_ok Qwen3VLProcessor
```

## Run OVO 1/8 Evaluation

The existing 1/8 script is:

```bash
scripts/run_ovo_8gpu_vllm_finetuned.sh
```

It uses:

```text
configs/batch_ovo_qwen3vl8b_finetuned_8gpu.yaml
/mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_1of8_stratified.json
```

The 1/8 split currently has 205 samples.

Run step-200 evaluation:

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5

OUTPUT_DIR=outputs/ovo_qwen3vl8b_sft_anchor_delta_step200_1of8 \
bash scripts/run_ovo_8gpu_vllm_finetuned.sh \
  /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_anchor_delta_init_anchor_step200_vllm
```

This starts 8 local vLLM servers:

```text
GPU 0 -> http://127.0.0.1:8000/v1
GPU 1 -> http://127.0.0.1:8001/v1
...
GPU 7 -> http://127.0.0.1:8007/v1
```

The evaluation then runs 16 workers over those endpoints.

Outputs:

```text
outputs/ovo_qwen3vl8b_sft_anchor_delta_step200_1of8/results.jsonl
outputs/ovo_qwen3vl8b_sft_anchor_delta_step200_1of8/results_summary.json
outputs/ovo_qwen3vl8b_sft_anchor_delta_step200_1of8/results_summary.md
outputs/ovo_qwen3vl8b_sft_anchor_delta_step200_1of8/traces/
outputs/ovo_qwen3vl8b_sft_anchor_delta_step200_1of8/worker_logs/
outputs/ovo_qwen3vl8b_sft_anchor_delta_step200_1of8/vllm_logs/
```

## Cleanup

If a run is interrupted and leaves vLLM processes around, clean them with the saved pid files:

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5

for f in outputs/ovo_qwen3vl8b_sft_anchor_delta_step200_1of8/vllm_pids/*.pid; do
  [ -f "$f" ] || continue
  pid="$(cat "$f")"
  kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
done
```

If ports are still occupied, check them:

```bash
ss -ltnp | grep -E ':800[0-7]\b'
```

## GPU Check

Before launching 8 vLLM replicas, check GPU usage:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

Do not launch the 8-replica OVO script while SFT/RL training is still using those GPUs.

## Current Evaluation Path

The current StreamWeave output protocol is `<anchor>/<delta>`.

The runtime accepts the old internal names in schema and metrics (`NoteRecord`, `BridgeRecord`, `event.kind == "note"`), but visible model output should use:

```xml
<state>...</state>
<answer>...</answer>
<anchor t="..."></anchor>
<delta t="...">...</delta>
```

For empty Memory, the first observation tag must be the first-frame anchor:

```xml
<anchor t="0.0-1.0"></anchor>
```

The validator/repair path enforces this in current code.
