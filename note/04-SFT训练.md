# SFT 训练

本文记录 StreamWeave V5 的 SFT 训练主线：数据入口、LLaMAFactory 注册、训练配置、启动/续跑命令、checkpoint 状态、导出模型和历史结论。SFT 数据清洗链路见 `02-SFT数据清洗.md`。

## 当前状态

当前主线训练是 `0516_4500`：

```text
数据文件: dataset2/sft_0516_4500.jsonl
数据行数: 67856
LLaMAFactory 数据集名: streamweave_sft_0516_4500
训练配置: SFT/LlamaFactory/configs/qwen3vl_8b_full_sft_streamweave_0516_4500.yaml
启动脚本: SFT/LlamaFactory/configs/train_streamweave_0516_4500.sh
输出目录: SFT/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_0516_4500
训练日志: SFT/LlamaFactory/logs/sft0516_4500_20260516_195458/train.out
```

截至当前记录：

```text
最新可恢复 checkpoint: checkpoint-100
global_step: 100
max_steps: 263
epoch: 0.3810431055
step 100 loss: 0.2023988962
step 100 lr: 7.173382108829826e-06
eval: 尚未发生，配置是 eval_steps=200
```

注意：`trainer_log.jsonl` 里已经出现了 step 101 的一条训练日志，loss 为 `0.1953287125`，但当前磁盘上没有 `checkpoint-101`。所以后续续跑应该从 `checkpoint-100` 恢复。当前检查没有发现仍在运行的 `llamafactory-cli` 或 `torchrun` 进程。

当前已经存在的 0516 导出模型目录：

```text
models/qwen3vl_sft_0516_step50
models/qwen3vl_sft_0516_step100
```

## 训练总览

| 训练 | 数据 | 配置 | 输出 / 导出 | 状态 |
| --- | --- | --- | --- | --- |
| 早期 LoRA smoke | `streamweave_sft`，122 examples | 历史 `examples/train_lora/qwen3vl_8b_lora_sft_streamweave.yaml` | `saves/qwen3-vl-8b/lora/streamweave_sft` | 已完成，只验证链路，非当前主线 |
| V2 3077 full SFT | `streamweave_sft_v2_3077` | `qwen3vl_8b_lora_sft_streamweave_v2_3077.yaml` | `saves/qwen3-vl-8b/full/streamweave_sft_v2_3077` | 配置和脚本保留，当前 saves 下未见输出目录 |
| answered-full SFT | `streamweave_sft_answered_full` | `qwen3vl_8b_full_sft_streamweave_answered_full.yaml` | `models/qwen3vl8b_streamweave_sft_answered_full_vllm` | 历史训练完成，旧笔记记录了完整指标 |
| answered-full init-anchor | `streamweave_sft_answered_full_anchor_delta_init_anchor` | `qwen3vl_8b_full_sft_streamweave_answered_full_anchor_delta_init_anchor.yaml` | `models/qwen3vl8b_sft_anchor_delta_step200_vllm` | 可见 step200 vLLM 导出模型；当前未找到完整训练日志 |
| 0511 note SFT | `dataset2/sft_0511_note.jsonl`，182998 行 | `qwen3vl_8b_full_sft_streamweave_0511_note.yaml` | `models/qwen_sft_0513` | 已完成，708/708 step |
| 0516 4500 SFT | `dataset2/sft_0516_4500.jsonl`，67856 行 | `qwen3vl_8b_full_sft_streamweave_0516_4500.yaml` | `models/qwen3vl_sft_0516_step50`，`models/qwen3vl_sft_0516_step100` | 当前主线，checkpoint-100 已保存 |

## 数据注册

数据注册文件：

```text
SFT/LlamaFactory/data_streamweave/dataset_info.json
```

当前保留的 SFT 注册名：

```text
streamweave_sft
streamweave_sft_v2_3077
streamweave_sft_answered_full
streamweave_sft_answered_full_anchor_delta_init_anchor
streamweave_sft_0511
streamweave_sft_0511_150k_6to4
streamweave_sft_0511_note
streamweave_sft_0516_4500
```

当前 0516 注册项：

```json
"streamweave_sft_0516_4500": {
  "file_name": "/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset2/sft_0516_4500.jsonl",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages",
    "images": "images"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
}
```

SFT 数据格式要求：

```json
{
  "messages": [
    {"role": "user", "content": "...<image>..."},
    {"role": "assistant", "content": "..."}
  ],
  "images": ["video/xxx/000000.jpg"]
}
```

要求：

- `messages` 按 `user -> assistant` 成对出现。
- 文本中的 `<image>` 数量必须等于 `images` 列表长度。
- `images` 是相对路径时，训练配置里的 `media_dir` 必须指向这些相对路径的根目录。
- 当前 dataset2 系列数据的图片路径通常是 `video/...jpg`，所以 `media_dir` 使用 `dataset2`。
- 每次新训练必须新增或明确复用 dataset name，并使用新的 `output_dir`，避免覆盖 checkpoint。

## 0516 4500 SFT

这是当前正在推进的 SFT 主线。

数据：

```text
dataset2/sft_0516_4500.jsonl
rows: 67856
dataset name: streamweave_sft_0516_4500
media_dir: /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset2
```

模型和训练方式：

```text
base model: /mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct
template: qwen3_vl_nothink
stage: sft
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
deepspeed: examples/deepspeed/ds_z3_config.json
```

多模态和长度参数：

```text
cutoff_len: 16384
image_max_pixels: 131072
video_max_pixels: 16384
preprocessing_num_workers: 32
dataloader_num_workers: 8
tokenized_path: SFT/LlamaFactory/cache/streamweave_sft_0516_4500
overwrite_cache: true
```

训练超参：

```text
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
learning_rate: 1.0e-5
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000
```

保存和评估：

```text
output_dir: saves/qwen3-vl-8b/full/streamweave_sft_0516_4500
logging_steps: 1
save_steps: 25
save_total_limit: 5
eval_strategy: steps
eval_steps: 200
val_size: 0.01
report_to: swanlab
swanlab_project: streamweave-sft
swanlab_run_name: qwen3vl-8b-full-streamweave-sft-0516-4500
```

启动命令：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory
bash configs/train_streamweave_0516_4500.sh
```

脚本实际环境：

```text
CONDA_ENV=/mmu_mllm_hdd/zhouhanshu/conda/envs/llama_0425
PATH=$CONDA_ENV/bin:$PATH
PYTHONPATH=/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory/src
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
FORCE_TORCHRUN=1
```

脚本实际调用：

```bash
/mmu_mllm_hdd/zhouhanshu/conda/envs/llama_0425/bin/llamafactory-cli \
  train configs/qwen3vl_8b_full_sft_streamweave_0516_4500.yaml
```

续跑命令：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory
RESUME=1 bash configs/train_streamweave_0516_4500.sh
```

续跑脚本行为：

- 自动找 `saves/qwen3-vl-8b/full/streamweave_sft_0516_4500/checkpoint-*` 中编号最大的 checkpoint。
- 复制临时 YAML。
- 把 `resume_from_checkpoint` 改成最新 checkpoint。
- 把 `overwrite_cache` 改成 `false`，避免续跑时重建 tokenizer cache。
- 如果不是 `RESUME=1` 且输出目录非空，脚本会拒绝启动，防止覆盖已有 checkpoint。

当前 checkpoint：

```text
checkpoint-25
checkpoint-50
checkpoint-75
checkpoint-100
```

`checkpoint-100/trainer_state.json`：

```text
global_step: 100
epoch: 0.3810431055013098
max_steps: 263
train_batch_size: 2
last logged loss at step 100: 0.2023988962173462
```

最后几步 loss：

```text
step 96: 0.2159143537
step 97: 0.2155754566
step 98: 0.2068932950
step 99: 0.1990048885
step 100: 0.2023988962
step 101: 0.1953287125  # trainer_log 中存在，但没有对应 checkpoint
```

当前判断：

- `checkpoint-100` 已保存，可以作为当前稳定恢复点。
- 还没到 `eval_steps=200`，所以没有 0516 的 eval loss。
- 如果继续训练，直接 `RESUME=1` 从 `checkpoint-100` 续跑。
- 如果要评测当前 100 step，使用已导出的 `models/qwen3vl_sft_0516_step100`。

## 0511 Note SFT

这是上一轮 dataset2 SFT 主线，训练已完成。

数据：

```text
dataset2/sft_0511_note.jsonl
rows: 182998
dataset name: streamweave_sft_0511_note
media_dir: /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset2
```

配置和脚本：

```text
SFT/LlamaFactory/configs/qwen3vl_8b_full_sft_streamweave_0511_note.yaml
SFT/LlamaFactory/configs/train_streamweave_0511.sh
SFT/LlamaFactory/configs/train_streamweave_0511_resume_debug.sh
```

启动：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory
bash configs/train_streamweave_0511.sh
```

续跑：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory
RESUME=1 bash configs/train_streamweave_0511.sh
```

成功完成日志：

```text
SFT/LlamaFactory/logs/resume_debug_20260514_144429/train.out
```

训练过程记录：

```text
最终从 checkpoint-500 恢复
训练到 708/708
epoch: 1.0
EXIT_STATUS: 0
```

最终指标：

```text
train_loss: 0.05421645726777066
eval_loss: 0.18262900412082672
train_runtime: 27331.9382s
eval_runtime: 243.3463s
train_samples_per_second: 6.628
train_steps_per_second: 0.026
```

当前磁盘状态：

```text
SFT/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_0511_note
```

当前该目录已经是最终 HF 模型目录，包含：

```text
model.safetensors
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
processor_config.json
trainer_state.json
train_results.json
eval_results.json
all_results.json
```

历史日志记录训练时曾保存：

```text
checkpoint-600
checkpoint-700
checkpoint-708
```

但当前磁盘上 `streamweave_sft_0511_note` 目录下没有这些 `checkpoint-*` 子目录，只有最终模型文件。

导出模型：

```text
models/qwen_sft_0513
```

用途：

- 作为 0513/0514 评测和后续 RL 起点候选。
- 需要和 base instruct、answered-full SFT、GRPO/RL checkpoint 同口径评测。

## Answered-Full SFT

这是 2026-05-08 左右的一轮历史 SFT，用 V5 协议蒸馏 Gemini teacher 的 stepwise 输出。

目标：

- 学 `<state>/<answer>/<anchor>/<delta>` 格式。
- 只保留有回答的 QA 条目。
- 过滤掉 delta 超过 20s 的 ShareGPT step。
- 判断这个 SFT 是否适合作为 RL 起点。

数据和注册：

```text
dataset name: streamweave_sft_answered_full
dataset_info file: SFT/LlamaFactory/data_streamweave/dataset_info.json
registered file: data_engine/sft/outputs/gemini_answered_full/llamafactory_sharegpt_anchor_delta_le20.jsonl
media_dir: /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/streamweave_data
```

旧笔记记录的数据统计：

```text
Gemini sample: 3956
accepted: 2491
failed: 1465
accepted rate: 62.97%
accepted steps: 25092
attempted steps: 36379
raw ShareGPT rows: 35583
delta <= 20s after filter: 32583
answered_steps: 11982
silent_steps: 20601
format_error_rows: 0
```

训练配置：

```text
SFT/LlamaFactory/configs/qwen3vl_8b_full_sft_streamweave_answered_full.yaml
SFT/LlamaFactory/configs/train_streamweave_answered_full.sh
```

训练参数：

```text
base model: /mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
cutoff_len: 12288
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-5
num_train_epochs: 3.0
save_steps: 200
eval_steps: 50
```

旧笔记记录的完成结果：

```text
steps: 759/759
epoch: 3
train_loss: 0.181242
eval_loss: 0.246001
train_runtime: 29410.8616s
elapsed: 8:10:10
train_samples_per_second: 3.29
train_steps_per_second: 0.026
```

历史 checkpoint 记录：

```text
checkpoint-200
checkpoint-400
checkpoint-600
checkpoint-759
```

当前可见导出模型：

```text
models/qwen3vl8b_streamweave_sft_answered_full_vllm
```

注意：当前 `SFT/LlamaFactory/saves` 下没有看到 `streamweave_sft_answered_full` 训练输出目录；这部分完整指标来自恢复出来的旧笔记。

## Answered-Full Init-Anchor SFT

这是 answered-full 的一个变体，注册名是：

```text
streamweave_sft_answered_full_anchor_delta_init_anchor
```

配置和脚本：

```text
SFT/LlamaFactory/configs/qwen3vl_8b_full_sft_streamweave_answered_full_anchor_delta_init_anchor.yaml
SFT/LlamaFactory/configs/train_streamweave_answered_full_anchor_delta_init_anchor.sh
```

与 answered-full 使用同一个注册数据文件：

```text
data_engine/sft/outputs/gemini_answered_full/llamafactory_sharegpt_anchor_delta_le20.jsonl
```

主要区别是 SwanLab run 和 output dir：

```text
swanlab_run_name: qwen3vl-8b-full-streamweave-answered-full-anchor-delta-init-anchor
output_dir: saves/qwen3-vl-8b/full/streamweave_sft_answered_full_anchor_delta_init_anchor
```

当前可见导出模型：

```text
models/qwen3vl8b_sft_anchor_delta_step200_vllm
```

当前没有在 `SFT/LlamaFactory/logs` 或 `SFT/LlamaFactory/saves` 下找到这轮完整训练结果，只能确认配置、脚本和 step200 vLLM 导出目录存在。

## 早期 V2 / LoRA 训练

早期文档里有一次 LoRA smoke 训练，用来证明 LLaMAFactory + Qwen3-VL + StreamWeave ShareGPT 数据链路能跑通。

历史 LoRA smoke：

```text
base model: /mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct
dataset: streamweave_sft
media_dir: /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data
output_dir: saves/qwen3-vl-8b/lora/streamweave_sft
```

旧笔记记录结果：

```text
Num examples: 122
Num epochs: 3
Total batch size: 8
Total optimization steps: 48
Train loss: 0.3225
Final adapter: saves/qwen3-vl-8b/lora/streamweave_sft/adapter_model.safetensors
```

结论：这轮只证明启动链路可用，不作为当前 0516 full SFT 或 RL 起点。

V2 3077 配置：

```text
SFT/LlamaFactory/configs/qwen3vl_8b_lora_sft_streamweave_v2_3077.yaml
SFT/LlamaFactory/configs/train_streamweave_v2_3077.sh
dataset: streamweave_sft_v2_3077
registered file: exp2/streamweave_v4/data_engine/sft/outputs/gemini_v2_full_3077_w128/llamafactory_sharegpt.jsonl
output_dir: saves/qwen3-vl-8b/full/streamweave_sft_v2_3077
```

这个 YAML 文件名带 `lora`，但当前内容是：

```text
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: true
freeze_language_model: false
```

当前没有在 `SFT/LlamaFactory/saves` 下看到 `streamweave_sft_v2_3077` 输出目录。

## 检查命令

检查 SFT JSONL 行数：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
wc -l dataset2/sft_0516_4500.jsonl dataset2/sft_0511_note.jsonl
```

检查 ShareGPT 结构：

```bash
DATA=/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset2/sft_0516_4500.jsonl
MEDIA=/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset2

jq -r 'select((.messages|length)!=2 or .messages[0].role!="user" or .messages[1].role!="assistant") | input_line_number' "$DATA" | head
jq -r 'select(([.messages[]?.content? | tostring | scan("<image>")] | length) != (.images|length)) | input_line_number' "$DATA" | head
jq -r '.images[]' "$DATA" | sort -u | while IFS= read -r p; do test -f "$MEDIA/$p" || printf "%s\n" "$p"; done | head
```

查看 0516 checkpoint：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
find SFT/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_0516_4500 -maxdepth 1 -type d | sort
```

查看 0516 trainer state：

```bash
jq '{global_step, epoch, max_steps, train_batch_size, log_history: (.log_history[-8:])}' \
  SFT/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_0516_4500/checkpoint-100/trainer_state.json
```

查看 0516 训练日志：

```bash
tail -200 \
  SFT/LlamaFactory/logs/sft0516_4500_20260516_195458/train.out
```

查看训练是否仍在跑：

```bash
ps -eo pid,args | rg '/mmu_mllm_hdd/zhouhanshu/conda/envs/llama_0425/bin/(llamafactory-cli|python)|torchrun' | rg -v 'codex|rg|bash -lc'
```

检查导出模型文件：

```bash
find models/qwen3vl_sft_0516_step100 -maxdepth 1 -type f | sort
```

## vLLM 兼容注意

经验：

- LLaMAFactory / Transformers 导出的 tokenizer 字段可能导致 vLLM crash。
- 导出目录里需要有 `preprocessor_config.json` 和 `video_preprocessor_config.json`。
- 如果 `tokenizer_config.json` 里有不兼容的 `extra_special_tokens`，需要清理后再跑 vLLM。
- 不能只看文件存在，导出后必须实际用 OVO/StreamingBench 加载一次。

当前可见的 SFT 导出模型：

```text
models/qwen_sft_0513
models/qwen3vl8b_streamweave_sft_answered_full_vllm
models/qwen3vl8b_sft_anchor_delta_step200_vllm
models/qwen3vl_sft_0516_step50
models/qwen3vl_sft_0516_step100
```

## 历史评测教训

第一次 V4 SFT 在 OVO 1/8 上退化：

| 模型 | Backward | Realtime | Forward | Total |
| --- | ---: | ---: | ---: | ---: |
| V4 base 1/8 | 59.91 | 78.13 | 38.72 | 58.92 |
| V4 SFT 1/8 | 42.70 | 69.71 | 32.33 | 48.25 |

结论：

- SFT 不能只看 loss。
- 必须检查 answer 分布、Unable 能力、时机判断和 OVO full。
- 从 SFT checkpoint 继续 RL 时，要明确标记 `sft-init`。
- SFT 模型不默认优于 base，需要和 base instruct、answered-full SFT、历史 GRPO/RL 模型同口径对比。

## 记录规范

每次新增 SFT 训练必须记录：

- 数据文件和行数。
- dataset_info 注册名。
- YAML 配置路径。
- 启动脚本和完整启动命令。
- base model。
- media_dir。
- output_dir。
- 是否 resume，以及从哪个 checkpoint resume。
- checkpoint 列表。
- SwanLab run name。
- train loss / eval loss / step / epoch / runtime。
- 导出模型路径。
- OVO / StreamingBench 回评结果。
