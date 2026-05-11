# StreamWeave Qwen3-VL-8B 训练启动说明

本文记录如何用 LLaMA-Factory 和 `llama_0425` 环境训练 Qwen3-VL-8B 的 StreamWeave SFT 数据。当前已跑通的配置是：

- 模型：`/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct`
- 数据注册：`data_streamweave/dataset_info.json`
- 训练配置：`examples/train_lora/qwen3vl_8b_lora_sft_streamweave.yaml`
- 输出目录：`saves/qwen3-vl-8b/lora/streamweave_sft`

## 新数据集准备

新数据集需要是 LLaMA-Factory 的 ShareGPT 多模态格式，推荐每行 JSONL 形如：

```json
{
  "messages": [
    {"role": "user", "content": "...<image>..."},
    {"role": "assistant", "content": "..."}
  ],
  "images": ["video/xxx/000000.jpg"]
}
```

注意：

- `messages` 需要按 `user -> assistant` 成对出现。
- 文本中的 `<image>` 数量必须等于 `images` 列表长度。
- `images` 可以写相对路径；训练配置里的 `media_dir` 要指向这些相对路径的根目录。
- 如果新数据仍是 `video/...jpg`，那么 `media_dir` 应该是包含 `video/` 目录的上级目录。

可用下面命令做基础检查，把路径替换成新数据集：

```bash
DATA=/path/to/new/llamafactory_sharegpt.jsonl
MEDIA=/path/to/new/media_root

wc -l "$DATA"
jq -r 'select((.messages|length)!=2 or .messages[0].role!="user" or .messages[1].role!="assistant") | input_line_number' "$DATA" | head
jq -r 'select(([.messages[]?.content? | tostring | scan("<image>")] | length) != (.images|length)) | input_line_number' "$DATA" | head
jq -r '.images[]' "$DATA" | sort -u | while IFS= read -r p; do test -f "$MEDIA/$p" || printf "%s\n" "$p"; done | head
```

如果最后三条检查没有输出，通常说明格式、占位符数量和图片路径都没明显问题。

## 注册新数据集

编辑 `data_streamweave/dataset_info.json`，新增一个数据集名称。示例：

```json
{
  "streamweave_sft": {
    "file_name": "/old/path/llamafactory_sharegpt.jsonl",
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
  },
  "streamweave_sft_new": {
    "file_name": "/path/to/new/llamafactory_sharegpt.jsonl",
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
}
```

`file_name` 可以使用绝对路径。这样不需要把数据复制到 LLaMA-Factory 的 `data/` 目录。

## 修改训练配置

可以直接复用：

```text
examples/train_lora/qwen3vl_8b_lora_sft_streamweave.yaml
```

新数据集时重点改这几项：

```yaml
dataset: streamweave_sft_new
media_dir: /path/to/new/media_root
output_dir: saves/qwen3-vl-8b/lora/streamweave_sft_new
```

建议每次新实验都换一个新的 `output_dir`，因为当前配置里有：

```yaml
overwrite_output_dir: true
```

如果不换目录，会覆盖同名输出。

常用可调参数：

```yaml
cutoff_len: 8192
image_max_pixels: 131072
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 5.0e-5
num_train_epochs: 3.0
```

如果显存不够，优先降 `image_max_pixels`，例如改成 `65536`。如果文本被截断明显，再提高 `cutoff_len`，但显存会增加。

## 8 卡启动命令

必须让 `llama_0425` 环境的 `torchrun` 排在 PATH 最前面，并显式设置 `PYTHONPATH`。否则可能会调用系统 Python，报 `ModuleNotFoundError: No module named 'llamafactory'`。

从 LLaMA-Factory 根目录启动：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory

PATH=/mmu_mllm_hdd/zhouhanshu/conda/envs/llama_0425/bin:$PATH \
PYTHONPATH=/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory/src \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
FORCE_TORCHRUN=1 \
/mmu_mllm_hdd/zhouhanshu/conda/envs/llama_0425/bin/llamafactory-cli train \
examples/train_lora/qwen3vl_8b_lora_sft_streamweave.yaml
```

也可以临时覆盖配置项，不改 YAML：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory

PATH=/mmu_mllm_hdd/zhouhanshu/conda/envs/llama_0425/bin:$PATH \
PYTHONPATH=/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory/src \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
FORCE_TORCHRUN=1 \
/mmu_mllm_hdd/zhouhanshu/conda/envs/llama_0425/bin/llamafactory-cli train \
examples/train_lora/qwen3vl_8b_lora_sft_streamweave.yaml \
dataset=streamweave_sft_new \
media_dir=/path/to/new/media_root \
output_dir=saves/qwen3-vl-8b/lora/streamweave_sft_new
```

## 训练完成后检查

查看输出文件：

```bash
find saves/qwen3-vl-8b/lora/streamweave_sft_new -maxdepth 2 -type f | sort | head -80
```

正常情况下会看到：

- `adapter_model.safetensors`
- `adapter_config.json`
- `trainer_state.json`
- `train_results.json`
- `training_loss.png`
- 若干 `checkpoint-*` 目录

查看 GPU 是否释放：

```bash
nvidia-smi
```

## 当前已完成训练的记录

本次训练使用：

```text
dataset: streamweave_sft
media_dir: /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data
output_dir: saves/qwen3-vl-8b/lora/streamweave_sft
```

结果：

```text
Num examples: 122
Num epochs: 3
Total batch size: 8
Total optimization steps: 48
Train loss: 0.3225
Final adapter: saves/qwen3-vl-8b/lora/streamweave_sft/adapter_model.safetensors
```
