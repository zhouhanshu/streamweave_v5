# StreamWeave SFT

本目录收纳 StreamWeave v5 使用的 SFT 微调框架。

- `LlamaFactory/`: 当前使用的 LlamaFactory 代码库与训练产物。
- `LlamaFactory/configs/train_streamweave_answered_full.sh`: answered-full SFT 启动脚本。
- `LlamaFactory/configs/qwen3vl_8b_full_sft_streamweave_answered_full.yaml`: answered-full SFT 训练配置。
- `LlamaFactory/data_streamweave/dataset_info.json`: LlamaFactory 数据集注册表。

当前 answered-full 数据集指向：

```text
/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/data_engine/sft/outputs/gemini_answered_full/llamafactory_sharegpt_anchor_delta_le20.jsonl
```

注意：旧的 `cache/streamweave_answered_full` 是历史 bridge 数据的 tokenizer cache。当前 anchor/delta 配置使用新的 `cache/streamweave_answered_full_anchor_delta`，避免复用旧缓存。
