# 第四部分：SFT 训练

## 当前状态

SFT 数据合成链路已经打通，但第一次 SFT 回评是负面结果。当前不再把“继续扩大同口径 SFT 数据”作为主阻塞项，主线已经切到 V5 GRPO。

2026-05-08 更新：新一轮 `answered-full` SFT 训练已完成，使用当前 V5 协议：

- 输出协议：`<state>` + `<answer>` + timestamp-only `<note t="..."></note>` + `<bridge>`，不再使用 `<eta>`。
- SFT 数据只保留有回答的 QA 条目，并过滤掉 bridge 超过 20s 的 ShareGPT step。
- 已建立 vLLM 兼容模型目录，并启动 OVO 1/8 回评。
- 当前评测仍在运行中：输出目录尚无合并后的 `results.jsonl` 和 `results_summary.*`，不能记录正式分数。

## 数据与链路

- SFT 标注入口：`exp2/streamweave_v4/dataset/streamweave_data/annotations_qa_filter_final.jsonl`
- raw data root：`exp2/streamweave_v4/dataset/streamweave_data`
- SFT 输出目录：`exp2/streamweave_v4/data_engine/sft/outputs/<run>/`
- 训练入口：`llamafactory_sharegpt.jsonl`
- 小规模验证：`data_engine/sft/outputs/gemini_final_8`
  - `accepted=7/8`
  - 导出 `136` 条 step-level SFT 样本

每个 SFT 输出目录一般包含：

```text
samples/*.json
sample_manifest.jsonl
sft_steps.jsonl
llamafactory_sharegpt.jsonl
dataset_info_streamweave_sft.json
summary.json
sft_jobs.sqlite
```

## 2026-05-08 answered-full SFT 实验记录

实验目的：

- 用当前 V5 协议蒸馏 Gemini teacher 的 stepwise StreamWeave 行为。
- 训练 Qwen3-VL-8B student 学会 `<state>/<answer>/<note>/<bridge>` 格式、回答时机、note anchor 和 bridge 压缩。
- 评估该 SFT 是否优于 base instruct，并判断是否适合作为后续 RL 起点。

数据合成与过滤：

- answered 标注入口：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/streamweave_data/annotations_qa_filter_answered.jsonl`
- Gemini 合成输出目录：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/data_engine/sft/outputs/gemini_answered_full`
- 训练用 ShareGPT 文件：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/data_engine/sft/outputs/gemini_answered_full/llamafactory_sharegpt_bridge_le20.jsonl`
- 合成统计：`3956` 条 sample，`2491` 条 accepted，`1465` 条 failed，`0` 条 error，accepted rate 约 `62.97%`。
- step 统计：`25092` 个 accepted step，`36379` 个 attempted step。
- variant rescue：`158` 行 step 来自 rescue，产出 `329` 个 correct variants。
- 原始 ShareGPT：`35583` 行。
- bridge <= 20s 过滤后：保留 `32583` 行，删除 `3000` 行，其中 answered step 删除 `1158` 行、silent step 删除 `1842` 行；过滤前最大 bridge duration 为 `103s`。
- 过滤后检查：`answered_steps=11982`，`silent_steps=20601`，`missing_answer_tag=0`，`format_error_rows=0`，`sharegpt_structure_error_rows=0`，`bridge_over_threshold_rows=0`。

LLaMAFactory 训练：

- 训练配置：`/mmu_mllm_hdd/zhouhanshu/test/exp3/LlamaFactory/configs/qwen3vl_8b_full_sft_streamweave_answered_full.yaml`
- dataset info：`/mmu_mllm_hdd/zhouhanshu/test/exp3/LlamaFactory/data_streamweave/dataset_info.json`
- 训练输出 checkpoint：`/mmu_mllm_hdd/zhouhanshu/test/exp3/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_answered_full`
- checkpoint：`checkpoint-200`、`checkpoint-400`、`checkpoint-600`、`checkpoint-759`，最终目录下也有合并后的 `model.safetensors`。
- 训练步数：`759/759`，`epoch=3`。
- 训练耗时：`29410.8616s`，约 `8:10:10`。
- 训练结果：`train_loss=0.181242`，`eval_loss=0.246001`。
- 吞吐：`train_samples_per_second=3.29`，`train_steps_per_second=0.026`，`eval_samples_per_second=11.159`。

vLLM 评测目录：

- vLLM 兼容模型目录：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm`
- 该目录的 `model.safetensors` 是新 SFT checkpoint 的硬链接；tokenizer、chat template 和 preprocessor 文件使用之前 vLLM 兼容目录的格式，避免 LLaMAFactory 导出的 `extra_special_tokens` list 触发 vLLM tokenizer crash。
- OVO 1/8 评测输出：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/outputs/ovo_qwen3vl8b_finetuned_1of8`
- 评测配置：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/configs/batch_ovo_qwen3vl8b_finetuned_8gpu.yaml`
- 数据口径：`ovo_bench_1of8_stratified.json`，共 `205` 条。
- 运行方式：`scripts/run_ovo_8gpu_vllm_finetuned.sh` 启动 8 个 vLLM server，`evaluation/eval_batch.py --workers 16` 写入 part 文件。
- 当前磁盘状态：`.results_parts/part_*.jsonl` 已写出至少 `84/205` 行，评测仍在增长；尚无顶层 `results.jsonl`、`results_summary.json` 或 `results_summary.txt`。
- 当前判断：这轮评测只能记录为“运行中”；最终分数必须等合并结果和 summary 落盘后再写入 `0508实验跑分.md`。

## 硬约束

- 只有整条样本 accepted 才能进入 `sft_steps.jsonl` 和 `llamafactory_sharegpt.jsonl`。
- `llamafactory_sharegpt.jsonl` 必须使用 production prompt，不能混入 teacher-only 指令或 retry feedback。
- `<note>` 必须是成对标签 `<note ...></note>`，不能是自闭合 `<note .../>`。
- `<bridge>` 必须覆盖合法 gap，包括窗口边界、相邻 note 之间和 open-tail 继承。
- QA step-level retry 只检查 `<answer>` 是否该空或非空；不再有 eta target。答案内容是否正确放在样本级 accepted 判定。

## 当前源码约束

按 2026-05-08 的 `data_engine/sft` 源码，SFT 生成路径的实际约束是：

- `env.evaluate_attempt(... repair=False)`：teacher raw XML 必须自身严格合法，不能靠 repair 产物进入 target。
- `apply_note_count_constraint()`：每 step 最多 `max_notes_per_step` 个 note，默认 1。
- `note_reminder_context()`：长时间没有 note 时给 teacher 软提醒，但不是硬约束。
- `apply_qa_answer_constraints()`：根据 QA History 和 task 判断 answer 应为空还是非空。
- `check_sample_answer()`：整条样本所有 emitted answer 都必须匹配 GT。
- answer step 可以额外采样 variants；accepted 且答案正确的 variant 可以在 finalize/export 阶段救回样本。

当前源码没有实现 annotated key-frame hard constraint：

- 没有 `_key_frame_context()`。
- 没有 `_apply_key_frame_quality_constraints()`。
- 没有“必须输出标注 key frame 且不能输出额外 note”的检查。

因此，旧笔记里“关键帧标注提示不能进入训练数据”的说法只适用于历史 V4/V3 产物或旧 prompt 设计；当前 V5 SFT 源码不再注入这类 teacher-only key-frame constraint。

## 第一次 SFT 回评结论

R14：`StreamWeave V4 + Qwen3-VL-8B SFT / OVO 1/8`

| Category | SFT R14 | Base R13 | Delta |
| --- | ---: | ---: | ---: |
| Backward | 42.70 | 59.91 | -17.21 |
| Realtime | 69.71 | 78.13 | -8.42 |
| Forward | 32.33 | 38.72 | -6.39 |
| Total | 48.25 | 58.92 | -10.67 |

判断：

- 第一次 SFT 不是正向蒸馏，而是明显退化。
- 退化任务包括 HLD、FPD、REC、ASI、ACR、ATR 等。
- 可能原因包括 answer 分布失衡、`Unable to answer` 能力被训没、训练配置过强或 prompt 对齐问题。
- 当前 SFT checkpoint 可作为 RL 起点实验，但不能默认视为优于 base instruct。

## 后续使用原则

- 若继续从 SFT checkpoint 做 RL，run name 和笔记中必须显式标注 `sft-init`。
- 若要验证 SFT 是否可修复，应先做 answer 分布统计、loss mask 检查和小规模回评，不要直接扩大数据。
- RL 结果必须同时对比 base instruct 和 SFT checkpoint，避免把 SFT 退化掩盖掉。
