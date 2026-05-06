# 第四部分：SFT 训练

## 当前状态

SFT 数据合成链路已经打通，但第一次 SFT 回评是负面结果。当前不再把“继续扩大同口径 SFT 数据”作为主阻塞项，主线已经切到 V5 GRPO。

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

## 硬约束

- 只有整条样本 accepted 才能进入 `sft_steps.jsonl` 和 `llamafactory_sharegpt.jsonl`。
- `llamafactory_sharegpt.jsonl` 必须使用 production prompt，不能混入 teacher-only 指令、关键帧标注提示或 retry feedback。
- `<note>` 必须是成对标签 `<note ...></note>`，不能是自闭合 `<note .../>`。
- `<bridge>` 必须覆盖合法 gap，包括窗口边界、相邻 note 之间和 open-tail 继承。
- QA step-level retry 只检查 eta 和 answer 空/非空状态；答案是否正确放在样本级 accepted 判定。

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
