# 第六部分：评测

## 当前状态

详细跑分只维护在 [`实验跑分.md`](./实验跑分.md)。本文件只保留回评原则和当前必须比较的对象。

## 当前必须比较的对象

下一轮新 SFT / V5 RL checkpoint 出来后，至少需要同口径比较：

| 模型/策略 | 用途 |
| --- | --- |
| `/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct` | base instruct 起点 |
| `streamweave_sft_v2_3077` | 当前 SFT-init 起点 |
| `models/qwen_sft_0513` | 2026-05-14 新 SFT，来自 `streamweave_sft_0511_note/checkpoint-700`，当前准备评测 |
| V5 GRPO checkpoint | RL 后模型 |

优先跑 OVO 小规模回评；确认正向后再跑 full。

## 2026-05-14 待评测 SFT

- 模型路径：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen_sft_0513`
- 来源 checkpoint：`SFT/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_0511_note/checkpoint-700`
- 训练状态：`708/708` steps 完成，最终 eval loss `0.1826`，训练正常退出。
- 导出状态：已导出为 HF 推理目录，排除训练状态文件；`AutoProcessor` 可正常识别为 `Qwen3VLProcessor`。
- 当前计划：先跑 OVO 1/8 或小规模回评，确认稳定和方向后再跑 full；结果必须写入 `note/实验跑分.md`。

## 评测原则

- `full` 和 `subset` 不能直接横比。
- `task macro` 和 `sample weighted` 必须分开写。
- `smoke/debug` 只说明链路是否跑通，不纳入主结论。
- `V3`、`V4`、`V5` 的 adapter、prompt 和 stepwise 口径不同，不能混成一个结论。
- 所有新结果必须记录模型路径、checkpoint、config、run name、样本数、错误数和输出目录。
- 如果是 RL checkpoint，必须同时记录训练数据、初始化模型、`save_freq`、总 step 和是否从 checkpoint 恢复。

## 当前基线

- V3 教师上限：`StreamWeave V3 + Gemini / OVO full = 65.81`
- V4 8B full 起点：`StreamWeave V4 + Qwen3-VL-8B base / OVO full = 51.43`
- V4 8B 1/8 起点：`StreamWeave V4 + Qwen3-VL-8B base / OVO 1/8 = 58.92`
- V4 SFT 1/8 负面结果：`48.25`

## 结果记录格式

每次新跑分至少记录：

- 代码目录、commit 或修改说明。
- 模型路径、checkpoint、服务地址或后端类型。
- benchmark、数据范围、样本数和错误数。
- `policy`、`prompt_profile`、`sample_fps`、`frames_per_step`、`memory_window`、`postprocess.mode`。
- 总分、分 category、分 task。
- 是否为 `task macro`，是否另有 `sample weighted`。
- run 输出目录和 summary 文件路径。
- 当前源码使用 `<state>` 协议和 timestamp-only anchor；若评测旧 checkpoint 或旧 trace，必须单独记录是否出现 `<eta>` 或 `frame="N"` 格式退化。
