# 第六部分：评测

## 当前状态

- 详细跑分已经单独移到 [`实验跑分.md`](./实验跑分.md)。
- 本文件只保留评测入口、对比原则和后续回评要求。
- 当前重点：等待 V4 第一次 SFT 后的新模型回评，并和同口径 baseline 比较。

## 当前评测入口

- 跑分主表：[`实验跑分.md`](./实验跑分.md)
- 当前命令入口：[`08-commands-and-tools.md`](./08-commands-and-tools.md)
- 当前 V4 SFT 说明：`../data_engine/sft/README.md`
- 当前 V4 SFT 数据合成输出：`../data_engine/sft/outputs/`

## 评测原则

- 只在同一模型、同一数据范围、同一 adapter 口径下比较 `plain / SimpleStream / StreamWeave`。
- `full` 和 `subset` 不能直接横比。
- `task macro` 和 `sample weighted` 必须分开写。
- `smoke / debug` 只说明链路是否跑通，不纳入主结论。
- `StreamWeave V3` 在 `2026-04-28` 前的 OVO 结果属于旧 adapter 口径，后续正式对比必须重跑。

## 当前需要回评的点

- V4 第一次 SFT smoke 后的 `OVO-Bench` 小规模结果。
- V4 第一次 SFT 稳定后的 `OVO-Bench` 和 `StreamingBench` 主结果。
- 新 adapter 下的 `Gemini / StreamWeave V3 / OVO 1/8`。
- 新 adapter 下的 `Qwen3-VL-8B / StreamWeave V3 / OVO full`。
- `SimpleStream last-4 / last-8 / last-16` 的同模型对照。
- 若加入 forward early-answer 约束，需要额外记录 `first_answer_time` 和早答惩罚口径。

## 结果记录格式

每次新跑分至少记录：

- 代码目录、config、commit 或修改说明。
- 模型、服务地址或后端类型。
- benchmark、数据范围、样本数和错误数。
- `policy`、`prompt_type`、`fps`、`chunk_duration`、`chunks_per_step`、`memory_window`。
- 总分、分 category、分 task。
- 是否为 `task macro`，是否另有 `sample weighted`。
- 是否使用新 OVO adapter，以及 forward query 注入时间。
