# 第四部分：SFT 训练

- 状态：SFT 数据合成链路已打通，第二轮数据合成进行中；第一次 SFT 训练准备启动
- 目标：
  - 让模型学会逐步决策、桥接表达、必要时沉默，以及关键视觉状态选择
  - 基于大模型 agent 轨迹训练更小的 student 模型
- 数据版本：
  - 当前 SFT 标注入口：`exp2/streamweave_v4/dataset/streamweave_data/annotations_qa_filter_final.jsonl`
  - 当前 raw data root：`exp2/streamweave_v4/dataset/streamweave_data`
  - 当前 SFT 输出目录统一放在：`exp2/streamweave_v4/data_engine/sft/outputs/<run>/`
  - 训练入口优先使用 `llamafactory_sharegpt.jsonl`
- 训练配置：
  - 先逐步监督，再完整 rollout 监督
  - 当前第一版先做 step-level SFT，确认模型能稳定输出 XML 协议
  - 后续再考虑 multi-turn / closed-loop 数据
- 关键指标：
  - 协议解析成功率
  - `note/bridge` 选择质量
  - `eta/answer` 时机质量
  - OVO-Bench 与 StreamingBench 上的基础表现
- 结果：
  - `gemini_final_8` 已完成小规模数据合成与巡检：`accepted=7/8`，导出 `136` 条 step-level SFT 样本
  - 第二轮 `1000` 条合成进行中，待巡检后用于第一次 SFT
- 风险/问题：
  - 协议学会了但内容质量差
  - answer step 少，可能导致模型过度输出空 answer
  - 多目标一起训可能造成互相干扰
  - bridge 文本过长或 open-tail 继承过多，可能污染训练风格
  - teacher 合成中的标注约束和 retry feedback 不能进入最终训练 prompt
- 下一步：
  - 完成 `1000` 条第二轮合成。
  - 巡检 accepted 样本后，用 `llamafactory_sharegpt.jsonl` 做第一次 SFT smoke。
  - 训练完成后先跑小规模 OVO/StreamingBench 回评，再决定是否扩大数据规模。

## 当前 SFT 数据文件

每个输出目录一般包含：

```text
samples/*.json
sample_manifest.jsonl
sft_steps.jsonl
llamafactory_sharegpt.jsonl
dataset_info_streamweave_sft.json
summary.json
sft_jobs.sqlite
```

字段含义：

- `samples/*.json`：每条样本的完整合成记录，包含 accepted/failed/error 状态、每步 attempts、raw output、retry 质量问题和最终 target。
- `sample_manifest.jsonl`：样本级索引，记录每条样本是否 accepted、失败原因和 step 数。
- `sft_steps.jsonl`：accepted-only 的 step-level 中间格式，用于审计和二次导出。
- `llamafactory_sharegpt.jsonl`：accepted-only 的 LLaMAFactory ShareGPT 数据。
- `dataset_info_streamweave_sft.json`：给 LLaMAFactory `dataset_info.json` 合并用的片段。
- `sft_jobs.sqlite`：动态多进程任务队列和断点续跑状态。

## 当前关键校验

SFT 合成时已经做硬校验：

- XML 必须能解析，并包含 `<eta>`、`<answer>`、`<bridge>`、`<note>` 等协议字段。
- `<note>` 必须是成对标签 `<note ...></note>`，不能是自闭合 `<note .../>`。
- `<note frame="N">` 必须引用当前 step 的局部 frame id，并命中标注约束。
- `<bridge>` 必须覆盖合法 gap，包括窗口边界、note 之间，以及 open-tail 继承。
- QA 校验只检查 eta 和 answer 的空/非空状态；答案是否正确放在样本级 accepted 判定里。
- eta 对 `backward/realtime/forward` 都要求落在目标可答时间窗口内，而不是必须等于某一个端点。

## 当前推荐训练数据

小规模 smoke 数据：

```text
exp2/streamweave_v4/data_engine/sft/outputs/gemini_final_8/llamafactory_sharegpt.jsonl
```

第二轮数据完成后优先使用：

```text
exp2/streamweave_v4/data_engine/sft/outputs/gemini_final_1000_w64/llamafactory_sharegpt.jsonl
```
