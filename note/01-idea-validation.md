# 第一部分：Idea 验证

## 归档状态

本文件只保留早期 idea validation 的结论，不再作为当前进度入口。当前主线已经切到 `exp3/streamweave_v5/RL`，实时状态见 `experiment-log.md`。

## 早期口径

- 主要 baseline：`SimpleStream recent4`
- 主要模型：`Qwen3-VL-8B`、`Qwen3-VL-32B-Instruct`
- 关键设置：`recent4 / chunk_duration=1.0 / fps=1.0`
- 早期 StreamWeave 代码：
  - `exp1/stream-weave_v2`
  - `exp1/streamweave` 只作历史参考

## 阶段性结论

- `Qwen3-VL-32B-Instruct plain` 已完成本地 baseline：
  - OVO full：`Total 61.19`
  - 相比 `SimpleStream / Qwen3-VL-8B / recent4` 的 `58.59` 更强。
- `StreamWeave-v2` 在 OVO `1/8` 同 ID 正常样本上与 `SimpleStream recent4` 持平：
  - `113/170 = 66.47%`
- 早期结果说明：
  - 只靠 prompt/记忆结构已有部分收益信号。
  - 退化也很明显，尤其历史记忆污染、过早作答、forward 路径和长上下文问题。
  - 需要切到更严格的 V3/V4 协议和后续训练链路。

## 已吸收进后续版本的经验

- API runner 必须区分请求失败和答错，不能把系统性失败算成合法 `0 分`。
- 本地 vLLM 的 Qwen3 reasoning 输出字段需要特别处理，不能只读 `message.content`。
- StreamWeave 不能只做“最近窗口 + 文本总结”，必须显式区分视觉锚点 `note` 和文本过渡 `bridge`。
- forward 类任务必须正确处理子视频/sample 展开，否则结果不可比。

详细跑分已收敛到 `实验跑分.md`，旧命令不再保留。
