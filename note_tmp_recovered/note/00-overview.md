# StreamWeave 总览

## 项目目标

`StreamWeave` 面向在线流式视频理解：用户可以在视频任意时间提问，模型需要判断应该依赖过去、当前还是未来证据，并在有限上下文下保留关键视觉状态。

核心目标：

- 保住 OCR、颜色、空间关系、对象身份和动作进展等视觉细节。
- 不把所有历史帧都塞进上下文，而是显式选择关键视觉锚点。
- 在 OVO-Bench 和 StreamingBench 上验证时机判断、记忆保留和回答正确性。

## 当前协议

- `anchor`：长期保留的视觉锚点，只保存当前 step 的关键帧和时间信息。
- `delta`：用文本压缩两个视觉锚点之间的过渡。
- `state`：当前 step 的视频状态总结和回答判断，只用于本轮推理，不写回 Memory。
- `answer`：当前能答则输出答案，证据不足则保持空/沉默。
- 每一步输入：`memory + qa_history + current frames`。
- 每一步输出：

```xml
<state>...</state>
<answer>...</answer>
<delta t="...">...</delta>
<anchor t="..."></anchor>
```

硬约束：

- 空 Memory 第一轮必须输出首帧 `<anchor>`。
- `<anchor>` 必须是成对标签 `<anchor ...></anchor>`，不能使用自闭合 `<anchor .../>`。
- 最终评测取最后一个非空 `<answer>`。
- 内部 schema/指标仍可保留 `note/bridge` 命名；模型可见协议只使用 `anchor/delta`。

## 当前主线

- 当前仓库：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5`
- SFT 框架：`SFT/LlamaFactory`
- RL 框架：`RL/`，包含 vendored `verl` 和 `streamweave_rl` 自定义链路。
- 当前 GRPO 入口：`RL/scripts/train_grpo_ovo_8gpu.sh`
- PPO 对照入口：`RL/scripts/train_ppo.sh`
- 当前评测与清洗入口在 `evaluation/`。
- 最新源码事实以 `10-source-code-current-state.md` 为准；实时实验状态以 `experiment-log.md` 为准。

当前阶段重点已经从早期 idea validation / 旧 VideoXum 数据构造，转到：

- 用 answered-full / anchor-delta SFT 模型做 OVO 和新数据回评。
- 对新扩充数据做 3 次推理，估计样本难度和稳定性。
- 按难度和时长分配 SFT/RL 数据，具体见 `数据清洗0510.md`。
- 继续维护 RL 的 GRPO/PPO 链路和回评。

## 历史验证结论

早期 idea validation 已完成，不再单独维护旧文档。保留结论如下：

| 口径 | Backward | Realtime | Forward | Total |
| --- | ---: | ---: | ---: | ---: |
| SimpleStream / Qwen3-VL-8B / recent4 / OVO full | 50.48 | 81.48 | 43.79 | 58.59 |
| Qwen3-VL-32B-Instruct plain / OVO full | 60.73 | 78.15 | 44.70 | 61.19 |
| StreamWeave V3 + Gemini / OVO full retry2 | 66.73 | 80.24 | 50.46 | 65.81 |
| StreamWeave V4 + Qwen3-VL-8B base / OVO full | 48.09 | 75.41 | 30.78 | 51.43 |
| StreamWeave V4 + Qwen3-VL-8B base / OVO 1/8 | 59.91 | 78.13 | 38.72 | 58.92 |
| StreamWeave V4 + Qwen3-VL-8B SFT / OVO 1/8 | 42.70 | 69.71 | 32.33 | 48.25 |

结论：

- V3 + Gemini 证明方法上限可接近或略超 AURA overall，但 Forward 仍弱。
- V4 8B base 是学生模型的重要基线，full 总分 `51.43`。
- 第一次 V4 SFT 在 1/8 上退化，不能当作已验证的正向 SFT。
- 早期结果说明，只靠 prompt/记忆结构有收益信号，但也存在历史记忆污染、过早作答、forward 处理弱和长上下文问题。

已吸收到后续版本的经验：

- API runner 必须区分请求失败和答错。
- 本地 vLLM / Qwen3 reasoning 输出字段需要兜底处理。
- StreamWeave 必须显式区分视觉锚点和文本过渡，不能只做最近窗口文本总结。
- forward 类任务必须正确展开子视频/sample，否则结果不可比。

详细跑分继续维护在 `实验跑分.md` 和 `0508实验跑分.md`，本文件不再展开旧命令。

## 数据口径

旧 OVO RL 数据：

```text
/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/ovo
```

历史文件：

```text
ovo_bench_new.json
ovo_rl.json
ovo_rl_lt120s.json
```

说明：

- `ovo_rl.json` 是单 query 口径，`backward/forward/realtime` 各约 `200` 条。
- `ovo_rl_lt120s.json` 是 `<120s` 子集，历史 RL 训练曾使用。
- 这些数据保留为历史和对照；后续主线将切到新的扩充数据。

旧 VideoXum/ActivityNet 数据：

```text
exp2/data/streamweave_data/annotations_filtered_30s300s_key10to40.jsonl
```

历史过滤条件：

- `30 <= duration <= 300`
- `0.10 <= key_frame_ratio <= 0.40`

历史结果：

- 原始 `14001`
- 保留 `12029`

说明：

- 该文件只是视频/关键帧池，不是当前 SFT/RL 训练入口。
- 旧 VideoXum-StreamQA 清洗计划不再继续展开。
- 如果以后重启旧数据，必须重新检查帧目录、frame_count、key_frame_ids、key_frame_scores 和 QA 是否泄露未来信息。

## 新数据清洗原则

后续不再把旧数据集作为主训练数据。新数据处理原则见 `数据清洗0510.md`，核心是：

- 用 SFT 后模型对每条样本推理 `3` 次，得到 `pass_rate`。
- `easy = 3/3`，`medium = 2/3`，`hard = 1/3`，`unsolved = 0/3`。
- easy 短样本优先进 SFT，easy 长样本进 RL。
- medium/hard 按长度加权，一部分进 SFT，一部分进 RL。
- unsolved 交给更强教师模型推理 `3` 次；教师全错丢弃，教师至少一次正确则保留正确轨迹进 SFT，同时样本进 RL。

## 更新规则

- 本文件只保留项目目标、协议、历史结论和数据口径摘要。
- 当前实验状态写 `experiment-log.md`。
- 当前源码事实写 `10-source-code-current-state.md`。
- SFT 训练细节写 `04-sft-training.md`。
- RL 训练细节写 `05-rl-training.md`。
- 新数据清洗和分配策略写 `数据清洗0510.md`。
