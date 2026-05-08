# 第五部分：RL 训练

## 当前状态

- 状态：GRPO stepwise 链路已经跑通，但最近一次 run 非正常中断。
- 代码目录：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/RL`
- 当前脚本：`RL/scripts/train_grpo_ovo_vllm_qwen3vl8b_full_8gpu_lt120s.sh`
- 训练链路：`vLLM rollout + Ray + verl_0425 + StreamWeave stepwise env`
- 当前数据：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/ovo/ovo_rl_lt120s.json`
- 数据规模：`293` 条 `<120s` 单 query OVO RL 样本。
- 当前模型起点：`/mmu_mllm_hdd/zhouhanshu/test/exp3/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_v2_3077`

## 最近一次 run

- run 目录：`RL/outputs/debug/grpo_ovo_qwen3vl8b_full_vllm_8gpu_lt120s_20260505.163625`
- 进度：`39/73`，约 `53%`
- elapsed：`10:26:12`
- 日志最后更新时间：北京时间 `2026-05-06 11:06:52`
- 后续检查时核心进程均已消失：`TaskRunner / AgentLoopWorker / WorkerDict / vLLMHttpServer`
- 目录内没有 `exit_code.txt`，没有 checkpoint。

判断：本次不像正常完成，更像外部中断、进程被杀或 shell 任务异常退出。

## 当前观测

- 数据/rollout 链路没有明显错误：
  - `response/aborted_ratio = 0.0`
  - `prompt_length/clip_ratio = 0.0`
  - `response_length/clip_ratio` 基本为 `0`
  - `format_score` 基本在 `0.96 ~ 1.0`
- reward 能产生有效信号：
  - `streamweave/format_score/mean` 基本接近 `1`
  - `streamweave/success_score/mean` 约 `0.09 ~ 0.87`
  - `trajectory_score/mean` 约 `0.54 ~ 0.93`
- 每 step 只有 `4 * 8 = 32` 条 trajectory，success 波动明显是预期现象。

## 性能瓶颈

step 38 观测：

```text
gen:          51s
old_log_prob: 222s
update_actor: 934s
step total:   1212s
```

当前主要瓶颈不在 vLLM 生成，而在训练侧，尤其 `old_log_prob` 和 `update_actor`。

优先排查方向：

- `use_remove_padding=False`
- FSDP2 offload
- 长多模态序列
- stepwise 展开后的 token 量
- actor micro batch / sequence balance
- checkpoint 和日志写入频率

## 配置问题

- `save_freq=100`，但总 step 只有 `73`，所以中途不会保存 checkpoint。下一次应改成 `5` 或 `10`。
- 当前脚本实际从 SFT checkpoint 开始 RL，而不是 base instruct。这个选择需要明确记录：
  - SFT 起点：`/mmu_mllm_hdd/zhouhanshu/test/exp3/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_v2_3077`
  - base instruct：`/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct`

## Reward 当前口径

当前已经可工作的 reward 信号优先包括：

- 格式分：XML/parser、字段完整性、stepwise 输出协议。
- 成功分：最终 answer/trajectory 是否满足 OVO 任务。
- trajectory 聚合：对同一 query 的多条 rollout 做组内优化。

后续再逐步增强：

- 时间约束：`eta` 合法性、早答/迟答惩罚。
- memory 成本：note 数、bridge token、long bridge、open-tail bridge。
- 语义保真：先做离线 evaluator，不要一开始塞进主训练阻塞链路。

## Bridge Reward 设计

Bridge 的目标不是写得长，而是把两个视觉 anchor 之间的状态变化压缩成后续推理可用的记忆。后续 reward 可以分三层推进：

- 结构 reward：
  - bridge 时间段必须准确覆盖 note/window gap。
  - open-tail bridge 必须正确继承起点。
  - bridge 不应和 note 时间重叠。
- 信息 reward：
  - bridge 应包含实体、动作、状态变化、数量变化、位置变化等可用于推理的信息。
  - 对空泛句、重复上一段 memory、没有对象或状态描述的 bridge 给惩罚。
- 一致性 reward：
  - bridge 不能和相邻 note 图像矛盾。
  - 可以先用 teacher/VLM judge 或 caption overlap 做弱 reward。
  - 更稳的路线是先离线生成 reference bridge，再训练轻量 reward model 或用相似度打分。

不要一开始把强 LLM/VLM judge 放进在线 RL 主链路：成本高、延迟大、稳定性差。更实用的路线是 SFT 阶段先生成高质量 bridge，RL 阶段主要奖励结构覆盖、非空泛、弱一致性和最终 QA 增益。

## RL 训练课程

提高图文交错记忆推理能力不能只调最终 reward，需要分阶段训练：

1. SFT warmup：先让模型学会 XML 协议、note/bridge 风格、eta/answer 时机，减少 RL 在格式探索上的浪费。
2. Process RL：先提高 `w_step`，重点训练 keyframe selection 和 bridge 行为；最终 QA reward 保留，但不要唯一主导。
3. Answer RL：等 note/bridge 行为稳定后，再提高 `w_success`，让模型学习利用 memory 和当前帧进行问答。

这条课程的核心假设是：先把“会记、会写、会按时间组织记忆”练稳，再让最终答案 reward 主导推理能力提升。

## 下一步

1. 修改 GRPO 脚本，把 `save_freq` 改成 `5` 或 `10`。
2. 确认模型起点：SFT checkpoint 还是 base instruct。
3. 重新跑 `<120s` OVO RL 子集，确保至少产出可恢复 checkpoint。
4. 在 checkpoint 可恢复后，再优化 `old_log_prob/update_actor`。
5. 跑完一个完整小实验后，做 OVO 小规模回评，而不是只看训练 reward。
