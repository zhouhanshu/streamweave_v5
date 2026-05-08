# 第五部分：RL 训练

## 当前状态

- 状态：GRPO stepwise 链路已经跑通；旧 8GPU run 没有 checkpoint，fused/chunked run 已产出可恢复 checkpoint 但尚未完整完成。
- 代码目录：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/RL`
- 当前唯一保留的 GRPO 启动脚本：`RL/scripts/train_grpo_ovo_8gpu.sh`
- PPO 启动脚本：`RL/scripts/train_ppo.sh`
- 训练链路：`vLLM rollout + Ray + verl_0425 + StreamWeave stepwise env`
- 当前数据运行路径：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/ovo/ovo_rl_lt120s.json`
- 数据规模：`293` 条 `<120s` 单 query OVO RL 样本。
- 当前模型起点：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm`
- 当前注意事项：历史 RL 输出已从 `RL/outputs` 删除，旧 launcher 已清理；后续结论只基于新 reward v2 run。

## 2026-05-08 RL 启动脚本清理

已删除旧 launcher：

- legacy baseline GRPO launcher
- legacy OVO GRPO launcher
- legacy long-name 8GPU GRPO launcher

保留入口：

- `RL/scripts/train_grpo_ovo_8gpu.sh`：最新 GRPO reward v2 入口。
- `RL/scripts/train_ppo.sh`：PPO 对照入口。
- `RL/scripts/run_smoke.sh`：smoke test，不是训练 launcher。

历史输出：

- `RL/outputs` 已清空，旧 `global_step_30/60` 和旧 debug run 均不再保留在磁盘。

## 当前 reward v2 策略

训练框架不变：

- 继续使用 `streamweave_stepwise_traj_grpo`。
- 同一 sample 下多个 rollout 做组内归一化。
- trajectory reward 广播到对应 step 的 response token。

默认权重：

```text
w_format = 0.3
w_step = 0.3
w_success = 0.4
score_scale = 2.0
trajectory_score = 0.3 * mean(format_score)
                 + 0.3 * mean(step_score)
                 + 0.4 * success_score
```

step score：

- 默认 `step_score = note_frequency_score`。
- 每个窗口最多 1 个 note。
- 连续 3 个窗口没有 note 会被惩罚。
- 正常 `note_frequency_score=2.0`；一个窗口超过 1 个 note 或连续 3 个窗口无 note 时为 `0.0`。

LLM-as-Judge：

- 评估 `keyframe_selection`、`bridge_quality`、`semantic_alignment`、`state_factuality`。
- 默认 `judge.enable=false` 且 `judge_weight=0.0`，不影响训练。
- 显式开启后才进入 `step_score`。
- judge 被 note frequency gate 住：note frequency 没拿满分时，不调用 judge，`judge_score=0`。

日志字段：

- `format_score`
- `step_score`
- `note_frequency_score`
- `judge_score`
- `success_score`
- `trajectory_score`
- `turn_reward`

## 2026-05-08 RL 实验记录

### 数据和起点

- RL 数据：`dataset/ovo/ovo_rl_lt120s.json`
- 数据规模：`293` 条 `<120s` 单 query OVO 样本。
- 初始化模型：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm`
- 对照模型应包括：`/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct` 和当前 SFT checkpoint。
- 训练 batch 口径：每 step 约 `4 * 8 = 32` 条 trajectory，因此 success/reward 的 step 间波动较大。

### Run A：旧 8GPU lt120s

基本信息：

- run 目录：`RL/outputs/debug/grpo_ovo_qwen3vl8b_full_vllm_8gpu_lt120s_20260505.163625`
- launcher：历史旧 8GPU 入口，脚本已删除。
- 进度：日志到 `Training Progress: 41/73`，约 `56%`。
- 最后关键日志时间：`2026-05-06 11:36:01` 附近，随后记录 step 41。
- `exit_code.txt`：存在，内容为 `0`。
- checkpoint：未发现 `global_step_*` 目录。
- 结论：`exit_code=0` 不能等同于完整完成；没有 checkpoint，也没有最终 completion 记录，本 run 不能作为可恢复训练或最终 RL 结果。

观测指标：

| step | trajectory | success | format | step_score | gen | old_log_prob | update_actor | step total | throughput |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 38 | 0.7813 | 0.5625 | 1.0000 | 0.0000 | 51.19s | 222.80s | 934.45s | 1212.68s | 129.23 |
| 41 | 0.8281 | 0.6563 | 1.0000 | 0.0000 | 34.62s | 120.33s | 506.22s | 664.86s | 118.70 |

判断：

- 数据/rollout 链路没有明显错误：`response/aborted_ratio=0.0`，`prompt_length/clip_ratio=0.0`，`response_length/clip_ratio` 基本为 `0`。
- reward 有信号：`success_score` 和 `trajectory_score` 有波动，不是全零。
- 主要瓶颈在训练侧：`old_log_prob` 和 `update_actor` 占绝大多数 step 时间。
- 该 launcher 的 `save_freq=100`，而总步数只有 `73`，所以中途不会保存 checkpoint，这是不能恢复的直接原因。

### Run B：fused/chunked lt120s

基本信息：

- run 目录：`RL/outputs/debug/grpo_ovo_qwen3vl8b_full_vllm_8gpu_lt120s_fused_chunked`
- launcher：`RL/scripts/train_grpo_ovo_8gpu.sh` 的旧 run name 版本。
- 目录名仍带 `8gpu_lt120s_fused_chunked`，记录时以实际 run 目录为准。
- checkpoint：`global_step_30` 和 `global_step_60`，各约 `50G`。
- resume 证据：日志显示发现 `global_step_60`、`Setting global step to 60`、`Resuming from .../global_step_60`。
- 当前进度：resume 后日志到 `Training Progress: 83/146`，约 `57%`。
- `exit_code.txt`：存在，内容为 `0`。
- 缺失项：未发现 `global_step_90`，也没有完整训练完成记录。
- 结论：该 run 已证明 checkpoint/resume 和性能优化有效，但还不是完整 RL 结果。

性能对比：

| step | gen | old_log_prob | update_actor | step total | throughput | max mem alloc/reserved |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| old run 38 | 51.19s | 222.80s | 934.45s | 1212.68s | 129.23 | 40.00G / 45.45G |
| fused 61 | 37.05s | 34.51s | 166.76s | 241.81s | 388.56 | 17.60G / 27.19G |
| fused 80 | 69.19s | 62.21s | 229.85s | 364.70s | 458.50 | 19.61G / 29.21G |
| fused 83 | 43.22s | 31.29s | 195.14s | 273.13s | 341.00 | 19.61G / 29.21G |

reward/format 观测：

| step | trajectory | success | format | step_score |
| ---: | ---: | ---: | ---: | ---: |
| 61 | 0.1938 | 0.0625 | 0.2969 | 0.0000 |
| 62 | 0.2038 | 0.2813 | 0.1483 | 0.0000 |
| 80 | 0.3874 | 0.1875 | 0.5302 | 0.0000 |
| 83 | 0.3021 | 0.3438 | 0.1286 | 0.0000 |

判断：

- fused/chunked 配置明显降低显存和训练侧耗时，性能方向是正的。
- 但 resume 后 `format_score` 明显偏低，`step_score` 仍为 `0.0`；这批历史日志不能证明当前 note-frequency step reward 有效。
- 在产出论文或主结果前，应重新跑一条明确使用当前 reward 配置的 run，或至少从 checkpoint 恢复后确认 reward 配置、格式率和 step_score。

## 配置问题与记录口径

- 旧异常 run 脚本已删除，不再作为可选入口。
- 当前最新 fused/chunked 脚本默认从 answered-full SFT vLLM 兼容模型开始 RL。
- 当前最新脚本是 `save_freq=30`、`resume_mode=auto`、`use_remove_padding=true`、`use_fused_kernels=true`、`enable_chunked_prefill=true`，但仍需完整实验验证。
- 后续每个 RL run 必须记录：launcher、run dir、数据路径、初始化模型、`save_freq`、`resume_mode`、remove-padding/fused/chunked 设置、checkpoint 列表、最后 progress、exit code、是否有最终 completion、关键 reward 指标、关键 timing 指标。

## 当前源码链路

- `StreamWeaveAgentDataset` 返回 dummy tensor 和 sample metadata，真实 prompt 在 agent loop 内按每一步 Memory 状态重建。
- `StreamWeaveRLEnv.reset()` 使用 `FrameStore.load_frames()`，RL 不自动抽帧；缺帧会 abort trajectory。
- 每个 StreamWeave step 都是独立 prompt-response row，通过 `group_idx`、`traj_idx`、`turn_idx` 保留 trajectory 结构。
- `env.step()` 对 raw XML 计算 quality/reward，但用 repaired action 推进 Memory，避免坏输出直接让后续环境无法运行。
- 当前主 GRPO estimator 是 `streamweave_stepwise_traj_grpo`，按 trajectory score 做组内归一化，再广播到该 trajectory 的所有 step response token。

## Reward 当前口径

当前已经可工作的 reward 信号优先包括：

- 格式分：XML/parser、字段完整性、stepwise 输出协议。
- step 分：当前主要是 note frequency，可选 LLM/VLM judge 默认关闭。
- 成功分：最终 answer/trajectory 是否满足 OVO 任务，OVO scorer 会按 MCQ、REC、SSR、CRR 分别处理。
- trajectory 聚合：对同一 query 的多条 rollout 做组内优化。

后续再逐步增强：

- 回答时机：早答/迟答、state 与 answer 是否一致、forward 是否提前答。
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

1. SFT warmup：先让模型学会 XML 协议、state/answer 判断、note/bridge 风格，减少 RL 在格式探索上的浪费。
2. Process RL：先提高 `w_step`，重点训练 keyframe selection 和 bridge 行为；最终 QA reward 保留，但不要唯一主导。
3. Answer RL：等 note/bridge 行为稳定后，再提高 `w_success`，让模型学习利用 memory 和当前帧进行问答。

这条课程的核心假设是：先把“会记、会写、会按时间组织记忆”练稳，再让最终答案 reward 主导推理能力提升。

## 下一步

1. 用最新 fused/chunked GRPO 入口从 answered-full SFT 模型启动 reward v2 RL。
2. 确认新 run 不会 resume 旧 checkpoint，并记录新 run name。
3. 检查日志是否出现 `note_frequency_score`、`judge_score`、`step_score`、`trajectory_score`。
4. 重新跑 `<120s` OVO RL 子集，确保至少产出可恢复 checkpoint。
5. 在 checkpoint 可恢复后，再优化 `old_log_prob/update_actor`。
6. 跑完一个完整小实验后，做 OVO 小规模回评，而不是只看训练 reward。
