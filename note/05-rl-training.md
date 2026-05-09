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
w_format = 0.1
w_step = 0.2
w_success = 0.7
score_scale = 2.0
trajectory_score = 0.1 * mean(format_score)
                 + 0.2 * mean(step_score)
                 + 0.7 * success_score
```

step score：

- 默认配置中 `note_frequency_weight=0.3`、`judge_weight=0.7`。
- 如果 judge 关闭，则 `step_score` 只来自 note frequency。
- 当前 8GPU GRPO launcher 默认开启 judge，因此 `step_score = 0.3 * note_frequency_score + 0.7 * judge_score`。
- 每个窗口最多 1 个 note。
- 连续 3 个窗口没有 note 会被惩罚。
- 正常 `note_frequency_score=2.0`；一个窗口超过 1 个 note 或连续 3 个窗口无 note 时为 `0.0`。

LLM-as-Judge：

- 评估 `keyframe_selection`、`bridge_quality`、`semantic_alignment`、`state_factuality`。
- 配置文件默认 `judge.enable=false`，但 judge 权重默认保留为 `0.7`；8GPU launcher 默认显式开启 judge。
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

### Run C：2026-05-09 judge-enabled GRPO 负结果

状态和结论：

- 当前结论：本轮已经停止，未观察到稳定学习趋势，不作为有效 RL checkpoint 或主结果使用。
- launcher：`RL/scripts/train_grpo_ovo_8gpu.sh`
- run name：`grpo_ovo_8gpu_judge_debug`
- run 目录：`RL/outputs/debug/grpo_ovo_8gpu_judge_debug`
- 数据：`dataset/ovo/ovo_rl_lt120s.json`，`293` 条 `<120s` OVO 单 query 样本。
- 初始化模型：`models/qwen3vl8b_streamweave_sft_answered_full_vllm`
- GPU：`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- 训练步数口径：`total_epochs=2`，总步数约 `146`；这个规模不足以稳定判断曲线趋势，但本轮早期指标没有显示出明确改善。

本轮最终训练参数：

```text
data.train_batch_size=32
data.gen_batch_size=4
data.val_batch_size=4
actor_rollout_ref.rollout.n=8
actor_rollout_ref.rollout.agent.num_workers=32
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768
actor_rollout_ref.rollout.max_model_len=16384
actor_rollout_ref.rollout.max_num_batched_tokens=32768
actor_rollout_ref.rollout.max_num_seqs=2048
actor_rollout_ref.rollout.gpu_memory_utilization=0.7
actor_rollout_ref.rollout.enable_chunked_prefill=True
actor_rollout_ref.rollout.free_cache_engine=True
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.use_fused_kernels=True
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.model.override_config.attn_implementation=sdpa
actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
actor_rollout_ref.actor.fsdp_config.param_offload=False
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
actor_rollout_ref.ref.fsdp_config.param_offload=False
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.kl_loss_coef=0.0
actor_rollout_ref.actor.entropy_coeff=0
algorithm.use_kl_in_reward=False
critic.enable=False
trainer.save_freq=20
trainer.resume_mode=auto
trainer.logger=["console","swanlab"]
```

环境和 Gemini judge 参数：

```text
STREAMWEAVE_REWARD_JUDGE_ENABLE=true
STREAMWEAVE_REWARD_JUDGE_WEIGHT=1.0
STREAMWEAVE_JUDGE_BACKEND=gemini
STREAMWEAVE_JUDGE_MODEL=gemini-2.5-flash
GOOGLE_APPLICATION_CREDENTIALS=/mmu_ssd3/group_lisize/hetu/xujia10/joint_tags/scripts/gemini_client/config.json
STREAMWEAVE_JUDGE_MAX_TOKENS=2048
STREAMWEAVE_JUDGE_TEMPERATURE=0.0
STREAMWEAVE_JUDGE_TOP_P=0.1
STREAMWEAVE_JUDGE_TIMEOUT_SECONDS=180.0
STREAMWEAVE_JUDGE_IMAGE_RESOLUTION=512
STREAMWEAVE_JUDGE_MAX_RETRIES=2
STREAMWEAVE_JUDGE_RETRY_BACKOFF_SECONDS=5.0
```

本轮 reward 策略：

- 输出协议：`<state>`、`<answer>`、timestamp-only `<note>`、`<bridge>`。
- 分值统一放大：`score_scale=2.0`，因此 format、note frequency、judge、success 的满分都是 `2.0`。
- trajectory 聚合权重：`w_format=0.3`、`w_step=0.3`、`w_success=0.4`。
- `format_score`：基于 raw XML 输出计算，修复后的 action 只用于推进环境，不用于格式奖励。
- `note_frequency_score`：每个窗口最多允许 `1` 个 note；一个窗口输出大于 `1` 个 note，或连续 `3` 个窗口没有 note，得 `0.0`；否则得 `2.0`。
- `judge_score`：开启 Gemini LLM-as-judge，维度为 `keyframe_selection`、`bridge_quality`、`semantic_alignment`、`state_factuality`，每维 `0.0-1.0` 后聚合并乘以 `2.0`。
- judge gate：如果当前 step 没拿到完整 note frequency 奖励，则不调用或不计 judge，`judge_score=0.0`。
- judge prompt 特殊规则：如果当前帧窗口正好是 `0.0-5.0`，只有输出且仅输出 `note t="0.0-1.0"` 时，`keyframe_selection` 才能给 `1.0`；其他 note 时刻、缺 note 或多个 note 都给 `0.0`。
- judge 输出结构：四个维度都要求返回 `score` 和简短 `reason`，用于 debug。
- `step_score`：judge 开启且权重大于 `0` 时，由 note frequency 和 judge 加权平均得到；当前 `judge_weight=1.0`，等价于二者各占一半。
- `success_score`：trajectory 结束时按 OVO answer scorer 计算，满分 `2.0`。
- `trajectory_score = 0.3 * mean(format_score) + 0.3 * mean(step_score) + 0.4 * success_score`。
- `turn_reward`：非终止 step 分摊 `format + step`，最后一个 step 额外加入 `success` 项；日志里的 `critic/score` 和 `critic/rewards` 主要对应 token-level turn reward，不等同于 trajectory score。

本轮算法策略：

- 算法：GRPO，`algorithm.adv_estimator=streamweave_stepwise_traj_grpo`。
- Stepwise rollout：每个视频窗口作为一条独立 prompt-response row 训练，通过 `group_idx`、`traj_idx`、`turn_idx` 保留完整 trajectory。
- 采样规模：`data.gen_batch_size=4` 且 `rollout.n=8`，所以每个 optimizer step 约 `4 * 8 = 32` 条 trajectory。
- Advantage：同一 query 的 `8` 条 rollout 组成一组，按 `trajectory_score` 做组内归一化，然后把同一 trajectory 的 advantage 广播到它所有 step 的 response token。
- critic：`critic.enable=False`，没有 value model。
- KL：`algorithm.use_kl_in_reward=False`、`actor.use_kl_loss=False`、`kl_loss_coef=0.0`，本轮没有显式 KL 约束。
- 熵：`entropy_coeff=0`，没有额外熵奖励。
- 日志：已增加 `streamweave/trajectory_score/*`、`streamweave/success_score/*`、`streamweave/step_score/*`、`streamweave/note_frequency_score/*`、`streamweave/judge_score/*`；后续重启后应同时看 `traj/score/*`、`traj/success/*` alias，方便 SwanLab 图表筛选。

调参和现象：

- 先后尝试过提高并发和显存利用：`gen_batch_size=4`、`agent.num_workers=32`、`micro_batch_per_gpu=4`、`max_num_batched_tokens=32768`、`vLLM gpu_memory_utilization=0.65/0.7`、actor/ref offload 关闭。
- 训练速度没有稳定改善，较激进配置下 `update_actor` 明显变慢，说明瓶颈不只是 vLLM 显存利用率。
- 代表性日志中，`format_score` 常接近 `2.0`，但 `success_score`、`step_score`、`judge_score` 波动大，没有稳定上升。
- 例：较激进配置 step 2，`trajectory_score/mean=1.5597`、`success_score/mean=1.25`、`step_score/mean=1.3743`、`judge_score/mean=1.3215`，`gen=124.31s`、`old_log_prob=54.75s`、`update_actor=261.20s`、`step=444.26s`。
- 例：中间配置 step 4，`trajectory_score/mean=1.5192`、`success_score/mean=0.9375`、`step_score/mean=1.7605`、`judge_score/mean=1.6903`，`gen=68.05s`、`old_log_prob=26.01s`、`update_actor=98.65s`、`step=196.55s`。
- 曾遇到 vLLM EngineCore 残留进程占用 GPU 0/5 约 `48GB`，导致 vLLM 启动时误判可用显存不足；后续失败后需要先检查 `nvidia-smi` 和 Ray/vLLM 残留。

复盘判断：

- 这轮不是链路打不通，而是 reward/算法组合没有在短训练内产生清晰学习信号。
- 由于总步数只有约 `146`，单看前十来步噪声很大；但用户已经停止本轮，记录为负结果。
- 后续不应简单继续堆显存或并发。优先重新审视 reward 密度、judge 噪声、KL/entropy 约束、trajectory-level credit assignment，以及是否先做更短、更稳定的离线/小集校准。

### 2026-05-09 DAPO-style group filtering 更新

- 新增 group 级指标：`traj/score_mean`、`traj/score_std`、`traj/valid_group_ratio`，同时保留 `traj/group_score_mean/*`、`traj/group_score_std/*` 这类展开统计。
- 新增 StreamWeave 版 DAPO group filtering：在 reward 提取之后、`old_log_prob` 之前，按 `algorithm.filter_groups.metric` 聚合同一 group 的 trajectory 分数。
- 当前默认 metric：`trajectory_score`，脚本环境变量为 `STREAMWEAVE_DAPO_FILTER_METRIC=trajectory_score`。
- 有效 group 定义：同一 group 内至少两条 rollout 的 trajectory score 存在方差，`std > min_std`。
- invalid group：同组 rollout 全部同分，例如全高分、全低分或全失败；这类 group 的 GRPO advantage 基本为零，会被过滤掉，不进入 logprob/update。
- DAPO clip-higher：当前 8GPU launcher 显式设置 `clip_ratio_low=0.2`、`clip_ratio_high=0.28`，并保持 `loss_agg_mode=token-mean`。
- 安全保护：如果一个 batch 里所有 group 都 invalid，则不做过滤，避免空 batch 直接中断训练。
- 当前实现是当前 stepwise 链路内的 DAPO-style group filtering；还没有实现官方 DAPO 的跨 batch 动态补采样。

### 2026-05-09 reward 权重调整

- trajectory 聚合权重从 `format=0.3, step=0.3, success=0.4` 调整为 `format=0.1, step=0.2, success=0.7`。
- 当前公式：`trajectory_score = 0.1 * mean(format_score) + 0.2 * mean(step_score) + 0.7 * success_score`。
- step 内部权重从 note/judge 各半调整为 `note_frequency_weight=0.3`、`judge_weight=0.7`。
- 目的：降低格式奖励占比，强化最终答案成功信号；step 部分更依赖 judge 内容质量，只保留 note frequency 作为频率约束。
- 当前 8GPU launcher 对应环境变量默认值：`STREAMWEAVE_REWARD_NOTE_WEIGHT=0.3`、`STREAMWEAVE_REWARD_JUDGE_WEIGHT=0.7`。

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
