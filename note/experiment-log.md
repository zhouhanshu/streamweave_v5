# 实验记录索引

## 当前状态

- 实验：`StreamWeave`
- 当前主线：`exp3/streamweave_v5/RL`
- 当前阶段：GRPO stepwise 训练链路已跑通，正在定位稳定性和训练侧性能问题。
- 当前入口脚本：`RL/scripts/train_grpo_ovo_vllm_qwen3vl8b_full_8gpu_lt120s.sh`
- 当前训练数据：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/ovo/ovo_rl_lt120s.json`
- 当前初始化模型：`/mmu_mllm_hdd/zhouhanshu/test/exp3/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_v2_3077`
- 当前最优先事项：
  - 把 `save_freq=100` 改成 `5` 或 `10`，避免中断后完全没有 checkpoint。
  - 确认 RL 起点到底使用 SFT checkpoint 还是 base instruct。
  - 优化 `old_log_prob` 和 `update_actor`，当前瓶颈不在 rollout 生成。

## 文件索引

- [00-overview.md](./00-overview.md)：目标、方法和当前主线。
- [04-sft-training.md](./04-sft-training.md)：SFT 数据与第一次 SFT 负面结果。
- [05-rl-training.md](./05-rl-training.md)：当前 GRPO/RL 训练状态。
- [实验跑分.md](./实验跑分.md)：正式跑分主表。
- [07-key-points.md](./07-key-points.md)：关键结论和避坑。
- [08-commands-and-tools.md](./08-commands-and-tools.md)：当前有效命令。
- [02-data-construction.md](./02-data-construction.md)、[数据合成.md](./数据合成.md)：数据构造历史，当前只作追溯。
- [09-streamweave-proposal-draft.md](./09-streamweave-proposal-draft.md)：长期方案草稿，非当前状态依据。

## 关键里程碑

### 2026-05-06 V5 GRPO stepwise 链路跑通

- `streamweave_v5/RL` 已能从 OVO RL 标注启动 GRPO stepwise 训练。
- 链路为 `vLLM rollout + Ray + verl_0425 + StreamWeave stepwise env`。
- 当前 OVO RL 原始数据：`ovo_rl.json`，约 `600` 条，`backward/forward/realtime` 各约 `200` 条。
- 当前训练使用 `<120s` 子集：`ovo_rl_lt120s.json`，共 `293` 条。
- 最近 run：
  - 目录：`RL/outputs/debug/grpo_ovo_qwen3vl8b_full_vllm_8gpu_lt120s_20260505.163625`
  - 进度：`Training Progress: 39/73`，约 `53%`
  - 日志耗时：`elapsed=10:26:12`
  - 日志最后更新时间：北京时间 `2026-05-06 11:06:52`
  - 后续检查时 `TaskRunner / AgentLoopWorker / WorkerDict / vLLMHttpServer` 均已不在
  - 目录内没有 `exit_code.txt`，没有 checkpoint
- 判断：本次更像外部中断、进程被杀或 shell 任务异常退出，不是正常完整跑完。
- 数据链路观察：
  - `response/aborted_ratio = 0.0`
  - `prompt_length/clip_ratio = 0.0`
  - `response_length/clip_ratio` 基本为 `0`
  - `format_score` 基本在 `0.96 ~ 1.0`
- Reward 观察：
  - `streamweave/format_score/mean` 基本接近 `1`
  - `streamweave/success_score/mean` 约 `0.09 ~ 0.87`
  - `trajectory_score/mean` 约 `0.54 ~ 0.93`
  - 每 step 只有 `4 * 8 = 32` 条 trajectory，success 波动大是预期现象。
- 性能瓶颈：
  - step 38：`gen=51s`，`old_log_prob=222s`，`update_actor=934s`，`step total=1212s`
  - 主要瓶颈在训练侧，尤其 `old_log_prob` 和 `update_actor`
  - 可能相关因素：`use_remove_padding=False`、FSDP2 offload、长多模态序列、stepwise 展开后 token 量大
- 配置问题：
  - `save_freq=100` 但总 step 只有 `73`，中途不会保存 checkpoint。
  - 当前脚本从 SFT checkpoint 开始 RL，不是 base instruct。

### 2026-05-02 V4 SFT 链路打通

- V4 SFT 合成链路已经打通：样本级落盘、accepted-only 导出、production prompt 训练数据。
- 小规模结果：`data_engine/sft/outputs/gemini_final_8`，`accepted=7/8`，导出 `136` 条 step-level SFT 样本。
- 关键约束：
  - teacher-only 指令、关键帧标注提示和 retry feedback 不进入训练数据。
  - 只有整条样本所有 step 合法、且样本级答案正确时才进入 `sft_steps.jsonl` 和 `llamafactory_sharegpt.jsonl`。
- 后续第一次 SFT 回评显示明显退化，因此 V4 SFT 数据/训练不能直接视为成功。

### 2026-04-30 V3 Gemini OVO full 主结果

- 口径：`StreamWeave V3 + gemini-2.5-pro + OVO-Bench full`
- 二轮 retry 后：`3035` samples，剩余错误 `6` 条。
- 主结果：`Total AVG = 65.81`
- 分类：`Backward 66.73 / Realtime 80.24 / Forward 50.46`
- 相对 AURA：overall `+0.51pp`，Backward 和 Realtime 略强，Forward 仍弱。
- 最大短板：Forward reasoning，尤其 `CRR`。

### V4 学生基线与第一次 SFT 负面结果

- R12：`V4 + Qwen3-VL-8B base / OVO full`
  - `Backward 48.09 / Realtime 75.41 / Forward 30.78 / Total 51.43`
  - 是后续学生模型 full 回评的底线。
- R13：`V4 + Qwen3-VL-8B base / OVO 1/8`
  - `Backward 59.91 / Realtime 78.13 / Forward 38.72 / Total 58.92`
  - `1/8` 子集明显偏简单，不能单独作为最终结论。
- R14：`V4 + Qwen3-VL-8B SFT / OVO 1/8`
  - `Backward 42.70 / Realtime 69.71 / Forward 32.33 / Total 48.25`
  - 相比 R13 下降 `10.67pp`，说明第一次 SFT 发生反向蒸馏。

## 当前结论

- RL 链路已经跑通，当前不应优先怀疑数据、prompt、env 或 rollout。
- 最近 run 没有 checkpoint 的直接原因是 `save_freq` 配置不合理。
- 当前性能优化应优先看训练侧：remove padding、offload、长序列 token 量、micro batch 和 FSDP 设置。
- SFT 结果曾明显退化，因此如果继续从 SFT checkpoint 开始 RL，需要明确这是实验选择，不应默认认为它优于 base instruct。
