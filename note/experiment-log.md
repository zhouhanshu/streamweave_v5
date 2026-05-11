# 实验记录索引

## 当前状态

- 实验：`StreamWeave`
- 当前主线：`exp3/streamweave_v5/RL`
- 当前阶段：第一次 GRPO 训练已完整跑完；使用 OVO-Bench `<120s` 子集，已导出 HuggingFace 格式模型 `models/qwen3vl_8b_grpo_0509`。
- 当前唯一保留的 GRPO 入口：`RL/scripts/train_grpo_ovo_8gpu.sh`
- PPO 入口：`RL/scripts/train_ppo.sh`
- 当前训练数据运行路径：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/ovo/ovo_rl_lt120s.json`
- 当前初始化模型：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm`
- 当前最优先事项：
  - 对 `models/qwen3vl_8b_grpo_0509` 做 OVO-Bench 回评，和 base instruct、answered-full SFT 做 full 对照。
  - base full 和 answered-full SFT full 跑分已落盘；后续继续做 RL 时必须同时和 base instruct、answered-full SFT 做 full 对照。
  - 复盘 judge-enabled GRPO 负结果，优先检查 reward 密度、judge 噪声、KL/entropy 约束和 trajectory-level credit assignment。
  - 后续新 run 必须继续记录 `note_frequency_score`、`judge_score`、`step_score`、`trajectory_score`，并在 SwanLab 中优先看 `traj/score/*` 而不是只看 `critic/score`。
  - 性能优化不要只堆 vLLM 显存利用率；当前瓶颈经常在 `update_actor`、`old_log_prob` 和调度同步。

## 文件索引

- [00-overview.md](./00-overview.md)：目标、方法和当前主线。
- [04-sft-training.md](./04-sft-training.md)：SFT 数据、answered-full 训练、评测状态和第一次 SFT 负面结果。
- [05-rl-training.md](./05-rl-training.md)：当前 GRPO/RL run、checkpoint、reward 和性能记录。
- [实验跑分.md](./实验跑分.md)：正式跑分主表。
- [07-key-points.md](./07-key-points.md)：关键结论和避坑。
- [08-commands-and-tools.md](./08-commands-and-tools.md)：当前有效命令。
- [10-source-code-current-state.md](./10-source-code-current-state.md)：当前源码实际协议、SFT/RL 实现和脚本口径。
- [数据合成0508.md](./数据合成0508.md)、[数据清洗0510.md](./数据清洗0510.md)：数据构造历史和新数据清洗策略；旧数据构造/清洗摘要已合并进 `00-overview.md`。
- [补充实验待办.md](./补充实验待办.md)：主线之外的补充实验 checklist。
- [09-streamweave-proposal-draft.md](./09-streamweave-proposal-draft.md)：长期方案草稿，非当前状态依据。

### 2026-05-08 源码口径同步

- 当前 `<eta>` 协议已经彻底不是主线；源码协议是 `<state>`、`<answer>`、timestamp-only `<anchor>`、`<delta>`。
- `data_engine/sft` 当前没有 `_key_frame_context()` 或 `_apply_key_frame_quality_constraints()`；SFT 硬约束是 anchor 数量、QA answer 空/非空时机和样本级答案正确性。
- RL reward 当前是 `format + step + success`，默认 `w_format=0.1`、`w_step=0.2`、`w_success=0.7`、`score_scale=2.0`；step 内部默认 `note_frequency_weight=0.3`、`judge_weight=0.7`。
- step score 默认来自 anchor frequency：每窗口最多 1 个 anchor，连续 3 个窗口无 anchor 惩罚；judge 默认关闭。
- judge 显式开启后评估 `keyframe_selection`、`bridge_quality`、`semantic_alignment`、`state_factuality`，且被 anchor frequency gate 住。
- 旧 GRPO launcher 已清理；当前只保留最新 fused/chunked GRPO 入口和 PPO 入口。
- 最新 GRPO 入口使用 `save_freq=30`、`resume_mode=auto`、`use_remove_padding=true`、`use_fused_kernels=true`、`enable_chunked_prefill=true`。
- 详细源码事实见 `10-source-code-current-state.md`。

## 关键里程碑

### 2026-05-09 第一次完整 GRPO 训练完成并导出模型

- 结论：第一次 GRPO 训练完整跑完，使用 OVO-Bench 的 `<120s` 子集。
- 数据：`dataset/ovo/ovo_rl_lt120s.json`，`293` 条单 query OVO 样本。
- 初始化模型：`models/qwen3vl8b_streamweave_sft_answered_full_vllm`
- launcher：`RL/scripts/train_grpo_ovo_8gpu.sh`
- run 目录：`RL/outputs/debug/grpo_ovo_8gpu_judge_debug`
- 最终 checkpoint：`RL/outputs/debug/grpo_ovo_8gpu_judge_debug/checkpoints/global_step_36`
- 导出模型：`models/qwen3vl_8b_grpo_0509`
- 导出格式：HuggingFace `from_pretrained` 目录，包含 4 个 safetensors 分片、`model.safetensors.index.json`、config、tokenizer、processor 和 video preprocessor 配置。
- 核心训练设置：`gen_batch_size=16`、`rollout.n=8`、DAPO filter `min_std=0.1`、`lr=5e-6`、`total_steps=36`。
- 奖励设置：`w_format=0.1`、`w_step=0.2`、`w_success=0.7`；step 内部 `note_frequency_weight=0.3`、`judge_weight=0.7`，judge 使用 `gemini-2.5-flash`。
- 最终 step 36：`traj/score_mean=1.4291`、`valid_group_ratio=0.5625`、`trajectory_score/mean=1.3386`、`success_score/mean=1.1111`、`step_score/mean=1.7859`。
- 导出验证：`AutoConfig` 识别为 `qwen3_vl`，tokenizer 为 `Qwen2TokenizerFast`，processor 为 `Qwen3VLProcessor`。

### 2026-05-09 Qwen3-VL-8B base OVO full 完成

- 模型：`/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct`
- 协议：StreamWeave V5 `state + note_t`
- 数据：OVO-Bench full，展开后 `3035` samples
- 输出：`outputs/ovo_qwen3vl8b_base_full_state_note_t`
- 运行：先 8 卡静态分片评测到 `2903/3035`，后用共享任务队列和 `RESUME=1` 在 GPU `2 3 4 5 6 7` 上补跑剩余 `132` 条。
- 主结果：Backward `60.59` / Realtime `75.10` / Forward `36.87` / Total `57.52`
- 结果文件：`results.jsonl`、`results_summary.json`、`results_summary.txt`

### 2026-05-08 answered-full SFT 训练完成，OVO 1/8 回评进行中

- 训练数据：`data_engine/sft/outputs/gemini_answered_full/llamafactory_sharegpt_anchor_delta_le20.jsonl`
- 数据规模：`3956` 条 sample，`2491` 条 accepted；过滤后 `32583` 行 ShareGPT step。
- 训练输出：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_answered_full`
- 训练结果：`759/759` steps，`train_loss=0.181242`，`eval_loss=0.246001`，耗时约 `8:10:10`。
- vLLM 兼容模型：`models/qwen3vl8b_streamweave_sft_answered_full_vllm`
- OVO 1/8 评测输出：`outputs/ovo_qwen3vl8b_finetuned_1of8`
- 当前评测状态：8 个 vLLM server 和 `eval_batch.py` 仍在运行，part 文件已写出至少 `84/205` 行且仍在增长；没有顶层 `results.jsonl` 或 `results_summary.*`，因此暂无正式分数。

### 2026-05-06 V5 GRPO stepwise 链路跑通

- `streamweave_v5/RL` 已能从 OVO RL 标注启动 GRPO stepwise 训练。
- 链路为 `vLLM rollout + Ray + verl_0425 + StreamWeave stepwise env`。
- 当前 OVO RL 原始数据：`ovo_rl.json`，约 `600` 条，`backward/forward/realtime` 各约 `200` 条。
- 当前训练使用 `<120s` 子集：`ovo_rl_lt120s.json`，共 `293` 条。
- 2026-05-08 更新：`RL/outputs` 已清空，旧 launcher 已删除；以下旧 run 只作为历史记录，不再有磁盘输出可追溯。
- 旧 8GPU run：
  - 目录：`RL/outputs/debug/grpo_ovo_qwen3vl8b_full_vllm_8gpu_lt120s_20260505.163625`
  - 进度：日志到 `Training Progress: 41/73`，约 `56%`
  - `exit_code.txt=0`，但无 completion 记录
  - 没有 `global_step_*` checkpoint
  - 判断：不能视为完整完成，也不能恢复训练
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
- fused/chunked run：
  - 目录：`RL/outputs/debug/grpo_ovo_qwen3vl8b_full_vllm_8gpu_lt120s_fused_chunked`
  - 已有 checkpoint：`global_step_30`、`global_step_60`
  - resume 后日志到 `83/146`
  - 性能明显改善：step 61 总耗时约 `242s`，step 80 约 `365s`，step 83 约 `273s`
  - 问题：resume 后 `format_score` 偏低，`step_score` 仍为 `0.0`；该 run 还不能证明当前 reward 配置有效。

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
- 历史 run 已清理，后续不再基于旧 checkpoint 继续训练。
- 最新 RL 起点是 answered-full SFT vLLM 兼容模型，reward v2 口径需要重新跑完整实验。
- 当前性能优化应优先看训练侧：remove padding、offload、长序列 token 量、micro batch 和 FSDP 设置。
- SFT 结果曾明显退化，因此如果继续从 SFT checkpoint 开始 RL，需要明确这是实验选择，不应默认认为它优于 base instruct。
- 后续记录 RL run 时必须写明具体 launcher、模型起点、数据路径、`save_freq`、resume mode、remove-padding/fused/chunked 设置。
