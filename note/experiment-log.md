# 实验记录索引

## 当前状态

- 实验：`StreamWeave`
- 当前主线：`exp3/streamweave_v5/RL`
- 当前阶段：RL0511 step60 已完成 OVO 1/8 回评，成为当前 Qwen/student 1/8 最强结果；第一次 GRPO 0509 full 仍是当前 V5/Qwen full 最强结果。
- 当前唯一保留的 GRPO 入口：`RL/scripts/train_grpo_ovo_8gpu.sh`
- PPO 入口：`RL/scripts/train_ppo.sh`
- 当前训练数据运行路径：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/ovo/ovo_rl_lt120s.json`
- 当前初始化模型：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm`
- 当前最优先事项：
  - 对 `models/exp1_rl0511_step60` 做 full 回评，确认 OVO 1/8 的正向收益能否迁移到 full。
  - 后续继续做 RL 时必须同时和 base instruct、answered-full SFT、GRPO0509 和 RL0511 step60 做同口径对照。
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

### 2026-05-13 RL0511 step60 OVO full 回评完成

- 模型：`models/exp1_rl0511_step60`，输出：`outputs/ovo_exp1_rl0511_step60_full`，0 errors，耗时 `8:34:21`。
- 主结果：Backward `47.70` / Realtime `77.72` / Forward `54.17` / Total `59.87`。
- 与三条 full 基线对比：

| 指标 | base | SFT | GRPO0509 | RL0511 | vs base | vs SFT | vs GRPO0509 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EPM | 55.22 | 58.59 | 59.60 | 58.92 | +3.70 | +0.33 | -0.68 |
| ASI | 61.49 | 61.49 | 62.84 | 59.46 | -2.03 | -2.03 | -3.38 |
| HLD | 65.05 | 59.68 | 60.22 | 24.73 | -40.32 | -34.95 | -35.49 |
| Backward | 60.59 | 59.92 | 60.88 | 47.70 | -12.89 | -12.22 | -13.18 |
| Realtime | 75.10 | 78.15 | 76.08 | 77.72 | +2.62 | -0.43 | +1.64 |
| Forward | 36.87 | 50.07 | 54.74 | 54.17 | +17.30 | +4.10 | -0.57 |
| Total | 57.52 | 62.71 | 63.90 | 59.87 | +2.35 | -2.84 | -4.03 |

- RL0511 step60 **不是** 当前 Qwen/student full SOTA；full SOTA 仍是 R17 GRPO0509 `63.90`。1/8 上记录的 `64.70` 不能外推。
- HLD `186/186` GT 全部 `Unable to answer`：base 对 `121` / SFT 对 `111` / GRPO0509 对 `112` / RL0511 仅 `46`。同 id `base 对但 RL 错 = 78`、`base 错但 RL 对 = 3`，HLD 净损 `75 ≈ 40.32pp`，是 Backward AVG 从 `60+` 降到 `47.70` 的全部原因。
- Full HLD trace 统计：RL0511 的 `140` 个 HLD 错题中，只有 `17` 个最终 state 明确出现 `Unable/cannot/no mention/not visible/ambiguous` 这类拒答线索但仍选错；大多数错误是 state 本身直接把不存在或未确认的目标物、地点、属性写成确定事实。正确 HLD 中 `23/46` 的 state 明确写了无法确认/无视觉证据，说明模型不是不会拒答，而是缺少 answer-time evidence gate。
- 反事实推演：若 HLD 保持 base 的 `65.05`，Backward AVG = `61.14`，Total = `64.34`，可超过 R17 GRPO0509 `63.90`、逼近 AURA full `65.30`。HLD 是 RL0511 step60 唯一致命短板。
- EPM 在 full 上未明显提升（`58.92` 持平 SFT `58.59`、低于 GRPO0509 `59.60`），昨天"记忆增强已生效"的 1/8 结论（EPM `67.57`）也是切片偏向，full 上记忆并没有比 SFT 更强；RL0511 实际显著受益的是 Forward CRR/SSR 和 Realtime FPD/STU/ACR。
- 修订昨天结论：
  - "RL0511 step60 是 Qwen/student SOTA" → 仅 1/8 SOTA，full 不成立。
  - "1/8 是 HLD 放大效应、full 应该恢复" → 错，full HLD `24.73` 比 1/8 `26.09` 还低。abstention 行为崩盘在 full 同样稳态。
  - "增强记忆已生效" → 仅 1/8 EPM 有显著提升，full 没有。
- 下一步固定：
  - 当前 V5/Qwen full SOTA 仍是 R17 GRPO0509 `63.90`。
  - 修 HLD 必须在 RL 数据 `dataset2/rl_0511.jsonl` 中显式注入 `Unable to answer`-as-gold 的 false-premise MCQ；当前 `3444/3444` 条 gold 没有 Unable，是 abstention 边界被推没的直接原因。
  - 修改 `success_score`：GT=Unable 时答 Unable +1、答具体选项 0；GT=具体选项时答 Unable 0、答对 +1。
  - 在 eval prompt 或 postprocess 加硬规则：state 含 `likely/appears/suggests/assume/unclear` 且选项有 Unable 时强制选 Unable；观察到的具体答案不在选项中时也选 Unable，不映射最近似选项。

### 2026-05-12 RL0511 step60 Backward 退化根因分析

- 问题：训练目标是增强记忆，但 1/8 回评 Backward AVG 从 base 1/8 `58.46` 退到 `52.27`。
- 三个 Backward task 拆开看，记忆其实变强、退化全部集中在 HLD：
  - EPM `51.35 → 59.46 → 67.57`（base → SFT → RL），`+16.22pp`。
  - ASI 三段全部 `63.16`，不变。
  - HLD `60.87 → 43.48 → 26.09`，`-34.78pp`。task-macro 把 EPM 收益和 HLD 退化平均，掉了 `6.19pp`，正好对上 AVG 降幅。
- HLD 的 gold 文本全是 `Unable to answer`：1/8 切片 `23/23`，full `186/186`。区别只在 Unable 对应的选项字母不同；full 字母分布为 A=71、B=61、C=27、D=26、E=1。
- 选中 Unable 对应字母的次数：Qwen base 1/8 `14/23`、SFT anchor 1/8 `10/23`、SFT rerun 1/8 `12/23`、RL0511 step60 `6/23`。同 id 对比中，`base 对但 RL 错` 共 `9` 条，`base 错但 RL 对` 仅 `1` 条（id `349`）；相对 SFT rerun，RL 丢 `6` 条且没有赚回。
- trace 实读两条典型失败样本：
  - id `341` 问 `Which belt did I select?`，GT `A. Unable to answer`，RL 选 `B. the second one`。`memory.txt` 全篇是 `sweater / tablet case / basket`，无任何 `belt` 字样。base/SFT 都正确拒答。
  - id `390` 问 `What did I put in bottle on sink?`，GT `B. Unable to answer`，RL 选 `A. oil`。`memory.txt` 是 motorcycle garage 场景，`bottle of oil placed on newspaper`，从未出现 `sink`。base 正确拒答，SFT 起开始硬猜，RL 延续。
- 新查 RL0511 训练集 `dataset2/rl_0511.jsonl`：`3444` 条全是 backward，`Unable to answer` 作为 gold 的数量是 `0`；含 Unable 选项的样本只有 `1` 条且 gold 不是 Unable。RL0511 配置里 `w_success=0.8`、`w_format=0.1`、`w_step=0.1`，judge 开启但 judge prompt 主要评估过程记忆质量，不直接校准 Unable/false-premise MCQ。
- 机制判断：RL0511 的最终成功奖励几乎从未奖励拒答，且没有 KL 约束（`use_kl_loss=false`、`use_kl_in_reward=false`）。这会把模型从 SFT 的保守拒答边界继续推向"有问题就从 memory/current frame 里找最像的具体选项"。HLD 这种 false-premise/Unable-only task 被直接打穿。
- trace 细分显示两类错误：一类是 state 里已经出现 `likely / appears / suggests / assume / Unable possible`，但最终仍选具体项（如 `309/341/365/390/414/478`）；另一类是 memory 或 state 直接把不存在的目标写实了（如 `430` 抽象画、`462` chandelier、`478` 红花图片），后续 answer scorer 只能按这个幻觉去选项。
- 待办：
  - 优先把 `models/exp1_rl0511_step60` 跑 OVO full 验证 HLD 在 full 上是否同样崩盘。
  - 在 eval prompt 或 answer postprocess 中加入硬规则：如果问题目标/地点/关系没有直接证据，或 state 含 `likely/appears/suggests/assume/unclear` 且存在 Unable 选项，必须选 Unable；如果观察到的答案不在选项中，也选 Unable，不能映射到最近似选项。
  - 后续 RL 数据必须补 hard-negative/false-premise MCQ，并显式包含 `Unable to answer` 为 gold 的样本；否则高 `w_success` 会继续奖励具体答案倾向。
  - 不应据此回退 RL：样本级同 id `233/364 = 64.01%` 仍是 student SOTA，EPM 大幅上升说明"增强记忆"目标已生效。

### 2026-05-12 RL0511 step60 OVO 1/8 回评完成

- 模型：`models/exp1_rl0511_step60`
- 协议：StreamWeave V5 `state + note_t`
- 数据：OVO-Bench `1/8`，`ovo_bench_1of8_stratified.json`，展开后 `364` samples
- 输出：`outputs/ovo_exp1_rl0511_step60_1of8`
- 评测配置：`prompt.profile=eval`、`postprocess.mode=eval_repair`、`sample_fps=1.0`、`frames_per_step=5`、`memory.window_seconds=180`
- 主结果：Backward `52.27` / Realtime `81.54` / Forward `60.30` / Total `64.70`
- 样本级正确率：`233/364 = 64.01%`
- 同 id 对比：高于 Gemini state+note_t 1/8 `+12` 题，高于 GRPO0509 full `+19` 题，高于 answered-full SFT full `+22` 题，高于 SFT anchor step200 1/8 `+9` 题。
- 过程指标：`num_steps=17663`，`model_call_count=17663`，`repair_count=280`，无 backend retry/error，平均调用延迟 `3.84s/call`。
- 结论：当前 Qwen/student OVO 1/8 SOTA；优势来自 Realtime 和 Forward，主要短板是 Backward/HLD。

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

### 2026-05-08 answered-full SFT 训练完成

- 训练数据：`data_engine/sft/outputs/gemini_answered_full/llamafactory_sharegpt_anchor_delta_le20.jsonl`
- 数据规模：`3956` 条 sample，`2491` 条 accepted；过滤后 `32583` 行 ShareGPT step。
- 训练输出：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_answered_full`
- 训练结果：`759/759` steps，`train_loss=0.181242`，`eval_loss=0.246001`，耗时约 `8:10:10`。
- vLLM 兼容模型：`models/qwen3vl8b_streamweave_sft_answered_full_vllm`
- OVO 1/8 评测输出：`outputs/ovo_qwen3vl8b_finetuned_1of8`
- 回评结果后来已落盘，正式分数见 `note/实验跑分.md` 和 `note/0508实验跑分.md`；answered-full SFT 1/8 为 `61.28`，rerun 为 `61.54`，full 为 `62.71`。

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
