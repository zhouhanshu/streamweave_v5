# 当前源码现状

最后同步：2026-05-08。本文只记录当前 `streamweave_v5` 源码实际行为，不记录早期方案设想。遇到 `docs/实验计划.md`、`verl修改计划.md`、`tmp_state_protocol_migration_plan.md`、`代码重构.md` 中的旧协议描述时，以本文和源码为准。

## 协议

当前模型输出协议是：

```xml
<state>...</state>
<answer>...</answer>
<delta t="...">...</delta>
<anchor t="..."></anchor>
```

硬规则来自 `streamweave/parser.py` 和 `streamweave/quality.py`：

- 必须恰好有一个 `<state>` 和一个 `<answer>`。
- 输出必须以 `<state>` 后接 `<answer>` 开头。
- `<answer>` 之后只能出现 `<delta>` 和 `<anchor>`。
- `<state>` 不能为空，不能包含 XML tag，也不写入 Memory。
- `<anchor>` 只允许 `t` 属性，必须是成对标签；不再接受 `frame`、`id` 或自闭合 anchor。
- `<delta>` 必须有非空文本和合法绝对时间区间。
- 至少要有一个 observation tag：`<delta>` 或 `<anchor>`。
- `<eta>` 已不是合法协议字段；旧 SFT 或旧 trace 不能和当前协议直接混训。

`qa_history` 是时间顺序日志；模型可以在后续窗口根据 Memory 和当前帧继续更新 `<answer>`。runtime 只在 QA History 从未出现过问题时丢弃非空 answer，避免无题样本写入答案。

## Runtime 与评测

核心链路：

```text
BenchmarkSample
  -> FrameStore.ensure_frames()
  -> StreamWeaveEnv
  -> build_prompt(memory + qa_history + current frames)
  -> backend.generate()
  -> score raw XML
  -> repair/apply
  -> commit memory
  -> RolloutTrace
```

评测路径使用 `evaluation/runner.py`、`evaluation/ovo_adapter.py`、`evaluation/streamingbench_adapter.py` 和 `streamweave/rollout.py`。Eval/SFT 可以通过 `FrameStore.ensure_frames()` 抽帧或复用已有帧；最终答案是 trace 中最后一个非空、已提交的 answer。

`eval_repair` 和 `rollout_repair` 只做一次后处理，不重试模型。raw 质量和 reward 仍按模型原始输出计算，执行 Memory 更新时使用 repaired/applied action。

## SFT 实际实现

当前 SFT 源码在 `data_engine/sft/`，主要入口是：

```text
data_engine/sft/run_pipeline.py
data_engine/sft/run_parallel_pipeline.py
data_engine/sft/export_llamafactory.py
```

实际生成链路：

```text
SamplePlan
  -> group frames
  -> StreamWeaveEnv.build_prompt()
  -> teacher backend generate
  -> env.evaluate_attempt(... repair=False)
  -> SFT-only constraints
  -> accepted raw XML commit
  -> sample-level answer check
  -> accepted-only ShareGPT export
```

当前 SFT 和 eval/RL 的关键差异：

- SFT 不使用 repair 产物做 target；只有 raw XML 本身 valid 的 attempt 才能 accepted。
- `target_xml` 等于 accepted teacher raw output。
- 失败 attempt 只进 sample debug record，不进入训练 JSONL。
- ShareGPT 默认重建 production prompt；不要用 recorded teacher prompt 训练，除非明确做诊断。

当前源码实际存在的 SFT-only constraints：

- `apply_note_count_constraint()`：每 step 最多 `max_notes_per_step` 个 anchor，默认 1。
- `note_reminder_context()`：长时间没有 anchor 时给 teacher 软提醒。
- `apply_qa_answer_constraints()`：只检查 answer 是否该空或非空。
- `check_sample_answer()`：整条样本级别检查 emitted answer 是否匹配 GT。
- answer step 会额外采样 variants，正确 variant 可以在 finalize/export 阶段救回样本。

当前源码没有实现的旧文档项：

- 没有 `_key_frame_context()`。
- 没有 `_apply_key_frame_quality_constraints()`。
- 没有“必须输出标注 key frame 且不能输出额外 anchor”的硬约束。

因此，笔记里若看到“关键帧标注提示不能混入训练数据”，应理解为历史 V4/V3 产物的注意事项；当前 V5 SFT 源码不再注入 annotated key-frame hard constraint。

## RL 实际实现

当前 RL 自有代码在 `RL/streamweave_rl/`，vendored 训练框架在 `RL/verl/`。

核心链路：

```text
RL/scripts/train_grpo*.sh
  -> python -m verl.trainer.main_ppo --config-name=streamweave_stepwise
  -> StreamWeaveAgentDataset
  -> StreamWeaveAgentLoop
  -> StreamWeaveRLEnv.reset()/step()
  -> StreamWeave reward
  -> custom advantage
```

`StreamWeaveAgentDataset` 只返回 dummy tensor 和 metadata；真正 multimodal prompt 在 agent loop 内按当前 memory 状态逐步渲染。

RL 环境和 Eval/SFT 的重要区别：

- RL 调 `FrameStore.load_frames()`，不自动抽帧。
- 缺帧、query 被 `max_steps` 截断、prompt 过长、空 response 等会 abort trajectory，并返回零奖励输出。
- 每个 StreamWeave step 都生成一条训练 row；`group_idx`、`traj_idx`、`turn_idx` 保留 trajectory 结构。
- env step 按 raw XML 算 quality/reward，但用 repaired action 推进 Memory，避免单步坏输出直接让后续环境崩掉。

当前 `streamweave_stepwise.yaml` 是公共基础配置，主要保留 dataset、runtime、memory、reward、judge 默认值、agent loop、batch 和 vLLM/FSDP 通用开关。启动脚本只负责具体数据集/模型/run 路径，以及打开 GRPO/PPO、judge、DAPO 等实验开关。

当前 `train_grpo_ovo_8gpu.sh` 实验口径：

- `algorithm.adv_estimator=streamweave_stepwise_traj_grpo`
- `critic.enable=false`
- `actor_rollout_ref.rollout.n=8`
- `trainer.stepwise_rollout=true`
- `trainer.stepwise_value_mask=true`
- `reward.w_format=0.1`
- `reward.w_step=0.2`
- `reward.w_success=0.7`
- `reward.score_scale=2.0`
- `enable_note_frequency_reward=true`
- judge 默认参数在 YAML 中配置为 Gemini Flash，当前 GRPO 脚本会打开 judge，`judge_weight=0.7` 时影响 `step_score`

当前 reward 组成：

- format score：来自 raw XML 的 valid/parser 结果，默认满分为 2.0。
- step score：当前主要是 anchor frequency，可选接 LLM/VLM judge。
- success score：最终 answer 用 dataset scorer 打分，OVO scorer 会按 task 处理 MCQ、REC、SSR、CRR，默认满分为 2.0。
- final turn 会额外加 `w_success * success_score` 到 turn reward。

当前 custom advantage：

- `streamweave_stepwise_traj_grpo`：同一 `group_idx` 内按 trajectory score 做组内归一化，再广播到该 trajectory 的所有 step response token。
- `streamweave_stepwise_ppo_gae`：按 `(group_idx, traj_idx, turn_idx)` 做 stepwise GAE，主要给 PPO/critic 路径预留。

## 当前训练脚本口径

旧 GRPO launcher 已清理。当前只保留一个 GRPO 入口和一个 PPO 入口：

```text
RL/scripts/train_grpo_ovo_8gpu.sh
```

这是当前最新 GRPO reward v2 入口：

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- 默认模型：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm`
- `save_freq=20`
- `resume_mode=auto`
- `use_remove_padding=true`
- `use_fused_kernels=true`
- `enable_chunked_prefill=true`
- `data.train_batch_size=32`，`data.gen_batch_size=16`
- `rollout.n=8`
- `algorithm.filter_groups.enable=true`
- `algorithm.filter_groups.min_std=0.1`

```text
RL/scripts/train_ppo.sh
```

这是 PPO 对照入口。

已删除：

- legacy baseline GRPO launcher
- legacy OVO GRPO launcher
- legacy long-name 8GPU GRPO launcher

后续笔记记录 run 时，必须写清楚具体脚本、模型起点、数据路径、`save_freq`、resume mode、remove-padding/fused/chunked 设置。

## 当前风险

- `pyproject.toml` 和若干 package docstring 仍有 V4 命名残留，不影响运行但会误导读者。
- 历史 docs 中大量 `<eta>`、`frame="N"`、RAFT 旧计划和 V4/V3 目标已经不是当前实现。
- SFT key-frame hard constraint 在旧文档里出现过，但当前源码没有。
- `configs/eval_ovo_qwen3vl32b_one.yaml` 已改为从 `SILICONFLOW_API_KEY` 读取密钥。
- `TraceWriter` 初始化会清空同一 trace 目录中的旧 trace 文件，重跑同一 sample 前要确认输出目录。
