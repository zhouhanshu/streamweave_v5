# 实验要点记录

## 当前关键结论

- 当前主线已经切到 `exp3/streamweave_v5/RL`。
- GRPO stepwise 训练链路已经跑通，当前问题是稳定跑完、保存 checkpoint 和优化训练侧性能。
- 历史 RL 输出已经清理，旧 GRPO launcher 已删除；旧 8GPU run 和旧 fused/chunked run 只作为笔记里的历史记录。
- 当前唯一保留的 GRPO 入口是 `train_grpo_ovo_8gpu.sh`，另保留 `train_ppo.sh`。
- 当前 GRPO 入口默认从 answered-full SFT vLLM 兼容模型初始化。
- 最新 GRPO 入口使用 `save_freq=30`、`resume_mode=auto`、remove padding、fused kernels 和 chunked prefill；它仍需要完整跑完验证。
- 当前性能瓶颈主要在 `old_log_prob` 和 `update_actor`，不是 vLLM 生成。
- V4 第一次 SFT 在 OVO 1/8 上明显退化，不能默认当作更优 RL 起点。
- V5 answered-full SFT 训练已完成，但 OVO 1/8 回评仍在运行中；没有 `results_summary.*` 前不能写正式分数。
- V3 + Gemini 在 OVO full 上达到 `65.81`，证明方法上限存在，但 Forward/CRR 仍是核心短板。
- RL 优化不要只押最终 QA reward；应分阶段做 `SFT warmup -> Process RL -> Answer RL`，先练 keyframe/bridge/记忆组织，再提高 `w_success` 训练最终推理。
- Bridge reward 应分为结构覆盖、信息密度、一致性三类；在线 RL 先用轻量信号，强 VLM/LLM judge 更适合离线生成 reference 或训练 reward model。

## 协议硬约束

- 输出固定为：

```xml
<state>...</state>
<answer>...</answer>
<bridge t="...">...</bridge>
<note t="..."></note>
```

- `state` 是当前 step 的状态总结和回答判断，不写回 Memory。
- `note` 只保存视觉锚点，不保存文字。
- `bridge` 保存文本压缩和绝对时间区间。
- `qa_history` 是时间顺序日志；runtime 用最新 QA role 判断是否还有未答问题，没有未答问题时会丢弃模型输出的非空 answer。
- `<note .../>` 自闭合格式非法，必须使用 `<note ...></note>`。
- `note` 只允许 `t` 属性，不再使用 `frame` 或局部 id。
- `bridge` gap 完整性是硬约束，包括窗口边界、相邻 note 和 open-tail 继承。
- `<eta>`、`frame="N"`、`<note .../>` 都是旧协议残留，不能进入当前 V5 训练 target。

## 源码事实

- SFT 当前没有 annotated key-frame hard constraint；实际约束是 note 数量、answer 空/非空时机和样本级答案正确性。
- SFT target 来自 accepted raw XML，`repair=False`，不能用 repaired action 当训练目标。
- RL 不自动抽帧，只读已存在的 `dataset_root/dataset_name/video/<video_id>/`。
- RL reward 当前是 `format + step + success`，默认 `w_format=0.3`、`w_step=0.3`、`w_success=0.4`、`score_scale=2.0`。
- step score 默认是 note frequency：每窗口最多 1 个 note，连续 3 个窗口无 note 惩罚。
- LLM-as-Judge 默认关闭；开启后评估 keyframe、bridge、semantic alignment 和 state factuality，并被 note frequency gate 住。
- 当前 GRPO 主 estimator 是 `streamweave_stepwise_traj_grpo`，trajectory advantage 会广播到同一 trajectory 的所有 step response token。

## 数据口径

- 早期 `annotations_filtered_30s300s_key10to40.jsonl` 只是 VideoXum/ActivityNet 的视频与关键帧池，不是 SFT 或 RL 训练入口。
- V4 SFT 入口是 `annotations_qa_filter_final.jsonl`。
- 当前 V5 RL 使用 OVO 单 query 数据：
  - `ovo_rl.json`：约 `600` 条
  - `ovo_rl_lt120s.json`：`293` 条
- OVO forward 样本会展开成子视频/sample；处理时必须确认 `sample_id` 和 `video_id` 是否已经展开，避免重复展开。

## 评测口径

- `full`、`1/8`、`1/4` 不能直接横比。
- `task macro` 和 `sample weighted` 必须分开写。
- V3/V4/V5 的 adapter 和 prompt 不同，跨版本只能做趋势判断，不能当严格同口径结论。
- Gemini/API 请求失败不能当作答错；需要区分 error 和 wrong。
- 如果结果突然 `0.00%`，先检查系统性 error、参数错误、认证/配额和是否错误读取 response 字段。

## 避坑

- 本地 `vllm` 使用 `--reasoning-parser qwen3` 时，答案可能在 `message.reasoning`，不一定在 `message.content`。
- Codex 沙箱里的 `localhost` 连通性判断可能失真；本地服务是否可用以外部实际 HTTP 响应为准。
- OVO 全量高并发容易触发视频解码或 API 资源问题；不稳时先降并发。
- Gemini Pro 不支持把 Flash 的 thinking 参数直接照搬；`thinking_budget=0` 可能导致大量 `400 INVALID_ARGUMENT`。
- retryable API 错误要 sample-level 重跑，不能把失败请求写成 done 后继续统计。
- 共享磁盘上控制 GPU 占用的 pid/log 文件必须按 hostname 隔离，否则两台机器会互相覆盖。

## 当前下一步

1. 等 answered-full SFT 评测完成并落盘，补全学生模型对比。
2. 用最新 fused/chunked GRPO 入口从 answered-full SFT 模型启动 reward v2 RL。
3. 检查新 run 日志是否包含 `note_frequency_score`、`judge_score`、`step_score`、`trajectory_score`。
4. 重跑 `<120s` OVO RL 子集并确保 checkpoint 可恢复。
5. 优化 `old_log_prob/update_actor`。
6. 用首个完整 RL checkpoint 做 OVO 小规模回评。
