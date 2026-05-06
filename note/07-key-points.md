# 实验要点记录

## 当前关键结论

- 当前主线已经切到 `exp3/streamweave_v5/RL`。
- GRPO stepwise 训练链路已经跑通，当前问题是稳定跑完、保存 checkpoint 和优化训练侧性能。
- 最近一次 GRPO run 到 `39/73` 后非正常中断，没有 checkpoint；这首先是实验配置和运行稳定性问题，不是“链路没跑通”。
- `save_freq=100` 对总 step `73` 的 run 不合理，下一次必须改成 `5` 或 `10`。
- 当前脚本从 SFT checkpoint `streamweave_sft_v2_3077` 初始化；如果目标是 base instruct，需要显式改模型路径并单独命名 run。
- 当前性能瓶颈主要在 `old_log_prob` 和 `update_actor`，不是 vLLM 生成。
- V4 第一次 SFT 在 OVO 1/8 上明显退化，不能默认当作更优 RL 起点。
- V3 + Gemini 在 OVO full 上达到 `65.81`，证明方法上限存在，但 Forward/CRR 仍是核心短板。

## 协议硬约束

- 输出固定为：

```xml
<eta>...</eta>
<answer>...</answer>
<bridge t="...">...</bridge>
<note t="..." frame="..."></note>
```

- `note` 只保存视觉锚点，不保存文字。
- `bridge` 保存文本压缩和绝对时间区间。
- `qa_history` 是时间顺序日志，不维护 active query 配对。
- `eta` 是视频内绝对秒级时间戳，不是相对 delay。
- `<note .../>` 自闭合格式非法，必须使用 `<note ...></note>`。
- `bridge` gap 完整性是硬约束，包括窗口边界、相邻 note 和 open-tail 继承。

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

1. 改 GRPO `save_freq`。
2. 明确 RL 初始化模型。
3. 重跑 `<120s` OVO RL 子集并确保 checkpoint 可恢复。
4. 优化 `old_log_prob/update_actor`。
5. 用首个完整 RL checkpoint 做 OVO 小规模回评。
