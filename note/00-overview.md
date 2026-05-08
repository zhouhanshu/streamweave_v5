# 实验总览

## 目标

`StreamWeave` 面向在线流式视频理解：用户可以在视频任意时间提问，模型需要判断应该依赖过去、当前还是未来证据，并在有限上下文下保留关键视觉状态。

核心目标：

- 保住 OCR、颜色、空间关系、对象身份和动作进展等视觉细节。
- 不把所有历史帧都塞进上下文，而是显式选择关键视觉锚点。
- 在 OVO-Bench 和 StreamingBench 上验证时机判断、记忆保留和回答正确性。

## 方法口径

- `note`：长期保留的视觉锚点，只保存当前 step 的关键帧和时间信息。
- `bridge`：用文本压缩两个视觉锚点之间的过渡。
- `state`：当前 step 的视频状态总结和回答判断，只用于本轮推理，不写回 Memory。
- `answer`：当前能答则输出答案，证据不足则保持空/沉默。
- 每一步输入：`memory + qa_history + current frames`。
- 每一步输出：

```xml
<state>...</state>
<answer>...</answer>
<bridge t="...">...</bridge>
<note t="..."></note>
```

## 当前主线

- 当前代码主线：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5`
- 当前训练主线：`streamweave_v5/RL` 的 GRPO stepwise。
- 当前唯一保留的 GRPO 启动脚本：`RL/scripts/train_grpo_ovo_8gpu.sh`
- PPO 启动脚本：`RL/scripts/train_ppo.sh`
- 当前数据运行路径：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/ovo/ovo_rl_lt120s.json`
- 历史数据来源：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/ovo/ovo_rl_lt120s.json`
- 当前模型起点：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm`

最新状态：GRPO 链路已经跑通。历史 RL 输出已经从 `RL/outputs` 清理；旧 GRPO launcher 也已删除，只保留最新 fused/chunked GRPO 入口和 PPO 入口。下一步从 answered-full SFT 后的模型启动 reward v2 口径 RL：`save_freq=30`、`resume_mode=auto`、remove padding、fused kernels、chunked prefill，reward 为 `0.3 format + 0.3 step + 0.4 success`，默认 `score_scale=2.0`，judge 默认关闭。

## 关键结果

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
- V5 RL 已经能跑，当前主要问题转为 checkpoint、稳定性和训练速度。

## 当前优先级

1. 等 answered-full SFT 的 OVO 1/8 回评完成并写入正式分数。
2. 用最新 fused/chunked GRPO 入口从 answered-full SFT 模型启动 RL。
3. 确认新 run 的日志出现 `note_frequency_score`、`judge_score`、`step_score` 和 `trajectory_score`。
4. 确保 `save_freq=30` 和 `resume_mode=auto` 能在中断后恢复。
5. 优化训练侧瓶颈：`old_log_prob` 和 `update_actor`。
6. 产出首个完整 RL checkpoint 后，再做 OVO 小规模回评。

## 主要路径

- V5 当前仓库：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5`
- V5 RL 输出：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/RL/outputs`
- OVO RL 数据运行路径：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/ovo`
- OVO RL 历史来源：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/ovo`
- V4 历史仓库：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4`
- V3 历史仓库：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v3`
- 8B base instruct：`/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct`
- 当前 RL 起点模型：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm`
