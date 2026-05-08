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
- 当前脚本：`RL/scripts/train_grpo_ovo_vllm_qwen3vl8b_full_8gpu_lt120s.sh`
- 当前数据：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/ovo/ovo_rl_lt120s.json`
- 当前模型起点：`/mmu_mllm_hdd/zhouhanshu/test/exp3/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_v2_3077`

最新状态：GRPO 链路已经跑通，但最近一次训练在 `39/73` 后非正常中断，没有 checkpoint。下一步不是重新怀疑数据链路，而是先修保存频率，再优化训练侧性能。

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

1. 把 GRPO 脚本的 `save_freq` 从 `100` 改成 `5` 或 `10`。
2. 确认继续从 SFT checkpoint 开始 RL，还是切回 base instruct。
3. 优化训练侧瓶颈：`old_log_prob` 和 `update_actor`。
4. 在可恢复 checkpoint 的前提下重跑 `<120s` OVO RL 子集。
5. 产出首个完整 RL checkpoint 后，再做 OVO 小规模回评。

## 主要路径

- V5 当前仓库：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5`
- V5 RL 输出：`/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/RL/outputs`
- OVO RL 数据：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/ovo`
- V4 历史仓库：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4`
- V3 历史仓库：`/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v3`
- 8B base instruct：`/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct`
- 当前 SFT checkpoint：`/mmu_mllm_hdd/zhouhanshu/test/exp3/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_v2_3077`
