# 第五部分：RL 训练

- 状态：未开始；当前先完成 V4 SFT 数据合成、第一次 SFT 和评测，随后开始搭建 RL 框架
- 目标：用 GRPO 联合优化格式正确性、回答正确性、时机选择与视觉保留成本
- 初始化模型：第一轮 SFT 模型冷启动，后续再决定是否加入 RAFT
- 训练数据：优先使用与 SFT 不重叠的新数据集或 heldout QA，避免直接在训练集上做 RL

## Reward 设计

### 总公式

```
R_traj =
  gate_format * (
      w_time  * mean_t(R_time_valid_t)
    + w_eta   * mean_t(R_eta_t)
    + w_ans   * R_answer_correct
    + w_mem   * mean_t(R_memory_t)
  )
```

### 各项说明

**`gate_format`（硬约束）**
- 任意 step 无法解析或缺少必要 XML 字段，整条 rollout 记 0 分。

**`R_time_valid_t`（0/1，step 级）**
- 检查当前 step 所有时间约束：
  - `note frame` 引用合法（属于当前 group）
  - `note t` 与 frame 真实时间匹配
  - `bridge` 绝对时间区间合法
  - 普通 bridge 落在当前 step 范围内
  - open-tail bridge 继承合法

**`R_eta_t`（step 级，连续）**
- 根据绝对时间戳 `eta_pred` 与 `eta_target` 的偏差计算：
  - 尚未可答时：`eta_target = t*`
  - 已可答并回答时：`eta_target = current_window_end`
- 6s 以内不惩罚；超过 6s 后随偏差增大线性惩罚；预测方向与真值方向相反且偏差较大时加重惩罚。

**`R_answer_correct`（0/1，trajectory 级）**
- 由 judge 判断最终答案与标准答案语义一致性。
- 同一条 rollout 的所有 step 共享同一个值。
- judge 只接收 question、reference answer 和 model answer，结果缓存。

**`R_memory_t`（step 级，连续）**
- 核心项：在语义保真的前提下最小化 memory token 数量。

```
R_memory_t = w_sem * SemSim(frames_t, memory_t) - lambda_tok * TokenCount(memory_t)
```

- `SemSim(frames_t, memory_t)`：当前 step 原始图片序列与转换后图文序列（bridge + 有效 note）之间的语义相似度，取值 [0, 1]。打分器使用多模态模型（**待定**，初步考虑已有 MLLM 如 Qwen3-VL-32B-Instruct），输入原始帧图片和图文 memory，输出语义保真度评分。
- `TokenCount(memory_t)`：当前 step memory 的 token 数量（bridge token + note token），归一化到 [0, 1]。
- `lambda_tok`：token 惩罚系数，作为实验超参控制语义与压缩之间的 trade-off。

### GRPO 配置

- 每条 QA 采样 8 条完整 rollout，组内相对优化。
- 加 KL 约束到 SFT+RAFT policy，防止格式和语言风格漂移。
- step 级指标先聚合成 trajectory score，再在同一 QA 的 8 条 rollout 内归一化。

## 关键指标

- `parser_valid_rate`
- `time_valid_rate`
- `answer_accuracy`
- `ETA MAE`
- `note_rate`
- `avg_bridge_length`

## 风险

- reward hacking：每轮训练后必须跑 heldout closed-loop eval，格式稳定性下降时先降学习率、增大 KL 或停止 RL。
- 稀疏性正则过强导致关键信息丢失。
- 稀疏性正则过弱退化为全保留。

## 下一步

先完成当前 SFT 数据合成、第一次 SFT 和评测，再实现 `reward.py`、`rollout_env.py`、`collect_rl_rollouts.py` 和训练脚本。

## 2026-05-02 当前衔接计划

RL 暂时不是当前阻塞项，但需要按 V4 SFT 已经固定下来的协议提前对齐：

- rollout env 必须复用 V4 的 XML parser、quality validator、FrameStore 和 production prompt。
- reward 里格式分应该包含 bridge gap 完整性、note/frame 合法性、时间边界、open-tail 继承和 QA eta/answer 状态。
- answer reward 需要和当前样本级 accepted 逻辑一致，避免把“是否回答”和“是否答对”混成同一个 step 级 retry 目标。
- memory reward 第一版先统计 note 数、bridge token 数、long bridge 和 open-tail bridge，再考虑引入语义相似度 evaluator。
- 训练前必须先有 SFT 模型作为格式稳定的初始化，否则 RL 很容易退化成格式错误采样。
