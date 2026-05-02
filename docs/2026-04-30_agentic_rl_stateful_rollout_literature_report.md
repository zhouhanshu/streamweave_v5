# Stateful Agent RL 与非拼接多步 rollout 文献报告

日期：2026-04-30

写作目标：

- 只讨论和 StreamWeave 当前训练问题直接相关的 agent RL 文献。
- 重点区分两类多轮训练：一种是把完整对话历史拼成长序列再做 mask；另一种是环境状态被动作更新后，每一步重新生成 observation/prompt，并把每步视作独立 transition。
- 结论服务于后续在 `verl` 中实现 StreamWeave RL：每步 fresh prompt、非拼接训练、step-aware credit assignment。

## 1. StreamWeave 对应的 RL 问题形式

StreamWeave 的 rollout 不是普通 chat transcript，也不是简单把上一轮 prompt 和 output 拼到下一轮输入。每一步的输入是由环境状态重新渲染出来的：

```text
s_t:
  video_progress_t
  memory_t
  qa_history_t
  current_frames_t

prompt_t = render(s_t):
  memory_t + qa_history_t + current_frames_t

a_t = model(prompt_t):
  <eta>...</eta>
  <answer>...</answer>
  <bridge t="...">...</bridge>
  <note t="..." frame="..."/>

s_{t+1} = env.step(s_t, a_t):
  apply bridge/note to memory
  append answer to qa_history when non-empty
  move to next video window
```

这更接近一个 partially observable Markov decision process，而不是多轮聊天记录。训练样本应该是 step-level transition：

```text
(prompt_t, action_t, reward_t, next_state_info)
```

下一步的 `prompt_{t+1}` 可以包含上一轮动作对环境状态产生的影响，但不应该把 `prompt_t + action_t` 原样拼成长上下文再训练。

## 2. 直接相关文献

### 2.1 Agent Lightning: Train ANY AI Agents with Reinforcement Learning

链接：

- arXiv: https://arxiv.org/abs/2508.03680
- HuggingFace paper page: https://huggingface.co/papers/2508.03680

核心相关点：

- 论文明确批评了传统 agent RL 中“sequence concatenation with masking”的做法。
- 它把 agent execution 建模为 MDP，并定义统一的数据接口，把复杂 agent trajectory 分解成 training transitions。
- LightningRL 中包含 credit assignment module，用来把整条轨迹的结果分配到各个 LLM invocation。

对 StreamWeave 的启发：

- 我们的每个视频 step 本质上就是一次 LLM invocation。
- `memory + qa_history + current_frames` 是当前 state 的 observation，而不是历史 transcript。
- 因此不应该用“拼接全轨迹 + response mask”的方案作为主实现，而应该记录每步 transition，并在完整 session 结束后做 credit assignment。

### 2.2 VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents

链接：

- arXiv: https://arxiv.org/abs/2510.16907
- GitHub: https://github.com/mll-lab-nu/VAGEN
- Project page: https://vagen-ai.github.io/

核心相关点：

- VAGEN 面向 VLM agent，多步任务被建模为 POMDP。
- 每一步 observation 是当前视觉状态和文本 prompt，动作改变环境，环境再返回新的 observation。
- 论文强调 partial observability 下需要显式 state estimation 和 transition modeling。
- 它设计了 dense turn-level supervision，并提出 turn-aware credit assignment。
- GitHub README 中直接给出两种训练方式：
  - `Multi-turn Concatenated Training`
  - `Multi-turn Non-Concatenated Training`
- 其中非拼接模式写明：each trajectory is split into multiple turn-level training instances，并提供配置：

```yaml
trainer:
  concat_multi_turn: False

algorithm:
  adv_estimator: no_concat_gae
```

对 StreamWeave 的启发：

- 这是目前最贴近我们需求的参考对象，因为它同样是 VLM、多步环境、当前 observation 驱动的 rollout。
- StreamWeave 的 `current_frames_t + memory_t` 可以看作 VAGEN 中的 visual observation 和 state belief。
- 我们应该优先参考 VAGEN 的 no-concat 训练思路，而不是普通 `verl` agent loop 的长 transcript 训练。

### 2.3 Reinforcement Learning for Long-Horizon Interactive LLM Agents / LOOP

链接：

- arXiv: https://arxiv.org/abs/2502.01600
- HuggingFace paper page: https://huggingface.co/papers/2502.01600

核心相关点：

- 论文研究 interactive digital agents，这类 agent 通过 API 与 stateful digital environments 交互。
- 训练被形式化为 POMDP。
- 目标环境 AppWorld 包含多个 app 和数据库状态，agent 每步执行 API 或代码会改变环境状态。
- LOOP 是一种 memory-efficient PPO variant，不使用 value network，只保留一份底层 LLM。

对 StreamWeave 的启发：

- 它证明“agent 动作改变环境状态，下一步根据新状态继续 rollout”的训练范式是成立的。
- 和 StreamWeave 类似，下一步输入不是简单历史拼接，而是环境状态经过渲染后的新 observation。
- 不过 LOOP 更偏 API/code agent，视觉 memory 和 note/bridge 这种显式记忆动作不是它的重点。

### 2.4 RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning

链接：

- arXiv: https://arxiv.org/abs/2504.20073

核心相关点：

- RAGEN 研究多轮 agent RL 中的 long-horizon decision making 和 stochastic environment feedback。
- 它提出 StarPO，即 State-Thinking-Actions-Reward Policy Optimization。
- 论文明确指出，如果没有 fine-grained、reasoning-aware reward，多轮 RL 容易学到 shallow strategies 或 hallucinated thoughts。

对 StreamWeave 的启发：

- 只用整条 session 的最终 answer reward 太粗，尤其无法有效训练 note/bridge 的中间决策。
- StreamWeave 已经有天然的 step-level reward 来源：格式、时间合法性、note 是否落在当前帧、bridge 是否忠实、memory token 成本、eta 是否合理。
- 这些 step-level reward 不应该只被平均进最终轨迹分，而应该参与每个 step 的 advantage。

### 2.5 WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning

链接：

- arXiv: https://arxiv.org/abs/2505.16421
- ACL Anthology: https://aclanthology.org/2025.emnlp-main.401/

核心相关点：

- WebAgent-R1 在真实 web-like 环境中做 online interaction。
- 每步 action 改变网页状态，后续 observation 来自新的页面状态。
- 它用 asynchronous trajectory rollout 和 multi-turn GRPO，从 online interaction 中学习。
- reward 主要是 task success 的 binary outcome reward。
- 实验显示，即使只有二值终局奖励，也能把小模型 web task success rate 明显提高。

对 StreamWeave 的启发：

- 终局 reward 可以作为 baseline work，但需要大量采样和较好的 warmup。
- WebAgent-R1 的成功不说明粗粒度 credit 一定足够，只说明在可验证 web task 上二值 outcome reward 有可能推动策略改善。
- StreamWeave 有更丰富的 step diagnostics，因此没有必要只依赖 binary final answer reward。

### 2.6 MobileRL: Online Agentic Reinforcement Learning for Mobile GUI Agents

链接：

- arXiv: https://arxiv.org/abs/2509.18119
- HuggingFace paper page: https://huggingface.co/papers/2509.18119
- GitHub: https://github.com/THUDM/MobileRL

核心相关点：

- MobileRL 面向 mobile GUI agent，当前屏幕状态会随着 agent action 改变。
- 它提出 Difficulty-Adaptive GRPO，即 ADAGRPO。
- 针对多步 agentic tasks，它引入 shortest-path reward adjustment，用任务长度重塑 reward。
- 还使用 difficulty-adaptive positive replay 和 failure curriculum filtering 来稳定训练。

对 StreamWeave 的启发：

- 多步 agent RL 里，采样效率和任务难度分布是实际问题。某些视频 QA 可能 8 条 rollout 都失败或都成功，GRPO 组内方差会很小。
- 可以在后续版本加入 reward variance filtering 或 difficulty-aware sampling。
- 但第一版不应过早引入 replay/curriculum，先把 no-concat rollout 和 step-aware advantage 做对。

### 2.7 GUI-Shepherd: Reliable Process Reward and Verification for Long-Sequence GUI Tasks

链接：

- arXiv: https://arxiv.org/abs/2509.23738
- HuggingFace paper page: https://huggingface.co/papers/2509.23738

核心相关点：

- GUI-Shepherd 明确指出 long-sequence GUI tasks 受 sparse rewards 和 credit assignment 困扰。
- 它训练 process reward model，给每一步 action 提供 dense feedback。
- 在 AndroidWorld online PPO 中，process reward 相比只用 outcome reward 有明显提升。

对 StreamWeave 的启发：

- 这支持我们把 raw action quality、memory quality、time validity 做成 step reward。
- 对 StreamWeave 来说，PRM 不一定第一版就要训练独立模型；可以先用规则和轻量 judge 组合：

```text
r_step_t =
  r_format_t
  + r_time_valid_t
  + r_eta_t
  + r_memory_t
  + r_bridge_grounded_t
  + r_note_valid_t
```

后续如果规则 reward 不够，可以再训练或接入 process reward model。

### 2.8 UI-TARS-2: Advancing GUI Agent with Multi-Turn Reinforcement Learning

链接：

- arXiv: https://arxiv.org/abs/2509.02544
- HuggingFace paper page: https://huggingface.co/papers/2509.02544

核心相关点：

- UI-TARS-2 是 GUI-centered agent，多轮 RL、环境稳定性、大规模 rollout 是它的核心问题。
- 技术报告强调 unified sandbox platform、multi-turn RL framework 和训练稳定性。
- 这类 GUI agent 任务和 StreamWeave 类似，都是“当前观测 + 动作 + 环境状态变化 + 新观测”的闭环。

对 StreamWeave 的启发：

- 工程上要把 rollout 环境封装清楚，保证可复现、可并行、可记录 trace。
- RL 训练不仅是 reward 公式，还依赖稳定的环境接口、失败处理、trace logging 和可验证评估。

### 2.9 AppWorld 与 WebShop：环境基准的早期参照

链接：

- AppWorld: https://arxiv.org/abs/2407.18901
- WebShop: https://arxiv.org/abs/2207.01206

核心相关点：

- AppWorld 提供 stateful app execution environment，任务成功通过 state-based unit tests 评估。
- WebShop 是早期 grounded language agent web navigation 环境，agent 需要多步浏览、搜索、选择商品。

对 StreamWeave 的启发：

- 这两篇不是我们要直接复现的训练算法，但它们说明 agent RL 的基础不是 transcript，而是 environment。
- 对 StreamWeave，`MemoryStore` 和视频窗口推进就是环境状态；`note/bridge/answer` 是动作；最终回答和中间 memory 质量是 reward。

## 3. 文献共识

这些工作给出的共识可以归纳为四点。

### 3.1 非拼接 rollout 是合理且必要的

在真实 agent 环境中，每一步 prompt/observation 往往由环境重新生成。上一轮输出会通过环境状态影响下一轮，但不应该被无脑拼成完整语言上下文。Agent Lightning 和 VAGEN 都明确支持 transition-level 或 no-concat training。

对应 StreamWeave：

```text
错误主方案：
  [prompt_0, action_0, prompt_1, action_1, ...] 拼成长序列
  然后 mask 掉 prompt 和环境 token

正确主方案：
  step 0: prompt_0 -> action_0
  step 1: prompt_1 -> action_1
  step 2: prompt_2 -> action_2
  ...
  每个 step 是独立训练样本
```

### 3.2 只用 trajectory-level advantage 可以 work，但通常不是最佳

WebAgent-R1 说明二值终局 reward 能推动 web agent 改进。MobileRL、GUI-Shepherd、RAGEN、VAGEN 则说明，对长序列 agent，稀疏 outcome reward 和粗 credit assignment 会带来训练不稳定、采样效率低和浅层策略。

对应 StreamWeave：

- `answer_correct` 是 trajectory-level reward。
- `note/bridge/eta/format/memory` 是 step-level reward。
- 如果把整条 session 的同一个 advantage 复制给每一步，能跑，但浪费了这些 step-level 信号。

### 3.3 Step-level reward 不等于只优化当前步

多步 agent 的关键是：当前 step 的动作会影响未来状态。比如错误 note 会污染 memory，导致后续 answer 错。合理做法不是只用 `r_t` 更新第 `t` 步，而是用从当前步开始的 return：

```text
G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + terminal_reward
```

这样，早期 note/bridge 会因为影响未来回答而获得 credit。

### 3.4 视觉 agent 更需要 state / memory 监督

VAGEN 的核心观点是 VLM agent 在部分可观测环境中需要 world model reasoning。StreamWeave 的 memory 正是显式 state belief。我们的 note/bridge 不只是输出格式，而是模型在构造后续可用状态。

因此奖励应当覆盖：

- 当前动作是否合法。
- 当前动作是否忠实当前窗口。
- 当前动作是否改善后续可答性。
- memory 是否足够短且保留关键证据。

## 4. 对 StreamWeave + verl 的实现建议

### 4.1 Rollout 数据结构

一条视频 QA 采样 `N=8` 个 session。每个 session 有 `T` 个 step，例如 20s 视频、5s 一步，则 `T=4`。

训练数据应保存为：

```text
sample_uid
session_id
step_idx
prompt_ids
response_ids
response_mask
raw_output
parsed_action
memory_before
memory_after
step_reward
terminal_reward
done
```

最终得到 `8 * T` 条 step-level training rows，而不是 8 条拼接轨迹。

### 4.2 Advantage 计算

推荐第一版主方案：

```text
r_step_{i,t} =
  w_format * r_format_{i,t}
  + w_time * r_time_valid_{i,t}
  + w_eta * r_eta_{i,t}
  + w_mem * r_memory_{i,t}
  + w_bridge * r_bridge_{i,t}
  + w_note * r_note_{i,t}

R_terminal_i =
  w_ans * answer_correct_i

G_{i,t} =
  r_step_{i,t}
  + gamma * r_step_{i,t+1}
  + ...
  + gamma^(T-t) * R_terminal_i
```

然后在同一条 QA 的 8 个 session 内，按相同 step index 做 group normalization：

```text
A_{i,t} =
  (G_{i,t} - mean_i(G_{i,t})) / (std_i(G_{i,t}) + eps)
```

最后把 `A_{i,t}` 赋给该 step response token：

```text
advantages[row_tokens] = A_{i,t} * response_mask
```

这个方案比 session-level broadcast 更合理，因为：

- `step 0` 的动作会看到完整未来 return。
- 同一时间步的 8 个 session 做相对比较，符合 GRPO 的 group baseline 思路。
- 每一步的本地 reward 真正影响对应 step 的更新。

### 4.3 和 VAGEN no-concat 的对应关系

VAGEN 的 no-concat 训练可以作为我们最直接的参考：

```text
VAGEN:
  observation_t = current visual state + prompt
  action_t = VLM output
  env.step(action_t) -> observation_{t+1}
  no-concat training instance per turn

StreamWeave:
  observation_t = memory_t + qa_history_t + current_frames_t
  action_t = note/bridge/eta/answer XML
  env.step(action_t) -> memory_{t+1}, qa_history_{t+1}, next_frames
  no-concat training instance per step
```

区别是：VAGEN 更多面向游戏、导航、SVG、ManiSkill 等环境；StreamWeave 的环境状态主要是视频时间推进和显式 memory store。

### 4.4 在 verl 中的落点

基于此前对 `verl` 的阅读，优先路径应该是：

- 使用支持 multi-output / 多 trajectory 的同步 PPO/GRPO 路径，而不是默认把 agent loop 输出当单条连续序列。
- 每个 session 返回 `list[AgentLoopOutput]`，每个 step 一个输出。
- 修正或扩展 advantage 计算逻辑：
  - 当前默认逻辑若把 final trajectory advantage broadcast 到所有 step，只能作为 baseline。
  - 主方案应新增 `streamweave_no_concat_return` 或类似 estimator。
- 组内归一化 key 应该是：

```text
(sample_uid, step_idx)
```

而不是只按：

```text
sample_uid
```

或：

```text
(sample_uid, session_id)
```

这样可以让同一条 QA、同一视频时间步上的 8 个动作互相比。

## 5. 推荐路线

### 第一阶段：可跑通且概念正确

- 不拼接 transcript。
- 每个 step 单独生成 `AgentLoopOutput`。
- 完整 session 结束后计算 `r_step` 和 `R_terminal`。
- 用 discounted return 得到 `G_{i,t}`。
- 同一 QA、同一 step index 的 8 个 session 做归一化 advantage。

### 第二阶段：提高 credit assignment 精度

- 将 `r_memory` 拆成 semantic utility 和 token cost。
- 对 bridge 引入 groundedness judge。
- 对 note 引入 frame/evidence relevance score。
- 对 `eta` 区分过早、过晚、不可答等待是否合理。

### 第三阶段：提高采样效率和稳定性

参考 MobileRL / WebAgent-R1 / RAGEN：

- reward variance filtering：丢弃 8 个 rollout 分数全一样的组，或降低权重。
- difficulty-aware sampling：对过难或过易样本调采样频率。
- failure filtering：严重格式错误导致环境无法推进时，给强负 reward 并截断或补齐 mask。
- positive replay：等第一版稳定后再考虑，不作为初版修改。

## 6. 结论

当前文献支持我们把 StreamWeave RL 定义成 stateful environment RL，而不是普通多轮 chat RL。最直接的参考是 Agent Lightning 的 transition-level agent RL 和 VAGEN 的 non-concatenated multi-turn VLM training。

因此 StreamWeave 在 `verl` 中的主设计应当是：

```text
fresh prompt per step
non-concatenated step-level training rows
full-session rollout before reward computation
discounted return per step
same-sample same-step group normalization
step response tokens receive step-specific advantage
```

粗粒度 session advantage broadcast 可以保留为 ablation baseline，但不应作为主方案。

