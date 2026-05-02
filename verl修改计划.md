# 在 verl 中实现 StreamWeave PPO/GRPO 训练的修改计划

本文档替换旧方案。旧方案里提到的 `main_ppo_sync.py` 不适用于当前本地
`/Users/zhs/learn/paper_reading/video/research/verl` checkout：我已经检查过，当前
verl 没有这个入口。后续应基于 `verl.trainer.main_ppo` 和
`verl.experimental.agent_loop` 做最小改动。

## 1. 当前结论

StreamWeave 的多轮形式不是普通 chat transcript，也不是典型 tool-agent 把历史
prompt/action 全部拼回上下文的形式。它更像一个有状态环境：

```text
state_t = memory_t + qa_history_t + current_frames_t
prompt_t = render(state_t)
action_t = model(prompt_t) -> XML: eta / answer / bridge / note
state_{t+1} = env.step(action_t)
prompt_{t+1} = render(state_{t+1})
```

也就是说，每一步的 prompt 都是由当前环境状态重新渲染出来的 fresh prompt。
上一轮输出不会作为完整 transcript 拼进下一轮；它只通过 `note`、`bridge` 和
QA 历史更新环境状态。

当前 verl 的 `rollout.multi_turn` / `tool_agent_loop` 可以做多轮交互，但它默认更偏
transcript/tool-call 风格，并且 `AgentLoopOutput` 当前主要假设一个输入 prompt 输出
一条 trajectory。StreamWeave 需要一个输入 QA 经过一次 session 产生多条 step row：

```text
1 条 QA 样本
  -> n 条 session rollout
  -> 每条 session 有 T 个 step
  -> 最终进入训练的是 n * T 条 step-level rows
```

例如 20s 视频、每 5s 一步，则 `T=4`；每条样本采样 `n=8` 条 session，最终是
`8 * 4 = 32` 条 step row。

## 2. 必须支持的四种训练模式

同一套 rollout 数据应支持四种算法，避免为每个算法写一套环境逻辑。

| 模式 | 用途 | 优势/回报分配 |
|---|---|---|
| `streamweave_traj_ppo` | PPO baseline，优先实现 | 每条 session 一个总分 `R_i`；每个 step 的 critic 预测 `V_{i,t}`；`A_{i,t}=R_i-V_{i,t}`；actor token 上广播该 step 优势 |
| `streamweave_traj_grpo` | GRPO baseline | 每条 session 一个总分 `R_i`；同一 QA 的 n 条 session 组内归一化；得到 `A_i` 后广播到该 session 所有 step |
| `streamweave_step_ppo` | 细粒度 PPO | 每个 step 有即时奖励 `r_{i,t}`；按 session 做 GAE 或 discounted return；每步优势可以不同 |
| `streamweave_step_grpo` | 细粒度 GRPO | 每个 step 计算 return `G_{i,t}`；同一 QA、同一 step_idx 的 n 条 session 组内归一化 |

优先级：先做 `streamweave_traj_ppo`，因为用户明确要 PPO baseline；其次做
`streamweave_traj_grpo`；最后再做两个细粒度版本。

## 3. Baseline 的奖励和优势定义

### 3.1 PPO baseline: `streamweave_traj_ppo`

一条 session 的总分：

```text
R_i = sum_t step_reward_{i,t} + terminal_success_i
```

其中 `step_reward` 主要来自格式和过程诊断，例如：

- XML 字段完整性：`eta` / `answer` / `bridge` / `note`
- `note` 时间范围是否合法
- `bridge` 时间范围是否合法
- open-tail bridge 是否和下一步视频窗口匹配
- 是否产生不可解析或明显破坏 memory update 的输出

`terminal_success` 是整条任务最终成功率或最终 QA 评价。

对 baseline PPO，不做 step credit assignment，先采用粗粒度 trajectory return：

```text
return_{i,t} = R_i
V_{i,t} = critic(prompt_{i,t}, response_{i,t}) 在 value_mask 位置的标量
A_{i,t} = return_{i,t} - V_{i,t}
```

然后将 `A_{i,t}` 广播到第 `t` 步 response 的所有有效 token 上。critic loss 只在
`value_mask` 指定的位置计算，通常取该步 response 的最后一个有效 token。

这不是 GRPO 组内归一化。PPO baseline 可以在 batch 内做 advantage whitening，但
语义上它仍然是 critic baseline，而不是同一 QA 内 n 条 rollout 的相对排序。

### 3.2 GRPO baseline: `streamweave_traj_grpo`

对同一条 QA 的 n 条 session，先计算每条 session 的总分：

```text
R_i = sum_t step_reward_{i,t} + terminal_success_i
```

再做组内归一化：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
```

然后将 `A_i` 直接广播到该 session 的所有 step、所有 response token。

因此，在 baseline GRPO 中，同一条 session 内每一步优势值相同。每一步打分的意义
是组成更稳定的 session 总分，而不是直接给每一步不同优势。后续细粒度版本才会让
不同 step 拿到不同优势。

### 3.3 细粒度 PPO: `streamweave_step_ppo`

每一步有即时奖励 `r_{i,t}`，终局成功奖励加到最后一步或作为 terminal value：

```text
G_{i,t} = r_{i,t} + gamma * r_{i,t+1} + ... + terminal_success_i
A_{i,t} = GAE(r_{i,t}, V_{i,t}, V_{i,t+1})
```

actor 的 token advantage 仍然是按 step 广播；critic loss 仍然只在 `value_mask`
位置计算。

### 3.4 细粒度 GRPO: `streamweave_step_grpo`

每条 session 内先算每个 step 的 return：

```text
G_{i,t} = r_{i,t} + gamma * r_{i,t+1} + ... + terminal_success_i
```

然后在同一 QA、同一 `step_idx=t` 的 n 条 session 内归一化：

```text
A_{i,t} = normalize({G_{j,t} | j in same QA group})
```

这样可以让第 1 步和第 3 步的优势不同，同时避免不同时间步之间奖励尺度混在一起。

## 4. 统一 rollout 数据结构

StreamWeaveAgentLoop 输出的每个 step row 至少需要携带这些字段：

```text
sample_uid          # 原始 QA 样本 id
session_id          # 一条完整 rollout session id
rollout_idx         # 同一 QA 下第几条采样，0..n-1
step_idx            # 当前 session 内第几步，0..T-1
session_length      # T
done                # 是否最后一步

prompt_ids
response_ids
attention_mask
position_ids
response_mask       # actor loss / logprob 的 token mask
value_mask          # critic value loss 的位置 mask，通常每个 step 只有 1 个 token
multi_modal_data

raw_output          # 模型原始 XML 输出
parsed_action       # 解析后的 eta / answer / bridge / note
memory_before
memory_after
qa_history_before
qa_history_after

step_reward
step_reward_breakdown
terminal_success
trajectory_score    # R_i，session 结束后回填到该 session 所有 step
step_return          # 细粒度算法使用，baseline 可为空或等于 trajectory_score
```

其中 `response_mask` 和 `value_mask` 必须分开：

- actor 要在整段 response token 上算 PPO/GRPO loss；
- critic 只需要在每个环境 step 的一个标量位置预测 value；
- 如果直接用 `response_mask` 做 critic value loss，会把一个 step 的同一个回报重复压到所有 token 上，既浪费也容易改变 loss 尺度。

## 5. verl 侧需要改哪些地方

### 5.1 `verl/experimental/agent_loop/agent_loop.py`

目标：让一个 AgentLoop 调用可以返回多条 step-level `AgentLoopOutput`。

当前逻辑更接近：

```text
one input prompt -> one AgentLoopOutput
```

需要改成兼容：

```text
one input prompt -> AgentLoopOutput 或 list[AgentLoopOutput]
```

具体改动：

1. 给 `AgentLoopOutput` 增加可选字段 `value_mask`。
2. StreamWeaveAgentLoop 内部跑完整个 session，但每个环境 step 构造一个
   `AgentLoopOutput`。
3. `_run_agent_loop` 接收单个 output 或 list outputs。
4. `_postprocess` flatten 所有 outputs，再 pad/concat 成 DataProto。
5. 将 `session_id`、`rollout_idx`、`step_idx`、`sample_uid`、`trajectory_score`、
   `step_reward`、`terminal_success` 等写入 `non_tensor_batch` 或 `extra_fields`。
6. 保留一个 `source_row_idx`，用于把原始 repeated batch 的 metadata 对齐到展开后的
   `B * n * T` rows。

这一步是最核心的 verl 改动，因为 StreamWeave 的训练单位不是一个完整 transcript，
而是一个 session 里的多个 fresh-prompt step。

### 5.2 `verl/trainer/ppo/ray_trainer.py`

目标：让 trainer 接受 rollout 后长度从 `B * n` 变成 `B * n * T`。

具体改动：

1. 在 repeat 原始 batch 时生成稳定的 `sample_uid` 和 `rollout_idx`。
2. rollout 返回后，不再假设 generated batch 长度一定等于 `B * n`。
3. 使用 `source_row_idx` 或 `sample_uid/session_id` 对齐原始输入 metadata。
4. 调用 advantage 计算时，把 `non_tensor_batch` 里的 StreamWeave metadata 传给
   custom estimator。
5. PPO baseline 配置必须开启 critic：

```yaml
algorithm:
  adv_estimator: streamweave_traj_ppo
critic:
  enable: true
```

verl 里 `need_critic` 对非 GAE estimator 默认不一定打开 critic，所以这里不能依赖
默认行为。

### 5.3 `verl/trainer/ppo/core_algos.py`

新增四个 advantage estimator：

```text
streamweave_traj_ppo
streamweave_traj_grpo
streamweave_step_ppo
streamweave_step_grpo
```

实现要点：

- `streamweave_traj_ppo`
  - 按 `session_id` 读取 `trajectory_score`
  - 每个 step 的 return 都设为该 session 的 `trajectory_score`
  - 从 critic values 的 `value_mask` 位置取 `V_{i,t}`
  - 算 `advantages = returns - values`
  - 将 step advantage 扩展到 response token 维度

- `streamweave_traj_grpo`
  - 按 `sample_uid` 分组
  - 每组内用 n 条 session 的 `trajectory_score` 做归一化
  - 同一 session 的所有 step 共享同一个 advantage
  - 扩展到 response token 维度

- `streamweave_step_ppo`
  - 按 `session_id` 和 `step_idx` 排序
  - 用 `step_reward`、`terminal_success`、`done` 和 `values` 做 GAE
  - 每个 step 产生一个 advantage

- `streamweave_step_grpo`
  - 先按 session 算每个 step 的 discounted return
  - 再按 `(sample_uid, step_idx)` 做组内归一化
  - 每个 step 产生一个 advantage

注意：verl 原生的 `compute_grpo_outcome_advantage` 是 row-level group by `uid`，不能
直接表达 baseline GRPO 的 “session 总分归一化后广播到多个 step rows”，所以这里
应新增 StreamWeave 专用 estimator，而不是强行复用原函数。

### 5.4 `verl/workers/critic/dp_critic.py`

目标：critic 支持 step-level `value_mask`。

具体改动：

1. 如果 batch 中存在 `value_mask`，critic forward 和 loss 都使用 `value_mask`；
2. 如果不存在，则保持原来的 `response_mask` 行为，避免破坏其他 recipe；
3. `compute_values` 返回的 `values` 仍是 token-level tensor，但只有 `value_mask`
   位置参与 loss；
4. PPO estimator 从 `value_mask` 位置抽取每个 step 的 scalar value。

### 5.5 `verl/workers/utils/losses.py`

目标：value loss 支持传入 `value_mask`。

具体改动：

1. `compute_value_loss` 增加可选参数或在调用侧把 `value_mask` 作为 mask 传入；
2. 保持原有 `response_mask` 路径兼容；
3. loss 归一化应按 `value_mask.sum()`，而不是 response token 数。

### 5.6 `verl/utils/model.py`

PPO baseline 如果 critic 使用 Qwen2.5-VL / Qwen3-VL 这类 VLM，需要保留/移植三个补丁：

1. ValueHead forward 支持 VLM 的 multimodal kwargs，例如 `pixel_values`、
   `image_grid_thw`、`video_grid_thw` 等；
2. 当顶层 config 没有 `hidden_size` 时，设置：

```python
config.hidden_size = config.text_config.hidden_size
```

3. critic loss 使用 `value_mask`，而不是默认 `response_mask`。

如果 critic 换成纯文本模型，前两个 VLM 补丁可以暂时不启用；但要跑 VLM PPO baseline，
这两个补丁必须保留。

### 5.7 新增 `verl/recipe/streamweave/`

建议新增一个独立 recipe，避免污染现有 examples：

```text
verl/recipe/streamweave/
  streamweave_agent_loop.py
  agent_loop_config.yaml
  prepare_data.py
  run_ppo_baseline.sh
  run_grpo_baseline.sh
  run_step_ppo.sh
  run_step_grpo.sh
```

其中 baseline PPO 先跑通：

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=streamweave_traj_ppo \
  critic.enable=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.agent.default_agent_loop=streamweave_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=/path/to/verl/recipe/streamweave/agent_loop_config.yaml \
  reward_model.enable=False \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.use_kl_loss=True
```

奖励在 StreamWeaveAgentLoop 内部完成并写入 rollout metadata，不需要先接 verl 原生
reward model。KL 可以按 verl PPO 原有 actor/ref 路径处理。

## 6. streamweave_v3 侧需要改哪些地方

### 6.1 新增 `streamweave/rl_env.py`

目标：把当前 `RolloutRunner` 拆成可被 verl agent loop 调用的 step-wise 环境。

建议接口：

```python
class StreamWeaveRLEnv:
    def reset(self, sample: dict) -> StepState: ...
    def build_prompt(self) -> list[ContentItem]: ...
    def step(self, raw_output: str) -> StepResult: ...
    def is_done(self) -> bool: ...
    def finalize(self) -> TrajectoryResult: ...
```

职责：

1. 初始化 video、QA、memory 和时间窗口；
2. 每一步用当前 memory、QA history 和 current frames 构造 fresh prompt；
3. 接收模型 raw XML 输出；
4. 用 raw output 计算格式和过程诊断；
5. 用 postprocess 后的 action 更新 memory 和 QA history；
6. session 结束后产出所有 step 的 rewards 和 trajectory score。

注意：reward 应基于 raw output 的质量诊断；环境状态更新可以使用 repaired action，
否则无效输出会让后续 step 无法继续。

### 6.2 新增 `streamweave/reward.py`

目标：集中定义 baseline 和细粒度算法共享的奖励。

建议输出：

```python
@dataclass
class StepReward:
    score: float
    breakdown: dict[str, float]

@dataclass
class TrajectoryReward:
    trajectory_score: float
    terminal_success: float
    step_rewards: list[StepReward]
    step_returns: list[float]
```

第一版只需要稳定支持：

```text
trajectory_score = sum(step_reward) + terminal_success
```

后续细粒度 PPO/GRPO 再打开 `step_returns`。

### 6.3 可选新增 `streamweave/verl_adapter.py`

目标：把 StreamWeave 的 `ContentItem`、图片帧、视频帧和 prompt 转成 verl/Qwen-VL
需要的数据结构。

职责：

1. `ContentItem` -> chat message；
2. frame paths / PIL / tensor -> `multi_modal_data`；
3. 生成 tokenizer/processor 所需的 `prompt_ids` 和 multimodal kwargs；
4. 记录 `frame_window`、`step_idx` 等 metadata。

如果 recipe 里直接实现转换也可以，但单独放 adapter 更容易测试。

### 6.4 测试

新增最少测试：

```text
tests/test_rl_env.py
tests/test_reward.py
tests/test_verl_masks.py
```

验收点：

1. 20s 视频、5s step 能产生 4 个 fresh prompts；
2. 每一步 prompt 中的 memory 是上一步 action 更新后的结果；
3. n=8 时，verl rollout 展开后是 32 条 step rows；
4. baseline GRPO 中，同一 session 的所有 step advantage 相同；
5. baseline PPO 中，critic loss 只在 `value_mask` 位置计算；
6. `value_mask.sum(dim=-1)==1`；
7. raw output 格式错误会影响 reward，但环境仍能通过 repaired action 尽量继续。

## 7. Baseline PPO 的完整数据流

以 20s 视频、每 5s 一步、`n=8` 为例：

```text
原始 QA 样本
  |
  | verl dataloader
  v
1 条 batch row
  |
  | actor_rollout_ref.rollout.n=8
  v
8 条 repeated rollout seeds
  |
  | StreamWeaveAgentLoop，每条 seed 跑完整 session
  v
每条 session 产生 4 个 step:
  step 0: prompt_0(memory_0 + frames_0) -> response_0 -> update memory_1
  step 1: prompt_1(memory_1 + frames_1) -> response_1 -> update memory_2
  step 2: prompt_2(memory_2 + frames_2) -> response_2 -> update memory_3
  step 3: prompt_3(memory_3 + frames_3) -> response_3 -> final answer
  |
  v
8 * 4 = 32 条 step rows
  |
  | reward.py
  v
每条 session 得到一个 trajectory_score R_i
  |
  | streamweave_traj_ppo
  v
每个 step:
  return_{i,t}=R_i
  advantage_{i,t}=R_i - V_{i,t}
  |
  | broadcast to response_mask tokens
  v
actor PPO loss + critic value loss(value_mask only)
```

这个流程里，每个 step 的 prompt 都是新渲染的，不拼接旧 prompt 和旧 response。

## 8. 为什么这样改动最小

1. 不重写 verl trainer，只扩展 agent_loop 输出从 1 条变多条；
2. 不把 StreamWeave 强行改成 chat transcript；
3. 训练仍复用 verl 的 PPO/GRPO actor、reference logprob、KL、critic、FSDP/Ray
   worker；
4. StreamWeave 自己只新增 step-wise env 和 reward，不破坏当前推理/评测路径；
5. baseline PPO、baseline GRPO、细粒度 PPO、细粒度 GRPO 共用一套 rollout 数据。

## 9. 实施顺序

### 阶段 1：先改 streamweave_v3，本地单元测试跑通

1. 新增 `streamweave/rl_env.py`；
2. 新增 `streamweave/reward.py`；
3. 必要时新增 `streamweave/verl_adapter.py`；
4. 测试 20s/5s 的 4 step fresh prompt 流程；
5. 测试 raw output reward 和 repaired action state update 分离。

### 阶段 2：改 verl 的 rollout 展开

1. 修改 `AgentLoopOutput`，支持 `value_mask`；
2. 修改 `_run_agent_loop` 和 `_postprocess`，支持 list outputs；
3. 新增 `verl/recipe/streamweave/streamweave_agent_loop.py`；
4. 验证 1 条 QA、n=8、T=4 能展开成 32 rows。

### 阶段 3：打通 PPO baseline

1. 修改 critic 使用 `value_mask`；
2. 移植 VLM critic 的 ValueHead 和 hidden_size 补丁；
3. 新增 `streamweave_traj_ppo` estimator；
4. 跑最小 batch smoke test，确认 actor/critic loss 都能反传。

### 阶段 4：打通 GRPO baseline

1. 新增 `streamweave_traj_grpo` estimator；
2. 检查同一 QA 下 n 条 session 的组内归一化；
3. 检查同一 session 内所有 step advantage 完全一致。

### 阶段 5：再做细粒度版本

1. 新增 step return / GAE；
2. 实现 `streamweave_step_ppo`；
3. 实现 `streamweave_step_grpo`；
4. 对比 baseline 和细粒度 credit assignment 的训练曲线。

## 10. 最终验收标准

必须满足：

1. StreamWeave 多轮是 fresh prompt 环境 rollout，不是 transcript 拼接；
2. 20s 视频、5s step、n=8 时，训练 batch 中有 32 条 step rows；
3. baseline PPO 可以跑，且 critic 开启；
4. baseline PPO 的 critic loss 只在 `value_mask` 上计算；
5. baseline PPO 每个 step 的 return 等于同一 session 的 `trajectory_score`；
6. baseline GRPO 每条 session 一个优势值，并广播到该 session 所有 step；
7. 细粒度 PPO/GRPO 的接口预留好，后续可以让不同 step 获得不同 advantage；
8. Qwen2.5-VL / Qwen3-VL 作为 critic 时，ValueHead forward 和 hidden_size 不报错；
9. 不改动 StreamWeave 当前非 RL 推理/评测主路径。

