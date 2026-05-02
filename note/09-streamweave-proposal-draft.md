# StreamWeave 提案整理版

- 状态：`draft`
- 日期：`2026-04-18`
- 主 benchmark：`OVO-Bench (Backward Tracing) + StreamingBench (60s Main)`
- 当前定位：`accuracy-first v1`

---

## 1. 问题与动机

### 1.1 场景定义

- 目标是在线流式视频理解。
- 用户可能在视频中的任意时间提出问题。
- 答案可能出现在提问之前、提问时或提问之后，因此模型必须判断回答时机。

benchmark 作用：

- `OVO-Bench`：要求模型根据时间位置判断应依赖过去、现在，还是等待未来证据。
- `StreamingBench`：要求时机判断、在线响应和连续交互能力。

核心约束：

- 只看最近窗口不够。
- 必须加入显式记忆机制。
- 记忆又不能让上下文无限膨胀，拖慢推理速度。

### 1.2 现有方法的根本矛盾

| 方法类型 | 代表工作 | 根本缺陷 |
| --- | --- | --- |
| 纯文本记忆 | VST, ThinkStream | 视觉信息一旦文本化不可逆，OCR、颜色、空间关系、对象身份会丢失 |
| 全帧保留 | SimpleStream, 滑窗 | 只能保最近帧，real-time 方面强，但没有长期压缩记忆 |
| 只解决 timing | MMDuet2, StreamReady | 没有解决历史视觉压缩问题 |
| 均匀 KV 压缩 | HERMES, ReKV | 压缩不区分关键视觉证据和过渡内容 |

### 1.3 核心洞察：视频编解码类比

- `I-frame`：完整保留关键状态
- `P-frame`：只保留相对上一关键帧的变化

迁移到 StreamWeave：

- `note = I-frame`：完整保留当前视觉状态
- `bridge = P-frame`：只描述相对上一个关键状态的变化

目标不是把历史全部压成文本，而是让模型学会区分：

- 哪些视觉证据必须长期保留为视觉状态
- 哪些过程内容可以压成短桥接文本

### 1.4 核心主张

**模型应该逐步学会区分“视觉上不可替代的证据”和“可被桥接文本取代的过渡”，同时只在当前前缀证据已经充分时才对外回答。**

- `state` 决策关注 `future utility / compression loss`
- `response` 决策关注 `prefix answerability / timing`

---

## 2. 核心方法

### 2.1 双通道决策

每个新 `chunk` 到来时，模型并行做两个决策：

- 状态决策：`note / bridge`
- 响应决策：`silent / answer`

说明：

- `note`：当前视觉状态永久保留，同时附极短内部标注
- `bridge`：不保留视觉状态，只写相对上一个 `active note` 的短变化文本
- `silent`：当前证据不足，继续等待
- `answer`：当前证据已足够，输出回答

两个决策并行、不互斥。

### 2.2 teacher 拆分，学生统一

数据构造阶段拆成两条 teacher：

- `state teacher`：判断当前 `chunk` 应该成为 `note` 还是 `bridge`
- `response teacher`：判断当前前缀应保持 `silent` 还是输出 `answer`

最终学生模型仍然是统一策略：

```text
f_theta(x_t) -> (a_t^state, a_t^resp, y_t^state, y_t^resp)
```

好处：

- 数据构造更干净
- 叙事更简洁
- `state` 和 `response` 仍可共享底层表示

### 2.3 输出协议

每一步使用固定 tag block。

运行时完整 schema 当前记为：

```text
<state>note|bridge</state>
<note>...</note>
<bridge>...</bridge>
<query>...</query>
<response>silent|answer</response>
<answer>...</answer>
```

约束：

- tag 顺序固定
- `note` 与 `bridge` 互斥
- `response=silent` 时 `<answer>` 必须为空
- `<note>` 和 `<bridge>` 不允许偷渡最终答案
- `note` 保持极短，尽量收紧到 `event + cue + reason`
- `bridge` 只写相对 `active note` 的变化，不写长摘要

### 2.4 混合状态结构

```text
x_t = {V_<t^keep, T_<t^bridge, W_t^recent, c_t, q_t}
```

- `V_keep`：所有历史 `note`
- `T_bridge`：历史桥接文本
- `W_recent`：最近窗口原始视觉内容
- `c_t`：当前 `chunk`
- `q_t`：当前有效 query

---

## 3. 训练路线

### 3.1 第一版的基本单位

- 决策单位：`1s chunk`
- 系统始终维护一个 `active note`
- 后续所有 `bridge` 都相对这个锚点书写
- 一旦新 `note` 出现，它接管 `active note`

### 3.2 结构化状态

第一版每个 `chunk` 维护五类结构化字段：

- `entities`
- `attributes_or_states`
- `relations`
- `ocr`
- `action`

`object/layout delta` 不作为最终输出字段，而是作为启发式判断信号。

### 3.3 v0 baseline

`v0` 先解决两件事：

1. 当前 `chunk` 是否像一个新的视觉锚点
2. 当前 `chunk` 能否相对 `active note` 压成一条足够短的 `bridge`

锚点分数来自三类信号：

- `PySceneDetect` 的场景边界
- `prediction novelty`
- `object/layout delta`

可写成：

```text
a_t = alpha1 * scene + alpha2 * pred + alpha3 * layout
```

压缩判决基于：

- `bridge` 预算 `B = 0.3 * N_frame_tokens`
- 结构化重建误差 `E_rec`

若满足以下任一条件，则当前 `chunk` 记为 `note`：

- 锚点分数足够高
- 在预算 `B` 内无法稳定生成短 `bridge`
- 结构化重建误差超过阈值

否则记为 `bridge`。

### 3.4 四层数据组织

1. 小规模金标准：`StreamingBench + OVO-Bench`
2. `response-only` 预训练数据：如 `Streamo-Instruct-465K`
3. `state-only` 合成数据：如 `LLaVA-Video-178K`、`ShareGPT4Video`、`ActivityNet`、`COIN`、`YouCook2`、`Charades`
4. 自举扩增数据：`Self-Instruct + STaR + process verification`

作用分工：

- 第一层定协议和目标
- 第二层先把响应时机学稳
- 第三层给状态轨迹提供高质量视觉过程监督
- 第四层做有约束的扩量

### 3.5 teacher 设计

`state teacher` 输入：

- 当前任务
- 当前 `active note`
- 历史 `bridge`
- 最近视觉窗口
- 当前 `chunk`
- 可选结构化状态

输出只允许：

```text
<state>note|bridge</state>
<note>...</note>
<bridge>...</bridge>
```

`response teacher` 输入：

- 当前任务
- 历史 `note`
- 历史 `bridge`
- 最近视觉窗口
- 当前 `chunk`

输出只允许：

```text
<response>silent|answer</response>
<answer>...</answer>
```

最终再拼成联合轨迹：

```text
<state>note|bridge</state>
<note>...</note>
<bridge>...</bridge>
<response>silent|answer</response>
<answer>...</answer>
```

### 3.6 训练主线

- 第一步：`joint step-wise SFT`
- 第二步：`short rollout consistency SFT`
- `response-only warmup` 和 `state-only warmup` 只作为回退选项

### 3.7 过滤链路

进入训练集前必须通过：

- 格式过滤
- 未来泄漏过滤
- 答案偷渡过滤
- 时序一致性过滤
- 重复和低信息过滤
- 一致性与验证过滤

原则是宁可多删，也不让脏轨迹进入正式训练集。

### 3.8 后训练

第一版先做轻量后训练验证 reward 是否有信号：

```text
r = r_format + r_timing + r_answer + r_silence + r_keep_cost
```

其中：

```text
r_keep_cost = -lambda * note_count / total_chunks
```

默认顺序：

- `RAFT / rejection sampling`
- `Reinforce-Rej`
- `GRPO` 放后面

---

## 4. 实验设计

### 4.1 主实验对照组

- `recent-only`
- `recent + text`
- `recent + note + bridge (learned)`

### 4.2 关键消融

- 全 `note` vs 全 `bridge`
- 去掉 `silent`
- 有无 RL
- 不同 `lambda`

### 4.3 评估指标

- 主战场：`OVO-Bench Backward Tracing`
- 约束项：`StreamingBench 60s Main`
- 时机指标：主动回答的 timing 质量
- 效率指标：视觉保留率 = `note` 数 / 总 `chunk` 数

核心图：

- 横轴：视觉保留率
- 纵轴：Backward Tracing 准确率

---

## 5. 当前实现落点

- `plain recent-window benchmark` 继续放在 `exp1/SimpleStream`
- `StreamWeave` 的 baseline 和后续 agent 实现放在 `exp1/streamweave`
- 第一版工程目标不是训练，而是先把：
  - `note / bridge`
  - `silent / answer`
  - parser
  - state
  - trace
  - benchmark adapter
  这条闭环跑通

## 6. 当前待定实现点

- `<query>` 是否作为最终学生输出中的显式 tag，还是仅作为输入上下文字段，需要在实现 parser 前定死。
- `q_t` 已改为“当前有效 query”，实现时不能再默认它只在视频开始给出。
