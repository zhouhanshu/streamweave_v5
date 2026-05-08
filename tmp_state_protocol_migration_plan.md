# StreamWeave `<state>` Protocol Migration Plan

临时计划文件。目的：先把推理/评测链路从 `<eta>` 切到 `<state>`，观察小样本效果；如果效果稳定，再迁移 SFT 数据合成、ShareGPT 导出和 RL 训练链路。

## 1. 背景和目标

当前协议要求模型每一步输出：

```xml
<eta>...</eta>
<answer>...</answer>
<note t="..."></note>
<bridge t="...">...</bridge>
```

实际推理链路里，`<eta>` 没有驱动调度，也不参与 memory commit 或最终打分。系统每个 step 都继续调用模型，最终答案来自最后一个非空 `<answer>`。因此 `<eta>` 的主要作用是格式监督和 SFT teacher schedule 约束。

新协议希望模型在回答前先显式整理当前临时认知状态：根据 Memory 理清已经发生的内容，根据 Current frames 更新当前状态，再根据 QA History 判断现在是否需要回答。这个状态只用于当前 step 的推理和 trace，不写回 Memory。

目标输出：

```xml
<state>...</state>
<answer>...</answer>
<note t="..."></note>
<bridge t="...">...</bridge>
```

## 2. `<state>` 的语义

`<state>` 是当前 step 的临时状态总结。

它应该覆盖：

- 从 Memory 中重建到目前为止已经发生了什么；
- 当前帧窗口新增了什么可见证据；
- QA History 中是否有 active question；
- 当前证据是否足够支持现在回答。

它不应该做：

- 不写回 Memory；
- 不替代 `<bridge>`；
- 不复制大段 Memory；
- 不输出 XML 标签；
- 不编造不可见细节；
- 不承担长期记忆功能。

长期记忆仍然只由三部分组成：

- `<note>`：关键视觉证据，进入 Memory；
- `<bridge>`：视频历史的文本压缩，进入 Memory；
- QA History：问题和已输出答案，进入 Memory。

`<state>` 只进入 raw output、trace、SFT target、RL response。

## 3. Prompt 文案

推理 prompt 中使用以下规则：

```text
State and answer:
Before writing <answer>, write one short <state> paragraph. Use Memory to understand what has already happened in the video, use the current frames to update that understanding, and use QA History to decide whether there is an active question that should be answered now. The state should connect the past and current evidence into a coherent view of the situation. Keep it factual and grounded in visible evidence or Memory; do not copy Memory verbatim, invent unseen details, or put XML tags inside <state>.

Write <answer> only when the available evidence is enough to answer the active question. If there is no active question, the question has already been answered and the current frames add no useful update, or the evidence is still insufficient, leave <answer></answer> empty.

For multiple-choice questions, choose the option best supported by Memory and the current frames. If the evidence is unclear and an unanswerable option is available, choose that option.
```

输出格式：

```xml
<state>...</state>
<answer>...</answer>
<note t="..."></note>
<bridge t="...">...</bridge>
```

## 4. 第一阶段：只改推理/评测链路

第一阶段已经按这个边界执行：只让普通 inference/eval rollout 使用 `<state>`，先不整体迁移 SFT 生成和训练数据。

### 4.1 修改范围

需要修改：

- `streamweave/schemas.py`
- `streamweave/parser.py`
- `streamweave/prompts.py`
- `streamweave/postprocess.py`
- `streamweave/quality.py`
- `streamweave/trace_io.py`
- `backend/base.py`
- `RL/streamweave_rl/agent_loop_stepwise.py` 的 abort fallback

暂时不迁移：

- `data_engine/sft/rollout_sft.py`
- `data_engine/sft/export_llamafactory.py`
- 旧 SFT ShareGPT 数据；
- RL 训练配置和 reward 设计；
- 历史 trace 的格式迁移。

### 4.2 Schema

`ModelAction` 主字段从：

```python
eta: float | None
answer: str
events: list[ModelEvent]
eta_present: bool
answer_present: bool
```

改为：

```python
state: str
answer: str
events: list[ModelEvent]
state_present: bool
answer_present: bool
```

为了第一阶段不让旧 SFT 代码因为 `raw_action.eta` 引用直接报错，临时保留 deprecated compatibility 字段：

```python
eta: float | None = None
eta_present: bool = False
```

这些字段不参与新推理协议。

### 4.3 Parser

严格解析规则：

- exactly one `<state>`；
- exactly one `<answer>`；
- 输出必须以 `<state>`、`<answer>` 开头；
- `<answer>` 后只能出现 `<note>` 和 `<bridge>`；
- 至少有一个 `<note>` 或 `<bridge>`；
- 当前 note 只允许使用 timestamp-only paired tag：`<note t="..."></note>`；
- 不再接受 `frame`/`id` 属性，也不再通过 frame id 绑定当前帧；
- self-closing note 继续判为格式问题；
- `<state>` 不能为空；
- `<state>` 内不能包含 XML-like tag。

旧 `<eta>` 不再是合法 token。模型如果仍输出 `<eta>`，strict validation 会出现 format issues；在 `eval_repair` 模式下，repair 仍可能从 `<answer>/<note>/<bridge>` 抽取可执行动作，但 format reward 会受到影响。

### 4.4 Prompt

需要同步修改 teacher/eval/production prompt：

- few-shot 示例改成 `<state>`；
- 删除 `<eta>` 时间预测说明；
- 加入 state/answer 说明；
- Output Format 改成 `<state>` 开头；
- 保留 note/bridge/open-tail/gap 规则。

### 4.5 Postprocess

`repair_for_execution()`：

- lenient parse 后保留 `state`；
- repair 只作用于 note/bridge/answer 的可执行部分；
- `state` 不写入 Memory；
- 如果没有任何 question 却输出 answer，仍按原逻辑 drop answer。

`apply_raw_action()`：

- 直接使用 parser 得到的 action；
- 不对 state 做 memory commit。

### 4.6 Quality Metrics

格式 reward 仍来自 strict parser。

新增观测 metrics：

- `state_length`
- `state_present`

第一阶段不做 state 内容 reward。原因：如果奖励 state 内容，模型容易写模板废话或 reward hacking。先观察 state 是否稳定、有用、长度是否可控。

### 4.7 Trace

`trace.txt` 和 `trace.jsonl` 中 applied output 改为：

```xml
<state>...</state>
<answer>...</answer>
...
```

Memory dump 不包含 state。

### 4.8 Mock Backend

MockBackend 改成输出新协议，便于本地 smoke：

```xml
<state>...</state>
<answer>...</answer>
...
```

无 question 时也输出非空 state 和空 answer。

## 5. 第一阶段验证计划

### 5.1 编译检查

运行：

```bash
python -m py_compile \
  streamweave/schemas.py \
  streamweave/parser.py \
  streamweave/quality.py \
  streamweave/postprocess.py \
  streamweave/prompts.py \
  streamweave/trace_io.py \
  backend/base.py \
  RL/streamweave_rl/agent_loop_stepwise.py
```

预期：无语法错误。

### 5.2 Parser 手工样例

合法样例：

```xml
<state>The memory is empty and the current window starts the video, with no active question to answer.</state>
<answer></answer>
<note t="0.0-1.0"></note>
<bridge t="1.0-5.0">The person begins preparing the workspace.</bridge>
```

预期：

- `parser_ok=True`
- `quality.valid=True`
- `state_length>0`

旧协议样例：

```xml
<eta></eta>
<answer></answer>
<bridge t="0.0-5.0">...</bridge>
```

预期：

- strict parser invalid；
- issue 包含缺少 state 或 text outside tags；
- 不作为新协议合格输出。

异常样例：

```xml
<state></state>
<answer></answer>
<bridge t="0.0-5.0">...</bridge>
```

预期：

- issue 包含 `state_empty`。

```xml
<state>Use <answer> now.</state>
<answer>A</answer>
<bridge t="0.0-5.0">...</bridge>
```

预期：

- issue 包含 `state_contains_xml`。

### 5.3 Mock Rollout

用 mock backend 跑一个小样本，确认：

- prompt 里是 `<state>` 协议；
- raw output 是 `<state>`；
- applied output 是 `<state>`；
- memory 里没有 state；
- final answer 仍只取最后一个非空 `<answer>`。

### 5.4 少量真实样本

建议先跑 selected OVO 样本：

- base Qwen3-VL-8B / vLLM；
- Gemini teacher_eval；
- 每类任务至少覆盖 backward、realtime、forward。

重点看：

- format valid rate；
- state 是否真的整合 Memory + Current frames + QA History；
- state 是否过长；
- answer 是否更早/更晚输出；
- forward 是否提前答；
- final score 是否明显变化；
- repair_count 是否上升。

## 6. 第二阶段：SFT 合成迁移

如果第一阶段效果可接受，再改 SFT。

### 6.1 需要修改的文件

主要文件：

- `data_engine/sft/rollout_sft.py`
- `data_engine/sft/export_llamafactory.py`
- `data_engine/sft/inspect_intermediate.py`
- `data_engine/sft/README.md`
- 相关 scripts/configs。

### 6.2 删除 eta schedule 约束

当前 SFT 里有：

- `_apply_qa_eta_answer_constraints()`
- `_expected_qa_output()`
- `_eta_matches()`
- `_format_eta_answer_template()`
- `_format_expected_eta_requirement()`
- `eta_window`
- `expected_eta`

迁移后：

- 改名为 `_apply_qa_state_answer_constraints()`；
- 不再检查 eta timestamp；
- 保留 answer 是否该空/非空；
- 增加 state 必须非空、不能包含 XML tag、长度上限/软上限；
- retry feedback 改为要求 state 解释当前是否该答，而不是要求 eta 落在某个窗口。

### 6.3 Answer Schedule 仍保留

虽然去掉 eta，但 answer 逻辑必须保留：

- no question：answer 必须空；
- already answered 且当前无有用更新：answer 必须空；
- realtime/backward active unanswered：当前窗口可答时要求 answer；
- forward clue 前：answer 必须空；
- forward clue 到达后：answer required。

这里要特别小心 forward。旧协议用 eta 预测 clue window；新协议只能靠 state 说明当前证据还不足，并保持 answer 为空。需要观察模型是否因此提前答。

### 6.4 SFT Target

新的 target 必须统一为：

```xml
<state>...</state>
<answer>...</answer>
<note t="..."></note>
<bridge ...>...</bridge>
```

旧 ShareGPT 数据不能和新协议直接混训，除非先做格式转换并重新校验质量。

当前 `data_engine/sft/rollout_sft.py`、ShareGPT 导出和旧 SFT 文档仍可能保留 `<eta>`、`frame="N"` 或 `<frame id="...">` 旧协议写法。这部分按第二阶段处理，第一阶段暂不迁移，但正式合成新数据前必须统一到 timestamp-only note。

### 6.5 Teacher Retry Feedback

重试反馈从：

```text
set <eta> to any timestamp inside ...
```

改为：

```text
write a grounded <state> that explains the current video state and whether the QA can be answered now; keep <answer> empty until the evidence is sufficient
```

## 7. 第三阶段：RL 迁移

RL 环境复用 `StreamWeaveEnv`，第一阶段后基本已经能跑新 parser/prompt，但正式训练前还需要处理：

- old checkpoint 输出 `<eta>` 的格式退化；
- reward metrics 从 eta 命名迁移到 state；
- smoke tests 更新；
- eval scripts 的实验名和日志字段更新；
- 训练数据里如果仍有旧 prompt/target，需要重新生成。

### 7.1 Reward

第一版 RL reward 仍建议：

- format score；
- final success score；
- 不奖励 state 内容。

后续可以观察后再加轻量 state 约束：

- state 非空；
- state 长度在合理范围内；
- state 不含 XML tag；
- state 不复制 Memory 大段内容。

不要直接奖励“state 看起来像推理”，否则容易把 reward 变成模板生成奖励。

### 7.2 Checkpoint 风险

旧 SFT checkpoint 学过 `<eta>`，新协议下可能：

- 继续输出 `<eta>`；
- 输出 `<state>` 但内容空泛；
- `<state>` 很长，挤压 answer/bridge token；
- answer 提前或延迟。

建议先做小规模新协议 SFT refresh，再接 RL。

## 8. 风险清单

### 8.1 格式风险

模型可能继续输出 `<eta>`。第一阶段 strict parser 会判错，但 eval_repair 可能仍提取 answer。这会造成 final score 和 format score 分离，需要同时看 `score` 和 `format_reward`。

### 8.2 State 空泛

模型可能输出固定模板，例如“The current frames are observed and I will answer if needed”。这对任务帮助小。第一阶段先通过 trace 人工检查，不急着加 reward。

### 8.3 State 过长

state 过长会增加 token，尤其长视频 Memory 已经很大。需要记录 `state_length`，观察是否需要 prompt 里更强地限制 one short paragraph。

### 8.4 State/Answer 不一致

可能出现 state 说证据不足，但 answer 给了选项。第一阶段不做自动一致性判断；SFT 第二阶段可以加 answer schedule 校验。

### 8.5 Forward 提前答

去掉 eta 后，forward 任务失去“预测未来 clue window”的显式监督。必须用 selected forward 样本检查是否提前输出 answer。

### 8.6 旧数据混用

旧 SFT target 是 `<eta>`。新旧协议混用会让模型格式摇摆。正式训练前必须明确数据版本。

## 9. 评估指标

第一阶段看这些指标：

- final score；
- format reward mean；
- parser_ok rate；
- repair_count mean；
- `state_length` mean/p95；
- answer 空/非空时机；
- forward 提前答率；
- traces 中 state 的人工质量。

建议 trace 人工检查字段：

- state 是否利用了 Memory；
- state 是否正确更新当前帧；
- state 是否正确判断 QA；
- answer 是否和 state 一致；
- bridge 是否仍承担长期记忆，而不是被 state 替代。

## 10. 回滚策略

如果第一阶段效果差，可以回滚以下推理文件：

- `streamweave/schemas.py`
- `streamweave/parser.py`
- `streamweave/prompts.py`
- `streamweave/postprocess.py`
- `streamweave/quality.py`
- `streamweave/trace_io.py`
- `backend/base.py`
- `RL/streamweave_rl/agent_loop_stepwise.py`

因为 SFT/训练数据还没迁移，第一阶段回滚成本较低。

## 11. 推荐执行顺序

1. 完成第一阶段推理代码修改。
2. 跑 py_compile 和 parser 手工样例。
3. 跑 mock rollout。
4. 用 Gemini teacher_eval 跑 5-10 个 selected OVO 错例，人工看 state。
5. 用 Qwen/vLLM 跑同一批样本，对比旧协议 trace。
6. 如果 state 质量可接受，再进入 SFT 迁移。
7. 重新合成小规模 SFT 数据。
8. 小规模 SFT refresh。
9. 小规模 RL warmup。
10. 最后做 full/1of8 OVO 对比。
