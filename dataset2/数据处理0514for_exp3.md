# 数据处理 0514 for exp3

日期：2026-05-14

## 当前最终入口

当前 exp3 RL 数据入口是：

```text
dataset2/rl_exp3.jsonl
```

对应统计文件：

```text
dataset2/rl_exp3.jsonl.summary.json
```

当前最终规模：

```text
总量: 2000
一问一答: 1699
一问多答: 301
```

其中一问多答全部来自 Streamo，是当前真正承担语义漂移监督的部分。CogStream 当前全部是一问一答，用来补充普通流式状态问答。

2026-05-14 追加了一个新的 CogStream 同源视频聚合标注：

```text
CogStream/rl_0514.jsonl
CogStream/rl_0514.jsonl.summary.json
```

这个文件没有覆盖 `annotations.jsonl`、`query_events.jsonl` 或 `query_events_0514_filtered.jsonl`。它目前还没有替换旧的 `rl_exp3.jsonl`，但已经确认会作为后续 RL 训练数据的一部分。

本次决策记录：

```text
CogStream/rl_0514.jsonl 纳入后续训练数据。
它提供一视频多 query 的实时 QA 监督，用来降低同源视频重复 rollout 开销。
后续重新生成最终训练入口时，应优先使用 CogStream/rl_0514.jsonl，而不是旧的 CogStream/query_events_0514_filtered.jsonl。
```

当前规模：

```text
rollout rows: 1049
query_events: 2567
answer_events: 2567
multi-query rows: 852
```

2026-05-14 追加了 Streamo 三分训练文件：

```text
Streamo-Instruct-465K/rl_0514_unable.jsonl
Streamo-Instruct-465K/rl_0514_one.jsonl
Streamo-Instruct-465K/rl_0514_multi.jsonl
Streamo-Instruct-465K/rl_0514_split.summary.json
```

这三个文件来自 `Streamo-Instruct-465K/query_events_0514_filtered.jsonl`，互斥划分，不覆盖原始 filtered 文件：

```text
unable: 925
one-answer: 2318
multi-answer: 301
total: 3544
```

2026-05-14 追加了统一 RL schema 的 normalized 文件：

```text
normalize_rl_schema_0514.py

CogStream/rl_0514_normalized.jsonl
Streamo-Instruct-465K/rl_0514_unable_normalized.jsonl
Streamo-Instruct-465K/rl_0514_one_normalized.jsonl
Streamo-Instruct-465K/rl_0514_multi_normalized.jsonl
```

这一步不覆盖前面的 `rl_0514*.jsonl`，但会直接覆盖四个 `*_normalized.jsonl` 输出。当前约定中，模型看到的问题文本统一写在 `query_events[].content`。Streamo 的 proactive/update 样本会在题目前添加英文前缀：

```text
Please answer the following question based on the video content. You may update your answer multiple times.
```

统一后的顶层字段只保留：

```text
dataset
video_id
video
frames_dir
sample_fps
frame_count
frame_id_base
query_events
```

统一后的 `query_events[]` 字段只保留：

```text
qid
time
content
answer_type
answer_policy
options
answer_events
```

其中 `answer_type` 只允许 `mcq` 和 `text`。`mcq` 必须有 `options`，`text` 不写 `options`。Streamo 是 `mcq`，CogStream 是 `text`。

统一后的 `answer_events[]` 字段只保留：

```text
time
gt
answer
content
```

`time` 是数字秒，不是字符串；当前全部按 1fps 四舍五入后写成 `xx.0`。`frame_count` 是整数。MCQ 的 `gt` 是选项字母，`answer` 是选项文本；text answer 不写 `gt`。

normalized 文件当前统计：

```text
CogStream/rl_0514_normalized.jsonl
  rows: 1049
  query_events: 2567
  answer_events: 2567
  multi-query rows: 852
  answer_type: text
  policy: answer_when_asked
  frame_count range: 20-110

Streamo-Instruct-465K/rl_0514_unable_normalized.jsonl
  rows: 925
  query_events: 925
  answer_events: 925
  answer_type: mcq
  policy: answer_when_asked
  frame_count range: 20-87

Streamo-Instruct-465K/rl_0514_one_normalized.jsonl
  rows: 2318
  query_events: 2318
  answer_events: 2318
  answer_type: mcq
  policy: update_when_changed
  proactive prefix added: 2318
  frame_count range: 40-110

Streamo-Instruct-465K/rl_0514_multi_normalized.jsonl
  rows: 301
  query_events: 301
  answer_events: 655
  multi-answer queries: 301
  answer_type: mcq
  policy: update_when_changed
  proactive prefix added: 301
  frame_count range: 52-110
```

四个 normalized 文件合计：

```text
rows: 4593
query_events: 6111
answer_events: 6465
```

已做校验：

```text
1. 只包含统一 schema 白名单字段。
2. 所有 time 都是数字秒，所有 frame_count 都是整数。
3. 所有 answer time <= frame_count。
4. 所有帧目录都能在当前 dataset2 下找到。
5. Streamo normalized 的 content 全部包含 Options:。
6. CogStream 多 query 间隔 >= 7s。
7. Streamo multi 的 answer 间隔 >= 7s。
```

后续建议：

```text
1. 后续重新构造最终训练入口时，优先使用四个 *_normalized.jsonl。
2. 不再让 RL 主训练读取旧的 query_events_0514_filtered.jsonl。
3. 新数据集接入时也先转成这个最小 schema，再进入合并/采样脚本。
```

## 流水线总览

本轮数据流水线分三步：

```text
原始 annotations.jsonl
  -> build_query_event_annotations.py
  -> query_events.jsonl
  -> filter_query_events_0514.py
  -> query_events_0514_filtered.jsonl
  -> build_rl_exp3.py
  -> rl_exp3.jsonl
```

2026-05-14 追加的 CogStream 多 query 分支是：

```text
CogStream/query_events_0514_filtered.jsonl
  -> group_cogstream_source_video.py
  -> CogStream/rl_0514.jsonl
```

每一步的职责不同：

```text
1. build_query_event_annotations.py
   把 Streamo / CogStream 原始标注统一成 query_events[].answer_events[] 格式。

2. filter_query_events_0514.py
   做 1fps 时间对齐、短视频截断、answer 间隔过滤、重复 options 删除、frame_count 对齐。

3. build_rl_exp3.py
   合并两个 filtered 文件，保留全部一问多答，对一问一答分层降采样，得到 2000 条训练入口。

4. group_cogstream_source_video.py
   只处理 CogStream filtered 文件，按 source_video_name 把同一个原视频的不同前缀窗口聚成一条样本内的多个 query_events。
```

原始文件没有被覆盖：

```text
Streamo-Instruct-465K/annotations.jsonl
CogStream/annotations.jsonl
```

中间文件也保留：

```text
Streamo-Instruct-465K/query_events.jsonl
CogStream/query_events.jsonl
Streamo-Instruct-465K/query_events_0514_filtered.jsonl
CogStream/query_events_0514_filtered.jsonl
```

## 本轮目标

本轮目标是从 Streamo 和 CogStream 的本地标注出发，构造一批短视频、1fps 对齐、可直接用于 exp3 GRPPO 的 RL 数据。最终训练入口固定为 `dataset2/rl_exp3.jsonl`。

## 阶段 1：原始标注统一成 query_events

处理脚本：

```text
build_query_event_annotations.py
```

运行命令：

```bash
python exp3/streamweave_v5/dataset2/build_query_event_annotations.py
```

### Streamo 标准化

输入：

```text
Streamo-Instruct-465K/annotations.jsonl
```

输出：

```text
Streamo-Instruct-465K/query_events.jsonl
Streamo-Instruct-465K/query_events.jsonl.summary.json
```

规则：

```text
1. 按 source video / ask_time / question / options 聚合。
2. 同一问题的多个未来答案合并到 answer_events。
3. 保持一条样本一个 query_event，即 q0。
4. answer_policy = update_when_changed。
5. task_family = semantic_drift。
```

结果：

```text
原始行数: 7224
原始 QA 数: 7260
聚合后样本: 5202
缺帧丢弃: 0
```

聚合后的 answer 数量分布：

```text
1 answer: 4005
2 answers: 726
3 answers: 271
4 answers: 104
5 answers: 50
6 answers: 26
7 answers: 8
8 answers: 3
9 answers: 6
11 answers: 2
12 answers: 1
```

来源分布：

```text
ActivityNet: 2680
LLaVA_Video: 1536
Youcookv2: 986
```

### CogStream 标准化

输入：

```text
CogStream/annotations.jsonl
```

输出：

```text
CogStream/query_events.jsonl
CogStream/query_events.jsonl.summary.json
```

规则：

```text
1. 只保留有本地帧目录的数据。
2. 按 video_id 聚合。
3. 同一视频同一时间点如果有多个 QA，固定 seed=511 随机保留一个。
4. 保持一条样本一个 query_event，即 q0。
5. answer_policy = answer_when_asked。
6. task_family = streaming_state。
```

结果：

```text
原始 QA: 25700
聚合后样本: 5154
随机丢弃同时间重叠 QA: 20546
缺帧丢弃: 0
invalid_rows: 0
```

保留后的 query_type 分布：

```text
Basic/Actions: 1513
Basic/Object: 1070
Basic/Attributes: 1050
Streaming/Sequence Perception: 996
Streaming/Causal Reasoning: 357
Streaming/Object Tracking: 80
Streaming/Dynamic Updating: 80
Global/Global Analysis: 6
Global/Overall Summary: 2
```

## 阶段 2：0514 短视频过滤

输入文件：

- `Streamo-Instruct-465K/query_events.jsonl`
- `CogStream/query_events.jsonl`

输出文件：

- `Streamo-Instruct-465K/query_events_0514_filtered.jsonl`
- `Streamo-Instruct-465K/query_events_0514_filtered.jsonl.summary.json`
- `CogStream/query_events_0514_filtered.jsonl`
- `CogStream/query_events_0514_filtered.jsonl.summary.json`

处理脚本：

- `filter_query_events_0514.py`

运行命令：

```bash
python exp3/streamweave_v5/dataset2/filter_query_events_0514.py
```

## 过滤规则

### 1. 视频时长过滤和截断

所有 `query_events[].time`、`answer_events[].time` 和 `answer_events[].evidence_time` 会先按 `sample_fps` 做四舍五入。本轮数据 `sample_fps = 1.0`，所以所有 query/answer 事件时间都会被对齐到 1fps 整秒。

训练侧使用 `frame_count` 决定 rollout 视频长度，不依赖帧目录里的真实帧数。本轮过滤的时长口径是：

```text
video_seconds = frame_count / sample_fps
```

当前两个数据集都是 `sample_fps = 1.0`，所以等价于：

```text
video_seconds = frame_count
```

目标范围：

```text
20s <= video_seconds <= 110s
```

如果原样本超过 `110s`，允许做截断，但必须满足：

```text
query_time <= 110s
first_answer_time <= 110s
```

满足条件后，保留 `time <= 110s` 的 answer events，丢弃 110s 之后的 answer events。然后把样本的声明视频结束位置对齐到最后一个保留 answer：

```text
frame_count = ceil(last_retained_answer_time)
realtime = last_retained_answer_time
duration = last_retained_answer_time  # 仅 Streamo 原本有 duration 字段时更新
```

这样做的原因是：训练时我们希望 answer 结束时视频也结束，不能让样本在最后一个监督 answer 后继续 rollout 很久。

### 2. answer 间隔过滤

对每个 query 内部的 `answer_events` 按时间排序。只要存在相邻两个 answer 的间隔小于 `7s`，整条 QA 样本直接丢弃。

规则：

```text
answer_events[i + 1].time - answer_events[i].time >= 7s
```

一问一答样本没有相邻 answer，因此天然通过这条规则。

### 3. options 重复过滤

对多选题 options 做大小写和空白归一。如果同一个 query 内存在重复选项，整条样本直接丢弃。

本轮主要处理的问题是：

```text
Fast edge / Fast Edge
Wall / wall
Railing / railing
```

### 4. 最后 answer 和 frame_count 对齐

过滤后要求：

```text
0 <= frame_count - last_answer_time <= 3
```

本轮脚本直接把：

```text
frame_count = ceil(last_answer_time)
```

因此当前输出里所有样本都是：

```text
frame_count - last_answer_time = 0
```

这满足“frame_count 只能大于或等于最后一个 answer 时间，允许大 2-3 秒”的要求。

## Streamo 结果

输入：

```text
5202 条
```

输出：

```text
3544 条
```

丢弃：

```text
1658 条
```

丢弃原因：

```text
first_answer_after_max_cutoff: 1224
query_after_max_cutoff: 358
answer_gap_too_small: 74
duplicate_options: 2
```

截断情况：

```text
truncated_rows: 318
removed_answer_events_by_truncation: 464
```

输出后的 answer 数量分布：

```text
1 个 answer_events: 3243
2 个 answer_events: 256
3 个 answer_events: 38
4 个 answer_events: 6
5 个 answer_events: 1
```

因此 Streamo 过滤后：

```text
一问一答: 3243
一问多答: 301
```

来源分布：

```text
ActivityNet: 2019
LLaVA_Video: 1190
Youcookv2: 335
```

视频时长统计，按过滤后的 `frame_count`：

```text
min: 40s
p25: 52s
median: 68s
mean: 70.32s
p75: 87s
p90: 101s
max: 110s
```

一问多答的 answer 更新间隔：

```text
n: 354
min: 7s
p25: 12.25s
median: 20s
mean: 22.11s
p75: 29s
p90: 39s
max: 64s
```

## CogStream 结果

输入：

```text
5154 条
```

输出：

```text
2611 条
```

丢弃：

```text
2543 条
```

丢弃原因：

```text
query_after_max_cutoff: 2543
```

截断情况：

```text
truncated_rows: 29
removed_answer_events_by_truncation: 0
```

输出后的 answer 数量分布：

```text
1 个 answer_events: 2611
```

因此 CogStream 过滤后：

```text
一问一答: 2611
一问多答: 0
```

split 分布：

```text
train: 1987
test: 624
```

query_type 分布：

```text
Basic/Actions: 751
Basic/Object: 528
Basic/Attributes: 509
Streaming/Sequence Perception: 386
Streaming/Causal Reasoning: 304
Streaming/Object Tracking: 71
Streaming/Dynamic Updating: 61
Global/Overall Summary: 1
```

视频时长统计，按过滤后的 `frame_count`：

```text
min: 20s
p25: 41s
median: 64s
mean: 63.83s
p75: 85.5s
p90: 101s
max: 110s
```

CogStream 是一问一答实时问答，过滤后仍然是：

```text
query_time == answer_time
query_to_first_answer_gap = 0s
```

## 阶段 2.5：CogStream 同源视频多 query 聚合

处理脚本：

```text
group_cogstream_source_video.py
```

运行命令：

```bash
python exp3/streamweave_v5/dataset2/group_cogstream_source_video.py
```

输入：

```text
CogStream/query_events_0514_filtered.jsonl
```

输出：

```text
CogStream/rl_0514.jsonl
CogStream/rl_0514.jsonl.summary.json
```

这一步不覆盖任何已有标注文件。

### 为什么不是按 video_id 聚合

CogStream 的 `annotations.jsonl` 里，同一个 `video_id` 往往对应多个 QA，但这些 QA 基本都落在同一个 `ask_time`。这类同一时刻多问题对当前 RL 训练不友好：一条视频在同一帧注入多个 query，rollout 侧会把多个 query 挤在同一个时间点，不能形成我们想要的流式时间轴。

因此当前脚本不恢复同一个 `video_id` 内被丢弃的同时间 QA，而是按更上游的：

```text
source_video_name
```

做聚合。

`source_video_name` 表示同一个原始视频。过滤后的 CogStream 中，同一个 `source_video_name` 下面通常有多个不同长度的前缀窗口，例如：

```text
0 -> 27s
0 -> 48s
0 -> 77s
0 -> 107s
```

这些窗口共享同一个原始视频，只是问题发生在不同时间点。聚合时选择最长的那个保留窗口作为 cover video，把其它较短前缀窗口里的 query 按时间塞进同一条样本的 `query_events`。

### 聚合后的 RL 输入形态

聚合后仍然使用标准 RL 输入格式：

```text
一行 = 一个视频 rollout
query_events = 这个视频里的多个独立问题
query_events[].answer_events = 该问题自己的监督答案
```

示例结构：

```json
{
  "dataset": "CogStream",
  "video_id": "cogstream_train_6NVr0cNiHPM_000000_000107",
  "frames_dir": "video/cogstream_train_6NVr0cNiHPM_000000_000107",
  "frame_count": 107,
  "source_video_name": "6NVr0cNiHPM",
  "task_family": "streaming_state_multi_query",
  "query_dependency": "independent_same_video",
  "source_query_count": 6,
  "query_events": [
    {"qid": "q0", "time": 24.0, "answer_policy": "answer_when_asked"},
    {"qid": "q1", "time": 41.0, "answer_policy": "answer_when_asked"},
    {"qid": "q2", "time": 77.0, "answer_policy": "answer_when_asked"},
    {"qid": "q3", "time": 87.0, "answer_policy": "answer_when_asked"},
    {"qid": "q4", "time": 97.0, "answer_policy": "answer_when_asked"},
    {"qid": "q5", "time": 107.0, "answer_policy": "answer_when_asked"}
  ]
}
```

这里的多个 query 之间没有构造成逻辑依赖。它们不是类似 StreamingBench SQA 那种“后一个问题依赖前一个问题或前一个答案”的链式问答，而是：

```text
同一个原始视频
不同时间点
多个彼此独立的问题
```

这正好符合当前想要降低训练开销的形态：同一条视频只 rollout 一次，在不同时间点注入多个独立 query。

### 聚合规则

当前规则：

```text
1. 按 source_video_name 分组。
2. 每组选择 frame_count 最大的样本作为 cover video。
3. 汇总组内所有 query_events。
4. 同一时间点如果出现多个 query，只保留一个。
5. 相邻 query 的时间间隔必须 >= 7s。
6. query_events 按 time 升序排列，qid 重新编号为 q0, q1, ...
7. 每个 query 标注 query_dependency = independent_same_video，depends_on = []。
```

同一时间点 query 的选择策略是 deterministic priority，不使用运行时随机。当前 filtered 输入中没有同时间冲突，因此这条规则没有实际触发。

### 聚合结果

输入：

```text
rows: 2611
source_video_names: 1049
query_events: 2611
```

输出：

```text
rows: 1049
query_events: 2567
```

丢弃：

```text
same_time_query_events: 0
min_gap_query_events: 44
```

聚合前每个 `source_video_name` 的候选 query 数量分布：

```text
1 query: 193
2 query: 381
3 query: 297
4 query: 136
5 query: 31
6 query: 11
```

聚合后每行的 query 数量分布：

```text
1 query: 197
2 query: 387
3 query: 307
4 query: 122
5 query: 29
6 query: 7
```

注意这里 1 query 从 193 变成 197，是因为有 4 个原本多 query 的同源视频，在应用 `>= 7s` query 间隔过滤后只剩下 1 个 query。

输出 split 分布：

```text
train: 815
test: 234
```

query_type 分布：

```text
Basic/Actions: 740
Basic/Object: 516
Basic/Attributes: 502
Streaming/Sequence Perception: 372
Streaming/Causal Reasoning: 304
Streaming/Object Tracking: 71
Streaming/Dynamic Updating: 61
Global/Overall Summary: 1
```

视频时长统计，按聚合后 cover video 的 `frame_count`：

```text
min: 20s
p25: 73s
median: 88s
p75: 101s
max: 110s
```

query 间隔统计：

```text
n: 1518
min: 7s
p25: 17s
median: 26s
p75: 37s
max: 82s
```

校验结果：

```text
validation.ok: true
duplicate_source_video_output_rows: 0
frame_dir_missing_rows: 0
```

## 阶段 2.6：Streamo unable / one / multi 三分

处理脚本：

```text
split_streamo_rl_0514.py
```

运行命令：

```bash
python exp3/streamweave_v5/dataset2/split_streamo_rl_0514.py
```

输入：

```text
Streamo-Instruct-465K/query_events_0514_filtered.jsonl
```

输出：

```text
Streamo-Instruct-465K/rl_0514_unable.jsonl
Streamo-Instruct-465K/rl_0514_one.jsonl
Streamo-Instruct-465K/rl_0514_multi.jsonl
Streamo-Instruct-465K/rl_0514_split.summary.json
```

三类文件互斥划分，合计仍为 filtered 的 3544 条。

unable 构造规则：

```text
1. 只从一问一答样本中抽。
2. query_time >= 20s。
3. answer_time - query_time >= 20s。
4. 视频声明长度截断到 query_time。
5. 问题只加前缀 "Based on the video so far"，不加入显式 unable 提示句。
6. 替换一个非 GT 选项为 "Unable to answer from the video so far"。
7. GT 改为 unable 选项对应字母。
8. answer_policy = answer_when_asked。
```

输出规模：

```text
unable: 925
one-answer: 2318
multi-answer: 301
```

unable 来源：

```text
LLaVA_Video: 450
ActivityNet: 417
Youcookv2: 58
```

校验结果：

```text
validation.ok: true
duplicate_sample_ids: 0
frame_dir_missing_rows: 0
duplicate_options: 0
forbidden_prompt_phrase: 0
```

## 合并后的当前数据形态

两个数据集过滤后总量：

```text
6155 条
```

其中：

```text
Streamo: 3544
CogStream: 2611
```

按 QA 结构：

```text
一问一答: 5854
一问多答: 301
```

真正保留语义漂移监督的是 Streamo 的 `301` 条一问多答样本。CogStream 仍然是普通流式一问一答状态问答。

## 校验结果

脚本运行后又独立扫了一遍两个输出文件，检查项：

```text
1. 每条样本 query_events 数量
2. frame_count 是否在 [20, 110]
3. 相邻 answer_events 间隔是否 >= 7s
4. frame_count 是否大于等于最后 answer 时间
5. frame_count - last_answer_time 是否 <= 3s
6. query_time 是否没有超过 frame_count
7. query/answer/evidence_time 是否都对齐到 1fps
8. options 归一后是否仍有重复
```

结果：

```text
Streamo bad = 0
CogStream bad = 0
```

两个输出文件里所有样本都满足本轮过滤要求。

## 阶段 3：exp3 RL 聚合文件

聚合脚本：

```text
build_rl_exp3.py
```

运行命令：

```bash
python exp3/streamweave_v5/dataset2/build_rl_exp3.py
```

输入：

```text
Streamo-Instruct-465K/query_events_0514_filtered.jsonl
CogStream/query_events_0514_filtered.jsonl
```

输出：

```text
dataset2/rl_exp3.jsonl
dataset2/rl_exp3.jsonl.summary.json
```

聚合规则：

```text
1. 保留全部一问多答样本。
2. 对一问一答样本做分层随机降采样。
3. 分层字段为 dataset / source_dataset / query_type / split。
4. 随机种子为 511。
5. 最终总量固定为 2000 条。
```

输入总量：

```text
6155 条
一问一答: 5854
一问多答: 301
```

输出总量：

```text
2000 条
一问一答: 1699
一问多答: 301
```

输出来源分布：

```text
Streamo-Instruct-465K: 1242
CogStream: 758
```

输出 answer 数量分布：

```text
1 个 answer_events: 1699
2 个 answer_events: 256
3 个 answer_events: 38
4 个 answer_events: 6
5 个 answer_events: 1
```

独立校验结果：

```text
rows = 2000
duplicate_sample_ids = 0
bad = 0
```

## 当前需要记住的边界

### 1. 语义漂移监督并不多

最终 `rl_exp3.jsonl` 里，一问多答有：

```text
301 / 2000
```

这些全部来自 Streamo。CogStream 当前没有一问多答，只是流式状态 QA。

### 2. CogStream 的 test split 仍在数据里

最终 `rl_exp3.jsonl` 里 split 分布是：

```text
train: 1819
test: 181
```

这些 `test` 来自 CogStream。如果后续需要严格避免 split 泄漏，需要在 `build_rl_exp3.py` 或 CogStream 过滤阶段加 `split=train` 约束。

### 3. rl_exp3 默认不 shuffle 输出

`build_rl_exp3.py` 默认 `shuffle_output=false`，会按输入文件顺序保留输出顺序。当前输入顺序是 Streamo 在前，CogStream 在后。

如果训练 dataloader 自己会 shuffle，这不是问题。如果需要文件内部也打散，可以运行：

```bash
python exp3/streamweave_v5/dataset2/build_rl_exp3.py --shuffle-output
```

### 4. root-level ground_truth 不是主要监督

当前监督主要在：

```text
query_events[].answer_events[]
```

不是 root-level `answer` / `gt` / `ground_truth`。当前 GRPPO env 支持 structured `query_events[].answer_events[]`，但如果旧数据加载器或旧 reward 逻辑只看 root-level `ground_truth`，会读不到真正答案。

### 5. Unable 数据还没有进入 rl_exp3

当前 `rl_exp3.jsonl` 没有新构造的 unable 样本。后续计划可以从 TimeChat / NeXTVideo / LLaVA-YouTube 构造 prefix-unable，建议另起文件，不直接覆盖当前 `rl_exp3.jsonl`。

## 多问题样本调查

这里区分两个口径：

```text
1. 行内多问题：同一行标注里有多个 qa_list / query_events。
2. 同视频多问题：同一个 video_id 或同一个上游 source video 有多行 QA。
```

当前最终训练入口：

```text
dataset2/rl_exp3.jsonl
```

行内多问题统计：

```text
row_question_count_dist = {1: 2000}
```

也就是说，当前最终文件里每一行都只有一个 `query_event`。没有任何一行是“一条样本里多个 query_events”。

按当前 `video_id` 看，最终文件里仍有极少数同视频多行：

```text
unique_videos: 1998
videos_with_multiple_rows: 2
max_rows_per_video: 2
```

按更上游的 source video 看，同源视频多行更多：

```text
source_units: 1742
source_units_with_multiple_rows: 226
max_rows_per_source_unit: 4
```

这说明 `rl_exp3.jsonl` 已经基本是“一行一个问题”，但仍可能有同一源视频的多个问题分别作为多行存在。

### 上游各数据集情况

#### Streamo

`Streamo-Instruct-465K/annotations.jsonl` 里有少量行内多问题：

```text
rows: 7224
row_question_count_dist: {1: 7188, 2: 36}
rows_with_multiple_questions: 36
```

但是按上游 `source_video_path` 看，同源视频多问题很多：

```text
source_units: 4030
source_units_with_multiple_rows: 1065
max_rows_per_source_unit: 27
```

经过 `build_query_event_annotations.py` 后，Streamo 被规范成一行一个 query：

```text
Streamo query_events.jsonl:
rows: 5202
row_question_count_dist: {1: 5202}
videos_with_multiple_rows: 29
max_rows_per_video: 2
```

经过 0514 过滤后：

```text
Streamo query_events_0514_filtered.jsonl:
rows: 3544
row_question_count_dist: {1: 3544}
videos_with_multiple_rows: 22
max_rows_per_video: 2
```

所以 Streamo 的多问题视频基本被拆成了多行；同一个 query 的多阶段答案则保留在 `answer_events` 里。

#### CogStream

CogStream 原始标注是最典型的“同视频多问题”：

```text
CogStream annotations.jsonl:
rows: 25700
unique_videos: 5154
videos_with_multiple_rows: 5154
max_rows_per_video: 15
```

但这些 QA 大量落在同一个视频同一个时间点。当前标准化脚本没有把它们保留成同一行多 query，而是固定 `seed=511` 随机保留同时间 bucket 中的一个：

```text
output_rows: 5154
discarded_overlapping_query_events: 20546
```

因此当前 CogStream 输出是：

```text
CogStream query_events.jsonl:
row_question_count_dist: {1: 5154}
videos_with_multiple_rows: 0
```

也就是说，按 `video_id` 口径看，CogStream 的原始同时间多问题没有被拆成多行保留，而是大部分被随机丢弃了。

但是按 `source_video_name` 口径看，CogStream 仍然能恢复成“一条原始视频多个不同时间点 query”。当前新增文件：

```text
CogStream/rl_0514.jsonl
```

就是按这个口径重组的结果：

```text
rows: 1049
query_events: 2567
row_question_count_dist:
  1 query: 197
  2 query: 387
  3 query: 307
  4 query: 122
  5 query: 29
  6 query: 7
```

这批多 query 不是逻辑链式问答。它们只是共享同一个原始视频，每个 query 的问题和答案独立存在，适合作为“一次视频 rollout，多次独立 query 注入”的训练数据。

#### NeXTVideo

NeXTVideo 在 dataset2 里已经是一行一个 QA，但同一个视频有多行问题：

```text
rows: 33128
unique_videos: 3720
videos_with_multiple_rows: 3720
max_rows_per_video: 19
```

这说明它非常适合重构成“一条视频样本多个 query_events”，但当前 `rl_exp3.jsonl` 没有直接使用 NeXTVideo。

#### LLaVA-Video-178K-YouTube

YouTube 也是一行一个 QA、同视频多行：

```text
rows: 338467
unique_videos: 19938
videos_with_multiple_rows: 19938
max_rows_per_video: 22
```

它的数据量很大，适合做大规模多问题重组，但质量和问题类型需要更强过滤。

#### LLaVA-Video-178K-PerceptionTest

PerceptionTest 也是一行一个 QA、同视频多行：

```text
rows: 4692
unique_videos: 1194
videos_with_multiple_rows: 1009
max_rows_per_video: 13
```

不过它的视频整体较短，当前不适合作为语义漂移主来源。

#### TimeChat

按 dataset2 的 `video_id` 看，TimeChat 是一行一个视频窗口、一行一个 QA：

```text
rows: 8921
unique_videos: 8921
videos_with_multiple_rows: 0
```

但是按上游 `source_video_id` 看，同一个原视频有多个窗口 / 多个问题：

```text
source_units: 1169
source_units_with_multiple_rows: 1150
max_rows_per_source_unit: 36
```

所以 TimeChat 不是当前 `video_id` 口径下的多问题样本，但在上游 source video 口径下仍然可以重组。

### 当前判断

如果我们希望训练“一条视频里有多个 query_events”，当前 `rl_exp3.jsonl` 不是这种数据。它现在是：

```text
一行一个 query_event
一个 query_event 下面可能有多个 answer_events
```

目前真正保留下来的多阶段监督是：

```text
Streamo 的一问多答: 301 条
```

如果要恢复“一视频多问题”的训练形态，优先顺序建议是：

```text
1. CogStream：按 source_video_name regroup，已经生成 rl_0514.jsonl。
2. NeXTVideo / LLaVA-YouTube / PerceptionTest：按 video_id regroup，但需要先确认问题时间是否都集中在最后一帧。
3. TimeChat：按 source_video_id regroup，但要处理不同窗口的时间偏移。
4. Streamo：可以按 source_video_path regroup，但要避免把不同截断窗口混成一个视频时间轴。
```
