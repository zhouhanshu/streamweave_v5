# 数据处理 0514 for exp3

日期：2026-05-14

## 目标

本轮处理基于已经规范化好的 `query_events.jsonl`，分别对 Streamo 和 CogStream 做二次过滤，得到适合 exp3 训练的短视频标注。

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

两个输出文件里所有样本都满足本轮三条过滤要求。

## exp3 RL 聚合文件

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
