# 第二部分：数据构造

- 状态：V4 当前 SFT 数据入口已可用，SFT 合成链路已打通；第二轮 SFT 数据合成正在进行
- 当前主线：基于 `streamweave_v4/dataset/streamweave_data/annotations_qa_filter_final.jsonl` 合成 step-level SFT 数据，随后启动第一次 SFT 与评测
- 当前环境：所有处理默认在 `simple` 环境下执行

## 历史视频/关键帧池

- 视频源：`exp2/data/ActivityNet_Captions/Activity_Videos`
- 标注源：`exp2/data/videoxum/train_videoxum.json`、`val_videoxum.json`、`test_videoxum.json`
- 早期视频/关键帧池：`exp2/data/streamweave_data`
- 早期原始标注：`exp2/data/streamweave_data/annotations_filtered_30s300s_key10to40.jsonl`
- 早期可用视频/关键帧样本规模：`12029` 条
- 历史目标数据集：`VideoXum-StreamQA`

注意：`VideoXum` 本身没有 QA，没有 answer timestamp。`annotations_filtered_30s300s_key10to40.jsonl` 只能作为视频、帧路径和 query-free 关键帧监督，不能直接作为 SFT 样本。

## 2026-05-02 V4 当前 SFT 数据状态

当前可训练数据不再直接使用早期的 `annotations_filtered_30s300s_key10to40.jsonl`，而是使用已经包含 QA 的 V4 标注文件：

```text
exp2/streamweave_v4/dataset/streamweave_data/annotations_qa_filter_final.jsonl
```

对应 raw data root：

```text
exp2/streamweave_v4/dataset/streamweave_data
```

当前 SFT 合成入口：

```text
exp2/streamweave_v4/data_engine/sft/run_pipeline.py
exp2/streamweave_v4/data_engine/sft/run_parallel_pipeline.py
```

当前输出原则：

- 每条样本先写入独立 `samples/*.json`，保留完整 attempts、raw output、quality 和 metadata。
- 只有样本级 `status=accepted` 才会进入 `sft_steps.jsonl` 和 `llamafactory_sharegpt.jsonl`。
- `llamafactory_sharegpt.jsonl` 使用 production prompt，不包含 teacher-only 合成提示、关键帧标注提示或 retry feedback。
- 已完成 `gemini_final_8` 小规模巡检：`accepted=7/8`，导出 `136` 条 step-level SFT 数据。
- 第二轮合成目标先跑 `1000` 条样本；完成后需要先巡检数据，再进入第一次 SFT 训练。

## streamweave_data 构造规则

- 合并 `train/val/test` 到一个 jsonl，不再按 split 拆文件，但每条保留原始 `split` 字段。
- 视频按 `1fps` 抽帧，输出到 `video_frame/{video_id}/{frame_id:06d}.jpg`。
- `frame_id` 使用 0-based 编号，直接对应 `vsum_onehot` 的 list 下标。
- 对 `vsum_onehot` 的 10 个标注者逐帧求平均，得到 `key_frame_scores`。
- 使用 `score > threshold` 生成 `key_frame_ids`。
- 当前全量数据与过滤文件来自 `threshold=0.3`。
- annotation 中保存 `video_id / split / duration / frame_count / timestamps / key_frame_scores / key_frame_ids / key_frame_count / key_frame_ratio` 等字段。
- annotation 中不保存 `tsum` 和 `vsum`。

## 已完成数据规模

- 全量构造样本：`14001`
- 全量构造失败：`0`
- split：
  - `train 8000`
  - `val 2001`
  - `test 4000`
- 全量平均视频帧数：`124.22`
- 全量平均关键帧数：`19.30`
- 全量平均关键帧比例：`16.51%`

## 当前过滤版本

过滤文件：

```text
exp2/data/streamweave_data/annotations_filtered_30s300s_key10to40.jsonl
```

过滤条件：

```text
30 <= duration <= 300
0.10 <= key_frame_ratio <= 0.40
```

过滤结果：

```text
total_records:   14001
kept_records:    12029
removed_records:  1972
kept_ratio:      85.92%
```

保留 split：

```text
train: 6869
val:   1724
test:  3436
```

## 当前判断

- 后续 QA 合成优先使用过滤后的 `12029` 条样本。
- `summary_filtered_30s300s_key10to40.json` 只是统计摘要，不是训练标注入口。
- 当前不做超分；保持原抽帧结果，由模型 processor 或 dataloader 负责 resize/pad。
- `TimeChat-Online-139K` 第一波子集先不进入主训练数据，只作为备用来源。

## VideoXum-StreamQA 目标

每条 QA 至少需要包含：

- `video_id / split / duration`
- `question_id`
- `question_type`: `backward | realtime | forward`
- `question`
- `answer`
- `t_query`
- `evidence_intervals`
- `valid_answer_intervals`
- `t_answer_start`
- `source_keyframes`
- `keyframe_scores`
- `atomic_fact`
- `verification`

三类问题定义：

| 类型 | 时间关系 | 训练语义 |
|---|---|---|
| `backward` | 证据在 `t_query` 之前 | 问题出现后应立即可答 |
| `realtime` | 证据在 `t_query` 附近可见 | 当前 step 可答 |
| `forward` | `t_query` 早于证据出现 | 证据出现前必须等待，之后才答 |

## 新数据构造流程

1. 读取过滤后的 `streamweave_data` annotation，建立视频、帧路径、关键帧分数索引。
2. 聚合 `vsum_onehot` 已得到的 `key_frame_scores`，选择 top keyframes 或阈值关键帧。
3. 将关键帧扩成局部窗口，例如 `[t-2s, t+2s]`。
4. 对窗口调用 teacher MLLM 生成带时间的 `atomic_facts`。
5. 从 `atomic_facts` 生成三类 QA candidate。
6. 做自动验证：
   - evidence-only 必须能答对。
   - `backward` 的 `0..t_query` prefix 必须能答。
   - `realtime` 的当前窗口必须能答。
   - `forward` 的 `0..t_query` prefix 必须不能答，`0..t_answer_start` 必须能答。
   - text-only 可答的题丢弃。
   - summary-only 可答的题标记或按目标丢弃。
7. 导出 `qa_verified.jsonl`。
8. 再导出 V3 teacher traces 和 `sft_steps.jsonl`。

## 建议输出结构

```text
exp2/data/videoxum_streamqa/
  raw_videoxum_index.jsonl
  keyframe_segments.jsonl
  atomic_facts.jsonl
  qa_candidates.jsonl
  qa_verified.jsonl
  teacher_traces.jsonl
  sft_steps.jsonl
  reports/
    verification_stats.json
    human_audit_sample.jsonl
```

## 下一步

1. 监控并完成 `1000` 条第二轮 SFT 数据合成。
2. 对 accepted 样本做结构巡检：图片路径、ShareGPT 格式、note/keyframe 对齐、bridge gap、QA eta/answer 状态。
3. 用 accepted 数据启动第一次 SFT smoke，再扩大到当前合成批次。
4. 第一次 SFT 评测后，再接入新的数据集继续合成 SFT 数据。
5. 并行准备 RL 框架，但在 SFT+评测结果稳定前不把 RL 作为主阻塞项。
