# 第三部分：数据清洗

## 归档状态

本文件保留早期 VideoXum/ActivityNet 数据清洗结论。当前 V5 GRPO 使用 OVO RL 数据，数据清洗不是当前阻塞项。

## 已完成的历史过滤

早期视频/关键帧池：

```text
exp2/data/streamweave_data/annotations_filtered_30s300s_key10to40.jsonl
```

过滤条件：

- `30 <= duration <= 300`
- `0.10 <= key_frame_ratio <= 0.40`

结果：

- 原始：`14001`
- 保留：`12029`
- 该文件只表示视频/关键帧池，不是 QA/SFT/RL 训练入口。

## 后续若重启数据清洗

优先检查：

- 视频帧目录存在。
- `frame_count` 与抽帧文件数一致或误差可解释。
- `key_frame_ids` 全部在 `[0, frame_count)` 内。
- `len(key_frame_scores) == frame_count`。
- `key_frame_ratio == key_frame_count / frame_count`。
- QA 不暴露未来信息、绝对时间、frame id 或答案。

当前不继续展开旧 VideoXum-StreamQA 清洗计划。
