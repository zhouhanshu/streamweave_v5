# 第二部分：数据构造

## 当前状态

数据构造已经不是当前阻塞项。当前 V5 RL 使用的是 OVO 单 query RL 数据；V4 的 VideoXum/ActivityNet SFT 数据链路保留为历史和可选后续扩展。

## 当前 RL 数据

目录：

```text
/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/ovo
```

文件：

```text
ovo_bench_new.json        原始 OVO 标注
ovo_rl.json              单 query RL 数据，约 600 条
ovo_rl_lt120s.json       <120s 子集，293 条
```

`ovo_rl.json` 口径：

- `backward/forward/realtime` 各约 `200` 条。
- 单 query 样本。
- `sample_id/video_id` 已处理为 dataset 可直接读取的形态。
- `forward` 样本已展开并移除 `test_info`，避免 RL dataset 再重复展开。

当前 GRPO 脚本使用：

```text
train_files = ovo_rl_lt120s.json
val_files   = ovo_rl_lt120s.json
```

## V4 SFT 数据

SFT 标注入口：

```text
/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data/annotations_qa_filter_final.jsonl
```

raw data root：

```text
/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data
```

已验证小样本：

```text
data_engine/sft/outputs/gemini_final_8
accepted=7/8
sft_steps=136
```

注意：第一次 SFT 回评退化，因此这条链路只能说明数据格式和导出已打通，不能说明训练效果已经正向。

## 历史视频/关键帧池

- 视频源：`ActivityNet_Captions`
- 标注源：`VideoXum`
- 已构造样本：`14001`
- 过滤后早期样本：`12029`
- 抽帧：`1fps`
- `key_frame_ids`：0-based，对应 `{frame_id:06d}.jpg`
- 过滤阈值：`threshold=0.3`

早期文件：

```text
exp2/data/streamweave_data/annotations_filtered_30s300s_key10to40.jsonl
```

该文件只是视频、帧路径和 query-free 关键帧池，不是 SFT/RL 训练入口。

## 后续原则

- 当前先不要扩大 VideoXum SFT 数据，优先稳定 V5 GRPO。
- 如果要回到 SFT 数据构造，先排查第一次 SFT 退化原因。
- 新数据接入必须复用 production prompt、XML parser、quality validator 和 accepted-only 导出规则。
