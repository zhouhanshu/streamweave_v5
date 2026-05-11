# dataset2 数据合法性规则

`dataset2` 下每个子目录是一个可独立评测/清洗的数据集。校验入口：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5

python dataset2/validate_dataset.py dataset2/NeXTVideo
```

默认只检查结构、字段、索引、manifest 和每个视频的首/中/尾帧。最终入库前建议跑完整帧检查：

```bash
python dataset2/validate_dataset.py dataset2/NeXTVideo --check-frames full
```

批量检查五个数据集：

```bash
for d in dataset2/CogStream \
         dataset2/LLaVA-Video-178K-PerceptionTest \
         dataset2/LLaVA-Video-178K-YouTube \
         dataset2/NeXTVideo \
         dataset2/TimeChat-Online-139K; do
  python dataset2/validate_dataset.py "$d"
done
```

报告默认写到：

```text
<dataset>/validation_report.json
```

## 目录结构

每个数据集目录必须包含：

```text
annotations.jsonl
video_index.jsonl
video/
```

其中：

- `annotations.jsonl`：一行一个 QA 样本。
- `video_index.jsonl`：一行一个视频/帧目录索引。
- `video/<video_id>/`：1fps 抽帧目录，包含 `000000.jpg` 这类连续帧和推荐的 `manifest.json`。

## annotations 字段

每条 annotation 必须满足：

- `video_id` 非空。
- `task` 属于 `backward`、`realtime`、`forward`。
- `question` 非空。
- `answer` 非空。
- `gt` 非空。
- `sample_fps` > 0。
- `frame_count` > 0。
- `frame_id_base` >= 0，当前默认应为 0。
- `realtime` 存在，并落在当前视频帧范围内。
- `frames_dir` 或 `video` 指向存在的帧目录。

可选字段：

- `sample_id`：推荐提供，但不是硬要求；缺失时后续 loader 可以用行号构造唯一 sample id。
- `options`：选择题必须提供。长度可以是 2 到 26，当前 NeXTVideo 存在 E 选项，所以不能假设只有 A-D。
- `ask_time`、`clue_time`、`target_timestamp`、`duration`：如果存在，必须是非负数。

## 选择题规则

如果存在 `options`：

- `options` 必须是 list。
- 选项数量必须在 2 到 26 之间。
- 每个选项文本不能为空。
- `gt` 可以是：
  - 字母：`A`、`B`、`C`、...
  - 数字：优先按 zero-based 兼容，也允许 one-based；如果 `answer` 与其中一个解释对应，则用 `answer` 消歧。
  - 选项文本：必须能和某个 option 归一化后匹配。
- `source_gt_letter` 如果存在，会优先用于校验。
- `answer` 应该和 `gt` 指向的 option 文本一致；不一致会记 warning。

## video_index 规则

`video_index.jsonl` 每行必须满足：

- `video_id` 非空且唯一。
- `frame_count` > 0。
- `sample_fps` > 0。
- `frame_id_base` >= 0。
- `frames_dir` 或 `video` 指向存在的帧目录。

annotation 中出现的每个 `video_id` 必须能在 `video_index.jsonl` 中找到。

如果 annotation 和 video_index 同时包含以下字段，必须一致：

```text
frame_count
sample_fps
frame_id_base
frames_dir/video
```

## 帧目录规则

帧目录推荐格式：

```text
video/<video_id>/
  000000.jpg
  000001.jpg
  ...
  manifest.json
```

默认 `--check-frames sample` 会检查：

- 帧目录存在。
- `manifest.json` 可读且 `status` 为 `complete` 或缺省。
- `manifest.frame_count` 与 annotation 的 `frame_count` 一致。
- 首帧、中间帧、末帧存在。

`--check-frames full` 会额外检查：

- `frame_id_base` 到 `frame_id_base + frame_count - 1` 的所有帧都存在。
- 目录中是否有超出该范围的额外帧。
- 实际帧数量是否等于 `frame_count`。

## 当前五个数据集

当前目录包括：

```text
CogStream
LLaVA-Video-178K-PerceptionTest
LLaVA-Video-178K-YouTube
NeXTVideo
TimeChat-Online-139K
```

这些数据都应该被转换成同一套 StreamWeave-style `annotations.jsonl + video_index.jsonl + video/` 结构，后续三次推理、难度估计和 SFT/RL 分配脚本都按这套结构接入。

## 评测/SFT 使用口径

当前五个数据集里，按 `options` 字段区分：

| Dataset | QA 形式 | 当前用途 |
| --- | --- | --- |
| `CogStream` | 开放式自然语言 QA，无 `options` | SFT-only，不跑自动评测 |
| `TimeChat-Online-139K` | 开放式自然语言 QA，无 `options` | SFT-only，不跑自动评测 |
| `NeXTVideo` | 选择题，有 `options`，存在 E 选项 | 可跑自动评测/清洗 |
| `LLaVA-Video-178K-PerceptionTest` | 选择题，有 `options` | 可跑自动评测/清洗 |
| `LLaVA-Video-178K-YouTube` | 选择题，有 `options` | 可跑自动评测/清洗 |

开放式自然语言 QA 当前不进入自动评测池，原因是没有稳定的规则 scorer；这类数据只用于 SFT 或后续教师轨迹蒸馏。选择题数据可以用规则 scorer 做 3 次推理成功率、难度估计和训练集分配。
