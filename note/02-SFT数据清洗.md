# SFT 数据清洗

本文只讲当前 SFT 数据从 `dataset2` 层原始转换标注到最终训练文件的链路。上游官方数据下载、抽帧和 `dataset2/<dataset>/video/` 的生成见 `01-数据准备.md`；训练命令见 `04-SFT训练.md`。

## 一句话链路

```text
dataset2 层原始转换标注
  例如 CogStream/annotations.jsonl、NeXTVideo/sft.jsonl、LLaVA-Video-*/sft.jsonl
  -> SFT 源标注抽样：dataset2/sft_0516_4500_source.jsonl
  -> Gemini teacher 逐 step 合成 StreamWeave XML
  -> data_engine/sft/outputs/sft0516_4500/llamafactory_sharegpt.jsonl
  -> hard filter
  -> dataset2/sft_0516_4500.jsonl
```

再往上当然还有官方原始 annotation 和原始视频，但那是 `01-数据准备.md` 的范围。本文说“原始标注”时，默认指 `dataset2` 里保留下来的统一转换标注，不指官方压缩包里的原始 JSON。

这里最容易混的是两层数据：

- `dataset2/<dataset>/annotations.jsonl` / `sft.jsonl` / `rl_tmp.jsonl` 是 dataset2 层的原始转换标注或任务源标注，不是最终训练数据。
- `dataset2/sft_0516_4500.jsonl` 是 step-level ShareGPT，每一行训练模型在一个 stream step 上输出 `<state>/<answer>/<anchor>/<delta>`。

例如 `dataset2/CogStream/annotations.jsonl` 这种文件，已经不是官方 JSON 原样，而是经过本项目处理后的统一标注：它已经绑定了 `video_id`、`frames_dir`、`frame_count`、`sample_fps=1.0`、`question/answer/gt`、`ask_time/clue_time` 和原始来源字段。CogStream 这份文件主要给 RL 重构用；当前 SFT 主线用的是 NeXTVideo / LLaVA-Video / TimeChat 这些数据集下的 `sft.jsonl`。

## 当前主文件

最终 SFT 训练文件：

```text
dataset2/sft_0516_4500.jsonl
```

规模：

```text
rows:           67856
answered_steps: 35384
silent_steps:   32472
```

配套文件：

```text
dataset2/sft_0516_4500_source.jsonl
dataset2/sft_0516_4500_source.summary.json
dataset2/sft_0516_4500.summary.json
dataset2/sft_0516_4500.check_report.json
```

中间输出目录：

```text
data_engine/sft/outputs/sft0516_4500/
```

当前 SFT 主线不使用 CogStream 和 Streamo。CogStream / Streamo 主要进入 RL 数据链路，见 `03-RL数据清洗.md`。

## Stage 0：dataset2 层原始转换标注

这一步由 `01-数据准备.md` 记录。结果是每个数据集都有统一帧目录，并生成 dataset2 层标注：

```text
dataset2/<dataset>/video/<video_id>/000000.jpg ...
dataset2/<dataset>/annotations.jsonl / sft.jsonl / rl_tmp.jsonl
```

其中 `annotations.jsonl` 是最接近“原始转换标注”的文件名，比如：

```text
dataset2/CogStream/annotations.jsonl
```

它的一行大致包含：

```text
dataset / split / sample_id / video_id
video / frames_dir
task / question / answer / gt
sample_fps / frame_count / frame_id_base
realtime / ask_time / clue_time
source_* 原始追溯字段
input_window_start / input_window_end
```

当前保留的 dataset2 层追溯标注：

| 文件 | rows | 用途 |
|---|---:|---|
| `dataset2/CogStream/annotations.jsonl` | 25700 | CogStream 原始转换标注，当前主要给 RL 重构用 |
| `dataset2/NeXTVideo/sft.jsonl` | 2000 | SFT 多 QA 视频源 |
| `dataset2/LLaVA-Video-178K-YouTube/sft.jsonl` | 5000 | SFT 多 QA 视频源 |
| `dataset2/LLaVA-Video-178K-YouTube/sft_extra_2000.jsonl` | 2000 | YouTube 额外 SFT 源 |
| `dataset2/LLaVA-Video-178K-PerceptionTest/sft.jsonl` | 500 | SFT 多 QA 视频源 |
| `dataset2/TimeChat-Online-139K/sft.jsonl` | 5000 | SFT 单 QA text 源 |

SFT 当前没有使用 CogStream，所以 SFT 的起点不是 `CogStream/annotations.jsonl`，而是下面这些已经按 SFT 任务保留下来的 `sft.jsonl` 文件。它们和 `CogStream/annotations.jsonl` 处在同一层：都是抽帧之后、最终训练之前的 dataset2 标注。

| SFT 源 | 当前 SFT 源标注 | 源 QA 形态 | 当前用途 |
|---|---|---|---|
| NeXTVideo | `dataset2/NeXTVideo/sft.jsonl` | 一个视频多个 MCQ | SFT 多 QA 视频源 |
| LLaVA-Video-178K-YouTube | `dataset2/LLaVA-Video-178K-YouTube/sft.jsonl` | 一个视频多个 MCQ | SFT 多 QA 视频源 |
| LLaVA-Video-178K-YouTube extra | `dataset2/LLaVA-Video-178K-YouTube/sft_extra_2000.jsonl` | 一个视频多个 MCQ | YouTube 额外 SFT 源 |
| LLaVA-Video-178K-PerceptionTest | `dataset2/LLaVA-Video-178K-PerceptionTest/sft.jsonl` | 一个视频多个 MCQ turn | SFT 多 QA 视频源 |
| TimeChat-Online-139K | `dataset2/TimeChat-Online-139K/sft.jsonl` | 一个视频一个开放问答 | SFT 单 QA text 源 |

这些文件已经是清洗过的 dataset2 统一 annotation。它们共同遵守：

```text
dataset
video_id
video / frames_dir
sample_fps = 1.0
frame_count
frame_id_base = 0
realtime
```

`frame_count` 是下游有效视频长度；SFT 合成按它切 step，不以文件夹真实帧数自动延长。

## Stage 1：SFT annotation 的两种格式

### 多 QA 视频格式

NeXTVideo、LLaVA-Video YouTube、PerceptionTest 都是多 QA 视频格式。每行是一条视频，`qa_list` 里有多个互相独立的问题：

```json
{
  "dataset": "NeXTVideo",
  "video_id": "nextvideo_1164_3238737531",
  "frames_dir": "NeXTVideo/video/nextvideo_1164_3238737531",
  "sample_fps": 1.0,
  "frame_count": 77,
  "realtime": 76.0,
  "qa_list": [
    {
      "qa_index": 0,
      "question": "how many children are in the video?",
      "options": ["one", "three", "seven", "two", "five"],
      "gt": "D",
      "answer": "two",
      "qa_id": "train_0"
    }
  ]
}
```

这类数据的关键点：

- 一行是一个视频，不是一个 QA。
- `qa_list` 里的问题没有历史依赖，只是共享同一个视频。
- SFT 合成时视频前缀只推理一次，最后一个 step 才为每个 QA 分支单独插入问题并生成答案。
- 只有答案匹配 GT 的 QA 分支会导出到 SFT。

### 单 QA text 格式

TimeChat 是单 QA 格式。每行就是一个视频窗口和一个自然语言问答：

```json
{
  "dataset": "TimeChat-Online-139K",
  "video_id": "timechat_activitynet_v_BYLxSOPFOuc_000246_000463_039962",
  "frames_dir": "video/timechat_activitynet_v_BYLxSOPFOuc_000246_000463_039962",
  "sample_fps": 1.0,
  "frame_count": 218,
  "realtime": 217.0,
  "question": "What can be inferred about the time of day based on the scenes featuring the snowy environment?",
  "answer": "It is nighttime, as indicated by the darkness and artificial lighting.",
  "gt": "It is nighttime, as indicated by the darkness and artificial lighting."
}
```

这类数据的关键点：

- 问题在 `realtime` 对应的 step 注入。
- answer step 必须输出非空 `<answer>`。
- 开放答案除了规则匹配，还允许 Gemini semantic judge 接受等价改写。

## Stage 2：抽 4500 条合成输入

脚本：

```text
dataset2/0516data/sft/build_sft0516_4500_subset.py
```

命令：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python \
  dataset2/0516data/sft/build_sft0516_4500_subset.py \
  --overwrite
```

抽样策略：

| 来源文件 | 可用 rows | 抽样 rows | 说明 |
|---|---:|---:|---|
| `dataset2/NeXTVideo/sft.jsonl` | 2000 | 1000 | 随机抽样，保持源顺序 |
| `dataset2/LLaVA-Video-178K-PerceptionTest/sft.jsonl` | 500 | 500 | 全量保留 |
| `dataset2/LLaVA-Video-178K-YouTube/sft.jsonl` | 5000 | 1000 | 随机抽样，保持源顺序 |
| `dataset2/LLaVA-Video-178K-YouTube/sft_extra_2000.jsonl` | 2000 | 1000 | 随机抽样，保持源顺序 |
| `dataset2/TimeChat-Online-139K/sft.jsonl` | 5000 | 1000 | 随机抽样，保持源顺序 |

输出：

```text
dataset2/sft_0516_4500_source.jsonl
dataset2/sft_0516_4500_source.summary.json
```

抽完后的分布：

```text
total_rows: 4500
LLaVA-Video-178K-YouTube:        2000
NeXTVideo:                       1000
TimeChat-Online-139K:            1000
LLaVA-Video-178K-PerceptionTest:  500
```

`sft_0516_4500_source.jsonl` 仍然是视频级 annotation。4500 行不等于最终训练 4500 行，因为后面会按视频长度切成很多 step。

脚本会检查 `frames_dir/video` 是否真实存在；缺帧目录会直接报错。

## Stage 3：Gemini teacher 合成 step-level XML

入口脚本：

```text
dataset2/0516data/sft/run_dataset2_sft0516_4500_distill.sh
```

启动：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
nohup bash dataset2/0516data/sft/run_dataset2_sft0516_4500_distill.sh \
  > sft0516_4500_distill.out 2>&1 &
```

核心参数：

```text
MODEL=gemini-2.5-flash
WORKERS=128
FRAMES_PER_STEP=5
MAX_IMAGE_SIDE=448
ANSWER_STEP_ROLLOUTS=3
```

实际调用：

```bash
/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python \
  data_engine/sft/run_parallel_pipeline.py \
  --source dataset2 \
  --raw-data-root dataset2 \
  --backend gemini \
  --num-workers 128 \
  --frames-per-step 5 \
  --max-image-side 448 \
  --input dataset2/sft_0516_4500_source.jsonl \
  --output-dir data_engine/sft/outputs/sft0516_4500 \
  --model gemini-2.5-flash \
  --answer-step-rollouts 3
```

合成时每条 source row 会先变成 `SamplePlan`：

```text
dataset2 annotation row
  -> 读取 frame_count 个 1fps frame
  -> 按 frames_per_step=5 分组
  -> 每个 step 构造 Memory + QA History + Current frames
  -> Gemini teacher 输出 XML target
  -> strict validate
  -> accepted action 推进 Memory
```

当前 XML target：

```xml
<state>...</state>
<answer>...</answer>
<delta t="0.0-4.0">...</delta>
<anchor t="4.0-5.0"></anchor>
<delta t="5.0-10.0">...</delta>
```

关键约束：

- `<state>` 必须有，并且在 `<answer>` 前面。
- `<answer>` 必须有标签；不该回答时内容为空。
- `<anchor>` 必须是成对标签，不能写自闭合。
- `<anchor>` 只使用 `t`，不使用 frame id 或 local id。
- `<delta>` 描述窗口间可观察变化。
- 当前 V5 不使用 `<eta>`。
- 训练 target 必须来自 teacher raw valid XML，不能用 repair 后的 action 当 target。

## Stage 4：单 QA 和多 QA 如何变成 step

### 单 QA

TimeChat 这类单 QA 样本按正常 stream 推进：

```text
step 0..N-2: 没有问题，<answer></answer>
step N-1: 注入 question，teacher 输出答案
```

整条样本要被接受，必须满足：

- 所有 step XML 都合法。
- answer step 输出非空答案。
- 最终答案匹配 GT；开放答案可用 semantic judge 判等价。

### 多 QA

NeXTVideo、YouTube、PerceptionTest 的多 QA 样本不是把多个问题串成历史对话。它们是同一个视频上的多个独立问题：

```text
prefix step 0..N-2: 不插问题，只合成视频 memory
final step branch q0: 插入 q0，生成 q0 answer
final step branch q1: 从同一个 prefix memory 复制，插入 q1，生成 q1 answer
...
```

这样做的原因：

- 同一视频的前缀推理只做一次。
- 多个 QA 不互相污染 QA History。
- 每个 QA 分支独立校验答案。
- 只有答案正确的分支导出；答案错的分支丢掉。

这就是为什么 4500 条 source annotation 最后能产生远多于 4500 行的 ShareGPT 训练数据。

## Stage 5：中间产物

Gemini 合成输出目录：

```text
data_engine/sft/outputs/sft0516_4500/
```

核心文件：

```text
sft_jobs.sqlite
sft_steps.jsonl
sample_manifest.jsonl
samples/
llamafactory_sharegpt.jsonl
summary.json
```

合成统计：

```text
input source rows:       4500
accepted source samples: 2376
failed source samples:   2124
sft_steps.jsonl rows:    59496
attempted steps:         87074
variant rescue rows:     11
variant rescue variants: 17
```

`llamafactory_sharegpt.jsonl` 在导出时会展开 answer variants，所以行数变成：

```text
llamafactory_sharegpt.jsonl rows before hard filter: 96092
```

## Stage 6：导出 ShareGPT

导出代码：

```text
data_engine/sft/export_llamafactory.py
```

最终训练行格式：

```json
{
  "messages": [
    {"role": "user", "content": "...production prompt with Memory / QA History / Current frames..."},
    {"role": "assistant", "content": "<state>...</state><answer>...</answer>..."}
  ],
  "images": ["NeXTVideo/video/.../000000.jpg"]
}
```

导出规则：

- 训练输入重建 production prompt，不使用 teacher prompt。
- 不把 retry feedback、teacher-only instruction 写入训练数据。
- `messages` 只有 user / assistant。
- `content` 里的 `<image>` 数量必须等于 `images` 数量。
- 训练时 LLaMAFactory 的 `media_dir` 指向 `dataset2/`。

## Stage 7：Hard Filter 和最终文件

入口脚本：

```text
dataset2/0516data/sft/finalize_sft0516_4500.sh
```

命令：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
bash dataset2/0516data/sft/finalize_sft0516_4500.sh
```

脚本做三件事：

```text
1. data_engine/sft/filter_sharegpt_hard.py
2. dataset2/0516data/sft/merge_sft0516_4500.py
3. data_engine/sft/check_sharegpt.py
```

Hard filter 规则：

- assistant target 里的 `<delta>` 跨度超过 20 秒，删除该行。
- user prompt 的 Memory 中已有 `<delta>` 跨度超过 20 秒，删除该行。
- 完全重复行删除。
- 图片路径以 `dataset2` 为根检查。

Hard filter 结果：

```text
input rows:       96092
kept:             67856
dropped:          28236
answered_kept:    35384
silent_kept:      32472
answered_dropped: 18474
silent_dropped:    9762
```

drop reason：

```text
memory_delta_over_threshold: 26577
target_delta_over_threshold:  5689
duplicate_row:                 610
```

注意：一行可能同时命中多个 drop reason，所以 reason 数量之和可以大于 dropped 行数。

最终输出：

```text
dataset2/sft_0516_4500.jsonl
dataset2/sft_0516_4500.summary.json
dataset2/sft_0516_4500.check_report.json
```

最终数据集分布：

```text
LLaVA-Video-178K-YouTube:        32516
NeXTVideo:                       21853
LLaVA-Video-178K-PerceptionTest:  8003
TimeChat-Online-139K:             5484
```

格式检查结果：

```text
rows: 67856
answered_steps: 35384
silent_steps: 32472
missing_answer_tag: 0
format_error_rows: 0
sharegpt_structure_error_rows: 0
delta_over_threshold_rows: 0
target_delta_over_threshold_rows: 0
memory_delta_over_threshold_rows: 0
```

## End-to-End 重建命令

从现有 `dataset2/*/sft.jsonl` 重新生成当前 0516 SFT：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5

/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python \
  dataset2/0516data/sft/build_sft0516_4500_subset.py \
  --overwrite

WORKERS=128 \
MODEL=gemini-2.5-flash \
FRAMES_PER_STEP=5 \
MAX_IMAGE_SIDE=448 \
ANSWER_STEP_ROLLOUTS=3 \
OVERWRITE_OUTPUTS=1 \
bash dataset2/0516data/sft/run_dataset2_sft0516_4500_distill.sh

bash dataset2/0516data/sft/finalize_sft0516_4500.sh
```

只补失败 job：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
RERUN_FAILED=1 bash dataset2/0516data/sft/run_dataset2_sft0516_4500_distill.sh
bash dataset2/0516data/sft/finalize_sft0516_4500.sh
```

复查命令：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
wc -l dataset2/sft_0516_4500.jsonl
cat dataset2/sft_0516_4500_source.summary.json
cat dataset2/sft_0516_4500.summary.json
cat dataset2/sft_0516_4500.check_report.json
```

## 历史 SFT 数据

上一轮主文件：

```text
dataset2/sft_0511.jsonl
dataset2/sft_0511_note.jsonl
```

规模：

```text
sft_0511.jsonl:      347695 rows
sft_0511_note.jsonl: 182998 rows
```

`sft_0511_note.jsonl` 是按完整视频/rollout 的 anchor/note 占比 `[9%, 15%]` 过滤得到。它是历史可用数据，不是当前 0516 默认训练文件。

answered-full SFT 历史数据：

```text
data_engine/sft/outputs/gemini_answered_full/llamafactory_sharegpt_anchor_delta_le20.jsonl
```

## 注意事项

- 不要把 `dataset2/<dataset>/sft.jsonl` 当成可以直接训练的 ShareGPT。
- 当前 SFT 训练入口是 `dataset2/sft_0516_4500.jsonl`，LLaMAFactory 注册见 `04-SFT训练.md`。
- SFT 数据不要混入 teacher-only prompt、retry feedback 或旧协议。
- `<anchor>` 必须用成对标签。
- `<eta>`、`frame="N"`、旧 note_t 协议不能进入当前 target。
- Open-ended QA 的答案校验比 MCQ 更弱，合成后必须回评。
- 大并发 Gemini 合成时不要让每个 worker 重复加载全量样本；0516 的 parallel pipeline 已修复为主进程加载后 fork 使用。
