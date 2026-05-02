# StreamWeave SFT 数据合成说明

这个目录负责把已有视频帧标注合成为 StreamWeave 的 step-level SFT 数据，并导出 LLaMAFactory 可读的 ShareGPT 多图格式。

当前框架已经支持：

- legacy extracted-frame 标注输入，也就是 `frames` source。
- OVO-Bench 适配输入，也就是 `ovo` source。
- 没有 QA 的纯视频记忆合成样本。
- 有 QA 的 realtime / backward / forward 样本。
- 标注关键帧强约束。
- teacher 合成时多轮 retry。
- sample 级别失败保留和 accepted 过滤。
- SQLite 动态多进程任务队列。
- 断点续跑。
- 进度条。
- LLaMAFactory ShareGPT 导出。
- 中间文件转可读 txt 以便人工检查。

目前默认面向这个数据集：

```text
/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data/annotations_qa_filter_final.jsonl
```

默认图片根目录：

```text
/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data
```

## 代码结构

```text
data_engine/sft/
  schemas.py                 # SFT 内部统一数据协议：SamplePlan / FrameRef / QueryPlan
  sample_sources.py          # source 分发入口，把不同数据集转成 SamplePlan
  frame_dataset.py           # extracted-frame / streamweave_data 标注适配器
  ovo_dataset.py             # OVO-Bench 适配器
  rollout_sft.py             # SFT 合成主逻辑：逐 step 调 teacher、校验、retry、sample 过滤
  run_pipeline.py            # 单进程 pipeline：intermediate / finalize / sharegpt
  run_parallel_pipeline.py   # 动态多进程队列，推荐全量使用
  export_llamafactory.py     # accepted step -> LLaMAFactory ShareGPT
  inspect_intermediate.py    # 中间 JSON/JSONL -> 人类可读 txt
  io_utils.py                # JSON/JSONL/路径工具
  README.md
```

StreamWeave 运行时相关逻辑在：

```text
streamweave/
  prompts.py       # teacher_synthesis / production prompt
  parser.py        # XML 解析和严格格式检查
  quality.py       # note/bridge 时间、gap、open-tail 等质量校验
  postprocess.py   # retry feedback 生成，以及正式推理 repair
  env.py           # Memory / QA / prompt / commit 的状态机
  memory.py        # note / bridge / QA history 的内存结构
```

## 总体流程

```text
annotation json/jsonl
  -> source adapter
  -> SamplePlan
  -> rollout_sft.py
  -> samples/<task_index>_<sample_id>.json
  -> sample_manifest.jsonl
  -> sft_steps.jsonl
  -> export_llamafactory.py
  -> llamafactory_sharegpt.jsonl
  -> dataset_info_streamweave_sft.json
```

核心思想是：**worker 阶段只写单样本 JSON，全局文件最后统一重建**。这样方便并发、断点续跑和 debug。

每条样本会先被转成统一的 `SamplePlan`：

```text
sample_id
video_id
qa_id
task
query_events
question_text
query_time
answer_time
frames
metadata
```

然后按 `--frames-per-step` 切成多个 step。默认是 5 帧一个 step。

每个 step 的核心输入输出是：

```text
memory_before + qa_history + current_frames -> target_xml
```

`target_xml` 是 teacher 生成并通过校验后的 XML：

```xml
<eta>...</eta>
<answer>...</answer>
<bridge t="...">...</bridge>
<note t="..." frame="N"></note>
```

注意：当前输出 note 必须使用成对标签：

```xml
<note t="36.0-37.0" frame="2"></note>
```

不要使用 self-closing：

```xml
<note t="36.0-37.0" frame="2"/>
```

## Prompt 逻辑

合成阶段默认使用：

```text
--prompt-type teacher_synthesis
```

导出训练数据时默认使用：

```text
--train-prompt-type production
```

也就是说：

```text
teacher_synthesis prompt + Gemini 输出 -> target_xml
production prompt + target_xml -> ShareGPT 训练样本
```

这样做的目的是：teacher 可以看到更多合成约束和 retry feedback，但学生训练时看到的是正式 runtime 的 production prompt。

如果要保留中间文件里记录下来的原始 prompt，可以导出时使用：

```bash
--train-prompt-type recorded
```

一般不建议这么做，因为 recorded prompt 会包含 teacher_synthesis 的合成提示和 retry 痕迹。

## 关键帧约束

标注里可以包含：

```text
key_frame_ids
selected_key_frame_ids
```

它们是 **全局帧 id**。

在每个 step 中，代码会把全局帧 id 转成当前窗口里的局部 `frame="N"`。比如当前窗口是：

```text
frame_id=1 global_frame_id=20 t=20.0-21.0
frame_id=2 global_frame_id=21 t=21.0-22.0
frame_id=3 global_frame_id=22 t=22.0-23.0
```

如果标注关键帧是 `global_frame_id=22`，teacher 必须输出：

```xml
<note t="22.0-23.0" frame="3"></note>
```

如果当前 step 有标注关键帧：

- 必须为每个 required local frame id 输出 exactly one `<note>`。
- 不允许漏掉标注帧。
- 不允许输出未标注帧。
- `<note t="...">` 必须和该 current frame 的时间范围一致。

如果当前 step 没有标注关键帧：

- 不允许输出任何 `<note>`。
- 只能使用 bridge-only observation。

这套约束只在 teacher 合成 prompt 中出现，不会进入默认 production 训练 prompt。使用新的标注文件时，需要保证标注关键帧目标和 production prompt 的一般语义不明显冲突。

## Bridge 和 Open-tail 规则

Bridge 不是自由摘要，它有严格结构约束。

基本规则：

- bridge 时间必须落在当前 step 或合法 open-tail 范围内。
- 事件不能重叠或乱序。
- note 和窗口边界之间如果存在时间 gap，必须有 exactly one bridge。
- 不能把一个 required gap 拆成多个 bridge。
- 不能输出不对应任何 required gap 的 bridge。

Open-tail 继承规则：

如果 Memory 最后一个 observation 是：

```xml
<bridge t="A-B">...</bridge>
```

并且后面没有 note，那么下一步第一个 bridge 要继承 `A`：

```xml
<bridge t="A-C">...</bridge>
```

如果当前 step 里没有 note，`C` 是当前窗口结束时间。

如果当前 step 里有 note，继承 bridge 只能到第一个 current note 的起点：

```xml
<bridge t="A-note_start">...</bridge>
<note t="note_start-note_end" frame="N"></note>
```

踩过的坑：模型经常因为“动作语义连续”把 bridge 往前扩，例如当前窗口是 `10.0-15.0`，但输出：

```xml
<bridge t="8.0-15.0">...</bridge>
```

如果 Memory 最后不是 bridge，而是 note，这种输出是错的。retry prompt 现在会明确提示 required gap，并要求使用 exact interval。

另一个坑是 open-tail + current note。模型容易输出：

```xml
<bridge t="10.0-25.0">...</bridge>
<note t="24.0-25.0" frame="5"></note>
```

这会让 bridge 覆盖 note。合法写法应该是：

```xml
<bridge t="10.0-24.0">...</bridge>
<note t="24.0-25.0" frame="5"></note>
```

## QA / eta / answer 规则

没有问题时：

```xml
<eta></eta>
<answer></answer>
```

问题已经回答过时，后续 step 也应保持：

```xml
<eta></eta>
<answer></answer>
```

realtime / backward：

- 问题进入 QA History 后，当前 active unanswered question 要回答。
- `<answer>` 必须非空。
- `<eta>` 只需要落在目标 step 的时间窗口内，不再要求必须等于窗口最后一帧时间。

例如问题在 `20.0` 出现，当前 step 是 `20.0-25.0`，下面两个都允许：

```xml
<eta>20.0</eta>
<eta>25.0</eta>
```

forward：

- 问题出现但还没到 clue window 时，`<answer>` 必须为空。
- `<eta>` 预测 clue_time 所在窗口内的任意时间戳。
- 到达 clue window 后，`<answer>` 必须非空，`<eta>` 仍然只需要在目标窗口内。

重要：retry 阶段只检查：

- eta 是否为空或落在目标窗口。
- answer state 是否应为空或非空。

retry 阶段不会把 GT answer 泄露给 teacher，也不会检查答案内容是否正确。

答案内容正确性在 sample 结束后做 sample-level 检查：

- 如果标注有 `options` 和 `gt`，会检查模型最终 answer 是否匹配 GT 选项。
- 如果标注没有 QA / GT，则要求整条样本不要输出 answer。
- 只有答案正确且所有 step 都通过，样本才会进入训练数据。

## Retry 机制

有两层 retry。

Backend retry：

- 处理网络错误、429、5xx、超时等后端调用问题。
- 在 `backend/retry.py`。
- Gemini safety block 通常视为 non-retryable。样本会失败并保留错误信息。

Synthesis retry：

- teacher 输出 XML 后先做 parser / quality / QA / key-frame 校验。
- 如果失败，会把 previous raw output 和 errors 拼进 retry prompt。
- 最多重试 `--max-attempts` 次，默认 3 次。

常见 retry 错误：

```text
text_outside_tags
note_tag_format
bridge_time_oob
event_overlap
missing_bridge_gap
invalid_bridge_gap
missing_open_tail_bridge
open_tail_start_mismatch
annotated_key_frame_note_mismatch
qa_eta_answer_mismatch
backend_generate_error
```

如果某个 step 所有 attempts 都失败，这条 sample 立即停止，之前已经成功的 step 也不会进入训练数据。完整过程仍保存在该 sample JSON 里。

## Sample 级状态

每个 sample 最终有一个状态：

```text
accepted
failed
error
```

`accepted` 必须同时满足：

- 所有 step 都 valid。
- 每个 target XML 非空。
- 如果有 GT answer，answer correctness 通过。
- 如果没有 GT answer，没有产生任何 answer。

`failed` 表示合成正常运行完或中途被校验拦住，但不满足 SFT 可用条件，例如：

- `synthesis_raw_retry_failed_at_step_N`
- `answer_incorrect`

`error` 表示 worker 级异常，例如图片读不到、代码异常、后端异常未被合成流程吸收。

只有：

```json
"status": "accepted",
"usable_for_sft": true
```

的样本会进入：

```text
sft_steps.jsonl
llamafactory_sharegpt.jsonl
```

失败和错误样本不会进入训练数据，但会保留在：

```text
samples/*.json
sample_manifest.jsonl
```

## 输出文件

一次运行输出目录一般是：

```text
output_dir/
  samples/
    000000_<sample_id>.json
    000001_<sample_id>.json
    ...
  sft_jobs.sqlite
  sample_manifest.jsonl
  sft_steps.jsonl
  llamafactory_sharegpt.jsonl
  dataset_info_streamweave_sft.json
  summary.json
```

文件含义：

```text
samples/*.json
  单样本完整记录，包含所有 step、attempts、retry feedback、quality、raw output、target_xml。

sft_jobs.sqlite
  动态多进程任务队列和断点状态。

sample_manifest.jsonl
  每条 sample 的最终状态索引。

sft_steps.jsonl
  只包含 accepted sample 的成功 step。每行是一个 step-level 中间表示。

llamafactory_sharegpt.jsonl
  LLaMAFactory ShareGPT 训练数据。

dataset_info_streamweave_sft.json
  LLaMAFactory dataset_info 片段。

summary.json
  本次运行汇总。
```

`samples/*.json` 是 debug 最重要的文件。里面每个 step 都有：

```text
memory_before
memory_before_text
qa_history
current_frames
raw_teacher_xml
target_xml
quality
prompt
attempts
accepted_attempt_index
metadata.raw_action
metadata.applied
```

## 单进程使用

跑单条样本时用 `run_pipeline.py`。例如跑第 3 条样本：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4

python data_engine/sft/run_pipeline.py \
  --input /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data/annotations_qa_filter_final.jsonl \
  --raw-data-root /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data \
  --backend gemini \
  --model gemini-2.5-pro \
  --offset 2 \
  --limit 1 \
  --frames-per-step 5 \
  --max-attempts 3 \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/data_engine/sft/outputs/debug_third_sample_single \
  --overwrite
```

不指定 `--stage` 时默认是：

```text
--stage all
```

会依次执行：

```text
intermediate -> finalize -> sharegpt
```

也就是同时生成单样本 JSON、manifest、sft_steps、ShareGPT。

只想先生成 sample JSON：

```bash
python data_engine/sft/run_pipeline.py ... --stage intermediate
```

## 多进程全量使用

全量推荐使用 `run_parallel_pipeline.py`。它现在是动态 SQLite 队列，不再把 offset 静态切给 worker。

全量新跑：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4

python data_engine/sft/run_parallel_pipeline.py \
  --num-workers 4 \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/data_engine/sft/outputs/gemini_final_full \
  --overwrite
```

上面命令已经默认使用：

```text
--input /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data/annotations_qa_filter_final.jsonl
--raw-data-root /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data
--backend gemini
--model gemini-2.5-pro
--frames-per-step 5
--max-attempts 3
```

小规模测试 8 条：

```bash
python data_engine/sft/run_parallel_pipeline.py \
  --num-workers 4 \
  --limit 8 \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/data_engine/sft/outputs/gemini_final_8 \
  --overwrite
```

断点续跑：

```bash
python data_engine/sft/run_parallel_pipeline.py \
  --num-workers 4 \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/data_engine/sft/outputs/gemini_final_full
```

注意：断点续跑时不要加 `--overwrite`。

重跑 failed / error 样本：

```bash
python data_engine/sft/run_parallel_pipeline.py \
  --num-workers 4 \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/data_engine/sft/outputs/gemini_final_full \
  --rerun-failed
```

只生成中间样本和 manifest，不导出 ShareGPT：

```bash
python data_engine/sft/run_parallel_pipeline.py \
  --num-workers 4 \
  --limit 8 \
  --output-dir /tmp/streamweave_sft_test \
  --overwrite \
  --no-sharegpt
```

## 进度条说明

运行时会打印：

```text
[progress] [########------------------------] 123/6956 accepted=100 failed=20 error=3 running=4 pending=6829 rate=0.42/min eta=...
```

字段含义：

```text
[########------------------------]
  进度条。# 是完成，- 是未完成。

123/6956
  已完成样本数 / 总样本数。完成包括 accepted、failed、error。

accepted
  通过所有校验且会进入训练数据的样本数。

failed
  合成正常运行但样本不可用的数量，例如 retry 用尽或 answer_incorrect。

error
  worker 级异常数量。

running
  当前正在被 worker 处理的样本数。

pending
  还没开始处理的样本数。

rate
  平均完成速度，单位 sample/min。

eta
  预计剩余时间。刚开始没有完成样本时显示 --。
```

## 断点续跑机制

动态并发使用：

```text
sft_jobs.sqlite
samples/*.json
```

作为状态源。

默认行为：

- 如果输出目录里已经有 completed sample JSON，并且没有 `--overwrite`，则不会重复跑。
- `accepted` / `failed` / `error` 都算 completed。
- 运行结束后会重新扫描所有 sample JSON，重建全局 `sample_manifest.jsonl`、`sft_steps.jsonl`、ShareGPT。

使用建议：

- 新实验用新 `--output-dir`。
- 重新从零跑同一个目录才加 `--overwrite`。
- 续跑不要加 `--overwrite`。
- 想只重跑失败样本，加 `--rerun-failed`。

## 查看中间过程

用 `inspect_intermediate.py` 可以把 sample JSON 或 `sft_steps.jsonl` 转成人类可读 txt。

```bash
python data_engine/sft/inspect_intermediate.py \
  data_engine/sft/outputs/debug_third_sample_single/samples/v_fzp5ooc727c_5_backward.json
```

默认输出：

```text
data_engine/sft/outputs/debug_third_sample_single/samples/v_fzp5ooc727c_5_backward.inspect.txt
```

指定输出：

```bash
python data_engine/sft/inspect_intermediate.py \
  data_engine/sft/outputs/debug_third_sample_single/samples/v_fzp5ooc727c_5_backward.json \
  --output /tmp/debug_third_sample_single.inspect.txt
```

只看某一步：

```bash
python data_engine/sft/inspect_intermediate.py \
  data_engine/sft/outputs/debug_third_sample_single/samples/v_fzp5ooc727c_5_backward.json \
  --step 4
```

只看失败 attempts：

```bash
python data_engine/sft/inspect_intermediate.py \
  data_engine/sft/outputs/debug_third_sample_single/samples/v_fzp5ooc727c_5_backward.json \
  --attempts failed
```

这个脚本会保留每一步实际变化的 prompt 输入：

```text
[Actual Input]
=== Memory ===
...
=== Current frames ===
...
=== QA History ===
...
=== Teacher Context ===
...
=== Retry Feedback ===
...
```

并删除每一步重复的大段内容：

```text
[STREAM_AGENT]
Few-shot Example
Task Instructions
Other Rules
Output Format
```

## 检查运行结果

看 summary：

```bash
cat data_engine/sft/outputs/gemini_final_8/summary.json
```

看每条样本状态：

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("data_engine/sft/outputs/gemini_final_8/sample_manifest.jsonl")
for line in p.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    print(
        row["sample_id"],
        row["status"],
        "answer_correct=", row["answer_correct"],
        "reason=", row.get("failure_reason"),
    )
PY
```

检查 ShareGPT 行数和图片占位：

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("data_engine/sft/outputs/gemini_final_8/llamafactory_sharegpt.jsonl")
rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
bad = []
for i, row in enumerate(rows):
    content = row["messages"][0]["content"]
    images = row.get("images") or []
    if content.count("<image>") != len(images):
        bad.append((i, content.count("<image>"), len(images)))
print("rows:", len(rows))
print("placeholder mismatches:", len(bad))
print(bad[:10])
PY
```

## LLaMAFactory 使用注意

导出的训练文件：

```text
llamafactory_sharegpt.jsonl
```

格式是：

```json
{
  "messages": [
    {"role": "user", "content": "...<image>..."},
    {"role": "assistant", "content": "<eta>...</eta>\n<answer>...</answer>\n..."}
  ],
  "images": ["video/xxx/000000.jpg", "..."]
}
```

`dataset_info_streamweave_sft.json` 是 dataset_info 片段，需要合并到 LLaMAFactory 的 `dataset_info.json`，或者把输出目录作为 dataset_dir 并把文件改成 LLaMAFactory 期望的位置。

图片路径一般是相对 `raw-data-root` 的，例如：

```text
video/v_xxx/000000.jpg
```

所以训练时 LLaMAFactory 的 `media_dir` / image root 应该指向：

```text
/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data
```

建议先用小规模输出做 smoke training，不要直接上全量训练。

## 常见坑和当前处理

### 1. SFT legacy frames 默认 1fps

`frame_dataset.py` 已经支持 `sample_fps` / `fps`。帧时间现在按：

```text
seconds_per_frame = 1 / fps
```

计算，不再硬编码每帧 1 秒。当前 `streamweave_data` 里 `sample_fps=1.0`，所以表现仍然是 1fps。

### 2. 没有 QA 的样本也要能跑

没有 `question/options/gt` 的样本会作为 `task=synthesis` 运行。要求整个样本：

```xml
<eta></eta>
<answer></answer>
```

如果模型在无 QA 样本中输出 answer，sample-level check 会把样本判 failed。

### 3. note 标签格式

当前输出必须使用：

```xml
<note t="..." frame="N"></note>
```

self-closing note 会被 `note_tag_format` 拦住。

### 4. 关键帧是全局 id，note frame 是局部 id

标注里的 `key_frame_ids` 是 global frame id。`<note frame="N">` 里的 `N` 是当前 step 内的 local frame id。

这个映射由合成代码完成，prompt 中直接给 teacher 当前 step 应该输出的 local frame id。

### 5. bridge gap 必须完整

prompt 说 note 和窗口边界之间要有 bridge，validator 也已经补上了硬约束。

如果漏 bridge，会报：

```text
missing_bridge_gap
```

如果 bridge 多写或时间不对应 required gap，会报：

```text
invalid_bridge_gap
duplicate_bridge_gap
```

### 6. open-tail 不是语义连续就能继承

只有 Memory 最后一个 observation 是 bridge 时，才允许继承旧 bridge 起点。

如果 Memory 最后是 note，当前 step 的 bridge 必须从当前窗口或 note boundary 开始。

### 7. QA retry 不泄露 GT answer

retry 阶段不会告诉模型正确选项是什么。它只告诉模型：

- eta 是否应该为空或落在哪个时间窗口。
- answer 应该为空还是非空。

答案内容正确性只在 sample 结束后过滤。

### 8. eta 现在是窗口约束

之前 eta 被要求等于目标窗口最后一帧时间，模型容易把 QA History 的问题时间写进 eta 导致失败。

现在规则改成：

```text
eta 只要落在目标窗口内即可
```

例如目标窗口 `20.0-25.0`，`20.0` 和 `25.0` 都可接受。

### 9. Gemini safety block

Gemini 可能返回：

```text
PROHIBITED_CONTENT
```

这种通常不是格式错误，而是后端安全策略拦截。当前会把该样本标记为 failed/error 并保留信息。全量跑时要接受少量样本因此失败，后续可以考虑换模型、改安全阈值或跳过后重跑。

### 10. accepted 比例不等于模型生成成功率

有些样本所有 step 都能生成，但最终答案不对，会因为：

```text
answer_incorrect
```

被过滤掉。这是预期行为，因为我们只把可作为监督目标的样本导出到训练集。

## 新增数据集方式

后续如果加入新数据集，不要改 `rollout_sft.py` 和 `export_llamafactory.py`。

推荐做法：

1. 新增一个平级 adapter 文件，例如：

```text
my_dataset.py
```

2. 把原始标注转换成 `schemas.py` 里的 `SamplePlan`。

3. 在 `sample_sources.py` 注册新的 source。

4. 确保每个 `FrameRef` 至少有：

```text
global_frame_id
start_time
end_time
image_path
frame_index
```

5. 如果有 QA，填好：

```text
task
query_events
query_time
answer_time
metadata.options
metadata.gt
metadata.answer
```

6. 如果有关键帧，放在：

```text
metadata.key_frame_ids
metadata.selected_key_frame_ids
```

这样 rollout、retry、sample-level filtering、ShareGPT 导出都可以复用。
