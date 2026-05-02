# 实验记录索引

## 当前状态

- 实验：`StreamWeave`
- 当前阶段：`SFT 数据合成已打通 / 第二轮数据合成进行中 / 首次 SFT 训练准备`
- 当前主文件：`data_engine/sft/README.md`、`数据合成.md`、`04-sft-training.md`、`08-commands-and-tools.md`
- 当前主代码：`exp2/streamweave_v4`
- 当前目标：先用现有 `streamweave_data` 合成稳定的 step-level SFT 数据，完成第一次 SFT 和评测；随后接入新的数据集继续合成 SFT 数据，并开始搭建 RL 框架。
- 当前 SFT 标注入口：`exp2/streamweave_v4/dataset/streamweave_data/annotations_qa_filter_final.jsonl`
- 当前原始帧目录：`exp2/streamweave_v4/dataset/streamweave_data`
- 当前教师后端：`Gemini / gemini-2.5-pro`
- 当前默认环境：`simple`
- 当前评测入口：等 SFT 数据和训练结果出来后，把新跑分写入 `实验跑分.md`，`06-evaluation.md` 只保留回评规则
- 当前进展：
  - `Qwen3-VL-32B-Instruct plain` baseline 已完成
  - `StreamWeave-v2 OVO 1/8` 探索性对照已完成
  - `exp2/streamweave_v4` 已成为当前 SFT 数据合成和后续训练主线
  - `VideoXum` 标注已下载并解压特征
  - `ActivityNet_Captions` 视频已下载解压，VideoXum 的 `14001` 个 video_id 全部匹配
  - `streamweave_data` 已全量构造，1fps 抽帧完成
  - SFT 合成链路已经打通：teacher synthesis 生成中间表示，最终导出 production prompt 的 LLaMAFactory ShareGPT 数据
  - 小规模 `gemini_final_8` 已完成，`7/8` 条样本 accepted，失败的第一条属于已知问题样本
  - 第二轮数据合成正在进行，目标先跑 `1000` 条样本，为第一次 SFT 训练和评测做准备

## 文件索引

- [00-overview.md](./00-overview.md)：实验目标、总体路线、当前策略
- [01-idea-validation.md](./01-idea-validation.md)：Idea 验证阶段记录
- [02-data-construction.md](./02-data-construction.md)：数据构造
- [03-data-cleaning.md](./03-data-cleaning.md)：数据清洗
- [04-sft-training.md](./04-sft-training.md)：SFT 训练
- [05-rl-training.md](./05-rl-training.md)：RL 训练
- [06-evaluation.md](./06-evaluation.md)：评测入口、对比原则与后续回评规则
- [实验跑分.md](./实验跑分.md)：独立跑分主表，区分 full/subset、旧 adapter、smoke/debug 和外部参考
- [07-key-points.md](./07-key-points.md)：关键结论与避坑
- [08-commands-and-tools.md](./08-commands-and-tools.md)：当前有效命令与环境
- [09-streamweave-proposal-draft.md](./09-streamweave-proposal-draft.md)：完整提案原文
- [数据合成.md](./数据合成.md)：数据下载、解压、构造与过滤记录
- [README.md](./README.md)：接手说明

## 维护规则

- 本文件只保留当前状态、索引和短更新。
- 详细执行过程写到对应阶段文件。
- 错误步骤不在这里展开，只在 `07-key-points.md` 留摘要。

## 简要更新

### 2026-05-02 StreamWeave V4 SFT 链路打通

- 当前主线从 `streamweave_v3` 数据构造方案推进到 `streamweave_v4` 的实际 SFT 数据合成链路。
- SFT pipeline 已重构为样本级落盘：
  - 每条视频/QA 样本单独写入 `data_engine/sft/outputs/<run>/samples/*.json`。
  - 只有整条样本所有 step 合法、且样本级答案正确时，才进入全局 `sft_steps.jsonl` 和 `llamafactory_sharegpt.jsonl`。
  - failed/error 样本保留在 `samples/` 和 `sample_manifest.jsonl` 中用于排查，但不进入训练数据。
- 多进程入口 `data_engine/sft/run_parallel_pipeline.py` 已改为动态任务队列：
  - 使用 SQLite job queue，worker 完成一个样本后继续领取下一个样本。
  - 支持断点续跑；不带 `--overwrite` 时会复用已完成 sample 文件。
  - 进度条显示 `accepted / failed / error / running / pending / rate / elapsed / eta`。
- 当前验证过的小规模结果：
  - 输出目录：`data_engine/sft/outputs/gemini_final_8`
  - 输入：`dataset/streamweave_data/annotations_qa_filter_final.jsonl`
  - 结果：`accepted=7/8`，`failed=1`，`error=0`，导出 `136` 条 step-level SFT 样本。
  - 已检查 accepted 样本：ShareGPT 图片路径存在、prompt 未混入 teacher-only 指令或 retry feedback、note 命中标注关键帧、QA eta 落入目标时间窗。
- 当前第二轮合成计划：
  - 目标：先跑 `1000` 条样本。
  - 推荐输出目录：`data_engine/sft/outputs/gemini_final_1000_w64`
  - 推荐并发：`64` workers。
  - 该轮完成后先做数据巡检，再启动第一次 SFT，并在 OVO/StreamingBench 口径下回评。
- 后续方向：
  - 使用新的数据集继续合成 SFT 数据，验证当前 pipeline 对新增 dataset schema 的兼容性。
  - 在第一轮 SFT 和评测结果稳定后，开始搭建 RL 框架，优先补 rollout env、reward、格式/时间/答案检查和训练脚本。

### 2026-04-28 跑分记录整理

- 新增 `实验跑分.md` 作为独立跑分主表。
- 已把本地正式 baseline、探索性跑分、旧 adapter 诊断结果、smoke/debug 和外部参考表分开记录。
- `06-evaluation.md` 已收缩为评测入口和回评规则，不再承载详细跑分。
- `StreamWeave V3` 在 `2026-04-28` 前的 `Gemini 1/8` 和 `Qwen3-VL-8B full` 结果标记为旧 adapter 口径，后续正式对比需要重跑。

### 2026-04-28 Gemini V3 新 adapter 1/8 高错误率记录

- 结果文件：`exp2/streamweave_v3/outputs/ovo_gemini_1of8/results.jsonl`
- 口径：`StreamWeave V3 / gemini-2.5-pro / OVO 1/8 / teacher prompt / workers=32`
- 新 adapter：forward query 在 `t=0.0` 注入，`CRR` 改成判断当前视觉内容是否足够回答。
- task macro：`Backward 70.22`，`Realtime 68.03`，`Forward 40.10`，`Total 59.45`
- sample weighted：`209/364 = 57.42%`
- error：`37/364`，其中 `16x ClientError 499 CANCELLED`，`16x ServerError 504 DEADLINE_EXCEEDED`，`4x ReadTimeout`，`1x Gemini response has no text content`
- 错误主要来自 Gemini/Vertex 请求取消和超时，不是视频路径错误；`workers=32` 对多 step、多图 Gemini Pro 请求过高，且当前 evaluator 没有 retry。
- 该结果先标记为待低并发重跑，不作为正式结论。
- 已更新 `exp2/streamweave_v3/evaluation/eval_batch.py`：
  - 支持 YAML 中的 `batch.max_retries`、`retry_backoff_seconds`、`retry_backoff_multiplier`、`retry_error_patterns`。
  - retry 是 sample-level：一条样本任意 step 因 `499/504/ReadTimeout` 等 transient error 失败，会从头重跑该样本，只写最后一次结果。
- 已将 `configs/batch_ovo_gemini_1of8.yaml` 改为 `timeout_seconds=240.0`、`workers=4`、`max_retries=2`，输出目录改为 `outputs/ovo_gemini_1of8_retry`，避免覆盖高错误率旧结果。
- 已将用户提供的 OVO-Bench 论文 SOTA 表明确记录到 `实验跑分.md`，并补充 R9 与论文逐项 best 的差距统计。

### 2026-04-27 StreamWeave V3 大更新

- 当时新主框架：`exp2/streamweave_v3`
- 新主文档：
  - `exp2/streamweave_v3/docs/实验计划.md`
  - `exp2/streamweave_v3/docs/数据构造.md`
- V3 单步协议固定为：

```xml
<eta>...</eta>
<answer>...</answer>
<bridge t="...">...</bridge>
<note t="..." frame="..."></note>
```

- 协议变化：
  - 每一步输入是 `memory + qa_history + current frames`。
  - `note` 只表示被选中的视觉帧和时间区间，不保存文字。
  - `bridge` 保存文本和绝对时间戳。
  - `qa_history` 是按时间顺序记录的问题和历史答案日志，不再做 active query 配对。
  - `eta` 是视频内绝对秒级时间戳，不是相对 delay。
- V3 当前代码状态：
  - 已有 `rollout / memory / prompts / parser / action_quality / action_postprocess / video_io` 主链路。
  - 已接入 OVO backward / realtime / forward，forward 的 `REC/SSR/CRR` 会使用 `{id}_{i}.mp4` 子视频。
  - `teacher` postprocessor 和 teacher 样本丢弃逻辑还没实现。
  - `data/` 和 `training/` 目前还是占位。
- 数据主线变化：
  - 之前的 `streamweave_data` 只是视频和 query-free 关键帧池。
  - `VideoXum` 没有 question、answer、answer timestamp，不能直接生成 SFT。
  - 现在必须先构造 `VideoXum-StreamQA`，合成 `backward / realtime / forward` 三类流式问题。
- `VideoXum-StreamQA` 第一版流水线：
  1. 聚合 `vsum_onehot` 得到 `key_score`。
  2. 将关键帧扩成局部时间窗口。
  3. 用 teacher MLLM 生成带时间的 `atomic_facts`。
  4. 基于事实生成 `backward / realtime / forward` QA candidate。
  5. 做 evidence/prefix/text-only/summary-only 验证。
  6. 导出 `qa_verified.jsonl`。
  7. 再转成 V3 teacher traces 和 step-level SFT。
- 下一步优先做数据原型，不直接开训练：
  - `load_videoxum.py`
  - `build_keyframe_segments.py`
  - `generate_atomic_facts.py`
  - `generate_streamqa.py`
  - `verify_streamqa.py`
  - `export_teacher_trace.py`
  - `inspect_data.py`

### 2026-04-27 当前实验入口整理

- 当前阶段切到 `数据构造 / SFT 数据准备`。
- `Idea 验证` 已完成一轮：
  - `Qwen3-VL-32B-Instruct plain` baseline 已补齐。
  - `StreamWeave-v2 OVO 1/8` 在同 ID 正常样本上与 `SimpleStream recent4` 持平，都是 `113/170 = 66.47%`。
  - `EPM / FPD / ACR` 有收益，但被 `ASI / HLD / STU / OJR / OCR` 退化抵消。
- 当前视频/关键帧池改为 `streamweave_data`：
  - 视频来自 `ActivityNet_Captions/Activity_Videos`。
  - 关键帧监督来自 `VideoXum` 的 `vsum_onehot`。
  - 构造脚本：`exp2/scripts/build_videoxum_dataset.py`。
- 当前原始标注文件：
  - `exp2/data/streamweave_data/annotations_filtered_30s300s_key10to40.jsonl`
  - `12029` 条。
- 当前过滤条件：
  - `30 <= duration <= 300`
  - `0.10 <= key_frame_ratio <= 0.40`
- 当前重要约定：
  - 所有处理默认在 `simple` 环境执行。
  - 当前过滤文件来自 `threshold=0.3`。
  - `key_frame_ids` 是 0-based，和抽帧文件名 `{frame_id:06d}.jpg` 对齐。
  - 暂时不做超分，后续由 processor/dataloader resize 或 pad。
- 下一步：
  - 基于 V3 文档先生成 `VideoXum-StreamQA` 数据原型。
  - 先检查三类 QA 的证据时间、prefix 可答性和 text-only/summary-only 偏置。
  - QA 验证通过后，再导出 teacher traces 和 SFT steps。

### 2026-04-21

- 主实验代码线切换：
  - 当前独立重写版代码库为 `exp1/stream-weave_v2`
  - 旧目录 `exp1/streamweave` 降为历史参考，不再作为当前主实验入口
- 当前判断：
  - 上一版积累的 prompt 问题，当前视为已在 `stream-weave_v2` 中整体重写处理
  - 新阶段的首要任务不再是继续堆 prompt patch，而是先把 `v2` 跑通
- `stream-weave_v2` 当前状态：
  - 当前默认使用 OpenAI-compatible API backend
  - 后续需要切到本地部署模型再继续正式实验
  - `anno_path / chunked_dir` 等路径可能暂时不完全对齐，跑 benchmark 前要先核对
- 补记一份 `OVO-Bench` 外部参考表到 `06-evaluation.md`：
  - 来源：用户提供
  - 性质：参考记录，非本地复现
  - 内容：`GPT-4o / Gemini-1.5-Pro / StreamAgent / Streamo-7B / ViSpeak / MiniCPM-o-4.5 / Qwen3-VL-8B-Inst. / AURA`
- 补记一份 `StreamingBench` 外部参考表到 `06-evaluation.md`：
  - 来源：用户提供
  - 性质：参考记录，非本地复现
  - 内容：`GPT-4o / Gemini-1.5-Pro / StreamAgent / ViSpeak / Qwen3-VL-8B-Inst. / MiniCPM-o-4.5 / AURA`
- 记录时显式区分：
  - 外部参考表
  - 本地 `recent4 / chunk1 / fps1` 跑分
  - 避免后续把不同评测口径混成一组横比

### 2026-04-17

- 建立 `note/` 多文件实验记录结构。
- 记录 `StreamWeave` 提案与当前阶段拆分。
- 确认 `SimpleStream` 复现结果可直接作为参考基线。

### 2026-04-18

- 将本地 `Qwen3-VL-32B-Instruct` 设为当前教师模型。
- 新增 OpenAI SDK API-only 评测脚本，替代旧的混合式 API 入口。
- 修复本地 `vllm + reasoning-parser qwen3` 下答案字段读取问题。
- 确认新脚本 smoke 可用，当前进入 `32B plain benchmark baseline` 全量评测阶段。
- 记录 `Qwen3-VL-32B-Instruct plain` 的 `StreamingBench` 全量结果：
  - `REAL 80.12`
  - `SQA 54.80`
  - `Proactive time 61.20 / answer 60.80`
- 记录 `Qwen3-VL-32B-Instruct plain` 的 `OVO` 全量结果：
  - `Backward 60.73`
  - `Realtime 78.15`
  - `Forward 44.70`
  - `Total 61.19`
- 对照 `SimpleStream / Qwen3-VL-8B / recent4`：
  - `Backward +10.25`
  - `Realtime -3.33`
  - `Forward +0.91`
  - `Total +2.60`
- 当前状态更新：`32B plain benchmark baseline` 已完成，后续主任务切换到 `exp1/streamweave` 的 baseline / agent 实现与同模型对比。
- 根据新版提案修正主笔记：
  - 术语统一为 `note / bridge / silent / answer`
  - 场景改为“任意时间 query”，不再限定 `query-at-start`
  - 数据构造改成 `state teacher / response teacher` 分离，学生模型统一
- 明确后续方法代码写在 `exp1/streamweave`，与 `exp1/SimpleStream` 的 plain benchmark 链路隔离。
- 在 `exp1/streamweave` 下落地首版代码骨架：
  - `observe -> state -> response -> memory -> trace`
  - 每个核心 `.py` 文件带 `main()` 自测入口
- 本地自测已通过：
  - `backend.py`
  - `structured_state.py`
  - `state_policy.py`
  - `response_policy.py`
  - `memory.py`
  - `trace.py`
  - `rollout.py`
  - `eval_streamingbench.py`
- 真实后端 smoke：
  - `streamweave/eval_ovo.py` 已能打到本地 `vllm`
  - 当前 `sample_id=1, max_chunks=6` 返回空答案，后续继续调 prompt

### 2026-04-19

- `exp1/streamweave` 代码结构完成一轮收缩：
  - 选帧部分删除 `reconstruction_error`
  - 选帧当前只用 `scene cut + 大模型评判`
  - `state_policy.py` 收成单文件实现
  - 删除旧的 `bridge_policy / note_policy / reconstruction / state_gate`
- `eval_ovo_batch.py` 完成样本级并发与边跑边写：
  - 支持 `--max-workers`
  - 运行中实时写 `results.jsonl / errors.jsonl`
  - 修掉了批量结果重复写入的问题
- 在 `exp1/SimpleStream` 下新增 Gemini OVO 评测链路：
  - `lib/gemini_api_eval.py`
  - `main_experiments/eval_gemini_ovo.py`
- 首次 `gemini-2.5-pro` 的 `OVO 1/4` 运行出现一次严重误判：
  - 结果表面上显示 `0.00%`
  - 实际不是模型全错，而是请求参数错误导致大量 `400 INVALID_ARGUMENT`
  - 根因是把 `gemini-2.5-flash` 的 `thinking_budget=0` 习惯直接套到 `gemini-2.5-pro`
- 本次错误暴露出两个流程问题：
  - 请求失败被继续按 `response=None` 参与打分，产生误导性的 `0 分`
  - 失败样本被写进 checkpoint 后还会被当成 `done`，影响后续重跑
- 已完成修复：
  - Gemini runner 默认改为 `--thinking_budget auto`
  - `gemini-2.5-pro` 自动回退到安全 budget，避免再次触发 `400`
  - checkpoint 读取时忽略失败记录
  - 若全部请求失败，脚本直接报错，不再输出伪 benchmark 分数
- 当前结论：
  - 旧目录 `ovo_gemini25pro_quarter_recent4` 为无效结果
  - 后续 Gemini benchmark 必须先确认 `results_incremental.jsonl` 中没有系统性 `error`
- 修复后，拿到首个有效的 `gemini-2.5-pro` OVO 结果：
  - 口径：`OVO 1/4 subset`
  - `Backward 69.42`
  - `Realtime 77.65`
  - `Forward 44.09`
  - `Total 63.72`
- 当前判断：
  - 这说明 Gemini 跑 `recent4` OVO 是可行的
  - 但该结果仍是子集结果，暂不与 `Qwen3-VL-32B-Instruct plain` 全量结果做正式结论比较

### 2026-04-25 StreamWeave-v2 OVO 1/8 记录

- 当前代码库: `/mmu_mllm_hdd/zhouhanshu/test/exp1/stream-weave_v2`
- 固定环境: `simple`
- 本地 OpenAI-compatible vLLM:
  - `base_url=http://127.0.0.1:8082/v1`
  - `api_key=EMPTY`
  - `model=Qwen3-VL-32B-Instruct`
- OVO 1/8 annotation:
  - `/mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_1of8_stratified.json`
  - 共 `205` 个 top-level samples
- 默认每步 chunk 数: `chunks_per_step=5`，不要再传 `--chunks-per-step 10`。

稳定运行命令:

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/stream-weave_v2
python eval_ovo_batch.py \
  --backend openai \
  --anno-path /mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_1of8_stratified.json \
  --workers 2 \
  --trace-root results \
  --output results/ovo_1of8_w2.json \
  --log-dir results/ovo_1of8_w2_logs
```

批量脚本行为:

- `eval_ovo_batch.py` 已支持每个 worker 单独日志，日志目录由 `--log-dir` 指定。
- 终端只保留进度条，具体错误和 worker 输出看对应 `worker_<pid>.log`。
- `workers=8` 不适合判断模型能力，之前主要失败来自 API timeout、vLLM 500、视频路径问题，不是纯模型答错。
- `workers=2` 当前更稳定。

最近一次 `workers=2` 快照:

- 进度: `71/205`
- 正常完成: `68`
- error: `3`
- 正常样本 accuracy: `35/68 = 51.47%`
- error 按 0 分计: `35/71 = 49.30%`
- EPM: `23/37 = 62.16%`
- HLD: `8/20 = 40.00%`
- ASI 后续局部统计: 正常完成 `19`，正确 `11`，错误 `8`

HLD 当前问题:

- 已完成 HLD 正常样本 `20`，正确 `8`，错误 `12`。
- HLD error: sample `341`, `381`, `406`，原因是上下文超过 `32768`。
- HLD 错题的 GT 全是 `Unable to answer`，但 StreamWeave 给了具体选项。
- 主要原因: memory/bridge 把局部视觉证据过度解释成确定答案；prompt 缺少“证据不足时优先 Unable”的约束。
- 典型样本:
  - `301`: trace 看到 rice cooker on countertop，最终映射到 bottom cabinet；GT 是 `Unable`。
  - `333`: trace 看到 ladle/pot/stovetop，最终映射到 sink；GT 是 `Unable`。
  - `325`, `349`, `588`: `<answer>` 和 bridge 解释存在自相矛盾。

ASI 当前问题:

- 已完成 ASI 正常样本 `19`，正确 `11`，错误 `8`。
- 错题: `494`, `502`, `510`, `518`, `525`, `533`, `557`, `588`。
- 主要原因: before/after 时间关系在 bridge summary 里保留不够；模型容易回答当前动作或最近动作；rollout 看到 `<answer>` 后就清 pending，没有利用 `<eta>` 判断是否还应等待。
- 典型样本:
  - `494`: 问 loading new battery 之后的动作，trace 仍在描述 removing old battery。
  - `502`: memory 说 yellowish liquid，但没有稳定绑定到 juice，最终选了 stir mixture。
  - `533`: bridge 同时包含 spreading/adjusting logs 和 tinder，最终选了最近动作。
  - `588`: bridge 说 book was placed down，但 `<answer>` 选了 Took。

与旧 Qwen3-VL-32B recent-window 对比:

- 旧结果文件:
  - `/mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/main_experiments/results/ovo_qwen3vl32_openai_api_full_fix/qwen3vl_results_20260418_214423.json`
- 在 `68` 个重叠正常样本上:
  - StreamWeave: `35/68 = 51.47%`
  - 旧 Qwen32 recent-window: `34/68 = 50.00%`
- EPM overlap:
  - StreamWeave: `23/37 = 62.16%`
  - 旧 Qwen32 recent-window: `19/37 = 51.35%`
- HLD/ASI 当前低于旧 baseline，核心是 Unable calibration 和时间关系处理。

Prompt/代码差异:

- SimpleStream Qwen3VL-32B recent-window 使用原始 OVO prompt + 最近 4 帧，不带 memory/XML。
- StreamWeave 使用 `HEADER + few-shot + memory + QA + current frames + FOOTER`，输出 XML，包括 `<analysis>`, `<answer>`, `<eta>`, `<notes>`, `<todo>`。
- 相关位置:
  - SimpleStream: `SimpleStream/lib/recent_window_eval.py`, `SimpleStream/lib/openai_api_eval.py`
  - StreamWeave: `stream-weave_v2/prompts.py`, `stream-weave_v2/rollout.py`

Gemini 接入方案:

- 如果目标是评估 StreamWeave，不要直接用 SimpleStream runner。
- 应在 `stream-weave_v2/backend.py` 增加 `GeminiChatBackend`，让 `eval_ovo.py` 保持同一套 StreamWeave rollout/prompt，只替换 backend。
- 可复用工具: `/mmu_mllm_hdd/zhouhanshu/tools/call_gemini.py`

待办:

- HLD: 增强 `Unable to answer` 策略，要求证据和选项严格匹配，否则选 Unable。
- ASI: 让 `<eta>` 真正控制等待逻辑，证据不足时不要清 pending。
- 控制 memory/context 长度，避免超过 `32768`。
- 官方 forward tasks 目前未完整支持，脚本仍按 `chunked_videos/<id>.mp4` 读取，和 forward 文件命名不匹配。

### 2026-04-25 StreamWeave-v2 OVO 1/8 完成汇总

- 结果文件：`exp1/stream-weave_v2/results/ovo_1of8_w2.json`
- annotation：`ovo_bench_1of8_stratified.json`，共 `205` 个 top-level samples。
- 正常完成样本：`170`
- 正常完成准确率：`113/170 = 66.47%`
- 若把 error 按 0 分计：`113/205 = 55.12%`
- error 共 `35`：
  - `14` 个 `Input length > 32768`
  - `21` 个 video read error，全部来自当前未正确支持的 forward 类任务：
    - `CRR 6`
    - `SSR 5`
    - `REC 10`
- 正常完成样本分 task：
  - `EPM 23/37 = 62.16%`
  - `HLD 8/20 = 40.00%`
  - `ASI 11/19 = 57.89%`
  - `STU 11/20 = 55.00%`
  - `OJR 15/20 = 75.00%`
  - `ATR 10/13 = 76.92%`
  - `FPD 8/11 = 72.73%`
  - `ACR 12/13 = 92.31%`
  - `OCR 15/17 = 88.24%`
- 与 `Qwen3-VL-32B-Instruct plain / SimpleStream recent4` 在同 `170` 个正常样本上对齐：
  - `StreamWeave-v2 113/170 = 66.47%`
  - `SimpleStream 113/170 = 66.47%`
  - `both correct 92`
  - `StreamWeave only 21`
  - `SimpleStream only 21`
  - `both wrong 36`
- 正常样本上的 task 差异：
  - 明显收益：`EPM +10.81pp`，`FPD +18.18pp`，`ACR +7.69pp`
  - 基本持平：`ATR 0pp`
  - 主要退化：`ASI -15.79pp`，`HLD -5.00pp`，`STU -5.00pp`，`OJR -5.00pp`，`OCR -5.88pp`
- 当前判断：
  - StreamWeave 的 memory 对历史物体/位置类问题已经有收益，尤其 `EPM`。
  - 但时序关系、证据充分性、长上下文控制仍是主要瓶颈。
  - 当前总分没有超过 SimpleStream，不是单一问题，而是 memory 带来的收益被 `ASI/HLD` 退化、context error 和 forward 路径错误抵消。

### 2026-04-25 第二阶段：数据构造启动

- 当前进入第二阶段：构造训练/实验数据。
- 后续新代码优先放在：`/mmu_mllm_hdd/zhouhanshu/test/exp2`。
- 第一批目标数据源：`TimeChat-Online-139K`。
- HF 仓库：`yaolily/TimeChat-Online-139K`。
- 下载策略：先下载全部标注信息 + 部分视频/帧源；第一波不下载 `QV-Highlights`。
- 目标本地目录：`/mmu_mllm_hdd/zhouhanshu/test/datasets/TimeChat-Online-139K`。

第一波下载文件:

```text
README.md
LICENSE
time_chat_online_139k_train_flt.jsonl
annotations_caption_flt.jsonl
OtherTrainingDataset/llava_video_100k.jsonl
OtherTrainingDataset/tarsier2_129k.jsonl
OtherTrainingDataset/videochat_flash_3k.jsonl
Youcook2.tar.gz.000
Youcook2.tar.gz.001
Youcook2.tar.gz.002
HiREST.tar.gz
COIN.tar.gz
ActivityNet.tar.gz
TVSum.tar.gz
```

第一波预计大小:

```text
YouCook2     46.66 GB
HiREST        5.54 GB
COIN          7.90 GB
ActivityNet   0.84 GB
TVSum         0.26 GB
标注          3.58 GB
合计         64.78 GB
```

第一波下载命令:

```bash
conda activate simple

python /mmu_mllm_hdd/zhouhanshu/test/download_repo.py \
  --source hf \
  --repo-type dataset \
  --repo-id yaolily/TimeChat-Online-139K \
  --output-root /mmu_mllm_hdd/zhouhanshu/test/datasets \
  --local-name TimeChat-Online-139K \
  --max-workers 8 \
  --allow-patterns \
    README.md \
    LICENSE \
    time_chat_online_139k_train_flt.jsonl \
    annotations_caption_flt.jsonl \
    'OtherTrainingDataset/*.jsonl' \
    'Youcook2.tar.gz.*' \
    HiREST.tar.gz \
    COIN.tar.gz \
    ActivityNet.tar.gz \
    TVSum.tar.gz
```

待第二波再决定是否下载:

```text
QV-Highlights.tar.gz.000 ~ QV-Highlights.tar.gz.008，合计约 177.17 GB
```

### 2026-04-27 OVO-Bench 推理评测待回填

- `Gemini` 推理口径已完成 `OVO-Bench 1/8`：
  - 结果文件：`exp2/streamweave_v3/outputs/ovo_gemini_1of8/results.jsonl`
  - task macro：`Backward 69.37`，`Realtime 70.97`，`Forward 53.81`，`Total 64.72`
  - 样本加权：`221/364 = 60.71%`
  - error：`10/364`，包括 `6x ClientError`、`2x ReadTimeout`、`1x ProxyError`、`1x ServerError`
- `Qwen3-VL-8B-Instruct / StreamWeave V3` 已完成 `OVO-Bench` 全量：
  - 结果文件：`exp2/streamweave_v3/outputs/ovo_qwen3vl8b_8gpu/results.jsonl`
  - 运行口径：`policy=streamweave`，`prompt_type=teacher`，`chunk_duration=1.0`，`fps=1.0`，`chunks_per_step=5`，`memory_window=120.0`
  - task macro：`Backward 43.20`，`Realtime 57.75`，`Forward 46.21`，`Total 49.05`
  - 样本加权：`1434/3035 = 47.25%`
  - error：`0/3035`
  - 相比旧参考基线 `SimpleStream / Qwen3-VL-8B / recent4`：`Backward -7.28pp`，`Realtime -23.73pp`，`Forward +2.42pp`，`Total -9.54pp`

### 2026-04-25 TimeChat-Online-139K 第一波下载启动

- 当前路径：`/mmu_mllm_hdd/zhouhanshu/test`
- 运行环境提示符：`base`
- 下载脚本：`/mmu_mllm_hdd/zhouhanshu/test/download_repo.py`
- HF 仓库：`yaolily/TimeChat-Online-139K`
- repo type：`dataset`
- 目标目录：`/mmu_mllm_hdd/zhouhanshu/test/datasets/TimeChat-Online-139K`
- workers：`8`
- 文件数：`14`
- 本轮策略：全部标注 + `Youcook2`、`HiREST`、`COIN`、`ActivityNet`、`TVSum`；不下载 `QV-Highlights`。
- 运行观测：启动后显示 `Downloading (incomplete total...): 409M/59.1G [01:17<3:19:44, 4.90MB/s]`。

实际运行命令:

```bash
python /mmu_mllm_hdd/zhouhanshu/test/download_repo.py \
  --source hf \
  --repo-type dataset \
  --repo-id yaolily/TimeChat-Online-139K \
  --output-root /mmu_mllm_hdd/zhouhanshu/test/datasets \
  --local-name TimeChat-Online-139K \
  --max-workers 8 \
  --allow-patterns \
    README.md \
    LICENSE \
    time_chat_online_139k_train_flt.jsonl \
    annotations_caption_flt.jsonl \
    'OtherTrainingDataset/*.jsonl' \
    'Youcook2.tar.gz.*' \
    HiREST.tar.gz \
    COIN.tar.gz \
    ActivityNet.tar.gz \
    TVSum.tar.gz
```

### 2026-04-28 StreamWeave V3 Gemini OVO-Bench full 运行中

- 启动时间记录：2026-04-28 11:26 UTC。
- 状态：running，结果待回填。
- 代码目录：`exp2/streamweave_v3`。
- config：`configs/batch_ovo_gemini_full.yaml`。
- 输出目录：`exp2/streamweave_v3/outputs/ovo_gemini_full_retry`。
- 主结果文件：`exp2/streamweave_v3/outputs/ovo_gemini_full_retry/results.jsonl`。
- worker logs：`exp2/streamweave_v3/outputs/ovo_gemini_full_retry/worker_logs`。
- 当前已创建 shard 文件：32 个；worker log 文件：32 个。
- Benchmark：OVO-Bench full，`ovo_bench_new.json`，展开后 `3035` samples。
- 后端：Gemini VertexAI，`gemini-2.5-pro`。
- 运行口径：`prompt_type=teacher`，`policy=streamweave`，`chunk_duration=1.0`，`fps=1.0`，`chunks_per_step=5`，`memory_window=120.0`。
- Gemini 参数：`max_tokens=2048`，`temperature=0.0`，`timeout_seconds=240.0`，`max_image_side=768`，`image_quality=85`。
- Batch：`workers=32`，`max_retries=2`，保留 transient retry patterns。
- 本轮 prompt 变更：只在 StreamWeave 总 prompt 中显式说明 `role="q"` 是 user question、`role="a"` 是 previous model answer；OVO 官方任务 prompt 未做额外改写。

### 2026-04-29 StreamWeave V3 Gemini OVO-Bench full 完成与错误重跑准备

- Full run 已完成：`3035/3035` samples，耗时 `7:13:06`。
- 主结果：`outputs/ovo_gemini_full_retry/results.jsonl`。
- Summary：`outputs/ovo_gemini_full_retry/results_summary.json` 和 `outputs/ovo_gemini_full_retry/results_summary.txt`。
- 当前总体：`Total AVG = 60.16%`。
- 分类结果：`Backward AVG = 65.65%`，`Realtime AVG = 69.45%`，`Forward AVG = 45.39%`。
- Forward 瓶颈仍是 `CRR = 31.25%`；`REC = 37.82%`，`SSR = 67.09%`。
- 错误样本：`230/3035`，其中 `217x ClientError`，`6x RuntimeError`，`5x ServerError`，`2x ReadTimeout`。
- 已新增 `evaluation/retry_failed_results.py`：从已有 `results.jsonl` 抽取错误样本，生成 retry config，重跑后只用成功结果替换原失败样本，并重新写 merged summary。
- OVO adapter 的 `sample_ids` 按 `annotation_id` 过滤；本轮 `230` 个失败 sample 对应 `179` 个 annotation，实际 retry 会跑 `414` 个展开样本，合并时只替换原始失败 sample。
- 已排查 `Gemini response has no text content`：当前 Gemini 抽取逻辑已先读 `response.text`，再读 `response.candidates[*].content.parts[*].text`，暂无证据表明是简单字段抽取错误；更可能是 Gemini 某次返回无 text part、空 candidate、safety/finish_reason 截断或高并发/配额压力下的异常空响应。
- 已增强 diagnostics：`streamweave/gemini_client.py` 在空文本时记录 `candidate_count`、`prompt_feedback`、`finish_reason`、`safety_ratings`、`part_count`、`part_types`。
- 已把 `Gemini response has no text content` 加入 retryable patterns：`evaluation/eval_batch.py`、`configs/batch_ovo_gemini_full.yaml`、`evaluation/retry_failed_results.py`。
- 验证已过：`compileall` 和 `git diff --check`。
- 详细记录见：`note/实验跑分.md` 的 “2026-04-29 V3 Gemini OVO full 评测与错误重跑进展”。
