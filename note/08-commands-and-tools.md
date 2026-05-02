# 常用命令与工具

## 基本约定

- 工作目录：`/mmu_mllm_hdd/zhouhanshu/test`
- 评测默认环境：`simple`
- 下面的命令默认已经执行：

```bash
conda activate simple
```

## StreamWeave V4 SFT 当前命令

代码目录：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4
```

Gemini 凭证：

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/mmu_ssd3/group_lisize/hetu/xujia10/joint_tags/scripts/gemini_client/config.json
```

当前 `run_parallel_pipeline.py` 默认参数已经对齐到当前主数据：

```text
--input dataset/streamweave_data/annotations_qa_filter_final.jsonl
--raw-data-root dataset/streamweave_data
--backend gemini
--model gemini-2.5-pro
--frames-per-step 5
--max-attempts 3
```

跑 8 条 smoke：

```bash
python data_engine/sft/run_parallel_pipeline.py \
  --num-workers 4 \
  --limit 8 \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/data_engine/sft/outputs/gemini_final_8 \
  --overwrite
```

当前第二轮 1000 条合成，64 workers：

```bash
python data_engine/sft/run_parallel_pipeline.py \
  --num-workers 64 \
  --limit 1000 \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/data_engine/sft/outputs/gemini_final_1000_w64 \
  --overwrite
```

断点续跑同一批数据时，不要加 `--overwrite`：

```bash
python data_engine/sft/run_parallel_pipeline.py \
  --num-workers 64 \
  --limit 1000 \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/data_engine/sft/outputs/gemini_final_1000_w64
```

单条样本 debug：

```bash
python data_engine/sft/run_pipeline.py \
  --offset 2 \
  --limit 1 \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/data_engine/sft/outputs/debug_sample_offset2 \
  --overwrite
```

把中间样本文件转成方便阅读的 txt，去掉重复任务指导和 few-shot：

```bash
python data_engine/sft/inspect_intermediate.py \
  --input /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/data_engine/sft/outputs/gemini_final_8/samples/<sample_file>.json \
  --output /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/data_engine/sft/outputs/gemini_final_8/<sample_file>.inspect.txt
```

当前输出文件说明：

```text
samples/*.json                         每条样本完整合成记录
sample_manifest.jsonl                  样本级 accepted/failed/error 索引
sft_steps.jsonl                        accepted-only step 中间格式
llamafactory_sharegpt.jsonl            accepted-only ShareGPT 训练数据
dataset_info_streamweave_sft.json      LLaMAFactory dataset_info 片段
summary.json                           汇总统计
sft_jobs.sqlite                        动态队列和断点续跑状态
```

注意：

- accepted-only 文件才用于训练。
- `samples/*.json` 可以用于排查失败样本，但不要直接喂训练。
- `--overwrite` 只用于新开实验；续跑必须去掉。
- 进度条里的 `elapsed` 是本轮运行耗时，`eta` 是按当前完成速率估计的剩余时间。

## Conda 备忘录

- `dvd`: `/mmu_mllm_hdd/zhouhanshu/conda/envs/dvd`
  - `Deep Video Discovery`
- `long`: `/mmu_mllm_hdd/zhouhanshu/conda/envs/long`
  - `LongVideo-R1`
- `simple`: `/mmu_mllm_hdd/zhouhanshu/conda/envs/simple`
  - `SimpleStream`
  - `torch` 为手动安装 wheel
  - 已安装 `openai`
  - 已安装 `PySceneDetect`：`scenedetect[opencv]`
- `test`: `/mmu_mllm_hdd/zhouhanshu/conda/envs/test`
  - 用途待补充
- `tmp`: `/mmu_mllm_hdd/zhouhanshu/conda/envs/tmp`
  - 特殊备用环境
  - 含 `torch 2.5.1+cu121`
  - 一般只用于 clone，不建议修改
- `verl`: `/mmu_mllm_hdd/zhouhanshu/conda/envs/verl`
  - 用途待补充
- `verl2`: `/mmu_mllm_hdd/zhouhanshu/conda/envs/verl2`
  - 用途待补充
- `verl3`: `/mmu_mllm_hdd/zhouhanshu/conda/envs/verl3`
  - 用途待补充
- `verl_test`: `/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_test`
  - 用途待补充
- `vllm`: `/mmu_mllm_hdd/zhouhanshu/conda/envs/vllm`
  - 本地 `vllm` 推理环境
- `base`: `/mmu_mllm_hdd/zhouhanshu/miniconda3`
  - 基础环境

## 下载工具

- 查看下载脚本帮助：

```bash
python /mmu_mllm_hdd/zhouhanshu/test/download_repo.py --help
```

- 下载 `Qwen3-VL-32B-Instruct` 到模型目录：

```bash
python /mmu_mllm_hdd/zhouhanshu/test/download_repo.py \
  --source hf \
  --repo-type model \
  --repo-id Qwen/Qwen3-VL-32B-Instruct \
  --output-root /mmu_mllm_hdd/Models
```

## 数据构造命令

### VideoXum

```bash
cd /mmu_mllm_hdd/zhouhanshu/test

python /mmu_mllm_hdd/zhouhanshu/test/download_repo.py \
  --source hf \
  --repo-type dataset \
  --repo-id jylins/videoxum \
  --output-root /mmu_mllm_hdd/zhouhanshu/test/exp2/data \
  --local-name videoxum \
  --max-workers 8
```

### ActivityNet_Captions

```bash
cd /mmu_mllm_hdd/zhouhanshu/test

python /mmu_mllm_hdd/zhouhanshu/test/download_repo.py \
  --source hf \
  --repo-type dataset \
  --repo-id friedrichor/ActivityNet_Captions \
  --output-root /mmu_mllm_hdd/zhouhanshu/test/exp2/data \
  --local-name ActivityNet_Captions \
  --max-workers 8
```

解压 ActivityNet 视频分卷：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp2/data/ActivityNet_Captions
cat ActivityNet_Videos.tar.part-* | tar -xf -
```

### 构造 streamweave_data

调试两条样本：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test

python exp2/scripts/build_videoxum_dataset.py \
  --limit 2 \
  --threshold 0.3
```

复现当前全量版本：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test

python exp2/scripts/build_videoxum_dataset.py \
  --threshold 0.3
```

当前主数据入口：

```text
exp2/data/streamweave_data/annotations_filtered_30s300s_key10to40.jsonl
```

注意：

- 当前过滤文件来自 `threshold=0.3` 的全量构造结果。
- `summary_filtered_30s300s_key10to40.json` 只是统计摘要，不是训练数据。
- 该文件只是 VideoXum 视频/关键帧池，不是最终 SFT QA 数据。

### StreamWeave V3 历史命令

历史代码目录：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v3
```

主文档：

```text
docs/实验计划.md
docs/数据构造.md
```

历史 V3 数据构造目标：

```text
exp2/data/videoxum_streamqa/
```

计划新增脚本：

```text
load_videoxum.py
build_keyframe_segments.py
generate_atomic_facts.py
generate_streamqa.py
verify_streamqa.py
export_teacher_trace.py
inspect_data.py
```

历史 QA 数据合成 8 workers 断点续跑命令：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v3

/mmu_mllm_hdd/zhouhanshu/conda/envs/simple/bin/python data/synthesize/run_pipeline_parallel.py \
  --input raw_data/anno.jsonl \
  --raw-data-root raw_data \
  --output-dir data/synthesize/outputs/full_parallel_w4 \
  --limit 0 \
  --workers 8 \
  --num-questions 8 \
  --keep-per-video 1 \
  --resume 2>&1 | tee -a data/synthesize/outputs/full_parallel_w4/run.log
```

注意：`--resume` 会读取 `progress.jsonl` 跳过已完成视频，并继续 append 到同一输出目录；续跑时不要加 `--overwrite`。

## 本地教师模型部署

- 模型：`Qwen3-VL-32B-Instruct`
- 模型目录：`/mmu_mllm_hdd/Models/Qwen3-VL-32B-Instruct`
- 环境：`vllm`
- 端口：`8082`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /mmu_mllm_hdd/Models/Qwen3-VL-32B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 \
  --port 8082 \
  --reasoning-parser qwen3 \
  --served-model-name Qwen3-VL-32B-Instruct
```

## 当前推荐评测入口

- 代码目录：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream
```

- 当前推荐脚本：
  - `main_experiments/eval_qwen3vl_ovo_api.py`
  - `main_experiments/eval_streamingbench_real_api.py`
  - `main_experiments/eval_streamingbench_sqa_api.py`
  - `main_experiments/eval_streamingbench_proactive_api.py`
  - `main_experiments/eval_gemini_ovo.py`

- 当前说明：
  - 使用 `openai` Python SDK
  - 适配本地 `vllm` 的 OpenAI-compatible endpoint
  - 已兼容 `--reasoning-parser qwen3` 下的 `message.reasoning`

- 废弃说明：
  - 旧的混合式 `--qa_backend api` / `--qa-backend api` 入口不再作为正式本地 `vllm` 评测入口

## Gemini / VertexAI 约定

- 凭证：

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/mmu_ssd3/group_lisize/hetu/xujia10/joint_tags/scripts/gemini_client/config.json
```

- Gemini OVO 当前使用：
  - `main_experiments/eval_gemini_ovo.py`
  - 默认 `--thinking_budget auto`

- 重要避坑：
  - 不要给 `gemini-2.5-pro` 传 `--thinking_budget 0`
  - 目前安全写法是直接用 `--thinking_budget auto`
  - 如果结果突然整片 `0.00%`，先检查 `result_dir/results_incremental.jsonl` 里是否大量 `error`

## StreamWeave v2 历史代码入口

- 代码目录：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/stream-weave_v2
```

- 说明：
  - 这是历史 v2 实验代码库，不是当前 V4 SFT 主线
  - `eval_ovo.py` 是自包含的 `OVO` 入口
  - 默认 backend 仍是 OpenAI-compatible API
  - 当时后续主任务是切到本地部署模型
  - 默认 `anno-path / chunked-dir` 可能暂时不完全对齐，正式跑前先核对

- 先跑本地 smoke：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/stream-weave_v2
python eval_ovo.py --smoke
```

- 跑 API 版 OVO 单样本：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/stream-weave_v2
python eval_ovo.py \
  --backend openai \
  --sample-id 0 \
  --max-chunks 12
```

- 若切到本地 `vllm`，优先尝试显式覆盖：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/stream-weave_v2
python eval_ovo.py \
  --backend openai \
  --sample-id 0 \
  --max-chunks 12 \
  --api-base-url http://127.0.0.1:8082/v1 \
  --api-key EMPTY \
  --model-name Qwen3-VL-32B-Instruct
```

## StreamWeave 历史首版代码入口

- 代码目录：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/streamweave
```

- 说明：
  - 当前是 `prompt-driven replay pipeline`
  - 每个核心 `.py` 文件都带 `main()` 自测入口
  - 默认先用 `MockBackend` 做离线验收
  - 再切本地 `vllm` 做真实 smoke
  - 场景切分当前使用 `PySceneDetect`

- 先跑主链路 demo：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/streamweave
python rollout.py
```

- 跑 StreamingBench 风格 demo：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/streamweave
python eval_streamingbench.py
```

- 跑 OVO 单样本 mock smoke：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/streamweave
python eval_ovo.py --backend mock --sample-id 1
```

- 跑 OVO 单样本真实后端 smoke：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/streamweave
python eval_ovo.py \
  --backend openai \
  --sample-id 1 \
  --max-chunks 6 \
  --api-base-url http://127.0.0.1:8082/v1 \
  --api-key EMPTY \
  --model-name Qwen3-VL-32B-Instruct
```

- 当前 debug trace 规则：
  - 默认会写到 `exp1/streamweave/results/debug_traces/`
  - 一条样本一个目录，例如：
    - `exp1/streamweave/results/debug_traces/ovo_1_EPM/`
  - 每个样本目录内包含：
    - `trace.jsonl`
    - `frames/`
  - `trace.jsonl` 中只记录占位符文件名，例如：
    - `chunk.frame_placeholders`
    - `saved_note_frame_placeholders`
    - `active_note_before.frame_placeholders`
    - `active_note_after.frame_placeholders`

- 如果要做更快的 OVO smoke，先限制前缀长度：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/streamweave
python eval_ovo.py \
  --backend openai \
  --sample-id 1 \
  --max-chunks 6 \
  --api-base-url http://127.0.0.1:8082/v1 \
  --api-key EMPTY \
  --model-name Qwen3-VL-32B-Instruct
```

## Full Benchmark 命令

### OVO

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream

python main_experiments/eval_qwen3vl_ovo_api.py \
  --api_base_url http://127.0.0.1:8082/v1 \
  --api_key EMPTY \
  --model_path Qwen3-VL-32B-Instruct \
  --anno_path /mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_new.json \
  --chunked_dir /mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/chunked_videos \
  --result_dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/main_experiments/results/ovo_qwen3vl32_openai_api_full_fix \
  --recent_frames_only 4 \
  --chunk_duration 1.0 \
  --fps 1.0 \
  --max_qa_tokens 256 \
  --max_concurrency 4
```

- 如果出现 `decord EOF retry`、`BlockingIOError`、`Resource temporarily unavailable`，直接降到：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream

python main_experiments/eval_qwen3vl_ovo_api.py \
  --api_base_url http://127.0.0.1:8082/v1 \
  --api_key EMPTY \
  --model_path Qwen3-VL-32B-Instruct \
  --anno_path /mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_new.json \
  --chunked_dir /mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/chunked_videos \
  --result_dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/main_experiments/results/ovo_qwen3vl32_openai_api_full_fix_c2 \
  --recent_frames_only 4 \
  --chunk_duration 1.0 \
  --fps 1.0 \
  --max_qa_tokens 256 \
  --max_concurrency 2
```

### OVO with Gemini

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream

export GOOGLE_APPLICATION_CREDENTIALS=/mmu_ssd3/group_lisize/hetu/xujia10/joint_tags/scripts/gemini_client/config.json

python main_experiments/eval_gemini_ovo.py \
  --model_name gemini-2.5-pro \
  --anno_path /mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_new.json \
  --chunked_dir /mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/chunked_videos \
  --result_dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/main_experiments/results/ovo_gemini25pro_recent4 \
  --recent_frames_only 4 \
  --chunk_duration 1.0 \
  --fps 1.0 \
  --max_concurrency 1 \
  --api_timeout 120 \
  --api_max_retries 2 \
  --thinking_budget auto
```

### StreamingBench REAL

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream

python main_experiments/eval_streamingbench_real_api.py \
  --api-base-url http://127.0.0.1:8082/v1 \
  --api-key EMPTY \
  --qa-model Qwen3-VL-32B-Instruct \
  --video-dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/data/streamingbench/videos \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/main_experiments/results/streamingbench_real_qwen3vl32_openai_api_full_fix \
  --recent-frames-only 4 \
  --chunk-duration 1.0 \
  --fps 1.0 \
  --max-qa-tokens 256 \
  --max-concurrency 4
```

### StreamingBench SQA

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream

python main_experiments/eval_streamingbench_sqa_api.py \
  --api-base-url http://127.0.0.1:8082/v1 \
  --api-key EMPTY \
  --qa-model Qwen3-VL-32B-Instruct \
  --video-dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/data/streamingbench/videos \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/main_experiments/results/streamingbench_sqa_qwen3vl32_openai_api_full_fix \
  --recent-frames-only 4 \
  --chunk-duration 1.0 \
  --fps 1.0 \
  --max-qa-tokens 256 \
  --max-concurrency 4
```

### StreamingBench Proactive

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream

python main_experiments/eval_streamingbench_proactive_api.py \
  --api-base-url http://127.0.0.1:8082/v1 \
  --api-key EMPTY \
  --qa-model Qwen3-VL-32B-Instruct \
  --video-dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/data/streamingbench/videos \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/main_experiments/results/streamingbench_proactive_qwen3vl32_openai_api_full_fix \
  --recent-frames-only 4 \
  --chunk-duration 1.0 \
  --fps 1.0 \
  --max-qa-tokens 256 \
  --max-concurrency 4
```

## Smoke 命令

### OVO backward

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream

python main_experiments/eval_qwen3vl_ovo_api.py \
  --api_base_url http://127.0.0.1:8082/v1 \
  --api_key EMPTY \
  --model_path Qwen3-VL-32B-Instruct \
  --anno_path /mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/OVO-Bench/data/ovo_bench_new.json \
  --chunked_dir /mmu_mllm_hdd/zhouhanshu/test/OVO-Bench/chunked_videos \
  --result_dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/main_experiments/results/ovo_qwen3vl32_api_openai_smoke_backward_fix_20260418 \
  --recent_frames_only 4 \
  --chunk_duration 1.0 \
  --fps 1.0 \
  --max_qa_tokens 256 \
  --max_concurrency 2 \
  --splits backward \
  --max_samples_per_split 1
```

### StreamingBench REAL

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream

python main_experiments/eval_streamingbench_real_api.py \
  --api-base-url http://127.0.0.1:8082/v1 \
  --api-key EMPTY \
  --qa-model Qwen3-VL-32B-Instruct \
  --video-dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/data/streamingbench/videos \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/main_experiments/results/streamingbench_real_qwen3vl32_api_openai_smoke_fix_20260418 \
  --recent-frames-only 4 \
  --chunk-duration 1.0 \
  --fps 1.0 \
  --max-qa-tokens 256 \
  --max-concurrency 2 \
  --max-questions 1
```

### StreamingBench SQA

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream

python main_experiments/eval_streamingbench_sqa_api.py \
  --api-base-url http://127.0.0.1:8082/v1 \
  --api-key EMPTY \
  --qa-model Qwen3-VL-32B-Instruct \
  --video-dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/data/streamingbench/videos \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/main_experiments/results/streamingbench_sqa_qwen3vl32_api_openai_smoke_fix_20260418 \
  --recent-frames-only 4 \
  --chunk-duration 1.0 \
  --fps 1.0 \
  --max-qa-tokens 256 \
  --max-concurrency 2 \
  --max-questions 1
```

### StreamingBench Proactive

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream

python main_experiments/eval_streamingbench_proactive_api.py \
  --api-base-url http://127.0.0.1:8082/v1 \
  --api-key EMPTY \
  --qa-model Qwen3-VL-32B-Instruct \
  --video-dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/data/streamingbench/videos \
  --output-dir /mmu_mllm_hdd/zhouhanshu/test/exp1/SimpleStream/main_experiments/results/streamingbench_proactive_qwen3vl32_api_openai_smoke_fix_20260418 \
  --recent-frames-only 4 \
  --chunk-duration 1.0 \
  --fps 1.0 \
  --max-qa-tokens 256 \
  --max-concurrency 2 \
  --max-questions 1
```

## Demo

```bash
python /mmu_mllm_hdd/zhouhanshu/test/exp1/demo.py
```

- 用途：快速验证 OpenAI-compatible 接口链路是否可用
