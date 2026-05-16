# dataset3 — 反幻觉评测(VideoHallucer / EventHallusion)

本目录在 streamweave_v5 框架之上自建一套评测管线,用来测**"图文交错记忆(anchor + delta) vs 纯文本记忆(delta only)"** 在反幻觉/抗混淆 benchmark 上的差异。

- **框架代码不改一行**,只 import:`streamweave.env`、`streamweave.rollout.RolloutRunner.run_multi_qa_sample`、`streamweave.frame_store.FrameStore`、`backend.factory.create_backend`、`streamweave.config.load_eval_config`
- **新增代码全部在 `dataset3/` 下**

最后更新:2026-05-16

## 核心命题

> 图文交错记忆(图片锚点 + 文本 delta)优于纯文本记忆,因为图片锚点保留原始视觉证据,模型在相似项之间不容易混淆。

跑同一个模型 × 两个 memory policy(`anchor_delta` / `delta_only`),对比反幻觉准确率。

## 数据集

### VideoHallucer(已接入)
- 来源:HuggingFace `bigai-nlco/VideoHallucer`(MIT)
- 题型:**对抗式 yes/no 配对**,basic 题真 + hallucination 题假,两题都答对才算"无幻觉"
- 7 个 subset:`object_relation` / `temporal` / `semantic_detail` / `interaction` / `external_factual` / `external_nonfactual` / `fact_detect`
- 规模:1150 pair / 2300 题 / 1322 视频(去重后 1072 个唯一视频)
- **特例**:`semantic_detail` 的 basic 和 hallucination 用**不同视频**(_a vs _b),其他 6 个 subset 同视频问两次
- 答案匹配:regex `\b(yes|no)\b`(对齐官方 `evaluations/evaluation_utils.py`)
- 核心指标:**Hallucination Rate** = basic 和 hallucination 都答对的 pair 占比

### EventHallusion(已接入)
- 来源:GitHub `Stevetich/EventHallusion`(视频在 Google Drive)
- 题型:yes/no QA(独立题,无配对),设计上故意"违反预期"测语言先验偏差
- 3 个 split:`entire` / `mix`(对应 `interleave/` 视频目录)/ `misleading`
- 规模:397 视频 / 409 题(`entire`/`misleading` 部分视频有 2 个 question,最多 7)
- 答案匹配:`startswith("yes"/"no")`(对齐官方 `eval.py` 的 `extract_pred`)
- 核心指标:**QA Accuracy**(per split + macro / micro overall)
- **暂不接入**:`entire`/`mix` 的 desc 描述题(需要 GPT-4o judge,本项目跳过)
- **命名坑**:`mix_*.json` 里的 id 是 `mix_001`,**视频文件是 `interleave/interleave_001.mp4`**。convert 脚本已自动做 prefix 替换,内部 `video_id = interleave_001`,`source_id = mix_001`。

## 目录结构

```
dataset3/
├── README.md                                  # 本文件
├── __init__.py
├── raw/                                       # 原始下载(只读)
│   ├── videohallucer/{subset}/{json + videos/}
│   ├── eventhallusion/{questions/, videos/videos/, eval.py, gpt4o_judge.py}
│   └── eventhallusion_repo/                   # GitHub 全量 clone
├── videohallucer/                             # 框架可读的 VH 工件
│   ├── videohallucer.json                     # 1350 entry,test_info 装 1-2 个 yes/no probe
│   └── video/<video_id>/                      # FrameStore 抽出的帧 + manifest.json
├── eventhallusion/                            # 框架可读的 EH 工件
│   ├── eventhallusion.json                    # 397 entry,test_info 装 1-7 个 yes/no probe
│   └── video/<video_id>/                      # 抽帧(video_id = mp4 stem,mix 实际叫 interleave_xxx)
├── outputs/<RUN_NAME>/                        # 评测结果
├── convert_videohallucer.py                   # VH:raw → videohallucer.json
├── convert_eventhallusion.py                  # EH:raw → eventhallusion.json
├── extract_frames.py                          # 统一抽帧(--benchmark videohallucer|eventhallusion)
├── videohallucer_loader.py                    # VH:JSON → BenchmarkSample
├── videohallucer_scorer.py                    # VH:yes/no + pair 配对 + per-subset 指标
├── eval_videohallucer.py                      # VH:单进程 CLI
├── eval_videohallucer_batch.py                # VH:多进程 CLI(--resume)
├── eventhallusion_loader.py                   # EH:JSON → BenchmarkSample
├── eventhallusion_scorer.py                   # EH:startswith yes/no + per-split QA accuracy
├── eval_eventhallusion.py                     # EH:单进程 CLI
├── eval_eventhallusion_batch.py               # EH:多进程 CLI(--resume)
├── configs/
│   ├── eval_videohallucer_anchor_delta.yaml   # VH:policy=streamweave(图文)
│   ├── eval_videohallucer_delta_only.yaml     # VH:policy=delta_only(纯文本)
│   ├── eval_eventhallusion_anchor_delta.yaml  # EH:policy=streamweave
│   └── eval_eventhallusion_delta_only.yaml    # EH:policy=delta_only
└── scripts/
    ├── run_videohallucer_8gpu.sh              # VH:启 8 个 vLLM + batch + 自动清理
    └── run_eventhallusion_8gpu.sh             # EH:同上
```

## 一次性数据准备(已完成,记录在此供参考)

### Step 1:下载 raw 数据

```bash
# VideoHallucer(14GB,HF 公开)
python /mmu_mllm_hdd/zhouhanshu/test/download_repo.py \
  --source hf --repo-type dataset \
  --repo-id bigai-nlco/VideoHallucer \
  --output-dir dataset3/raw/videohallucer

# EventHallusion questions(GitHub,几 MB)
git clone https://github.com/Stevetich/EventHallusion.git dataset3/raw/eventhallusion_repo
mkdir -p dataset3/raw/eventhallusion
cp -r dataset3/raw/eventhallusion_repo/questions dataset3/raw/eventhallusion/
cp dataset3/raw/eventhallusion_repo/{eval.py,gpt4o_judge.py} dataset3/raw/eventhallusion/

# EventHallusion 视频(4.7GB,Google Drive)
pip install gdown
cd dataset3/raw
gdown 1IPmx6Y80UrXwVPmZJh6zjCPHtlsw4p9n -O videos.zip
unzip videos.zip -d eventhallusion/videos/
```

### Step 2:转换 JSON

```bash
# VideoHallucer(秒级)
python dataset3/convert_videohallucer.py
# → dataset3/videohallucer/videohallucer.json (1350 entry / 2300 题)

# EventHallusion(秒级)
python dataset3/convert_eventhallusion.py
# → dataset3/eventhallusion/eventhallusion.json (397 entry / 409 题)
```

### Step 3:抽帧

```bash
# VideoHallucer(1072 视频,1fps,约几分钟)
python dataset3/extract_frames.py --benchmark videohallucer --workers 16

# EventHallusion(397 短视频,1fps,约 20 秒)
python dataset3/extract_frames.py --benchmark eventhallusion --workers 16
```

抽帧参数:`sample_fps=1.0`,`jpeg_quality=95`(跟 `configs/eval_ovo.yaml` 一致)。manifest 走 FrameStore 标准格式,后续 `FrameStore.load_frames` 直接能读。

### 依赖

`extract_frames.py` 走 `streamweave.video_extract`,需要:
```bash
pip install decord pillow
```

## 运行评测

### 调试 / Smoke(单进程,mock backend)

```bash
# VideoHallucer
python dataset3/eval_videohallucer.py \
  --config dataset3/configs/eval_videohallucer_anchor_delta.yaml --limit 5

# EventHallusion
python dataset3/eval_eventhallusion.py \
  --config dataset3/configs/eval_eventhallusion_anchor_delta.yaml --limit 5
```

`mock` backend 永远输出 `<answer>A</answer>`,yes/no 提取为空、hit=0,只验证管线连通性。

### 全量评测(8 卡 vLLM)

**典型用法**:同一模型,跑两 benchmark × 两 policy = 4 个 run。

```bash
MODEL=/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct

# VideoHallucer × 2 policy
./dataset3/scripts/run_videohallucer_8gpu.sh   "$MODEL" anchor_delta
./dataset3/scripts/run_videohallucer_8gpu.sh   "$MODEL" delta_only

# EventHallusion × 2 policy
./dataset3/scripts/run_eventhallusion_8gpu.sh  "$MODEL" anchor_delta
./dataset3/scripts/run_eventhallusion_8gpu.sh  "$MODEL" delta_only
```

**环境变量(两个脚本通用)**:

| 变量 | 默认 | 说明 |
|---|---|---|
| `GPUS` | `0 1 2 3 4 5 6 7` | 要用的 GPU id,空格或逗号分隔 |
| `PORT_BASE` | `8000` | vLLM 起始端口,实际端口 = `PORT_BASE + gpu_id` |
| `WORKERS` | `2 × GPU 数` | 评测 worker 进程数 |
| `RESUME` | `0` | `1` 表示复用 shard 续跑 |
| `ALLOW_EXISTING_SERVERS` | `0` | `1` 表示端口已被占用时不报错 |
| `LIMIT` | (空) | 调试用,只跑前 N 个 entry |
| `TASK` | (空) | 调试用,只跑某一 subset / split |
| `RUN_NAME` | `<vh|eh>_<policy>_<model>_<date>` | 覆盖 run 名称 |
| `OUTPUT_DIR` | `dataset3/outputs/<RUN_NAME>` | 覆盖输出目录 |
| `PYTHON` / `VLLM` | conda 默认路径 | 覆盖二进制 |

**调试示例**:

```bash
# 只跑 8 个 entry × semantic_detail
LIMIT=8 TASK=semantic_detail \
  ./dataset3/scripts/run_videohallucer_8gpu.sh "$MODEL" anchor_delta

# 失败后续跑(复用已完成 shard)
RESUME=1 ./dataset3/scripts/run_eventhallusion_8gpu.sh "$MODEL" anchor_delta

# 只用 GPU 4-7
GPUS="4 5 6 7" ./dataset3/scripts/run_videohallucer_8gpu.sh "$MODEL" anchor_delta
```

### 多进程 batch eval(脚本之外手动调用)

如果已经有别处起好的 vLLM endpoint,可以直接调:

```bash
python dataset3/eval_videohallucer_batch.py \
  --config dataset3/configs/eval_videohallucer_anchor_delta.yaml \
  --backend vllm --model /path/to/model \
  --endpoints http://h1:8000/v1 http://h2:8000/v1 \
  --workers 16 --output-name my_run

# EventHallusion 同款
python dataset3/eval_eventhallusion_batch.py \
  --config dataset3/configs/eval_eventhallusion_anchor_delta.yaml \
  --backend vllm --model /path/to/model \
  --endpoints http://h1:8000/v1 http://h2:8000/v1 \
  --workers 16 --output-name my_run
```

## 输出结构

```
dataset3/outputs/<RUN_NAME>/
├── per_qa.jsonl          # 每行一个 question 的推理结果
├── per_pair.jsonl        # 仅 VH:每行一个 pair 的 basic/halluc/overall hit
├── summary.json          # per-subset/split + overall
├── summary.txt           # 人读的对照表
├── .shards/              # 每 worker 一份 shard,供 --resume 复用
├── vllm_logs/            # vLLM 服务端 stdout
├── vllm_pids/            # vLLM 进程 pid(脚本清理用)
└── worker_logs/          # 评测 worker 的 stdout/stderr
```

### VideoHallucer summary.txt(示意)

```
Subset                   pairs     basic    halluc   overall
------------------------------------------------------------
external_factual           200    0.8500    0.4200    0.3550
external_nonfactual        200    0.8500    0.6750    0.6100
fact_detect                 50    0.7800    0.5800    0.4800
interaction                124    0.7258    0.5645    0.4194
object_relation            200    0.9300    0.6850    0.6400
semantic_detail            200    0.8900    0.5450    0.4950
temporal                   176    0.7841    0.4943    0.4148
------------------------------------------------------------
OVERALL (macro)           1150    0.8298    0.5662    0.4877
```

### EventHallusion summary.txt(示意)

```
Split            videos  questions   correct    qs_acc  no_match
----------------------------------------------------------------
entire              109        114        80    0.7018         3
misleading           95        102        61    0.5980         2
mix                 193        193       125    0.6477         1
----------------------------------------------------------------
OVERALL (macro)                  409       266    0.6492         6
OVERALL (micro)                  409       266    0.6504
```

(以上数字均为示意,真实跑出来填回这里)

### 怎么读对照结果

对比 `outputs/<bench>_anchor_delta_*/summary.txt` 和 `outputs/<bench>_delta_only_*/summary.txt`:

**VideoHallucer**:
- 看 **OVERALL overall_acc** 差值 — 总体证据
- 看 **per-subset overall_acc** 差值 — 图文记忆贡献最大的 subset
  - **预测最强** ⭐⭐⭐:`semantic_detail`(双视频细粒度区分)、`object_relation`(物体识别)
  - 预测中:`temporal`、`interaction`、`external_factual`
  - 预测弱:`external_nonfactual`(纯文本+常识能答)、`fact_detect`(主战场在文本)

**EventHallusion**:
- 看 **OVERALL qs_acc** 差值
- per-split:
  - **预测最强** ⭐⭐⭐:`entire`(反预期事件,如 beluga 突然吓人 — 必须看到那一刻的帧)、`mix`(穿插反预期事件)
  - 预测中:`misleading`(视觉前提误判,看准就行)
- 跨 benchmark:VH 测**物体/属性级混淆**,EH 测**事件/时序级反预期**。两者命中差异方向应一致,但 EH 对**短视频内瞬时事件**更敏感

## 设计要点(给后来者)

### 为什么 1 entry 装 1-N 个 question(`test_info` 模式)

很多视频会被问 2-7 个 question(VH 的配对题、EH 的多题视频)。直接拆 N 个独立 sample 会让前缀 rollout 算 N 次,浪费算力。

借用框架自带的 `RolloutRunner.run_multi_qa_sample`:在前 K-1 步建好 memory,最后一步 `deepcopy(env)` 后给每个 query 各跑一遍 — 正好就是"末步多 query 分支"的实现,而且**支持任意 N**(不仅是 2)。

`*_loader.py` 把每个 entry 的 `test_info` 装进 `BenchmarkSample.metadata["qa_list"]`,框架自然走 multi-qa 流程。

### 为什么 VH 的 `semantic_detail` 是 2 个 entry

basic 用视频 _a,hallucination 用视频 _b,**视频不同就没法共享前缀** rollout,只能拆成 2 个独立 sample。两条 entry 共享同一 `pair_id`,scorer 按 pair_id 配对算 hit。

### 视频 dedup(VH)

`external_factual` 和 `external_nonfactual` 的视频 100% 重合,`fact_detect` 的 50 个视频又是 ef/en 的子集。按 `video_id`(mp4 stem)dedup 后,1322 个 raw mp4 只需抽 **1072 帧目录**,共享 frame_dir → 抽帧只跑一次。

### EH 的 mix → interleave 命名映射

`mix_questions.json` 里的 id 是 `mix_001`,但 raw zip 解压后视频是 `videos/videos/interleave/interleave_001.mp4`。`convert_eventhallusion.py` 自动做 prefix 替换:
- `source_id` = `mix_001`(原始 id,保留给 scorer 用)
- `video_id` = `interleave_001`(实际 mp4 stem,作为 frame_dir 名)
- `video` = `interleave/interleave_001.mp4`(配 `video_root` 拼路径)

### 严格 token / 设置对齐

每个 benchmark 的两份 config(`anchor_delta` / `delta_only`)**只在 `policy:` 一行不同**,其余(prompt profile、frames_per_step、resolution、reward 配置、backend、采样温度...)全部锁死。任何对照差异都只能归因于 memory policy。

### VH 和 EH 的 scorer 差异

| | VH | EH |
|---|---|---|
| 答案提取 | regex `\b(yes|no)\b` | `startswith("yes"/"no")` |
| 配对 | basic + halluc(按 pair_id) | 无配对,逐题独立 |
| 主指标 | Overall Acc(配对都对) | QA Accuracy |
| 出错 fallback | 拿不到 yes/no → hit=0 | 同 |

## 与框架原代码的关系

| 框架模块 | 用途 | 是否修改 |
|---|---|---|
| `streamweave.env.StreamWeaveEnv` | memory 状态机 | ❌ 只 import |
| `streamweave.rollout.RolloutRunner` | 末步多 query 分支(`run_multi_qa_sample`) | ❌ 只 import |
| `streamweave.frame_store.FrameStore` | 帧加载 / 抽取 | ❌ 只 import |
| `streamweave.config.load_eval_config` | YAML 解析 | ❌ 只 import |
| `streamweave.schemas.BenchmarkSample, QueryEvent` | 数据载体 | ❌ 只 import |
| `backend.factory.create_backend` | backend 工厂(mock / vllm / openai / gemini) | ❌ 只 import |
| `evaluation/runner.py`, `evaluation/eval_batch.py` | OVO/StreamingBench 的 CLI | ❌ 不复用(单 query 假设,跟 multi-qa 不兼容);只**模仿** pattern |

## TODO

- [ ] **真实模型跑 VideoHallucer 全量对照**(anchor_delta vs delta_only)
- [ ] **真实模型跑 EventHallusion 全量对照**
- [ ] EventHallusion `entire` / `mix` 的描述题接 GPT-4o judge(本 repo 跳过)
- [ ] 对照表加入实际数字
