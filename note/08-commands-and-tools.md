# 常用命令与工具

## 基本约定

当前工作目录：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
```

常用环境：

```bash
conda activate simple
```

## V5 GRPO 训练

当前唯一保留的 GRPO 启动脚本：

```bash
bash RL/scripts/train_grpo_ovo_8gpu.sh
```

PPO 入口保留用于对照或后续 critic/PPO 路径实验：

```bash
bash RL/scripts/train_ppo.sh data.train_files=/path/to/train.parquet data.val_files=/path/to/val.parquet
```

脚本当前关键路径：

```text
model: /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm
base instruct: /mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct
data: /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/ovo/ovo_rl_lt120s.json
outputs: /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/RL/outputs
```

最新 GRPO 脚本当前是：

```text
save_freq=30
resume_mode=auto
use_remove_padding=true
use_fused_kernels=true
enable_chunked_prefill=true
reward: w_format=0.3, w_step=0.3, w_success=0.4, score_scale=2.0
judge: disabled by default, judge_weight=0.0
```

历史 RL 输出已清理：

```text
RL/outputs
```

## GPU 占用工具

共享磁盘上 `gtest.sh` 已改为按 hostname 写 pid/log，避免两台机器互相覆盖。

```bash
bash /mmu_mllm_hdd/zhouhanshu/test/gtest.sh status
bash /mmu_mllm_hdd/zhouhanshu/test/gtest.sh start --gpus 0-7
bash /mmu_mllm_hdd/zhouhanshu/test/gtest.sh stop
```

默认文件形如：

```text
/mmu_mllm_hdd/zhouhanshu/test/gpu_hold.<hostname>.pid
/mmu_mllm_hdd/zhouhanshu/test/gpu_hold.<hostname>.log
```

查占 GPU 的 Python/Ray/vLLM 进程：

```bash
pgrep -af 'gpu_hold.py|TaskRunner|AgentLoopWorker|WorkerDict|vLLMHttpServer|ray::|vllm'
```

如果当前环境能访问驱动，再用：

```bash
nvidia-smi
```

## OVO RL 数据

当前 RL 数据目录：

```text
/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/ovo
```

历史来源目录：

```text
/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/ovo
```

关键文件：

```text
ovo_bench_new.json        原始 OVO 标注
ovo_rl.json              单 query RL 数据，约 600 条
ovo_rl_lt120s.json       <120s 子集，293 条，当前训练使用
```

数据口径：

- `backward/realtime/forward` 三类单 query 样本。
- `forward` 样本已经展开并去掉 `test_info`，避免 RL dataset 再次重复展开。
- 训练脚本的 `train_files` 和 `val_files` 当前都指向 V5 仓库下的 `ovo_rl_lt120s.json`。

## SFT 历史数据

当前 answered-full SFT 已完成训练，模型和评测路径如下：

```text
LLaMAFactory checkpoint:
/mmu_mllm_hdd/zhouhanshu/test/exp3/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_answered_full

vLLM-compatible model:
/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm

training ShareGPT:
/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/data_engine/sft/outputs/gemini_answered_full/llamafactory_sharegpt_bridge_le20.jsonl

current OVO 1/8 eval output:
/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/outputs/ovo_qwen3vl8b_finetuned_1of8
```

8 卡 OVO 1/8 回评命令：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
./scripts/run_ovo_8gpu_vllm_finetuned.sh \
  /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm
```

旧 SFT 合成链路已经打通，但第一次 SFT 回评退化，只作为历史和可选 RL 初始化来源。

```text
SFT checkpoint:
/mmu_mllm_hdd/zhouhanshu/test/exp3/LlamaFactory/saves/qwen3-vl-8b/full/streamweave_sft_v2_3077

SFT 小规模合成:
/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/data_engine/sft/outputs/gemini_final_8

V4 SFT 标注入口:
/mmu_mllm_hdd/zhouhanshu/test/exp2/streamweave_v4/dataset/streamweave_data/annotations_qa_filter_final.jsonl
```

如果需要重新跑 SFT 合成，先读 `04-sft-training.md`，不要把旧的 1000 条合成命令当作当前主线。

## 评测记录

正式跑分只维护在：

```text
note/实验跑分.md
```

新回评至少记录：

- 模型路径/checkpoint
- 代码目录和 run name
- benchmark、数据范围、样本数、错误数
- prompt/adapter 口径
- 总分、category 分、task 分
- 输出目录和 summary 文件

## 环境备注

- `simple`：主要评测和数据处理环境。
- `llama_zhsdnc`：历史训练/部分数据检查环境。
- Codex 沙箱里的 `localhost` 可能不代表外部服务真实可达，vLLM/Ray 服务以实际外部日志和 HTTP 响应为准。
