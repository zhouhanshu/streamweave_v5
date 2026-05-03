# 2026-04-20 verl + vLLM 调通记录

## 目标

在本机裸机环境上跑通：

- `verl`
- `vLLM 0.11.0`
- `Qwen3-1.7B`
- `GSM8K`
- 4 卡训练：`CUDA_VISIBLE_DEVICES=4,5,6,7`

约束：

- 必须使用 `vllm`
- 不使用 Docker
- 不使用 `flash-attn`
- 改用 `sdpa`

最终状态：

- 训练已启动并进入实际 step 更新
- 最新可见进度已到 `Training Progress: 7/58`
- 运行目录：
  - `outputs/debug/qwen3_1_7b_gsm8k_v071_vllm_sdpa_4gpu_20260420.092324`


## 最终可用组合

代码：

- 仓库：`/mmu_mllm_hdd/zhouhanshu/test2/test1/verl`
- 版本：`v0.7.1`

环境：

- 环境：`/mmu_mllm_hdd/zhouhanshu/conda/envs/verl312_vllm0110_ray2492`
- `python=3.12.13`
- `torch=2.8.0+cu128`
- `vllm=0.11.0`
- `transformers=4.57.1`
- `ray=2.49.2`
- `datasets=4.8.4`
- `tensordict=0.10.0`
- `flash_attn=ABSENT`

模型与数据：

- 模型：`/mmu_mllm_hdd/Models/Qwen3-1.7B`
- 训练集：`/mmu_mllm_hdd/zhouhanshu/data/gsm8k/train.parquet`
- 验证集：`/mmu_mllm_hdd/zhouhanshu/data/gsm8k/test.parquet`


## 本机约束

机器硬约束是整个排查的根：

- 系统 `glibc=2.31`
- 本机 `nvcc=12.1`

这带来两个直接后果：

1. `flash-attn` 官方现成 wheel 在本机不可用
   - 安装能过
   - `import flash_attn` 时会报：
   - `GLIBC_2.32 not found`

2. `vllm 0.11.0` 官方 wheel 要求的 `torch` 是 `2.8.0`
   - 这条线天然更接近 `cu128`
   - 和本机 `nvcc 12.1` 不完全对齐
   - 但这不是主阻塞，因为独立 `vllm` 已实测能起


## 关键版本问题

### 1. vLLM 版本问题

- `Qwen3-VL` 官方要求 `vllm >= 0.11.0`
- 当前任务最终固定在 `vllm==0.11.0`

结论：

- 如果要官方 `vllm 0.11.0`，核心 torch 栈必须跟到：
  - `torch==2.8.0`
  - `torchvision==0.23.0`
  - `torchaudio==2.8.0`


### 2. flash-attn 问题

尝试过的方向：

- 官方 `flash-attn` wheel
- 旧版本 `flash-attn`
- 继续维持 `flash_attention_2`

结论：

- 本机 `glibc 2.31` 卡住了官方 wheel
- 当前任务最终放弃 `flash-attn`
- 训练改走 `sdpa`


### 3. sglang / torch 版本冲突

排查过程中确认：

- 当前较新的 `sglang 0.5.8` 会把 `torch` 拉到 `2.9.1`
- 但 `vllm 0.11.0` 需要 `torch 2.8.0`

结论：

- `sglang` 和 `vllm 0.11.0` 不能在这条训练线上共用同一套目标版本
- 当前最终路线完全切到 `vllm`


### 4. ray 版本问题

观察到：

- `ray 2.55.0` 下，`verl + vllm` 集成更早出问题
- 降到 `ray 2.49.2` 后，流程能稳定走得更远

最终选择：

- `ray==2.49.2`

结论：

- `ray 2.55.0` 不是唯一根因
- 但它明显不是当前这条栈的好版本
- `2.49.2` 更适合 `v0.7.1 + vllm 0.11.0`


## 后端排查结论

### 结论 1：不是 vLLM 本体起不来

单独测试过：

- 不经过 `verl`
- 直接起 4 卡 `vllm` OpenAI server
- `Qwen3-1.7B` 可以成功启动

这说明：

- `vllm 0.11.0 + 4xA800 + Qwen3-1.7B` 本身是能在这台机器上起来的
- 真正出问题的是 `verl` 对 `vllm` 的封装路径


### 结论 2：问题最初在 verl 的 vLLM 集成层

最早死掉的是：

- `vLLMHttpServer` 这个 Ray actor
- 表现为 actor bootstrap 极早期直接退出
- 没有正常 Python traceback

最终定位：

- `verl/workers/rollout/vllm_rollout/__init__.py` 的 eager import 链太早触发了重型导入
- 导致 `vLLMHttpServer` actor 在非常早期就死


## 为调通所做的代码修改

### 1. lazy import，修掉 vLLM rollout 早期崩溃

文件：

- `verl/workers/rollout/vllm_rollout/__init__.py`

修改：

- 去掉顶层：
  - `from .vllm_rollout import ServerAdapter`
- 改成：
  - `__getattr__` 按需 lazy import `ServerAdapter`

原因：

- 避免仅仅导入 `vllm_async_server.py` 时就提前触发整条 rollout import 链

效果：

- `vLLMHttpServer` actor 能正常启动
- 旧的 actor bootstrap 崩溃消失


### 2. 移除训练路径对 flash_attn 的硬依赖

文件：

- `verl/utils/attention_utils.py`

修改：

- 原来直接依赖：
  - `flash_attn.bert_padding`
- 现在改成：
  - 优先尝试 `flash_attn.bert_padding`
  - 如果没有 `flash_attn`，就回退到：
    - `einops.rearrange`
    - `transformers.modeling_flash_attention_utils._index_first_axis`
    - `_pad_input`
    - `_unpad_input`

原因：

- 不安装 `flash-attn` 时，旧 log-prob 路径会直接 `ModuleNotFoundError`

效果：

- 当前训练线不再要求本机必须装 `flash-attn`


### 3. 修掉 actor 在 compute_log_prob 前停留在 CPU 的问题

文件：

- `verl/workers/engine/base.py`

根因：

- `engine_workers.py` 在 rollout 权重同步后，会手动把 actor engine 挪到 CPU
- 但 `BaseEngineCtx._context_switch()` 只有在 `param_offload=True` 时才会在进入 eval/train 前把模型挪回 GPU
- 当前配置里 `param_offload=False`
- 于是 actor 在 `compute_log_prob()` 前模型还留在 CPU
- FSDP 报：
  - `AssertionError: Expects tensor to be on the compute device cuda:0, was on cpu`

修改：

- 在 `_context_switch()` 中，不再只依赖 `is_param_offload_enabled`
- 进入目标 device 时，强制把 model/grad 拉回 device

效果：

- 训练越过了之前 `compute_log_prob()` 的 CPU device assertion


### 4. 新增独立训练脚本

文件：

- `examples/grpo_trainer/run_qwen3-1.7b_gsm8k_4gpu_vllm_sdpa.sh`

内容方向：

- `vllm`
- `sdpa`
- `bf16`
- 4 卡
- GSM8K
- 保守 smoke-test 参数
- 自动输出 debug artifacts

附加调试能力：

- 保存 `train.log`
- 保存 Ray logs 快照
- 保存 `git_describe/git_status/python_version/pip_list/nvidia_smi/exit_code`


## 当前训练参数

当前脚本的关键运行参数：

- `+actor_rollout_ref.model.override_config.attn_implementation=sdpa`
- `actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16`
- `actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16`
- `actor_rollout_ref.rollout.name=vllm`
- `actor_rollout_ref.rollout.enforce_eager=True`
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.35`
- `actor_rollout_ref.rollout.max_num_batched_tokens=2048`
- `actor_rollout_ref.rollout.max_num_seqs=64`
- `actor_rollout_ref.rollout.n=2`
- `trainer.val_before_train=False`
- `trainer.total_epochs=1`

这些参数的作用是：

- 优先保证跑通
- 牺牲一部分吞吐换稳定性


## 当前实际表现

从最新训练日志读取到：

- step 1:
  - `timing_s/step=110.12s`
  - `perf/throughput=571.79`
- step 6:
  - `timing_s/step=59.05s`
  - `perf/throughput=1097.19`
- step 7:
  - `timing_s/step=56.77s`
  - `perf/throughput=1091.23`

解释：

- step 1 明显包含 warmup 和初次稳定化开销
- 稳态后单步大致在 `57s~59s`
- 当前观测吞吐大约在 `1.08k~1.10k tokens/s`


## 速度会降多少

这里必须区分两件事：

1. 当前已经能跑的真实速度
2. 如果以后装回 `flash-attn`，可能会快多少

当前真实可见速度：

- 稳态吞吐大约 `1.1k tokens/s`

关于不用 `flash-attn` 的性能损失：

- 这里没有本机同任务、同参数、同模型的 `flash-attn` 成功 baseline
- 所以不能给“实测差值”
- 只能给工程估计

保守估计：

- 当前 `sdpa` 相比可用的 `flash_attention_2`
- 在这条 1.7B、长响应、log-prob 参与较多的 GRPO 路线上
- 预计整体吞吐损失大约在：
  - `10% ~ 30%`

更可能受影响的阶段：

- rollout generation
- old_log_prob/ref log_prob

日志上也能看出：

- `gen` 大约 `16s~17s`
- `old_log_prob` 大约 `7s~8s`
- `ref` 大约 `7s~8s`
- `update_actor` 大约 `23s~25s`

所以：

- 当前最重的部分并不只有 attention
- 即便以后补上 `flash-attn`，也不会出现“速度翻倍”这种量级


## 踩过的主要坑

### 1. `flash-attn` wheel 不兼容本机 glibc

现象：

- 能安装
- 不能 import
- 报：
  - `GLIBC_2.32 not found`

结论：

- 官方 wheel 不适合当前机器


### 2. 当前 main 分支和旧环境不匹配

现象：

- main 分支对 `sglang` / `torch` 的要求更高
- 与原先环境冲突严重

处理：

- 单独 clone 一份仓库
- 切到 `v0.7.1`


### 3. `vllm` 路线早期 actor 崩溃

根因：

- `vllm_rollout/__init__.py` eager import

处理：

- 改成 lazy import


### 4. 没装 `flash-attn` 时训练路径崩在 log-prob

根因：

- `attention_utils.py` 直接 import `flash_attn.bert_padding`

处理：

- 增加 transformers fallback


### 5. actor 在 log-prob 前留在 CPU

根因：

- 手动 CPU offload 和 `_context_switch()` 恢复逻辑不一致

处理：

- 修 `BaseEngineCtx._context_switch()`


## 当前剩余风险

### 1. 当前是“跑通版”，不是“最终高性能版”

现在这套参数明显偏保守：

- `enforce_eager=True`
- `gpu_memory_utilization=0.35`
- `rollout.n=2`
- `max_num_batched_tokens=2048`
- `max_num_seqs=64`

优点：

- 稳

缺点：

- 吞吐偏低


### 2. `flash-attn` 仍未解决

当前训练靠 `sdpa`，不是 `flash_attention_2`

影响：

- 吞吐损失
- 更难做到最终最优性能


### 3. 改动属于本地修复，未验证上游兼容面

尤其是：

- `verl/workers/engine/base.py`
- `verl/utils/attention_utils.py`
- `verl/workers/rollout/vllm_rollout/__init__.py`

这些改动当前对本机任务有效，但还没有做更大范围回归验证。


### 4. 当前训练仍在进行中

当前日志显示训练已经在稳态推进，但还没有完整跑完整个 epoch 并完成收尾验证。


## 当前建议

### 短期

先把这轮训练继续跑完，观察：

- 是否完整结束
- 是否有 checkpoint
- 是否在后续 step 中再次出现 device / actor / Ray 崩溃


### 中期

如果这轮训练稳定结束，再做两件事：

1. 适度放大 rollout 参数
2. 再决定是否值得继续攻 `flash-attn`


### 长期

如果以后必须追更高吞吐，优先级建议是：

1. 解决本机可用的 `flash-attn`
2. 再逐步增大：
   - `rollout.n`
   - `gpu_memory_utilization`
   - `max_num_batched_tokens`
   - `max_num_seqs`


## 最终一句话结论

这次真正把 `verl + vllm` 调通，靠的不是继续换参数，而是：

- 固定到 `v0.7.1 + vllm 0.11.0 + torch 2.8.0 + ray 2.49.2`
- 放弃 `flash-attn`，改走 `sdpa`
- 修掉 `verl` 本地三处集成问题：
  - vLLM rollout 早期 eager import
  - log-prob 路径对 `flash_attn` 的硬依赖
  - actor engine 在 phase 切换时未回 GPU

当前这条线已经不是“起不来”，而是**已经在实际训练**。


## 2026-04-20 补充：Qwen3-VL-8B + geo3k + 8 卡 GRPO smoke test

目标：

- 模型：`/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct`
- 数据：
  - `/mmu_mllm_hdd/zhouhanshu/test2/test1/data/geo3k/train.parquet`
  - `/mmu_mllm_hdd/zhouhanshu/test2/test1/data/geo3k/test.parquet`
- 路线：
  - `verl`
  - `vllm`
  - `sdpa`
  - 不使用 `flash-attn`
  - 单机 `8xA800`

新增脚本：

- `examples/grpo_trainer/run_qwen3_vl-8b_geo3k_8gpu_vllm_sdpa.sh`

脚本关键设置：

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- `trainer.n_gpus_per_node=8`
- `actor_rollout_ref.actor.strategy=fsdp2`
- `actor_rollout_ref.ref.strategy=fsdp2`
- `actor_rollout_ref.rollout.name=vllm`
- `+actor_rollout_ref.model.override_config.attn_implementation=sdpa`
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.30`
- `actor_rollout_ref.rollout.max_model_len=6144`
- `actor_rollout_ref.rollout.max_num_batched_tokens=6144`
- `actor_rollout_ref.rollout.max_num_seqs=16`
- `actor_rollout_ref.rollout.n=2`

### 这条线额外遇到的问题

#### 1. `ref.model.override_config` 配错

错误：

- `TypeError: FSDPActorConfig.__init__() got an unexpected keyword argument 'model'`

原因：

- 新 worker 实现下，`ref` 不接受 `ref.model.*` 这一层覆盖

处理：

- 去掉 `+actor_rollout_ref.ref.model.override_config.attn_implementation=sdpa`
- 只保留全局 `actor_rollout_ref.model.override_config.attn_implementation=sdpa`


#### 2. vLLM 默认读取模型原始超长上下文，KV cache 不够

错误：

- vLLM 按模型配置把 `max seq len` 视为 `262144`
- 直接报 KV cache 不足，无法服务请求

处理：

- 显式固定：
  - `actor_rollout_ref.rollout.max_model_len=6144`
  - `actor_rollout_ref.rollout.max_num_batched_tokens=6144`

结论：

- 这一步之后，8 个 `vLLMHttpServer` 都能成功起来


#### 3. `geo3k` reward 依赖缺少 `mathruler`

错误：

- `ModuleNotFoundError: No module named 'mathruler'`

定位：

- `verl/utils/reward_score/geo3k.py` 依赖：
  - `from mathruler.grader import extract_boxed_content, grade_answer`

处理：

- 把 `mathruler` 安装到工作区本地：
  - `verl/.vendor/mathruler`
- 在脚本里增加：
  - `VENDOR_DIR=${REPO_ROOT}/.vendor`
  - `export PYTHONPATH="${VENDOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}"`

结论：

- reward 路径恢复正常
- 不需要改动当前 conda 环境本体


### 实际训练状态

最终有效 run：

- `outputs/debug/qwen3_vl_8b_geo3k_vllm_sdpa_8gpu_20260420.124412`

训练过程确认到的关键点：

- 8 个 `vLLMHttpServer` 都成功启动
- CUDA graph capture 成功
- `AgentLoopManager` 已建立
- `update_weights done` 已出现
- `Training Progress` 已推进到：
  - `1/32`
  - `2/32`
  - `3/32`

中断前可见的训练指标：

- `step:1`
  - `critic/rewards/mean=0.2671875`
  - `timing_s/step=81.998`
  - `perf/throughput=191.08`
- `step:2`
  - `critic/rewards/mean=0.1757812`
  - `timing_s/step=64.546`
  - `perf/throughput=263.15`

结论：

- `Qwen3-VL-8B-Instruct + geo3k + 8卡 + vllm + sdpa` 在本机当前环境下已经不只是“能初始化”
- 而是已经进入了实际 RL 训练 step


### 这次为什么停下

不是崩溃。

这轮是人工手动中断，用于先确认链路能否稳定跑过旧崩点。

停止方式：

- 前台 `Ctrl-C`

退出状态：

- `exit_code=130`

收尾确认：

- 训练相关进程已退出
- GPU 上没有残留计算进程


### 当前结论更新

截至 `2026-04-20`：

- 纯文本线：
  - `Qwen3-1.7B + GSM8K + 4卡 + vllm + sdpa` 已完成完整训练收尾
- 多模态线：
  - `Qwen3-VL-8B + geo3k + 8卡 + vllm + sdpa` 已成功进入实际训练
  - 当前只是人工在 `3/32` 附近停下，不是新的结构性崩溃
