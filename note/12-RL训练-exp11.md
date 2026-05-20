# RL 训练 exp11

本文记录 exp11 系列 RL 实验配置。跑完后的指标、checkpoint、导出模型和评测结果后续补充到本文。

## 实验目标

exp11 重新从较早的 SFT 模型启动 RL，观察在更低 KL、更低 answer weight、重算 silence reward 后，模型是否能改善：

- 该答时不答。
- 不该答时乱答。
- 依靠大量正确沉默刷高 answer reward。
- judge step reward 与 answer reward 的训练信号是否更稳定。

## 代码口径

当前核心代码改动：

```text
RL/streamweave_rl/env.py
RL/streamweave_rl/agent_loop_stepwise.py
```

answer attempt 硬编码奖励：

```text
GRPPO_ANSWER_ATTEMPT_REWARD_OFFSET = 0.1
```

answer reward 语义：

```text
该答但没答: 0
该答且答错: silence_reward_value + 0.1
该答且答对: 1.0 + silence_reward_value + 0.1
```

在当前 `silence_reward_value=0.1` 下：

```text
该答且答错: 0.2
该答且答对: 1.2
```

silence reward 后处理：

```text
位置: RL/streamweave_rl/agent_loop_stepwise.py
时机: trajectory 结束后，verl 聚合 reward_extra_info 到 non_tensor_batch 前
```

只对 silence 部分重算，不改 answer turn 的 reward：

```text
K = 当前 trajectory 中 answer supervision turn 数
  + 当前 trajectory 中非 answer supervision 但模型输出 answer 的 turn 数

不该答且模型没答:
  grppo_answer_reward = silence_reward_value / K

不该答但模型乱答:
  grppo_answer_reward = 0

该答的 turn:
  grppo_answer_reward 保持 env 原始值
```

同步字段：

```text
extra_fields["grppo_answer_reward"]
extra_fields["reward_extra_info"]["grppo_answer_reward"]
```

说明：

- 训练 advantage 使用 `non_tensor_batch["grppo_answer_reward"]`。
- `turn_reward` / `reward_score` 目前不作为 GRPPO answer advantage 的来源。
- trace 中如果看 `turn_reward`，它可能不完全反映后处理后的 answer reward。

## 共同配置

基础：

```text
base config: RL/configs/streamweave_stepwise.yaml
agent config: RL/configs/streamweave_agent_stepwise.yaml
dataset_root: dataset2
train_file: dataset2/rl_0516_filter2.jsonl
val_file: dataset2/rl_0515_val.jsonl
source_model: models/qwen_sft_0513
prompt_profile: eval
judge: gemini
judge_prompt_version: streamweave_grppo_judge_v1
```

训练参数：

```text
actor lr: 1e-6
actor.use_kl_loss: true
actor.kl_loss_coef: 0.002
actor.kl_loss_type: low_var_kl
critic.enable: false
ppo_micro_batch_size_per_gpu: 4
log_prob_micro_batch_size_per_gpu: 4
ppo_max_token_len_per_gpu: 32768
log_prob_max_token_len_per_gpu: 32768
clip_ratio_low: 0.2
clip_ratio_high: 0.28
```

rollout：

```text
rollout.n: 8
temperature: 1.0
top_p: 0.95
rollout.max_model_len: 12288
rollout.max_num_batched_tokens: 65536
rollout.max_num_seqs: 3072
rollout.gpu_memory_utilization: 0.7
val_kwargs.n: 1
val_kwargs.do_sample: false
val_kwargs.temperature: 0
```

GRPPO：

```text
algorithm.adv_estimator: streamweave_stepwise_grppo
algorithm.use_kl_in_reward: false
algorithm.filter_groups.enable: false
algorithm.grppo_answer_decay: 0.3
algorithm.grppo_step_weight: 1.0
algorithm.grppo_answer_weight: 0.8
algorithm.grppo_norm_by_std: true
algorithm.grppo_min_std: 0.01
algorithm.grppo_filter_groups.enable: true
algorithm.grppo_filter_groups.min_std: 0.01
algorithm.grppo_forced_answer_postprocess_enable: false
algorithm.stepwise_validation_score_key: grppo_target_trajectory_score
```

reward：

```text
grppo_process_weight: 0.7
grppo_format_weight: 0.15
grppo_note_frequency_weight: 0.15
grppo_answer_event_mode: timeline
grppo_silence_reward: true
grppo_silence_reward_value: 0.1
grppo_target_answer_weight: 1.0
grppo_target_format_weight: 0.0
```

保存：

```text
resume_mode: auto
save_freq: 10
test_freq: 30
total_epochs: 2
max_actor_ckpt_to_keep: 5
max_critic_ckpt_to_keep: 5
```

## 当前最好模型对应配置：exp10

说明：

```text
当前最好模型来自 exp10 早期实际运行配置。这里记录的是当时 Hydra 展开的真实配置，
不是当前 RL/scripts/train_exp10.sh 文件的最新内容；train_exp10.sh 后续被改过参数。
```

当前主力模型：

```text
model: models/qwen3vl_rl_exp10_step20
source checkpoint: RL/outputs/runs/exp10/checkpoints/global_step_20
run_dir: RL/outputs/runs/exp10
framework: verl
model_engine: dp
```

数据与模型：

```text
source_model: models/qwen_sft_0513
runtime actor model path: RL/outputs/runs/exp10/model_config
train_file: dataset2/rl_0516_filter2.jsonl
val_file: dataset2/rl_0515_val.jsonl
dataset_name: mixed_rl_exp3
prompt_profile: eval
require_question: true
```

规模：

```text
run_name: exp10
nnodes: 1
n_gpus_per_node: 8
total_gpus: 8
train_batch_size: 16
gen_batch_size: 16
val_batch_size: 16
rollout.n: 8
agent.num_workers: 32
real rollout trajectories per step: 16 * 8 = 128
```

actor / PPO：

```text
actor.optim.lr: 1e-5
actor.optim.lr_scheduler_type: constant
actor.optim.total_training_steps: 320
actor.use_kl_loss: true
actor.kl_loss_coef: 0.001
actor.kl_loss_type: low_var_kl
actor.ppo_mini_batch_size: 16
actor.ppo_micro_batch_size_per_gpu: 4
actor.ppo_epochs: 1
actor.ppo_max_token_len_per_gpu: 32768
actor.clip_ratio_low: 0.2
actor.clip_ratio_high: 0.28
actor.grad_clip: 1
actor.checkpoint.load_contents: ["model", "optimizer", "extra"]
actor.checkpoint.save_contents: ["model", "optimizer", "extra"]
critic.enable: false
```

rollout / logprob：

```text
rollout.name: vllm
rollout.mode: async
rollout.temperature: 1.0
rollout.top_p: 0.95
rollout.max_model_len: 12288
rollout.max_num_batched_tokens: 65536
rollout.max_num_seqs: 3072
rollout.gpu_memory_utilization: 0.7
rollout.log_prob_micro_batch_size_per_gpu: 4
rollout.log_prob_max_token_len_per_gpu: 32768
ref.log_prob_micro_batch_size_per_gpu: 4
ref.log_prob_max_token_len_per_gpu: 32768
val_kwargs.n: 1
val_kwargs.do_sample: false
val_kwargs.temperature: 0
```

reward / GRPPO：

```text
algorithm.adv_estimator: streamweave_stepwise_grppo
algorithm.use_kl_in_reward: false
algorithm.filter_groups.enable: false
algorithm.grppo_step_weight: 1.0
algorithm.grppo_answer_decay: 0.3
algorithm.grppo_answer_weight: 0.6
algorithm.grppo_norm_by_std: true
algorithm.grppo_min_std: 0.01
algorithm.grppo_filter_groups.enable: true
algorithm.grppo_filter_groups.min_std: 0.01
algorithm.grppo_silence_reward_value: 0.05
algorithm.grppo_forced_answer_postprocess_enable: false
algorithm.stepwise_validation_score_key: grppo_target_trajectory_score
data.streamweave.reward.grppo_process_weight: 0.7
data.streamweave.reward.grppo_format_weight: 0.15
data.streamweave.reward.grppo_note_frequency_weight: 0.15
data.streamweave.reward.grppo_answer_event_mode: timeline
data.streamweave.reward.grppo_silence_reward: true
data.streamweave.reward.grppo_silence_reward_value: 0.05
data.streamweave.reward.grppo_target_answer_weight: 1.0
data.streamweave.reward.grppo_target_format_weight: 0.0
```

judge：

```text
judge.backend: gemini
judge.model: gemini-2.5-flash
judge.prompt_version: streamweave_grppo_judge_v1
judge.temperature: 0
judge.top_p: 0.1
judge.timeout_seconds: 180
judge.max_tokens: 2048
judge.max_image_side: 512
judge.image_quality: 80
```

保存与评测记录：

```text
save_freq: 10
test_freq: 30
total_epochs: 2
resume_mode: auto
max_actor_ckpt_to_keep: 5
max_critic_ckpt_to_keep: 5
exported_model: models/qwen3vl_rl_exp10_step20
evaluation_note: note/模型评测.md
```

## exp11_32

脚本：

```text
RL/scripts/train_exp11_32.sh
```

规模：

```text
run_name: exp11_32
nnodes: 4
n_gpus_per_node: 8
total_gpus: 32
train_batch_size: 32
gen_batch_size: 32
val_batch_size: 32
agent.num_workers: 128
real rollout trajectories per step: 32 * 8 = 256
```

输出：

```text
run_dir: RL/outputs/runs/exp11_32
train_log: RL/outputs/runs/exp11_32/train.log
checkpoints: RL/outputs/runs/exp11_32/checkpoints
```

节点：

```text
10.82.121.78
10.82.121.83
10.82.122.16
10.82.122.215
```

启动流程：

```bash
# 每台机器先清理
/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/ray stop --force
rm -rf /dev/shm/*ray* /dev/shm/plasma* /dev/shm/sem.* /dev/shm/torch_* /dev/shm/nccl* /dev/shm/vllm*
```

head：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
EXP11_32_RAY_ROLE=head RAY_HEAD_IP=10.82.121.78 bash RL/scripts/train_exp11_32.sh
```

workers：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
EXP11_32_RAY_ROLE=worker RAY_HEAD_IP=10.82.121.78 RAY_NODE_IP=10.82.121.83 bash RL/scripts/train_exp11_32.sh
```

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
EXP11_32_RAY_ROLE=worker RAY_HEAD_IP=10.82.121.78 RAY_NODE_IP=10.82.122.16 bash RL/scripts/train_exp11_32.sh
```

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
EXP11_32_RAY_ROLE=worker RAY_HEAD_IP=10.82.121.78 RAY_NODE_IP=10.82.122.215 bash RL/scripts/train_exp11_32.sh
```

资源检查：

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 /mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/python -c "import ray; ray.init(address='10.82.121.78:6379'); nodes=ray.nodes(); print([(n['NodeManagerAddress'], n['Alive'], n['Resources'].get('GPU',0), n['Resources'].get('CPU',0)) for n in nodes]); print('alive_gpu=', sum(n['Resources'].get('GPU',0) for n in nodes if n['Alive']))"
```

预期：

```text
alive_gpu=32
```

driver：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
EXP11_32_RAY_ROLE=driver RAY_HEAD_IP=10.82.121.78 bash RL/scripts/train_exp11_32.sh
```

## exp11_24

脚本：

```text
RL/scripts/train_exp11_24.sh
```

创建原因：

```text
10.82.122.215 暂不可用，因此准备 3 机 24 卡版本。
```

规模：

```text
run_name: exp11_24
nnodes: 3
n_gpus_per_node: 8
total_gpus: 24
train_batch_size: 24
gen_batch_size: 24
val_batch_size: 24
agent.num_workers: 96
real rollout trajectories per step: 24 * 8 = 192
```

输出：

```text
run_dir: RL/outputs/runs/exp11_24
train_log: RL/outputs/runs/exp11_24/train.log
checkpoints: RL/outputs/runs/exp11_24/checkpoints
```

节点：

```text
10.82.121.78
10.82.121.83
10.82.122.16
```

启动流程：

```bash
# 每台机器先清理
/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/ray stop --force
rm -rf /dev/shm/*ray* /dev/shm/plasma* /dev/shm/sem.* /dev/shm/torch_* /dev/shm/nccl* /dev/shm/vllm*
```

head：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
EXP11_24_RAY_ROLE=head RAY_HEAD_IP=10.82.121.78 bash RL/scripts/train_exp11_24.sh
```

workers：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
EXP11_24_RAY_ROLE=worker RAY_HEAD_IP=10.82.121.78 RAY_NODE_IP=10.82.121.83 bash RL/scripts/train_exp11_24.sh
```

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
EXP11_24_RAY_ROLE=worker RAY_HEAD_IP=10.82.121.78 RAY_NODE_IP=10.82.122.16 bash RL/scripts/train_exp11_24.sh
```

资源检查：

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 /mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/python -c "import ray; ray.init(address='10.82.121.78:6379'); nodes=ray.nodes(); print([(n['NodeManagerAddress'], n['Alive'], n['Resources'].get('GPU',0), n['Resources'].get('CPU',0)) for n in nodes]); print('alive_gpu=', sum(n['Resources'].get('GPU',0) for n in nodes if n['Alive']))"
```

预期：

```text
alive_gpu=24
```

driver：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
EXP11_24_RAY_ROLE=driver RAY_HEAD_IP=10.82.121.78 bash RL/scripts/train_exp11_24.sh
```

## exp12_16

脚本：

```text
RL/scripts/train_exp12_16.sh
```

创建原因：

```text
在 exp11 配置基础上降低 answer advantage 权重，观察 answer reward 对整体训练稳定性和乱答/漏答指标的影响。
使用 2 机 16 卡，batch 改为 16。
```

相对 exp11_24 的主要改动：

```text
run_name: exp12_16
nnodes: 2
total_gpus: 16
train_batch_size: 16
gen_batch_size: 16
val_batch_size: 16
agent.num_workers: 64
algorithm.grppo_answer_weight: 0.4
real rollout trajectories per step: 16 * 8 = 128
```

保持不变的关键参数：

```text
source_model: models/qwen_sft_0513
train_file: dataset2/rl_0516_filter2.jsonl
val_file: dataset2/rl_0515_val.jsonl
actor lr: 1e-6
actor.kl_loss_coef: 0.002
algorithm.grppo_answer_decay: 0.3
algorithm.grppo_norm_by_std: true
algorithm.grppo_min_std: 0.01
algorithm.grppo_filter_groups.min_std: 0.01
grppo_silence_reward_value: 0.1
rollout.n: 8
save_freq: 10
resume_mode: auto
max_actor_ckpt_to_keep: 5
```

输出：

```text
run_dir: RL/outputs/runs/exp12_16
train_log: RL/outputs/runs/exp12_16/train.log
checkpoints: RL/outputs/runs/exp12_16/checkpoints
```

启动流程：

```bash
# 每台机器先清理
/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/ray stop --force
rm -rf /dev/shm/*ray* /dev/shm/plasma* /dev/shm/sem.* /dev/shm/torch_* /dev/shm/nccl* /dev/shm/vllm*
```

head：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
EXP12_16_RAY_ROLE=head RAY_HEAD_IP=10.82.121.78 bash RL/scripts/train_exp12_16.sh
```

worker：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
EXP12_16_RAY_ROLE=worker RAY_HEAD_IP=10.82.121.78 RAY_NODE_IP=10.82.121.83 bash RL/scripts/train_exp12_16.sh
```

资源检查：

```bash
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 /mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/python -c "import ray; ray.init(address='10.82.121.78:6379'); nodes=ray.nodes(); print([(n['NodeManagerAddress'], n['Alive'], n['Resources'].get('GPU',0), n['Resources'].get('CPU',0)) for n in nodes]); print('alive_gpu=', sum(n['Resources'].get('GPU',0) for n in nodes if n['Alive']))"
```

预期：

```text
alive_gpu=16
```

driver：

```bash
cd /mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5
EXP12_16_RAY_ROLE=driver RAY_HEAD_IP=10.82.121.78 bash RL/scripts/train_exp12_16.sh
```

## 观察指标

启动后优先看：

```text
grppo/target_answer_rate
grppo/target_answer_model_answer_rate
grppo/target_answer_missing_rate
grppo/no_target_model_answer_rate
grppo/no_target_silence_rate
streamweave/grppo_judge_step_reward/mean
streamweave/grppo_step_reward/mean
streamweave/grppo_answer_reward/mean
streamweave/grppo_answer_credit/mean
streamweave/grppo_reward/mean
grppo/step_valid_cohorts
grppo/answer_valid_cohorts
actor/ppo_kl
actor/kl_loss
actor/pg_loss
actor/grad_norm
```

期望方向：

```text
target_answer_model_answer_rate 上升
target_answer_missing_rate 下降
no_target_model_answer_rate 不明显上升
no_target_silence_rate 保持较高
grppo_answer_reward / grppo_answer_credit 更有区分度
step_valid_cohorts 和 answer_valid_cohorts 不应过低
```

## 结果待填

### exp11_32

```text
start_time:
end_time:
latest_step:
latest_checkpoint:
swanlab_url:
exported_model:
OVO full:
OVO 1/8:
StreamingBench:
notes:
```

### exp11_24

```text
start_time:
end_time:
latest_step: 30
latest_checkpoint: RL/outputs/runs/exp11_24/checkpoints/global_step_30
swanlab_url:
exported_model: models/qwen3vl_rl_exp11_24_step30
OVO full:
OVO 1/8:
StreamingBench:
notes:
```

### exp12_16

```text
start_time:
end_time:
latest_step:
latest_checkpoint:
swanlab_url:
exported_model:
OVO full:
OVO 1/8:
StreamingBench:
notes:
```
