# 2026-05-06 Remove Padding Change Log

This note records the recent StreamWeave RL changes made while debugging the slow GRPO run and enabling `use_remove_padding=True`.

## Scope

All changes are limited to:

- `RL/scripts/train_grpo_ovo_vllm_qwen3vl8b_full_4gpu_3_4_6_7_lt120s_fused_chunked.sh`
- `RL/verl/verl/trainer/ppo/core_algos.py`
- `RL/verl/verl/trainer/ppo/ray_trainer.py`
- `RL/verl/verl/workers/actor/dp_actor.py`
- `RL/verl/verl/workers/critic/dp_critic.py`

No files outside `streamweave_v5/RL` were changed for this step.

## Training Script

File:

- `RL/scripts/train_grpo_ovo_vllm_qwen3vl8b_full_4gpu_3_4_6_7_lt120s_fused_chunked.sh`

This is a separate experiment script. The original `train_grpo_ovo_vllm_qwen3vl8b_full_8gpu_lt120s.sh` was not modified.

Current important settings:

```bash
RUN_NAME="grpo_ovo_qwen3vl8b_full_vllm_8gpu_lt120s_fused_chunked"
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.use_fused_kernels=True
actor_rollout_ref.actor.strategy=fsdp
actor_rollout_ref.ref.strategy=fsdp
data.train_batch_size=32
data.gen_batch_size=4
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.rollout.gpu_memory_utilization=0.6
actor_rollout_ref.rollout.max_num_seqs=2048
actor_rollout_ref.rollout.enable_chunked_prefill=True
actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True
trainer.total_epochs=2
```

Rationale:

- `use_remove_padding=True` is the main speed experiment. The previous run showed the bottleneck was `old_log_prob + update_actor`, not vLLM rollout.
- `use_fused_kernels=True` is kept for the fused log-prob path.
- `fsdp` is used instead of `fsdp2` because `fsdp2 + fused_kernels` hit a mixed `torch.Tensor` / `DTensor` error in the fused LM head path.
- `enable_chunked_prefill=True` is kept for rollout-side prefill behavior, though earlier timing showed rollout is not the main bottleneck.
- `disable_mm_preprocessor_cache=True` is restored after vLLM raised `Expected a cached item for mm_hash=...` in the multimodal processor cache during async image rollout.
- `data.train_batch_size` and `actor_rollout_ref.actor.ppo_mini_batch_size` are set to 32 so the actor mini-batch passes verl validation and matches the VAGEN script's explicit value.
- `data.gen_batch_size` is added and set to 4. verl uses this field as the actual dataloader/rollout batch size when present, so each StreamWeave rollout step still starts from 4 original samples to limit Ray object-store spill.
- `actor_rollout_ref.rollout.max_num_seqs` was raised from 16 to 2048. VAGEN does not set this in its script, but its rollout base config defaults to 1024, so 2048 is the strict "higher than VAGEN effective value" setting.
- `actor_rollout_ref.rollout.gpu_memory_utilization` is set to 0.6, matching the VAGEN Sokoban script's explicit value.
- `trainer.total_epochs` is set to 2. With the current 293-sample lt120s OVO file, `data.gen_batch_size=4`, and `drop_last=True`, the progress bar is `floor(293 / 4) * 2 = 146` trainer steps.

## Actor Remove-Padding Fix

File:

- `RL/verl/verl/workers/actor/dp_actor.py`

Problem:

When `actor_rollout_ref.model.use_remove_padding=True`, the actor enters the remove-padding branch. The original code used `index_first_axis(...)` to unpad `position_ids` and then called `.transpose(0, 1)`.

This crashed with:

```text
IndexError: Dimension out of range
```

The deeper issue is not just the crash. In this environment, `index_first_axis(...)` can squeeze `position_ids` unexpectedly. For Qwen3-VL / StreamWeave, this is unsafe because the agent loop builds a 4-channel `position_ids` tensor:

```text
(batch, 4, seq_len)
```

The first channel is text position ids and the remaining three channels are Qwen3-VL MRoPE vision position ids. Losing that channel dimension breaks the model semantics.

Change:

```python
position_indices = indices.to(dtype=torch.long)
if position_ids.dim() == 3:
    position_ids_flat = rearrange(position_ids, "c b s ... -> (b s) c ...")
    position_ids_rmpad = torch.index_select(position_ids_flat, 0, position_indices)
    position_ids_rmpad = position_ids_rmpad.transpose(0, 1).unsqueeze(1)
else:
    position_ids_flat = rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ...")
    position_ids_rmpad = torch.index_select(position_ids_flat, 0, position_indices)
    position_ids_rmpad = position_ids_rmpad.transpose(0, 1)
```

Expected shapes:

```text
text-only position_ids:     (1, nnz)
Qwen3-VL StreamWeave mrope: (4, 1, nnz)
```

The unused `index_first_axis` import was removed from this file.

## Critic Remove-Padding Fix

File:

- `RL/verl/verl/workers/critic/dp_critic.py`

The same `position_ids` unpadding logic was updated in the critic path.

Even though the current GRPO config has `critic.enable=False`, this keeps the PPO / critic path consistent and avoids the same crash if critic training is enabled later.

The unused `index_first_axis` import was also removed from this file.

## Validation Done

Static check:

```bash
python -m py_compile \
  RL/verl/verl/workers/actor/dp_actor.py \
  RL/verl/verl/workers/critic/dp_critic.py
```

Result: passed.

Shape sanity checks:

```text
C=1  -> fixed shape (1, 1, nnz)
C=3  -> fixed shape (3, 1, nnz)
C=4  -> fixed shape (4, 1, nnz)
2D   -> fixed shape (1, nnz)
```

Packed sequence boundary check:

```text
text positions: [0, 1, 2, 3, 0, 1, 2]
packed ids:     [0, 0, 0, 0, 1, 1, 1]
```

This confirms that when remove-padding packs multiple samples into one sequence, the text-position reset can still identify sample boundaries for the SDPA packed-sequence mask path.

## Stepwise Legacy Batch Padding Fix

File:

- `RL/verl/verl/trainer/ppo/ray_trainer.py`

Problem:

With `trainer.use_legacy_worker_impl=enable`, verl dispatches `DataProto` to data-parallel workers by equal chunks. StreamWeave stepwise rollout emits one row per turn, and different trajectories have different turn counts. A batch that starts as `data.gen_batch_size=4`, repeats with `rollout.n=8`, and then expands by turns can therefore produce a row count like 433.

The legacy dispatch path then crashed before log-prob computation:

```text
AssertionError: only support equal chunk. Got size of DataProto 433 and chunk 8.
```

Change:

- `_compute_old_log_prob(...)` now pads the legacy `DataProto` before `actor_rollout_wg.compute_log_prob(...)`, then unpads the returned log-prob tensors.
- `_compute_ref_log_prob(...)` does the same for the legacy ref-log-prob path.
- `_update_actor(...)` now pads the legacy actor-update batch to the actor mini-batch divisor and DP divisor, then zeros `response_mask` and `advantages` for the padding rows before dispatch.
- `_compute_values(...)` and `_update_critic(...)` now apply the same legacy padding pattern for PPO / critic-enabled runs. The current GRPO script has `critic.enable=False`, but this keeps the stepwise framework consistent.
- `agg_loss(...)` now adds a small epsilon to zero-mask-sensitive denominators. This prevents all-padding micro-batches from producing `0/0` when `ppo_micro_batch_size_per_gpu=1`.

For the current config:

```text
rollout.n = 8
actor.ppo_mini_batch_size = 32
dp_size = 8
```

The actor update divisor is `lcm(8, 32 * 8) = 256`. So a 433-row stepwise batch is padded to 512 rows, each of 8 ranks receives 64 rows, and each rank can split its local batch into two 32-row actor mini-batches. The extra rows are duplicated only as transport padding and have zero loss weight.

Important detail:

The padding rows are appended at the end before legacy DP dispatch. With 433 -> 512 rows, the last rank can receive only padding rows. Since padding rows have `response_mask=0`, loss aggregation must be zero-mask safe; otherwise an all-padding micro-batch would compute `0/0`. The epsilon change makes that path return zero loss instead of NaN.

## Remaining Risk

This change fixes the observed `position_ids` shape bug and preserves Qwen3-VL / StreamWeave MRoPE channels. It does not prove the full training loop is now fully compatible with `use_remove_padding=True`.

The next run should verify:

- The job passes `old_log_prob`.
- The job passes `update_actor`.
- `timing_s/old_log_prob` and `timing_s/update_actor` decrease compared with the dense-padding run.
- No new `remove_padding + Qwen3-VL + fused + fsdp` compatibility error appears.

## Comparison With VAGEN Sokoban Script

Compared files:

- StreamWeave current script: `RL/scripts/train_grpo_ovo_vllm_qwen3vl8b_full_4gpu_3_4_6_7_lt120s_fused_chunked.sh`
- VAGEN reference script: `/mmu_mllm_hdd/zhouhanshu/test/VAGEN/examples/train/sokoban/train_grpo_qwen3vl4b.sh`

Important settings now aligned:

```text
actor_rollout_ref.model.use_remove_padding=True
actor_rollout_ref.model.use_fused_kernels=True
actor_rollout_ref.model.enable_gradient_checkpointing=True
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.actor.kl_loss_coef=0.0
actor_rollout_ref.actor.kl_loss_type=low_var_kl
actor_rollout_ref.actor.entropy_coeff=0
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.ref.fsdp_config.param_offload=True
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.n=8
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.rollout.free_cache_engine=True
actor_rollout_ref.rollout.enable_chunked_prefill=True
trainer.save_freq=100
trainer.critic_warmup=0
trainer.nnodes=1
```

Remaining major differences:

| Area | StreamWeave current script | VAGEN Sokoban script | Note |
| --- | --- | --- | --- |
| Entry point | `python -m verl.trainer.main_ppo` | `python3 -m vagen.main_ppo` | Different framework wrapper and config stack. |
| Base config | `configs/grpo_stepwise.yaml` | `vagen/configs/vagen_multiturn.yaml` | StreamWeave uses stepwise rollout/reward logic. |
| Dataset | OVO JSON via `StreamWeaveAgentDataset` | Sokoban YAML via `AgenticDataset` | Task semantics differ. |
| Model | Qwen3-VL-8B StreamWeave SFT checkpoint | `Qwen/Qwen3-VL-4B-Instruct` | StreamWeave model is larger and already SFT-tuned. |
| GPUs | 8 | 4 | StreamWeave uses more GPUs but also a larger model/context. |
| Train batch size | 32 | 32 | Now aligned with VAGEN's explicit value; used by verl validation. |
| Gen batch size | 4 | not set | StreamWeave uses this as actual dataloader/rollout batch size to limit Ray object-store spill. |
| Actor mini batch | 32 | 32 | Now aligned with VAGEN's explicit PPO mini-batch size. |
| Actor micro batch per GPU | 1 | 1 | Now aligned with VAGEN. |
| Ref log-prob micro batch per GPU | 1 | 1 | Now aligned with VAGEN. |
| Rollout log-prob micro batch per GPU | 1 | 1 | Now aligned with VAGEN. |
| Actor max token len per GPU | 16384 | not explicitly set in script | StreamWeave pins the long context budget explicitly. |
| Prompt length | 15360 | 1000 | This is the biggest remaining compute difference. User requirement: do not reduce StreamWeave window. |
| Response length | 1024 | 4000 | StreamWeave keeps shorter answer budget; VAGEN allows long game trajectories. |
| Rollout backend | `vllm` | `sglang` | User environment uses vLLM. |
| Rollout GPU memory utilization | 0.6 | 0.6 | Now aligned with VAGEN's explicit vLLM/SGLang memory utilization value. |
| Rollout max model len | 16384 | not explicitly set in script | VAGEN relies on defaults / engine behavior. |
| Rollout max num batched tokens | 16384 | 10000 | StreamWeave allows larger prefill/token batches. |
| Rollout max num seqs | 2048 | script does not set it; VAGEN rollout default is 1024 | StreamWeave now exceeds VAGEN's effective default. The actual concurrency is still bounded by `max_num_batched_tokens` and long prompt lengths. |
| Rollout enforce eager | `False` | `True` | StreamWeave allows CUDA graph / compiled path; VAGEN disables it. |
| Agent workers | 16 | not the same field; VAGEN uses `rollout.agent.agent_loop_config_path` | StreamWeave uses verl agent loop workers for stepwise video rollout. |
| Algorithm | `streamweave_stepwise_traj_grpo` | `grpo` | StreamWeave has custom trajectory/stepwise advantage handling. |
| Reward | format + success reward from StreamWeave scorer | environment reward | Not comparable directly. |
| Critic | `critic.enable=False` | script configures critic model but does not explicitly set `critic.enable` | Current StreamWeave GRPO does not use critic. |
| Validation before train | `False` | `True` | StreamWeave skips pre-train validation for speed. |
| Test frequency | `-1` | `20` | StreamWeave disables validation during this debug run. |
| Logger | `console`, `swanlab` | `console`, `wandb` | Different experiment logger. |
| Output/debug artifacts | custom debug dump under `RL/outputs/debug/...` | experiment dir logs/checkpoints/rollout data | StreamWeave script captures Ray logs, pip list, git status, nvidia-smi. |

Interpretation:

- The critical compute-side VAGEN settings have now been mostly matched: remove padding, fused kernels, gradient checkpointing, FSDP offload, async rollout, `n=8`, chunked prefill, free cache engine.
- The main remaining speed difference is not a missing VAGEN trick; it is StreamWeave's much longer context window plus larger 8B model.
- `rollout.name` remains intentionally different: StreamWeave uses `vllm`, while the VAGEN script uses `sglang`.
- `enforce_eager` is different. If vLLM stability issues appear, `enforce_eager=True` can be tested, but it is not expected to fix the current `old_log_prob/update_actor` training-side bottleneck.
- Actor mini-batch size is aligned with VAGEN's explicit value. Actual rollout/data loading is intentionally capped with `data.gen_batch_size=4` because StreamWeave expands each sample into multi-step video trajectories and large Ray objects. Rollout GPU memory utilization remains aligned with VAGEN's explicit value.
