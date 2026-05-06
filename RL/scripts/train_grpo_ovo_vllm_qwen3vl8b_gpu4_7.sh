#!/usr/bin/env bash
set -xeuo pipefail

RL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
V5_DIR="$(cd -- "${RL_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/python}"
MODEL_PATH="${MODEL_PATH:-/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct}"
DATASET_ROOT="${DATASET_ROOT:-/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset}"
TRAIN_FILE="${TRAIN_FILE:-${DATASET_ROOT}/ovo/ovo_rl.json}"
VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"

TIMESTAMP="$(date -u +%Y%m%d.%H%M%S)"
RUN_NAME="${RUN_NAME:-grpo_ovo_qwen3vl8b_vllm_gpu4_7}"
RUN_DIR="${RUN_DIR:-${RL_DIR}/outputs/debug/${RUN_NAME}_${TIMESTAMP}}"
LOG_FILE="${RUN_DIR}/train.log"

mkdir -p "${RUN_DIR}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export TOKENIZERS_PARALLELISM=false
export STREAMWEAVE_RL_DIR="${RL_DIR}"
export STREAMWEAVE_MODEL_PATH="${MODEL_PATH}"
export STREAMWEAVE_DATASET_ROOT="${DATASET_ROOT}"
MAX_MODEL_LEN="${STREAMWEAVE_MAX_MODEL_LEN:-16384}"
MAX_RESPONSE_LENGTH="${STREAMWEAVE_MAX_RESPONSE_LENGTH:-512}"
MAX_PROMPT_LENGTH="${STREAMWEAVE_MAX_PROMPT_LENGTH:-$((MAX_MODEL_LEN - MAX_RESPONSE_LENGTH))}"
MAX_NUM_BATCHED_TOKENS="${STREAMWEAVE_MAX_NUM_BATCHED_TOKENS:-16384}"
ROLLOUT_N="${STREAMWEAVE_ROLLOUT_N:-2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${STREAMWEAVE_GPU_MEMORY_UTILIZATION:-0.7}"
ROLLOUT_MAX_NUM_SEQS="${STREAMWEAVE_MAX_NUM_SEQS:-8}"
AGENT_WORKERS="${STREAMWEAVE_AGENT_WORKERS:-4}"
REWARD_WORKERS="${STREAMWEAVE_REWARD_WORKERS:-4}"
ACTOR_PARAM_OFFLOAD="${STREAMWEAVE_ACTOR_PARAM_OFFLOAD:-True}"
ACTOR_OPTIMIZER_OFFLOAD="${STREAMWEAVE_ACTOR_OPTIMIZER_OFFLOAD:-True}"
USE_REMOVE_PADDING="${STREAMWEAVE_USE_REMOVE_PADDING:-False}"
IMAGE_RESOLUTION="${STREAMWEAVE_IMAGE_RESOLUTION:-768}"
export STREAMWEAVE_MAX_MODEL_LEN="${MAX_MODEL_LEN}"
export STREAMWEAVE_MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}"
export STREAMWEAVE_MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}"
export STREAMWEAVE_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}"
export STREAMWEAVE_IMAGE_RESOLUTION="${IMAGE_RESOLUTION}"
export PYTHONPATH="${RL_DIR}:${RL_DIR}/verl:${V5_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ulimit -n 65535

if (( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH > MAX_MODEL_LEN )); then
    echo "Invalid lengths: max_prompt_length(${MAX_PROMPT_LENGTH}) + max_response_length(${MAX_RESPONSE_LENGTH}) > max_model_len(${MAX_MODEL_LEN})" >&2
    exit 2
fi

dump_debug_artifacts() {
    local exit_code=$?
    set +e

    echo "${exit_code}" > "${RUN_DIR}/exit_code.txt"
    git -C "${V5_DIR}" status --short > "${RUN_DIR}/git_status.txt" 2>&1
    "${PYTHON_BIN}" --version > "${RUN_DIR}/python_version.txt" 2>&1
    "${PYTHON_BIN}" -m pip list > "${RUN_DIR}/pip_list.txt" 2>&1

    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi > "${RUN_DIR}/nvidia_smi.txt" 2>&1
    fi

    if [ -d /tmp/ray/session_latest/logs ]; then
        mkdir -p "${RUN_DIR}/ray_logs"
        cp -a /tmp/ray/session_latest/logs/. "${RUN_DIR}/ray_logs/"
    fi
}

trap dump_debug_artifacts EXIT

echo "Debug artifacts will be saved to ${RUN_DIR}"

cd "${RL_DIR}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    --config-path="${RL_DIR}/configs" \
    --config-name=grpo_stepwise \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.return_raw_chat=True \
    data.return_multi_modal_inputs=False \
    data.filter_overlong_prompts=False \
    data.trust_remote_code=True \
    data.streamweave.dataset_name=ovo \
    data.streamweave.prompt_profile=eval \
    data.streamweave.policy=streamweave \
    data.streamweave.runtime.sample_fps=1.0 \
    data.streamweave.runtime.frames_per_step=5 \
    data.streamweave.runtime.max_frames=0 \
    data.streamweave.runtime.max_steps=0 \
    data.streamweave.runtime.resolution="${IMAGE_RESOLUTION}" \
    data.streamweave.dataset.dataset_root="${DATASET_ROOT}" \
    data.streamweave.dataset.dataset_name=ovo \
    data.streamweave.memory.window_seconds=120.0 \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_MODEL_LEN}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_MODEL_LEN}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    algorithm.adv_estimator=streamweave_stepwise_traj_grpo \
    algorithm.use_kl_in_reward=False \
    critic.enable=False \
    reward.num_workers="${REWARD_WORKERS}" \
    trainer.stepwise_rollout=True \
    trainer.stepwise_value_mask=True \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=streamweave_rl \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.default_local_dir="${RUN_DIR}/checkpoints" \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    "$@" 2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
