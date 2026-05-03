set -xeuo pipefail

MODEL_PATH=/mmu_mllm_hdd/Models/Qwen3-1.7B
TRAIN_FILE=/mmu_mllm_hdd/zhouhanshu/data/gsm8k/train.parquet
VAL_FILE=/mmu_mllm_hdd/zhouhanshu/data/gsm8k/test.parquet
TIMESTAMP=$(date -u +%Y%m%d.%H%M%S)
RUN_DIR=outputs/debug/qwen3_1_7b_gsm8k_v071_vllm_sdpa_4gpu_${TIMESTAMP}
LOG_FILE=${RUN_DIR}/train.log

mkdir -p "${RUN_DIR}"

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PYTHONFAULTHANDLER=1

dump_debug_artifacts() {
    local exit_code=$?
    set +e

    echo "${exit_code}" > "${RUN_DIR}/exit_code.txt"
    git describe --tags > "${RUN_DIR}/git_describe.txt" 2>&1
    git status --short > "${RUN_DIR}/git_status.txt" 2>&1
    python3 --version > "${RUN_DIR}/python_version.txt" 2>&1
    python3 -m pip list > "${RUN_DIR}/pip_list.txt" 2>&1

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

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.experiment_name='qwen3_1_7b_gsm8k_v071_vllm_sdpa_4gpu' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=1000 \
    trainer.total_epochs=1 "$@" 2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
