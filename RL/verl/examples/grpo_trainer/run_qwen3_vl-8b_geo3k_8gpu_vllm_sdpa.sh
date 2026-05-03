#!/usr/bin/env bash
set -xeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=/mmu_mllm_hdd/zhouhanshu/conda/envs/verl312_vllm0110_ray2492/bin/python

MODEL_PATH=${MODEL_PATH:-/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/mmu_mllm_hdd/zhouhanshu/test2/test1/data/geo3k/train.parquet}
VAL_FILE=${VAL_FILE:-/mmu_mllm_hdd/zhouhanshu/test2/test1/data/geo3k/test.parquet}
HF_HOME=${HF_HOME:-/mmu_mllm_hdd/zhouhanshu/test2/test1/.cache/huggingface}
VENDOR_DIR=${VENDOR_DIR:-${REPO_ROOT}/.vendor}
TIMESTAMP=$(date -u +%Y%m%d.%H%M%S)
RUN_DIR=${REPO_ROOT}/outputs/debug/qwen3_vl_8b_geo3k_vllm_sdpa_8gpu_${TIMESTAMP}
LOG_FILE=${RUN_DIR}/train.log

mkdir -p "${RUN_DIR}" "${HF_HOME}" "${VENDOR_DIR}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HF_HOME
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export PYTHONFAULTHANDLER=1
export PYTHONPATH="${VENDOR_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

dump_debug_artifacts() {
    local exit_code=$?
    set +e

    echo "${exit_code}" > "${RUN_DIR}/exit_code.txt"
    git -C "${REPO_ROOT}" describe --tags > "${RUN_DIR}/git_describe.txt" 2>&1
    git -C "${REPO_ROOT}" status --short > "${RUN_DIR}/git_status.txt" 2>&1
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

cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.max_model_len=6144 \
    actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    algorithm.use_kl_in_reward=False \
    trainer.use_legacy_worker_impl=disable \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen3_vl_8b_geo3k_vllm_sdpa_8gpu' \
    trainer.default_local_dir="${RUN_DIR}/checkpoints" \
    trainer.resume_mode=disable \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=1000 \
    trainer.test_freq=1000 \
    trainer.total_epochs=1 "$@" 2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
