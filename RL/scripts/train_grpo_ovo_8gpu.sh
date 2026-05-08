#!/usr/bin/env bash
set -xeuo pipefail

RL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
V5_DIR="$(cd -- "${RL_DIR}/.." && pwd)"
PYTHON_BIN="/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/python"

RUN_NAME="${STREAMWEAVE_RUN_NAME:-grpo_ovo_8gpu}"
RUN_DIR="${STREAMWEAVE_RUN_DIR:-${RL_DIR}/outputs/debug/${RUN_NAME}}"
LOG_FILE="${RUN_DIR}/train.log"
RAY_TMPDIR="/tmp/swray_$$"

mkdir -p "${RUN_DIR}" "${RAY_TMPDIR}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export RAY_TMPDIR="${RAY_TMPDIR}"
export TOKENIZERS_PARALLELISM=false
export STREAMWEAVE_RL_DIR="${RL_DIR}"
export STREAMWEAVE_MODEL_PATH="${STREAMWEAVE_MODEL_PATH:-/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/models/qwen3vl8b_streamweave_sft_answered_full_vllm}"
export STREAMWEAVE_DATASET_ROOT="/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset"
export STREAMWEAVE_IMAGE_RESOLUTION=512
export SWANLAB_LOG_DIR="${RUN_DIR}/swanlab"
export PYTHONPATH="${RL_DIR}:${RL_DIR}/verl:${V5_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export STREAMWEAVE_TRACE_FIRST_ROLLOUT="${STREAMWEAVE_TRACE_FIRST_ROLLOUT:-1}"
export STREAMWEAVE_TRACE_TRAJ_INDEX="${STREAMWEAVE_TRACE_TRAJ_INDEX:-0}"
export STREAMWEAVE_TRACE_MAX_CHARS="${STREAMWEAVE_TRACE_MAX_CHARS:-2000}"
export STREAMWEAVE_TRACE_GRPO_GROUPS="${STREAMWEAVE_TRACE_GRPO_GROUPS:-1}"
if [[ "${STREAMWEAVE_JUDGE_BACKEND:-openai_compatible}" == "gemini" ]]; then
    export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/mmu_ssd3/group_lisize/hetu/xujia10/joint_tags/scripts/gemini_client/config.json}"
fi

ulimit -n 65535

if ! "${PYTHON_BIN}" -c "import swanlab" >/dev/null 2>&1; then
    echo "trainer.logger includes swanlab, but swanlab is not installed in ${PYTHON_BIN}." >&2
    echo "Install it first: ${PYTHON_BIN} -m pip install swanlab" >&2
    exit 2
fi
if [[ "${STREAMWEAVE_REWARD_JUDGE_ENABLE:-false}" == "true" && "${STREAMWEAVE_JUDGE_BACKEND:-openai_compatible}" == "gemini" && ! -f "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
    echo "Gemini judge requires GOOGLE_APPLICATION_CREDENTIALS to point to an existing file." >&2
    echo "Current GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS:-<unset>}" >&2
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

    if [ -d "${RAY_TMPDIR}/session_latest/logs" ]; then
        mkdir -p "${RUN_DIR}/ray_logs"
        cp -a "${RAY_TMPDIR}/session_latest/logs/." "${RUN_DIR}/ray_logs/"
    elif [ -d /tmp/ray/session_latest/logs ]; then
        mkdir -p "${RUN_DIR}/ray_logs"
        cp -a /tmp/ray/session_latest/logs/. "${RUN_DIR}/ray_logs/"
    fi
}

trap dump_debug_artifacts EXIT

echo "Debug artifacts will be saved to ${RUN_DIR}"
echo "StreamWeave trace first_rollout=${STREAMWEAVE_TRACE_FIRST_ROLLOUT} traj=${STREAMWEAVE_TRACE_TRAJ_INDEX} max_chars=${STREAMWEAVE_TRACE_MAX_CHARS} grpo_groups=${STREAMWEAVE_TRACE_GRPO_GROUPS}"

cd "${RL_DIR}"

"${PYTHON_BIN%/python}/ray" stop --force >/dev/null 2>&1 || true

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    --config-path="${RL_DIR}/configs" \
    --config-name=grpo_stepwise \
    data.train_files="/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/ovo/ovo_rl_lt120s.json" \
    data.val_files="/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset/ovo/ovo_rl_lt120s.json" \
    data.train_batch_size=32 \
    +data.gen_batch_size=4 \
    data.val_batch_size=4 \
    data.max_prompt_length=15360 \
    data.max_response_length=1024 \
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
    data.streamweave.runtime.resolution=512 \
    data.streamweave.reward.w_format=0.3 \
    data.streamweave.reward.w_step=0.3 \
    data.streamweave.reward.w_success=0.4 \
    data.streamweave.reward.score_scale="${STREAMWEAVE_REWARD_SCORE_SCALE:-2.0}" \
    data.streamweave.reward.max_notes_per_step="${STREAMWEAVE_REWARD_MAX_NOTES_PER_STEP:-1}" \
    data.streamweave.reward.stale_note_after_steps="${STREAMWEAVE_REWARD_STALE_NOTE_AFTER_STEPS:-3}" \
    data.streamweave.reward.note_frequency_penalty_score="${STREAMWEAVE_REWARD_NOTE_PENALTY_SCORE:-0.0}" \
    data.streamweave.reward.judge.enable="${STREAMWEAVE_REWARD_JUDGE_ENABLE:-false}" \
    data.streamweave.reward.judge_weight="${STREAMWEAVE_REWARD_JUDGE_WEIGHT:-0.0}" \
    data.streamweave.reward.judge.backend="${STREAMWEAVE_JUDGE_BACKEND:-openai_compatible}" \
    data.streamweave.reward.judge.model="${STREAMWEAVE_JUDGE_MODEL:-}" \
    data.streamweave.reward.judge.base_url="${STREAMWEAVE_JUDGE_BASE_URL:-}" \
    data.streamweave.reward.judge.api_key_env="${STREAMWEAVE_JUDGE_API_KEY_ENV:-STREAMWEAVE_JUDGE_API_KEY}" \
    data.streamweave.reward.judge.max_tokens="${STREAMWEAVE_JUDGE_MAX_TOKENS:-768}" \
    data.streamweave.reward.judge.temperature="${STREAMWEAVE_JUDGE_TEMPERATURE:-0.0}" \
    data.streamweave.reward.judge.top_p="${STREAMWEAVE_JUDGE_TOP_P:-0.1}" \
    data.streamweave.reward.judge.timeout_seconds="${STREAMWEAVE_JUDGE_TIMEOUT_SECONDS:-180.0}" \
    data.streamweave.reward.judge.image_quality="${STREAMWEAVE_JUDGE_IMAGE_QUALITY:-80}" \
    data.streamweave.reward.judge.max_image_side="${STREAMWEAVE_JUDGE_IMAGE_RESOLUTION:-512}" \
    data.streamweave.reward.judge.max_retries="${STREAMWEAVE_JUDGE_MAX_RETRIES:-2}" \
    data.streamweave.reward.judge.retry_backoff_seconds="${STREAMWEAVE_JUDGE_RETRY_BACKOFF_SECONDS:-5.0}" \
    data.streamweave.reward.judge.retry_backoff_multiplier="${STREAMWEAVE_JUDGE_RETRY_BACKOFF_MULTIPLIER:-2.0}" \
    data.streamweave.reward.judge.failure_score="${STREAMWEAVE_JUDGE_FAILURE_SCORE:-0.0}" \
    data.streamweave.reward.judge.score_on_invalid="${STREAMWEAVE_JUDGE_SCORE_ON_INVALID:-false}" \
    data.streamweave.dataset.dataset_root="/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset" \
    data.streamweave.dataset.dataset_name=ovo \
    data.streamweave.memory.window_seconds=120.0 \
    actor_rollout_ref.model.path="${STREAMWEAVE_MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=16384 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.max_num_seqs=2048 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.agent.num_workers=16 \
    algorithm.adv_estimator=streamweave_stepwise_traj_grpo \
    algorithm.use_kl_in_reward=False \
    critic.enable=False \
    reward.num_workers=8 \
    trainer.stepwise_rollout=True \
    trainer.stepwise_value_mask=True \
    trainer.balance_batch=False \
    trainer.val_before_train=False \
    trainer.use_legacy_worker_impl=enable \
    trainer.critic_warmup=0 \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=streamweave_rl \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.default_local_dir="${RUN_DIR}/checkpoints" \
    trainer.resume_mode=auto \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    ray_kwargs.ray_init.num_cpus=64 \
    +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
    +ray_kwargs.ray_init.include_dashboard=False \
    trainer.save_freq=30 \
    trainer.test_freq=-1 \
    trainer.total_epochs=2 \
    "$@" 2>&1 | tee "${LOG_FILE}"

exit "${PIPESTATUS[0]}"
