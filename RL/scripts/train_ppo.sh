#!/usr/bin/env bash
set -euo pipefail

RL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
V5_DIR="$(cd -- "${RL_DIR}/.." && pwd)"
PYTHON_BIN="/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/python"

RUN_NAME="ppo_test4rl_8gpu"
RUN_DIR="${RL_DIR}/outputs/runs/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/train.log"
RAY_TMPDIR="/tmp/swray_$$"

DATASET_ROOT="${V5_DIR}/dataset2"
DATASET_NAME="NeXTVideo"
TRAIN_FILE="${DATASET_ROOT}/test4rl.jsonl"
VAL_FILE="${TRAIN_FILE}"

SOURCE_MODEL_PATH="${V5_DIR}/models/qwen3vl8b_sft_anchor_delta_step200_vllm"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
JUDGE_ENABLE="false"
JUDGE_BACKEND="gemini"
TRACE_FIRST_ROLLOUT="1"
TRACE_SAMPLE_EVERY="64"

for arg in "$@"; do
    case "${arg}" in
        data.streamweave.reward.judge.enable=*)
            JUDGE_ENABLE="${arg#*=}"
            ;;
        data.streamweave.reward.judge.backend=*)
            JUDGE_BACKEND="${arg#*=}"
            ;;
    esac
done

mkdir -p "${RUN_DIR}" "${RAY_TMPDIR}"

MODEL_PATH="$("${PYTHON_BIN}" "${RL_DIR}/scripts/prepare_qwen3vl_model_config.py" "${SOURCE_MODEL_PATH}" "${RUN_DIR}/model_config")"

is_true() {
    case "${1,,}" in
        true|1|yes|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

RUNTIME_ENV=(
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    HYDRA_FULL_ERROR=1
    PYTHONFAULTHANDLER=1
    RAY_DEDUP_LOGS=0
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
    RAY_TMPDIR="${RAY_TMPDIR}"
    TOKENIZERS_PARALLELISM=false
    STREAMWEAVE_RL_DIR="${RL_DIR}"
    PYTHONPATH="${RL_DIR}:${RL_DIR}/verl:${V5_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    STREAMWEAVE_TRACE_FIRST_ROLLOUT="${TRACE_FIRST_ROLLOUT}"
    STREAMWEAVE_TRACE_SAMPLE_EVERY="${TRACE_SAMPLE_EVERY}"
)

if is_true "${JUDGE_ENABLE}" && [[ "${JUDGE_BACKEND}" == "gemini" ]]; then
    GEMINI_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/mmu_ssd3/group_lisize/hetu/xujia10/joint_tags/scripts/gemini_client/config.json}"
    RUNTIME_ENV+=(GOOGLE_APPLICATION_CREDENTIALS="${GEMINI_CREDENTIALS}")
else
    GEMINI_CREDENTIALS=""
fi

ulimit -n 65535

if is_true "${JUDGE_ENABLE}" && [[ "${JUDGE_BACKEND}" == "gemini" && ! -f "${GEMINI_CREDENTIALS}" ]]; then
    echo "Gemini judge requires GOOGLE_APPLICATION_CREDENTIALS to point to an existing file." >&2
    echo "Current GOOGLE_APPLICATION_CREDENTIALS=${GEMINI_CREDENTIALS:-<unset>}" >&2
    exit 2
fi

dump_run_artifacts() {
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

trap dump_run_artifacts EXIT

echo "Run artifacts will be saved to ${RUN_DIR}"
echo "StreamWeave model source=${SOURCE_MODEL_PATH}"
echo "StreamWeave model path=${MODEL_PATH}"
echo "StreamWeave dataset=${TRAIN_FILE}"
echo "StreamWeave validation file=${VAL_FILE} (validation is disabled while trainer.test_freq=-1)"
echo "StreamWeave PPO judge enable=${JUDGE_ENABLE} reward judge_weight is forced to 0.0 by default"
echo "StreamWeave base config=${RL_DIR}/configs/streamweave_stepwise.yaml"
echo "StreamWeave trace first_rollout=${TRACE_FIRST_ROLLOUT} sample_every=${TRACE_SAMPLE_EVERY} sample_filter=${STREAMWEAVE_TRACE_SAMPLE_ID:-<none>}"

cd "${RL_DIR}"

env "${RUNTIME_ENV[@]}" "${PYTHON_BIN%/python}/ray" stop --force >/dev/null 2>&1 || true

DATA_ARGS=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size=32
    +data.gen_batch_size=16
    data.val_batch_size=4
    data.max_prompt_length=15360
    data.max_response_length=1024
    data.streamweave.dataset_name="${DATASET_NAME}"
    data.streamweave.dataset.dataset_root="${DATASET_ROOT}"
    data.streamweave.dataset.dataset_name="${DATASET_NAME}"
)

MODEL_ARGS=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    critic.model.path="${MODEL_PATH}"
)

ACTOR_ARGS=(
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.use_kl_loss=true
    actor_rollout_ref.actor.kl_loss_coef="${STREAMWEAVE_PPO_KL_LOSS_COEF:-0.001}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
)

CRITIC_ARGS=(
    critic.enable=true
    critic.ppo_mini_batch_size=8
    critic.ppo_micro_batch_size_per_gpu=1
    critic.model.fsdp_config.model_dtype=bfloat16
    critic.model.use_remove_padding=true
)

ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.max_model_len=16384
    actor_rollout_ref.rollout.max_num_batched_tokens=32768
    actor_rollout_ref.rollout.max_num_seqs=2048
    actor_rollout_ref.rollout.agent.num_workers=32
)

ALGO_ARGS=(
    algorithm.adv_estimator=streamweave_stepwise_ppo_gae
    algorithm.use_kl_in_reward=false
    data.streamweave.reward.judge_weight=0.0
)

TRAINER_ARGS=(
    trainer.use_legacy_worker_impl=enable
    trainer.critic_warmup=0
    trainer.experiment_name="${RUN_NAME}"
    trainer.default_local_dir="${RUN_DIR}/checkpoints"
    trainer.resume_mode=auto
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    ray_kwargs.ray_init.num_cpus=64
    +ray_kwargs.ray_init.object_store_memory=40000000000
    +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}"
    +ray_kwargs.ray_init.include_dashboard=false
    trainer.test_freq=-1
)

set +e
env "${RUNTIME_ENV[@]}" "${PYTHON_BIN}" -m verl.trainer.main_ppo \
    --config-path="${RL_DIR}/configs" \
    --config-name=streamweave_stepwise \
    "${DATA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ACTOR_ARGS[@]}" \
    "${CRITIC_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${ALGO_ARGS[@]}" \
    "${TRAINER_ARGS[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
train_status="${PIPESTATUS[0]}"
set -e

exit "${train_status}"
