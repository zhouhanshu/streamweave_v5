#!/usr/bin/env bash
set -euo pipefail

RL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
V5_DIR="$(cd -- "${RL_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/python}"

RUN_NAME="${RUN_NAME:-exp7_smoke}"
RUN_DIR="${RL_DIR}/outputs/runs/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/train.log"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/swray_${RUN_NAME}_$$}"

DATASET_ROOT="${V5_DIR}/dataset2"
DATASET_NAME="mixed_rl_exp3"
TRAIN_FILE="${TRAIN_FILE:-${DATASET_ROOT}/rl_0515_train.jsonl}"
VAL_FILE="${VAL_FILE:-${DATASET_ROOT}/rl_0515_val.jsonl}"

SOURCE_MODEL_PATH="${SOURCE_MODEL_PATH:-${V5_DIR}/models/qwen_sft_0513}"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
JUDGE_ENABLE="${JUDGE_ENABLE:-true}"
JUDGE_BACKEND="${JUDGE_BACKEND:-gemini}"
JUDGE_PROMPT_VERSION="streamweave_grppo_judge_v1"

ADV_ESTIMATOR="streamweave_stepwise_grppo"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-2}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-2}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-16}"
ROLLOUT_N="${ROLLOUT_N:-4}"
MAX_STEPS="${MAX_STEPS:-0}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

GRPPO_ANSWER_DECAY="${GRPPO_ANSWER_DECAY:-0.7}"
GRPPO_PROCESS_WEIGHT="${GRPPO_PROCESS_WEIGHT:-0.7}"
GRPPO_FORMAT_WEIGHT="${GRPPO_FORMAT_WEIGHT:-0.25}"
GRPPO_NOTE_FREQUENCY_WEIGHT="${GRPPO_NOTE_FREQUENCY_WEIGHT:-0.25}"
GRPPO_STEP_WEIGHT="${GRPPO_STEP_WEIGHT:-1.0}"
GRPPO_ANSWER_WEIGHT="${GRPPO_ANSWER_WEIGHT:-0.5}"
GRPPO_NORM_BY_STD="${GRPPO_NORM_BY_STD:-false}"
GRPPO_MIN_STD="${GRPPO_MIN_STD:-0.05}"
GRPPO_FILTER_GROUPS_ENABLE="${GRPPO_FILTER_GROUPS_ENABLE:-true}"
GRPPO_FILTER_MIN_STD="${GRPPO_FILTER_MIN_STD:-0.05}"
GRPPO_ANSWER_EVENT_MODE="${GRPPO_ANSWER_EVENT_MODE:-timeline}"
GRPPO_SILENCE_REWARD="${GRPPO_SILENCE_REWARD:-true}"
GRPPO_SILENCE_REWARD_VALUE="${GRPPO_SILENCE_REWARD_VALUE:-0.2}"
GRPPO_FORCED_ANSWER_POSTPROCESS_ENABLE="${GRPPO_FORCED_ANSWER_POSTPROCESS_ENABLE:-false}"
GRPPO_TARGET_ANSWER_WEIGHT="${GRPPO_TARGET_ANSWER_WEIGHT:-1.0}"
GRPPO_TARGET_FORMAT_WEIGHT="${GRPPO_TARGET_FORMAT_WEIGHT:-0.0}"
STEPWISE_VALIDATION_SCORE_KEY="grppo_target_trajectory_score"
ACTOR_USE_KL_LOSS="${ACTOR_USE_KL_LOSS:-true}"
ACTOR_KL_LOSS_COEF="${ACTOR_KL_LOSS_COEF:-0.001}"
ACTOR_KL_LOSS_TYPE="${ACTOR_KL_LOSS_TYPE:-low_var_kl}"

TRACE_SAMPLE_EVERY="${TRACE_SAMPLE_EVERY:-1}"
TRACE_TRAJ_INDEX="${TRACE_TRAJ_INDEX:-0}"
TRACE_MAX_CHARS="${TRACE_MAX_CHARS:-8000}"
DEBUG_GROUPS="${DEBUG_GROUPS:-2}"
DEBUG_TRAJS="${DEBUG_TRAJS:-4}"
DEBUG_DUMP_DIR="${DEBUG_DUMP_DIR:-${RUN_DIR}/grppo_debug}"
ROLLOUT_DUMP_DIR="${ROLLOUT_DUMP_DIR:-${RUN_DIR}/rollout_data}"

count_cuda_ids() {
    local ids="${1}"
    if [[ -z "${ids}" ]]; then
        echo 0
        return
    fi
    local without_commas="${ids//,/}"
    echo $(( ${#ids} - ${#without_commas} + 1 ))
}

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-$(count_cuda_ids "${GPU_IDS}")}"

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

if [[ "${STREAMWEAVE_ALLOW_EXISTING_RL:-0}" != "1" ]] && pgrep -f "verl.trainer.main_ppo" >/dev/null 2>&1; then
    echo "Another verl.trainer.main_ppo process is already running." >&2
    echo "Refusing to start ${RUN_NAME} because this script stops Ray before launch." >&2
    echo "Set STREAMWEAVE_ALLOW_EXISTING_RL=1 only after you are sure no active run should be preserved." >&2
    pgrep -af "verl.trainer.main_ppo" >&2 || true
    exit 2
fi

mkdir -p "${RUN_DIR}" "${RAY_TMPDIR}" "${DEBUG_DUMP_DIR}" "${ROLLOUT_DUMP_DIR}"

MODEL_PATH="$("${PYTHON_BIN}" "${RL_DIR}/scripts/prepare_qwen3vl_model_config.py" "${SOURCE_MODEL_PATH}" "${RUN_DIR}/model_config")"

RUNTIME_ENV=(
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    CUDA_VISIBLE_DEVICES="${GPU_IDS}"
    HYDRA_FULL_ERROR=1
    PYTHONFAULTHANDLER=1
    RAY_DEDUP_LOGS=0
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
    RAY_TMPDIR="${RAY_TMPDIR}"
    STREAMWEAVE_CONSOLE_METRICS=compact
    TOKENIZERS_PARALLELISM=false
    STREAMWEAVE_RL_DIR="${RL_DIR}"
    PYTHONPATH="${RL_DIR}:${RL_DIR}/verl:${V5_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
    STREAMWEAVE_TRACE_FIRST_ROLLOUT=1
    STREAMWEAVE_TRACE_SAMPLE_EVERY="${TRACE_SAMPLE_EVERY}"
    STREAMWEAVE_TRACE_TRAJ_INDEX="${TRACE_TRAJ_INDEX}"
    STREAMWEAVE_TRACE_MAX_CHARS="${TRACE_MAX_CHARS}"
    STREAMWEAVE_TRACE_GRPPO_GROUPS=1
    STREAMWEAVE_DEBUG_GRPPO_DUMP=1
    STREAMWEAVE_DEBUG_GRPPO_DUMP_DIR="${DEBUG_DUMP_DIR}"
    STREAMWEAVE_DEBUG_GRPPO_GROUPS="${DEBUG_GROUPS}"
    STREAMWEAVE_DEBUG_GRPPO_TRAJS="${DEBUG_TRAJS}"
    STREAMWEAVE_DEBUG_GRPPO_MAX_TEXT_CHARS="${TRACE_MAX_CHARS}"
)

if is_true "${JUDGE_ENABLE}" && [[ "${JUDGE_BACKEND}" == "gemini" ]]; then
    GEMINI_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/mmu_ssd3/group_lisize/hetu/xujia10/joint_tags/scripts/gemini_client/config.json}"
    RUNTIME_ENV+=(GOOGLE_APPLICATION_CREDENTIALS="${GEMINI_CREDENTIALS}")
else
    GEMINI_CREDENTIALS=""
fi

ulimit -n 65535

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
    echo "Missing ${RUN_NAME} data file:" >&2
    echo "  train=${TRAIN_FILE}" >&2
    echo "  val=${VAL_FILE}" >&2
    exit 2
fi
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
    if [ -d "${RAY_TMPDIR}/session_latest/logs" ]; then
        mkdir -p "${RUN_DIR}/ray_logs"
        cp -a "${RAY_TMPDIR}/session_latest/logs/." "${RUN_DIR}/ray_logs/"
    elif [ -d /tmp/ray/session_latest/logs ]; then
        mkdir -p "${RUN_DIR}/ray_logs"
        cp -a /tmp/ray/session_latest/logs/. "${RUN_DIR}/ray_logs/"
    fi
}

trap dump_run_artifacts EXIT

echo "Smoke run artifacts will be saved to ${RUN_DIR}"
echo "Smoke debug dump dir=${DEBUG_DUMP_DIR}"
echo "Smoke rollout dump dir=${ROLLOUT_DUMP_DIR}"
echo "StreamWeave model source=${SOURCE_MODEL_PATH}"
echo "StreamWeave model path=${MODEL_PATH}"
echo "StreamWeave train file=${TRAIN_FILE}"
echo "StreamWeave validation file=${VAL_FILE}"
echo "StreamWeave ${RUN_NAME} groups=${TRAIN_BATCH_SIZE} traj_per_group=${ROLLOUT_N} real_batch=$((TRAIN_BATCH_SIZE * ROLLOUT_N)) total_training_steps=${TOTAL_TRAINING_STEPS}"
echo "StreamWeave ${RUN_NAME} grppo reward process_weight=${GRPPO_PROCESS_WEIGHT} format_weight=${GRPPO_FORMAT_WEIGHT} note_frequency_weight=${GRPPO_NOTE_FREQUENCY_WEIGHT}"
echo "StreamWeave ${RUN_NAME} grppo answer_decay=${GRPPO_ANSWER_DECAY} step_weight=${GRPPO_STEP_WEIGHT} answer_weight=${GRPPO_ANSWER_WEIGHT} norm_by_std=${GRPPO_NORM_BY_STD}"
echo "StreamWeave ${RUN_NAME} grppo answer_event_mode=${GRPPO_ANSWER_EVENT_MODE} silence_reward=${GRPPO_SILENCE_REWARD} silence_reward_value=${GRPPO_SILENCE_REWARD_VALUE}"
echo "StreamWeave ${RUN_NAME} grppo step_filter enable=${GRPPO_FILTER_GROUPS_ENABLE} min_std=${GRPPO_FILTER_MIN_STD} advantage_min_std=${GRPPO_MIN_STD}"
echo "StreamWeave ${RUN_NAME} actor_kl use=${ACTOR_USE_KL_LOSS} coef=${ACTOR_KL_LOSS_COEF} type=${ACTOR_KL_LOSS_TYPE}"
echo "StreamWeave trace sample_every=${TRACE_SAMPLE_EVERY} traj_index=${TRACE_TRAJ_INDEX} debug_groups=${DEBUG_GROUPS} debug_trajs=${DEBUG_TRAJS}"

cd "${RL_DIR}"

env "${RUNTIME_ENV[@]}" "${PYTHON_BIN%/python}/ray" stop --force >/dev/null 2>&1 || true

DATA_ARGS=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size="${TRAIN_BATCH_SIZE}"
    +data.gen_batch_size="${GEN_BATCH_SIZE}"
    data.val_batch_size="${VAL_BATCH_SIZE}"
    data.val_max_samples="${VAL_MAX_SAMPLES}"
    data.max_prompt_length=6144
    data.max_response_length=2048
    data.streamweave.dataset_name="${DATASET_NAME}"
    data.streamweave.dataset.dataset_root="${DATASET_ROOT}"
    data.streamweave.dataset.dataset_name="${DATASET_NAME}"
    data.streamweave.runtime.max_steps="${MAX_STEPS}"
)

MODEL_ARGS=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
)

ACTOR_ARGS=(
    actor_rollout_ref.actor.optim.lr=1e-5
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.use_kl_loss="${ACTOR_USE_KL_LOSS}"
    actor_rollout_ref.actor.kl_loss_coef="${ACTOR_KL_LOSS_COEF}"
    actor_rollout_ref.actor.kl_loss_type="${ACTOR_KL_LOSS_TYPE}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
)

ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.max_model_len=8192
    actor_rollout_ref.rollout.max_num_batched_tokens=65536
    actor_rollout_ref.rollout.max_num_seqs=256
    actor_rollout_ref.rollout.agent.num_workers=4
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.val_kwargs.do_sample=false
    actor_rollout_ref.rollout.val_kwargs.temperature=0
)

ALGO_ARGS=(
    data.streamweave.reward.judge.enable="${JUDGE_ENABLE}"
    data.streamweave.reward.judge.backend="${JUDGE_BACKEND}"
    +data.streamweave.reward.judge.prompt_version="${JUDGE_PROMPT_VERSION}"
    +data.streamweave.reward.grppo_process_weight="${GRPPO_PROCESS_WEIGHT}"
    +data.streamweave.reward.grppo_format_weight="${GRPPO_FORMAT_WEIGHT}"
    +data.streamweave.reward.grppo_note_frequency_weight="${GRPPO_NOTE_FREQUENCY_WEIGHT}"
    +data.streamweave.reward.grppo_answer_event_mode="${GRPPO_ANSWER_EVENT_MODE}"
    +data.streamweave.reward.grppo_silence_reward="${GRPPO_SILENCE_REWARD}"
    +data.streamweave.reward.grppo_silence_reward_value="${GRPPO_SILENCE_REWARD_VALUE}"
    +data.streamweave.reward.grppo_target_answer_weight="${GRPPO_TARGET_ANSWER_WEIGHT}"
    +data.streamweave.reward.grppo_target_format_weight="${GRPPO_TARGET_FORMAT_WEIGHT}"
    algorithm.adv_estimator="${ADV_ESTIMATOR}"
    algorithm.use_kl_in_reward=false
    algorithm.filter_groups.enable=false
    +algorithm.grppo_answer_decay="${GRPPO_ANSWER_DECAY}"
    +algorithm.grppo_step_weight="${GRPPO_STEP_WEIGHT}"
    +algorithm.grppo_answer_weight="${GRPPO_ANSWER_WEIGHT}"
    +algorithm.grppo_norm_by_std="${GRPPO_NORM_BY_STD}"
    +algorithm.grppo_min_std="${GRPPO_MIN_STD}"
    +algorithm.grppo_filter_groups.enable="${GRPPO_FILTER_GROUPS_ENABLE}"
    +algorithm.grppo_filter_groups.min_std="${GRPPO_FILTER_MIN_STD}"
    +algorithm.grppo_silence_reward_value="${GRPPO_SILENCE_REWARD_VALUE}"
    +algorithm.grppo_forced_answer_postprocess_enable="${GRPPO_FORCED_ANSWER_POSTPROCESS_ENABLE}"
    +algorithm.stepwise_validation_score_key="${STEPWISE_VALIDATION_SCORE_KEY}"
    critic.enable=false
)

TRAINER_ARGS=(
    trainer.use_legacy_worker_impl=enable
    trainer.critic_warmup=0
    trainer.logger='["console"]'
    trainer.project_name=streamweave_rl
    trainer.experiment_name="${RUN_NAME}"
    trainer.default_local_dir="${RUN_DIR}/checkpoints"
    trainer.resume_mode=disable
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}"
    trainer.nnodes=1
    ray_kwargs.ray_init.num_cpus=64
    +ray_kwargs.ray_init.object_store_memory=40000000000
    +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}"
    +ray_kwargs.ray_init.include_dashboard=false
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.val_before_train=false
    trainer.total_epochs="${TOTAL_EPOCHS}"
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
    trainer.rollout_data_dir="${ROLLOUT_DUMP_DIR}"
)

set +e
env "${RUNTIME_ENV[@]}" "${PYTHON_BIN}" -m verl.trainer.main_ppo \
    --config-path="${RL_DIR}/configs" \
    --config-name=streamweave_stepwise \
    "${DATA_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${ACTOR_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${ALGO_ARGS[@]}" \
    "${TRAINER_ARGS[@]}" \
    "$@" 2>&1 | tee -a "${LOG_FILE}"
train_status="${PIPESTATUS[0]}"
set -e

echo "Smoke debug files:"
find "${DEBUG_DUMP_DIR}" -maxdepth 1 -type f -name '*.jsonl' -print 2>/dev/null | sort || true
find "${ROLLOUT_DUMP_DIR}" -maxdepth 1 -type f -name '*.jsonl' -print 2>/dev/null | sort || true

exit "${train_status}"
