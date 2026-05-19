#!/usr/bin/env bash
set -euo pipefail

RL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
V5_DIR="$(cd -- "${RL_DIR}/.." && pwd)"
PYTHON_BIN="/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/python"

RUN_NAME="exp6"
RUN_DIR="${RL_DIR}/outputs/runs/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/train.log"
RAY_TMPDIR="/tmp/swray_exp6_$$"

DATASET_ROOT="${V5_DIR}/dataset2"
DATASET_NAME="mixed_rl_exp3"
TRAIN_FILE="${DATASET_ROOT}/rl_exp3.jsonl"
VAL_FILE="${DATASET_ROOT}/rl_exp3.jsonl"

SOURCE_MODEL_PATH="${V5_DIR}/models/qwen3vl8b_sft_anchor_delta_step200_vllm"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
JUDGE_ENABLE="true"
JUDGE_BACKEND="gemini"
JUDGE_PROMPT_VERSION="streamweave_grppo_judge_v1"
TRACE_FIRST_ROLLOUT="1"
TRACE_SAMPLE_EVERY="64"

ADV_ESTIMATOR="streamweave_stepwise_grppo"
TRAIN_BATCH_SIZE=16
GEN_BATCH_SIZE=16
VAL_BATCH_SIZE=16
VAL_MAX_SAMPLES=200
ROLLOUT_N=8
MAX_STEPS=0
TEST_FREQ=30
SAVE_FREQ=20
TOTAL_EPOCHS=2

GRPPO_ANSWER_DECAY=0.7
GRPPO_PROCESS_WEIGHT=0.7
GRPPO_FORMAT_WEIGHT=0.25
GRPPO_NOTE_FREQUENCY_WEIGHT=0.25
GRPPO_STEP_WEIGHT=1.0
GRPPO_ANSWER_WEIGHT=0.5
GRPPO_NORM_BY_STD=false
GRPPO_MIN_STD=0.03
GRPPO_FILTER_GROUPS_ENABLE=true
GRPPO_FILTER_MIN_STD=0.03
GRPPO_ANSWER_EVENT_MODE="timeline"
GRPPO_SILENCE_REWARD=true
GRPPO_SILENCE_REWARD_VALUE=0.1
GRPPO_FORCED_ANSWER_POSTPROCESS_ENABLE=false
GRPPO_TARGET_ANSWER_WEIGHT=1.0
GRPPO_TARGET_FORMAT_WEIGHT=0.0
STEPWISE_VALIDATION_SCORE_KEY="grppo_target_trajectory_score"
ACTOR_USE_KL_LOSS=true
ACTOR_KL_LOSS_COEF=0.001
ACTOR_KL_LOSS_TYPE="low_var_kl"

if [[ "${STREAMWEAVE_ALLOW_EXISTING_RL:-0}" != "1" ]] && pgrep -f "verl.trainer.main_ppo" >/dev/null 2>&1; then
    echo "Another verl.trainer.main_ppo process is already running." >&2
    echo "Refusing to start ${RUN_NAME} because the script will stop Ray before launch." >&2
    echo "Set STREAMWEAVE_ALLOW_EXISTING_RL=1 only after you are sure no active run should be preserved." >&2
    pgrep -af "verl.trainer.main_ppo" >&2 || true
    exit 2
fi

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
    STREAMWEAVE_CONSOLE_METRICS=compact
    TOKENIZERS_PARALLELISM=false
    STREAMWEAVE_RL_DIR="${RL_DIR}"
    SWANLAB_LOG_DIR="${RUN_DIR}/swanlab"
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

if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
    echo "Missing exp6 data file:" >&2
    echo "  train=${TRAIN_FILE}" >&2
    echo "  val=${VAL_FILE}" >&2
    exit 2
fi
if ! "${PYTHON_BIN}" -c "import swanlab" >/dev/null 2>&1; then
    echo "trainer.logger includes swanlab, but swanlab is not installed in ${PYTHON_BIN}." >&2
    echo "Install it first: ${PYTHON_BIN} -m pip install swanlab" >&2
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
    "${PYTHON_BIN}" -m pip list > "${RUN_DIR}/pip_list.txt" 2>&1

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
echo "StreamWeave train file=${TRAIN_FILE}"
echo "StreamWeave validation file=${VAL_FILE}"
echo "StreamWeave exp6 adv_estimator=${ADV_ESTIMATOR}"
echo "StreamWeave exp6 judge prompt_version=${JUDGE_PROMPT_VERSION}"
echo "StreamWeave exp6 grppo reward process_weight=${GRPPO_PROCESS_WEIGHT} format_weight=${GRPPO_FORMAT_WEIGHT} note_frequency_weight=${GRPPO_NOTE_FREQUENCY_WEIGHT}"
echo "StreamWeave exp6 grppo answer_decay=${GRPPO_ANSWER_DECAY} step_weight=${GRPPO_STEP_WEIGHT} answer_weight=${GRPPO_ANSWER_WEIGHT} norm_by_std=${GRPPO_NORM_BY_STD}"
echo "StreamWeave exp6 grppo answer_event_mode=${GRPPO_ANSWER_EVENT_MODE} silence_reward=${GRPPO_SILENCE_REWARD} silence_reward_value=${GRPPO_SILENCE_REWARD_VALUE} forced_postprocess=${GRPPO_FORCED_ANSWER_POSTPROCESS_ENABLE}"
echo "StreamWeave exp6 grppo target_reward answer_weight=${GRPPO_TARGET_ANSWER_WEIGHT} format_weight=${GRPPO_TARGET_FORMAT_WEIGHT} validation_score_key=${STEPWISE_VALIDATION_SCORE_KEY}"
echo "StreamWeave exp6 actor_kl use=${ACTOR_USE_KL_LOSS} coef=${ACTOR_KL_LOSS_COEF} type=${ACTOR_KL_LOSS_TYPE}"
echo "StreamWeave exp6 grppo step_filter enable=${GRPPO_FILTER_GROUPS_ENABLE} min_std=${GRPPO_FILTER_MIN_STD}"
echo "StreamWeave exp6 scale train_batch=${TRAIN_BATCH_SIZE} gen_batch=${GEN_BATCH_SIZE} val_batch=${VAL_BATCH_SIZE} val_max_samples=${VAL_MAX_SAMPLES} rollout.n=${ROLLOUT_N} max_steps=${MAX_STEPS}"
echo "StreamWeave judge enable=${JUDGE_ENABLE}"
echo "StreamWeave base config=${RL_DIR}/configs/streamweave_stepwise.yaml"
echo "StreamWeave trace first_rollout=${TRACE_FIRST_ROLLOUT} sample_every=${TRACE_SAMPLE_EVERY}"

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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.use_kl_loss="${ACTOR_USE_KL_LOSS}"
    actor_rollout_ref.actor.kl_loss_coef="${ACTOR_KL_LOSS_COEF}"
    actor_rollout_ref.actor.kl_loss_type="${ACTOR_KL_LOSS_TYPE}"
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
)

ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.n="${ROLLOUT_N}"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=0.95
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7
    actor_rollout_ref.rollout.max_model_len=8192
    actor_rollout_ref.rollout.max_num_batched_tokens=65536
    actor_rollout_ref.rollout.max_num_seqs=2048
    actor_rollout_ref.rollout.agent.num_workers=32
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
    trainer.logger='["console","swanlab"]'
    trainer.project_name=streamweave_rl
    trainer.experiment_name="${RUN_NAME}"
    trainer.default_local_dir="${RUN_DIR}/checkpoints"
    trainer.resume_mode=auto
    trainer.n_gpus_per_node=8
    trainer.nnodes=1
    ray_kwargs.ray_init.num_cpus=64
    +ray_kwargs.ray_init.object_store_memory=40000000000
    +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}"
    +ray_kwargs.ray_init.include_dashboard=false
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq="${TEST_FREQ}"
    trainer.total_epochs="${TOTAL_EPOCHS}"
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

exit "${train_status}"
