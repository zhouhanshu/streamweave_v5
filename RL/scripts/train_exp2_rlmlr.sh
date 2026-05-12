#!/usr/bin/env bash
set -euo pipefail

RL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
V5_DIR="$(cd -- "${RL_DIR}/.." && pwd)"
PYTHON_BIN="/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/python"

RUN_NAME="exp2_rlmlr"
RUN_DIR="${RL_DIR}/outputs/runs/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/train.log"
RAY_TMPDIR="/tmp/swray_$$"

DATASET_ROOT="${V5_DIR}/dataset2"
DATASET_NAME="mixed_rl_0512"
TRAIN_FILE="${DATASET_ROOT}/rl_0512_train.jsonl"
VAL_FILE="${DATASET_ROOT}/rl_0512_val.jsonl"

SOURCE_MODEL_PATH="${V5_DIR}/models/qwen3vl8b_streamweave_sft_answered_full_anchor_delta_init_anchor_step200_vllm"

GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
JUDGE_ENABLE="true"
JUDGE_BACKEND="gemini"
TRACE_FIRST_ROLLOUT="1"
TRACE_SAMPLE_EVERY="64"
RLMLR_ALPHA="0.8"
RLMLR_NORM_BY_STD="false"
RLMLR_OUTCOME_KEY="success_score"
RLMLR_STATE_COMPONENTS="[step_score,format_score]"
RLMLR_STATE_WEIGHTS="[1.0,0.2]"

for arg in "$@"; do
    case "${arg}" in
        data.streamweave.reward.judge.enable=*)
            JUDGE_ENABLE="${arg#*=}"
            ;;
        data.streamweave.reward.judge.backend=*)
            JUDGE_BACKEND="${arg#*=}"
            ;;
        algorithm.rlmlr_alpha=*|+algorithm.rlmlr_alpha=*)
            RLMLR_ALPHA="${arg#*=}"
            ;;
        algorithm.rlmlr_norm_by_std=*|+algorithm.rlmlr_norm_by_std=*)
            RLMLR_NORM_BY_STD="${arg#*=}"
            ;;
        algorithm.rlmlr_outcome_key=*|+algorithm.rlmlr_outcome_key=*)
            RLMLR_OUTCOME_KEY="${arg#*=}"
            ;;
        algorithm.rlmlr_state_components=*|+algorithm.rlmlr_state_components=*)
            RLMLR_STATE_COMPONENTS="${arg#*=}"
            ;;
        algorithm.rlmlr_state_weights=*|+algorithm.rlmlr_state_weights=*)
            RLMLR_STATE_WEIGHTS="${arg#*=}"
            ;;
    esac
done

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
    echo "Missing exp2 split files. Generate them first:" >&2
    echo "  ${PYTHON_BIN} ${RL_DIR}/scripts/prepare_rl0512_split.py --val-size 80" >&2
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
echo "StreamWeave train file=${TRAIN_FILE}"
echo "StreamWeave validation file=${VAL_FILE}"
echo "StreamWeave RLMLR alpha=${RLMLR_ALPHA} norm_by_std=${RLMLR_NORM_BY_STD} outcome_key=${RLMLR_OUTCOME_KEY}"
echo "StreamWeave RLMLR state=${RLMLR_STATE_COMPONENTS} weights=${RLMLR_STATE_WEIGHTS}"
echo "StreamWeave judge enable=${JUDGE_ENABLE}"
echo "StreamWeave base config=${RL_DIR}/configs/streamweave_stepwise.yaml (RLMLR via CLI overrides)"
echo "StreamWeave trace first_rollout=${TRACE_FIRST_ROLLOUT} sample_every=${TRACE_SAMPLE_EVERY}"

cd "${RL_DIR}"

env "${RUNTIME_ENV[@]}" "${PYTHON_BIN%/python}/ray" stop --force >/dev/null 2>&1 || true

DATA_ARGS=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size=16
    +data.gen_batch_size=16
    data.val_batch_size=16
    data.max_prompt_length=6144
    data.max_response_length=2048
    data.streamweave.dataset_name="${DATASET_NAME}"
    data.streamweave.dataset.dataset_root="${DATASET_ROOT}"
    data.streamweave.dataset.dataset_name="${DATASET_NAME}"
)

MODEL_ARGS=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
)

ACTOR_ARGS=(
    actor_rollout_ref.actor.optim.lr=1e-5
    actor_rollout_ref.actor.ppo_mini_batch_size=16
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768
)

ROLLOUT_ARGS=(
    actor_rollout_ref.rollout.n=8
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
    algorithm.adv_estimator=streamweave_stepwise_rlmlr
    +algorithm.rlmlr_alpha="${RLMLR_ALPHA}"
    +algorithm.rlmlr_norm_by_std="${RLMLR_NORM_BY_STD}"
    +algorithm.rlmlr_outcome_key="${RLMLR_OUTCOME_KEY}"
    +algorithm.rlmlr_state_components="${RLMLR_STATE_COMPONENTS}"
    +algorithm.rlmlr_state_weights="${RLMLR_STATE_WEIGHTS}"
    algorithm.use_kl_in_reward=false
    algorithm.filter_groups.enable=false
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
    trainer.save_freq=20
    trainer.test_freq=20
    trainer.total_epochs=2
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
    "$@" 2>&1 | tee "${LOG_FILE}"
train_status="${PIPESTATUS[0]}"
set -e

exit "${train_status}"
