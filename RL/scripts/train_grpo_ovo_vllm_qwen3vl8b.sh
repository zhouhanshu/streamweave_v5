#!/usr/bin/env bash
set -euo pipefail

RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V5_DIR="$(cd "${RL_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/mmu_mllm_hdd/zhouhanshu/conda/envs/verl_0425/bin/python}"
MODEL_PATH="${STREAMWEAVE_MODEL_PATH:-/mmu_mllm_hdd/Models/Qwen3-VL-8B-Instruct}"
DATASET_ROOT="${STREAMWEAVE_DATASET_ROOT:-/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/dataset}"
TRAIN_FILE="${STREAMWEAVE_TRAIN_FILE:-${DATASET_ROOT}/ovo/ovo_rl.json}"
VAL_FILE="${STREAMWEAVE_VAL_FILE:-${TRAIN_FILE}}"
EXP_NAME="${STREAMWEAVE_EXPERIMENT_NAME:-grpo_ovo_qwen3vl8b_vllm}"
CKPT_DIR="${STREAMWEAVE_CKPT_DIR:-${RL_DIR}/checkpoints/${EXP_NAME}}"

N_GPUS="${STREAMWEAVE_N_GPUS_PER_NODE:-8}"
TRAIN_BATCH_SIZE="${STREAMWEAVE_TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${STREAMWEAVE_VAL_BATCH_SIZE:-8}"
ROLLOUT_N="${STREAMWEAVE_ROLLOUT_N:-8}"
AGENT_WORKERS="${STREAMWEAVE_AGENT_WORKERS:-8}"
TP_SIZE="${STREAMWEAVE_TENSOR_MODEL_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${STREAMWEAVE_GPU_MEMORY_UTILIZATION:-0.6}"
MAX_MODEL_LEN="${STREAMWEAVE_MAX_MODEL_LEN:-32768}"
MAX_RESPONSE_LENGTH="${STREAMWEAVE_MAX_RESPONSE_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${STREAMWEAVE_MAX_PROMPT_LENGTH:-$((MAX_MODEL_LEN - MAX_RESPONSE_LENGTH))}"
MAX_NUM_BATCHED_TOKENS="${STREAMWEAVE_MAX_NUM_BATCHED_TOKENS:-32768}"
LIMIT_IMAGES="${STREAMWEAVE_LIMIT_IMAGES:-8}"

SAMPLE_FPS="${STREAMWEAVE_SAMPLE_FPS:-1.0}"
FRAMES_PER_STEP="${STREAMWEAVE_FRAMES_PER_STEP:-5}"
MAX_FRAMES="${STREAMWEAVE_MAX_FRAMES:-0}"
MAX_STEPS="${STREAMWEAVE_MAX_STEPS:-0}"
MEMORY_WINDOW_SECONDS="${STREAMWEAVE_MEMORY_WINDOW_SECONDS:-120.0}"

export STREAMWEAVE_RL_DIR="${RL_DIR}"
export STREAMWEAVE_MODEL_PATH="${MODEL_PATH}"
export STREAMWEAVE_DATASET_ROOT="${DATASET_ROOT}"
export STREAMWEAVE_MAX_MODEL_LEN="${MAX_MODEL_LEN}"
export STREAMWEAVE_MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}"
export STREAMWEAVE_MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}"
export STREAMWEAVE_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}"
export PYTHONPATH="${RL_DIR}:${RL_DIR}/verl:${V5_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

if (( MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH > MAX_MODEL_LEN )); then
  echo "Invalid lengths: max_prompt_length(${MAX_PROMPT_LENGTH}) + max_response_length(${MAX_RESPONSE_LENGTH}) > max_model_len(${MAX_MODEL_LEN})" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "Train file not found: ${TRAIN_FILE}" >&2
  exit 2
fi
if [[ ! -f "${VAL_FILE}" ]]; then
  echo "Val file not found: ${VAL_FILE}" >&2
  exit 2
fi
if [[ ! -d "${DATASET_ROOT}/ovo/video" ]]; then
  echo "OVO frame directory not found: ${DATASET_ROOT}/ovo/video" >&2
  exit 2
fi

echo "[StreamWeave RL] python=${PYTHON_BIN}"
echo "[StreamWeave RL] model=${MODEL_PATH}"
echo "[StreamWeave RL] train=${TRAIN_FILE}"
echo "[StreamWeave RL] val=${VAL_FILE}"
echo "[StreamWeave RL] dataset_root=${DATASET_ROOT}"
echo "[StreamWeave RL] backend=vllm rollout_n=${ROLLOUT_N} gpus=${N_GPUS} tp=${TP_SIZE}"
echo "[StreamWeave RL] lengths prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH} model=${MAX_MODEL_LEN} batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
echo "[StreamWeave RL] experiment=${EXP_NAME}"

exec "${PYTHON_BIN}" -m verl.trainer.main_ppo \
  --config-path="${RL_DIR}/configs" \
  --config-name=grpo_stepwise \
  "data.train_files=${TRAIN_FILE}" \
  "data.val_files=${VAL_FILE}" \
  "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
  "data.val_batch_size=${VAL_BATCH_SIZE}" \
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}" \
  "data.max_response_length=${MAX_RESPONSE_LENGTH}" \
  "data.trust_remote_code=true" \
  "data.streamweave.dataset_name=ovo" \
  "data.streamweave.prompt_profile=eval" \
  "data.streamweave.policy=streamweave" \
  "data.streamweave.runtime.sample_fps=${SAMPLE_FPS}" \
  "data.streamweave.runtime.frames_per_step=${FRAMES_PER_STEP}" \
  "data.streamweave.runtime.max_frames=${MAX_FRAMES}" \
  "data.streamweave.runtime.max_steps=${MAX_STEPS}" \
  "data.streamweave.dataset.dataset_root=${DATASET_ROOT}" \
  "data.streamweave.dataset.dataset_name=ovo" \
  "data.streamweave.memory.window_seconds=${MEMORY_WINDOW_SECONDS}" \
  "actor_rollout_ref.model.path=${MODEL_PATH}" \
  "actor_rollout_ref.model.trust_remote_code=true" \
  "actor_rollout_ref.rollout.name=vllm" \
  "actor_rollout_ref.rollout.mode=async" \
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}" \
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}" \
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}" \
  "+actor_rollout_ref.rollout.limit_images=${LIMIT_IMAGES}" \
  "actor_rollout_ref.rollout.agent.num_workers=${AGENT_WORKERS}" \
  "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE}" \
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
  "trainer.n_gpus_per_node=${N_GPUS}" \
  "trainer.experiment_name=${EXP_NAME}" \
  "trainer.default_local_dir=${CKPT_DIR}" \
  "$@"
