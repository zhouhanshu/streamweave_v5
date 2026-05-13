#!/usr/bin/env bash
set -euo pipefail

LF_ROOT=/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/SFT/LlamaFactory
CONDA_ENV=/mmu_mllm_hdd/zhouhanshu/conda/envs/llama_0425
BASE_CONFIG=configs/qwen3vl_8b_full_sft_streamweave_0511_note.yaml
OUTPUT_DIR=saves/qwen3-vl-8b/full/streamweave_sft_0511_note
LOG_ROOT=logs

set_yaml_value() {
  local file=$1
  local key=$2
  local value=$3

  if grep -q "^${key}:" "$file"; then
    sed -i "s|^${key}:.*|${key}: ${value}|" "$file"
  else
    printf '\n%s: %s\n' "$key" "$value" >> "$file"
  fi
}

watch_loop() {
  local log_dir=$1
  local output_abs="$LF_ROOT/$OUTPUT_DIR"

  while true; do
    date -u '+===== %Y-%m-%d %H:%M:%S UTC ====='
    ps -eo pid,ppid,pgid,sid,stat,etime,pcpu,pmem,args \
      | grep -E 'llamafactory-cli|torchrun|launcher.py|resume_debug|tmp.*yaml' \
      | grep -v grep || true

    if [ -f "$output_abs/trainer_log.jsonl" ]; then
      stat -c 'trainer_log mtime=%y size=%s path=%n' "$output_abs/trainer_log.jsonl" || true
      tail -n 3 "$output_abs/trainer_log.jsonl" || true
    fi

    echo
    sleep 60
  done >> "$log_dir/ps_watch.log" 2>&1
}

run_training() {
  local log_dir=$1
  local run_config=$2
  local watch_pid=$3

  trap 'kill "$watch_pid" 2>/dev/null || true' EXIT

  cd "$LF_ROOT"

  export PATH="$CONDA_ENV/bin:$PATH"
  export PYTHONPATH="$LF_ROOT/src:$log_dir/debug_site${PYTHONPATH:+:$PYTHONPATH}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
  export FORCE_TORCHRUN=1
  export PYTHONUNBUFFERED=1

  export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-INFO}"
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
  export TORCH_SHOW_CPP_STACKTRACES="${TORCH_SHOW_CPP_STACKTRACES:-1}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,COLL,ENV}"
  export NCCL_DEBUG_FILE="$log_dir/nccl_%h_%p.log"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-600}"
  export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
  export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"

  {
    echo "RUN_CONFIG=$run_config"
    echo "LOG_DIR=$log_dir"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "START_TIME_UTC=$(date -u '+%Y-%m-%d %H:%M:%S')"
    echo
    "$CONDA_ENV/bin/llamafactory-cli" train "$run_config"
    status=$?
    echo
    echo "END_TIME_UTC=$(date -u '+%Y-%m-%d %H:%M:%S')"
    echo "EXIT_STATUS=$status"
    exit "$status"
  } >> "$log_dir/train.out" 2>&1
}

if [ "${1:-}" = "_watch" ]; then
  watch_loop "$2"
  exit 0
fi

if [ "${1:-}" = "_run" ]; then
  run_training "$2" "$3" "$4"
  exit 0
fi

cd "$LF_ROOT"

existing_launcher=$(pgrep -af "$LF_ROOT/src/llamafactory/launcher.py" | grep -v 'pgrep -af' || true)
if [ -n "$existing_launcher" ]; then
  echo "Refusing to start: an existing LlamaFactory launcher.py process is still running." >&2
  printf '%s\n' "$existing_launcher" >&2
  exit 1
fi

latest_checkpoint=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
if [ -z "$latest_checkpoint" ]; then
  echo "No checkpoint-* found under $LF_ROOT/$OUTPUT_DIR" >&2
  exit 1
fi

run_id=$(date '+%Y%m%d_%H%M%S')
log_dir="$LF_ROOT/$LOG_ROOT/resume_debug_$run_id"
mkdir -p "$log_dir/debug_site"

run_config="$log_dir/qwen3vl_8b_full_sft_streamweave_0511_note.resume.yaml"
cp "$BASE_CONFIG" "$run_config"

set_yaml_value "$run_config" resume_from_checkpoint "$latest_checkpoint"
set_yaml_value "$run_config" overwrite_cache false

if [ "${USE_SWANLAB:-0}" = "1" ]; then
  set_yaml_value "$run_config" report_to swanlab
  set_yaml_value "$run_config" use_swanlab true
else
  set_yaml_value "$run_config" report_to none
  set_yaml_value "$run_config" use_swanlab false
fi

cat > "$log_dir/debug_site/sitecustomize.py" <<'PY'
import faulthandler
import os
import signal
import sys

faulthandler.enable(all_threads=True)

def dump_stack(signum, frame):
    print(f"\n===== SIGUSR1 stack dump pid={os.getpid()} =====", file=sys.stderr, flush=True)
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

signal.signal(signal.SIGUSR1, dump_stack)
PY

cat > "$log_dir/dump_stacks.sh" <<SH
#!/usr/bin/env bash
set -euo pipefail
for p in \$(pgrep -f "$run_config"); do
  kill -USR1 "\$p"
done
SH
chmod +x "$log_dir/dump_stacks.sh"

cat > "$log_dir/kill_run.sh" <<SH
#!/usr/bin/env bash
set -euo pipefail
pkill -TERM -f "$run_config" || true
sleep 5
pkill -9 -f "$run_config" || true
if [ -f "$log_dir/ps_watch.pid" ]; then
  kill "\$(cat "$log_dir/ps_watch.pid")" 2>/dev/null || true
fi
SH
chmod +x "$log_dir/kill_run.sh"

printf '%s\n' "$log_dir" > "$LF_ROOT/$LOG_ROOT/latest_resume_debug"

nohup bash "$0" _watch "$log_dir" >/dev/null 2>&1 &
watch_pid=$!
echo "$watch_pid" > "$log_dir/ps_watch.pid"

nohup bash "$0" _run "$log_dir" "$run_config" "$watch_pid" >/dev/null 2>&1 &
train_pid=$!
echo "$train_pid" > "$log_dir/train.pid"

echo "Started resume debug run"
echo "checkpoint: $latest_checkpoint"
echo "log_dir: $log_dir"
echo "train_pid: $train_pid"
echo "watch_pid: $watch_pid"
echo "train_log: $log_dir/train.out"
echo "run_config: $run_config"
