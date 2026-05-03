#!/usr/bin/env bash
set -euo pipefail

RL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V5_DIR="$(cd "${RL_DIR}/.." && pwd)"

export STREAMWEAVE_RL_DIR="${RL_DIR}"
export PYTHONPATH="${RL_DIR}:${RL_DIR}/verl:${V5_DIR}:${PYTHONPATH:-}"

python -m verl.trainer.main_ppo \
  --config-path="${RL_DIR}/configs" \
  --config-name=ppo_stepwise \
  "$@"
