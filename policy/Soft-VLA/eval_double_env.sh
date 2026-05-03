#!/bin/bash
# Evaluate Soft-VLA on a RoboTwin task using a dual-environment setup:
#   * Policy server  – Soft-VLA venv (.venv, has flax / torch / softvla)
#   * Simulator client – RoboTwin conda env (has sapien for rendering)
#
# Usage (from RoboTwin root):
#   bash policy/Soft-VLA/eval_double_env.sh \
#       <task_name> <task_config> \
#       <train_config_name> <model_name> \
#       <checkpoint_id> <softvla_step> <seed> <gpu_id> [num_denoise_steps]
#
# The Python interpreters can be overridden via environment variables:
#   ROBOTWIN_PYTHON  – default: ~/anaconda3/envs/RoboTwin/bin/python
#   SOFTVLA_PYTHON   – default: <repo_root>/.venv/bin/python
#
# Example:
#   bash policy/Soft-VLA/eval_double_env.sh \
#       beat_block_hammer demo_clean \
#       softvla_robotwin_ft robotwin_finetune \
#       10000 16 42 0

policy_name=Soft-VLA
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
checkpoint_id=${5}
softvla_step=${6}
seed=${7}
gpu_id=${8}
num_denoise_steps=${9:-10}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# ── interpreter paths ──────────────────────────────────────────────────────────
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_SCRIPT_DIR}/../../../.." && pwd)"

ROBOTWIN_PYTHON_DEFAULT="${HOME}/anaconda3/envs/RoboTwin/bin/python"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-${ROBOTWIN_PYTHON_DEFAULT}}"

SOFTVLA_PYTHON_DEFAULT="${_REPO_ROOT}/.venv/bin/python"
SOFTVLA_PYTHON="${SOFTVLA_PYTHON:-${SOFTVLA_PYTHON_DEFAULT}}"

if [ ! -x "${ROBOTWIN_PYTHON}" ]; then
    echo "ERROR: RoboTwin python not found at ${ROBOTWIN_PYTHON}" >&2; exit 1
fi
if [ ! -x "${SOFTVLA_PYTHON}" ]; then
    echo "ERROR: Soft-VLA venv python not found at ${SOFTVLA_PYTHON}" >&2; exit 1
fi

echo -e "\033[32m[server] Using Soft-VLA venv: ${SOFTVLA_PYTHON}\033[0m"
echo -e "\033[34m[client] Using RoboTwin conda: ${ROBOTWIN_PYTHON}\033[0m"

# ── find a free port ───────────────────────────────────────────────────────────
FREE_PORT=$(${ROBOTWIN_PYTHON} - << 'EOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
EOF
)
echo -e "\033[33mUsing socket port: ${FREE_PORT}\033[0m"

# ── launch policy server (venv Python, background) ────────────────────────────
SERVER_LOG="/tmp/softvla_server_${FREE_PORT}.log"
echo -e "\033[32m[server] Launching policy_model_server (log: ${SERVER_LOG})...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
"${SOFTVLA_PYTHON}" script/policy_model_server.py \
    --port ${FREE_PORT} \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --policy_name ${policy_name} \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --checkpoint_id ${checkpoint_id} \
    --softvla_step ${softvla_step} \
    --num_denoise_steps ${num_denoise_steps} \
    --seed ${seed} >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

# Print server log in background so errors are visible in the terminal
tail -f "${SERVER_LOG}" &
TAIL_PID=$!

# Kill server and tail when this script exits (success or error)
trap "echo -e '\033[31m[cleanup] Killing server (PID=${SERVER_PID})\033[0m'; kill ${SERVER_PID} 2>/dev/null; kill ${TAIL_PID} 2>/dev/null" EXIT

# ── launch simulator client (RoboTwin conda Python, foreground) ───────────────
echo -e "\033[34m[client] Starting eval_policy_client...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
"${ROBOTWIN_PYTHON}" script/eval_policy_client.py \
    --port ${FREE_PORT} \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --policy_name ${policy_name} \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${model_name} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --checkpoint_id ${checkpoint_id} \
    --softvla_step ${softvla_step} \
    --num_denoise_steps ${num_denoise_steps} \
    --seed ${seed}

echo -e "\033[33m[main] eval_policy_client has finished; server will be terminated.\033[0m"
