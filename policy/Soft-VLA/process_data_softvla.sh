#!/bin/bash
# Convert RoboTwin raw HDF5 data into the processed format for Soft-VLA fine-tuning.
#
# Usage (from RoboTwin root OR from policy/Soft-VLA/):
#   bash policy/Soft-VLA/process_data_softvla.sh <task_name> <task_config> <expert_data_num>
#
# Example:
#   bash policy/Soft-VLA/process_data_softvla.sh beat_block_hammer demo_clean 50
#
# Output: policy/Soft-VLA/processed_data/<task_name>-<task_config>-<expert_data_num>/

task_name=${1}
task_config=${2}
expert_data_num=${3}

if [ -z "$task_name" ] || [ -z "$task_config" ] || [ -z "$expert_data_num" ]; then
    echo "Usage: bash process_data_softvla.sh <task_name> <task_config> <expert_data_num>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Resolve Python interpreter (same logic as finetune.sh)
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
if [ -n "${SOFTVLA_PYTHON:-}" ]; then
    PYTHON="${SOFTVLA_PYTHON}"
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
else
    echo "ERROR: No Soft-VLA python found." >&2
    echo "  Expected: ${REPO_ROOT}/.venv/bin/python (run \`uv sync\` at repo root)" >&2
    echo "  Or set SOFTVLA_PYTHON to a python with the Soft-VLA stack installed." >&2
    exit 1
fi
echo -e "\033[36mPython: ${PYTHON}\033[0m"

${PYTHON} scripts/process_data.py "$task_name" "$task_config" "$expert_data_num"
