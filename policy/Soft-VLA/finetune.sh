#!/bin/bash
# Fine-tune Soft-VLA on a RoboTwin task (Phase 2: soft-prompt + action head).
#
# Usage (from RoboTwin root):
#   bash policy/Soft-VLA/finetune.sh \
#       <task_name> <task_config> <exp_name> \
#       <src_ckpt_dir> <src_ckpt_step> \
#       <expert_data_num> <gpu_id> \
#       [extra args passed through to finetune.py]
#
# Examples:
#   # Basic (50 episodes, start from step 30000 checkpoint)
#   bash policy/Soft-VLA/finetune.sh \
#       beat_block_hammer demo_clean my_finetune \
#       policy/Soft-VLA/checkpoints/softvla_robotwin/base_model 30000 \
#       50 0
#
#   # With expert Gemma unfreezing at step 5000
#   bash policy/Soft-VLA/finetune.sh \
#       beat_block_hammer demo_clean robotwin_finetune \
#       checkpoints/dtw_fullft_v4_re_phase2 5000 \
#       50 0 --unfreeze_expert_step 5000
#
#   # With W&B logging enabled
#   bash policy/Soft-VLA/finetune.sh \
#       beat_block_hammer demo_clean my_finetune \
#       policy/Soft-VLA/checkpoints/softvla_robotwin/base_model 30000 \
#       50 0 --wandb_enabled --wandb_project my_project --wandb_entity my_team
#
#   # Resume an existing W&B run (e.g. after preemption)
#   bash policy/Soft-VLA/finetune.sh \
#       beat_block_hammer demo_clean my_finetune \
#       policy/Soft-VLA/checkpoints/softvla_robotwin/base_model 30000 \
#       50 0 --resume --wandb_enabled --wandb_run_id <wandb_run_id>
#
# Checkpoint layout after training (compatible with eval.sh):
#   policy/Soft-VLA/checkpoints/softvla_robotwin_ft/<exp_name>/<step>/
#       model.safetensors
#       optimizer.pt
#       metadata.pt
#       assets/robotwin/norm_stats.json
#
# Evaluation after fine-tuning:
#   bash policy/Soft-VLA/eval.sh \
#       <task_name> <task_config> \
#       softvla_robotwin_ft <exp_name> \
#       <softvla_step> <seed> <gpu_id>

task_name=${1}
task_config=${2}
exp_name=${3}
src_ckpt_dir=${4}
src_ckpt_step=${5}
expert_data_num=${6}
gpu_id=${7}
shift 7  # pass remaining args to finetune.py

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export SOFTVLA_PYTHON="/lustre/meat124/Soft-VLA/.venv/bin/python"
PYTHON="${SOFTVLA_PYTHON}"
echo -e "\033[36mPython: ${PYTHON}\033[0m"
echo -e "\033[36mPython version: $(${PYTHON} -V 2>&1)\033[0m"

# Keep training interpreter deterministic even inside tmux/conda shells.
unset VIRTUAL_ENV
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
hash -r

"${PYTHON}" scripts/finetune.py \
    --task_name "$task_name" \
    --task_config "$task_config" \
    --exp_name "$exp_name" \
    --src_ckpt_dir "$src_ckpt_dir" \
    --src_ckpt_step "$src_ckpt_step" \
    --expert_data_num "$expert_data_num" \
    "$@"
