#!/bin/bash
# Evaluate Soft-VLA on a RoboTwin task.
#
# Usage (from RoboTwin root):
#   bash policy/Soft-VLA/eval.sh \
#       <task_name> <task_config> \
#       <train_config_name> <model_name> \
#       <checkpoint_id> <softvla_step> <seed> <gpu_id> [num_denoise_steps]
#
#   checkpoint_id    : Checkpoint step number to load (sub-dir under model_name/).
#                      E.g. 5000 loads <model_name>/5000/model.safetensors.
#   softvla_step     : Number of actions to execute per inference call
#                      (action chunk execution length, like pi0_step in PI0).
#                      E.g. 16 executes the first 16 actions from the predicted chunk.
#   num_denoise_steps: (optional, default=10) Flow-matching ODE integration steps for
#                      action generation. More steps = higher quality but slower inference.
#                      This is SEPARATE from softvla_step.
#
# Example:
#   bash policy/Soft-VLA/eval.sh \
#       beat_block_hammer   demo_clean   \
#       softvla_robotwin_ft my_exp    \
#       5000   16   42   0
#
#   # With custom denoising steps (20 ODE steps, execute 8 actions per chunk):
#   bash policy/Soft-VLA/eval.sh \
#       beat_block_hammer   demo_clean   \
#       softvla_robotwin_ft my_exp    \
#       5000   8   0   0   20

policy_name=Soft-VLA
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
checkpoint_id=${5}
softvla_step=${6}
seed=${7}
gpu_id=${8}
num_denoise_steps=${9:-10}   # optional, default 10

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# Force RoboTwin simulator environment Python (must include sapien).
ROBOTWIN_PYTHON="/home/yonsei_meat/miniconda3/envs/RoboTwin/bin/python"
if [ ! -x "${ROBOTWIN_PYTHON}" ]; then
    echo "ERROR: RoboTwin python not found at ${ROBOTWIN_PYTHON}" >&2
    exit 1
fi

# cd ../.. # move to RoboTwin root

PYTHONWARNINGS=ignore::UserWarning \
"${ROBOTWIN_PYTHON}" script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --checkpoint_id ${checkpoint_id} \
    --softvla_step ${softvla_step} \
    --num_denoise_steps ${num_denoise_steps} \
    --seed ${seed} \
    --policy_name ${policy_name}
