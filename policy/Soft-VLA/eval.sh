#!/bin/bash
# Evaluate Soft-VLA on a RoboTwin task.
#
# Usage (from RoboTwin root):
#   bash policy/Soft-VLA/eval.sh \
#       <task_name> <task_config> \
#       <train_config_name> <model_name> \
#       <softvla_step> <seed> <gpu_id>
#
# Example:
#   bash policy/Soft-VLA/eval.sh \
#       block_hammer_beat   default   \
#       softvla_robotwin    my_exp    \
#       50   0   0

policy_name=Soft-VLA
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
softvla_step=${5}
seed=${6}
gpu_id=${7}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../.. # move to RoboTwin root

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --softvla_step ${softvla_step} \
    --seed ${seed} \
    --policy_name ${policy_name}
