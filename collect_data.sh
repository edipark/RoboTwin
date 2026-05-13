#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}
export LD_PRELOAD=/home/meat124/anaconda3/envs/RoboTwin/lib/libnvidia_malloc_compat.so${LD_PRELOAD:+:$LD_PRELOAD}

# Force NVIDIA Vulkan ICD on cluster nodes where autodetection is flaky.
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]; then
	export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
fi
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export LD_LIBRARY_PATH=/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config
rm -rf data/${task_name}/${task_config}/.cache
