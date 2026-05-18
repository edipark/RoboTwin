#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G
# export LD_PRELOAD=/home/meat124/anaconda3/envs/RoboTwin/lib/libnvidia_malloc_compat.so${LD_PRELOAD:+:$LD_PRELOAD}

# # Force NVIDIA Vulkan ICD on cluster nodes where autodetection is flaky.
# if [ -f /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json ]; then
#     export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
# fi
# export __GLX_VENDOR_LIBRARY_NAME=nvidia
# export LD_LIBRARY_PATH=/usr/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

policy_name=pi05
task_name=${1}
task_config=${2}
train_config_name=${3}
model_name=${4}
seed=${5}
gpu_id=${6}
expert_check=${7:-1}
checkpoint_id=${8:-1000}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# resolve script location so eval.sh works from any working directory
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_SCRIPT_DIR}/.venv/bin/activate"

# make imageio-ffmpeg binary available as 'ffmpeg'
_FFMPEG_DIR="$(python -c "import imageio_ffmpeg, os; print(os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()))" 2>/dev/null)"
if [ -n "${_FFMPEG_DIR}" ]; then
    # imageio-ffmpeg binary is named ffmpeg-linux64-*, create a symlink named 'ffmpeg'
    _FFMPEG_BIN="$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null)"
    _VENV_BIN="${_SCRIPT_DIR}/.venv/bin"
    if [ -n "${_FFMPEG_BIN}" ] && [ ! -f "${_VENV_BIN}/ffmpeg" ]; then
        ln -s "${_FFMPEG_BIN}" "${_VENV_BIN}/ffmpeg"
    fi
fi
cd "${_SCRIPT_DIR}/../.." # move to RoboTwin root

# Resolve actual checkpoint layout:
# Some runs save to checkpoints/{config}/{model_name}/{ckpt}/  (flat)
# Others save to  checkpoints/{config}/{config}/{model_name}/{ckpt}/  (nested)
_flat_path="policy/${policy_name}/checkpoints/${train_config_name}/${model_name}/${checkpoint_id}"
_nested_path="policy/${policy_name}/checkpoints/${train_config_name}/${train_config_name}/${model_name}/${checkpoint_id}"
if [[ ! -d "${_flat_path}" && ! -d "${_nested_path}" ]]; then
    echo "[error] checkpoint not found for checkpoint_id=${checkpoint_id}" >&2
    echo "        tried: ${_flat_path}" >&2
    echo "        tried: ${_nested_path}" >&2
    _flat_base="policy/${policy_name}/checkpoints/${train_config_name}/${model_name}"
    _nested_base="policy/${policy_name}/checkpoints/${train_config_name}/${train_config_name}/${model_name}"
    if [[ -d "${_flat_base}" ]]; then
        echo "[hint] available steps under ${_flat_base}:" >&2
        find "${_flat_base}" -maxdepth 1 -mindepth 1 -type d -printf '  %f\n' | sort -n >&2 || true
    fi
    if [[ -d "${_nested_base}" ]]; then
        echo "[hint] available steps under ${_nested_base}:" >&2
        find "${_nested_base}" -maxdepth 1 -mindepth 1 -type d -printf '  %f\n' | sort -n >&2 || true
    fi
    exit 2
elif [[ -d "${_nested_path}" && ! -d "${_flat_path}" ]]; then
    model_name="${train_config_name}/${model_name}"
fi

# Compatibility fix for checkpoints that store stats at:
#   assets/local/<repo_id>/norm_stats.json
# while pi_model currently resolves asset_id as "local" and expects:
#   assets/local/norm_stats.json
_ckpt_path="policy/${policy_name}/checkpoints/${train_config_name}/${model_name}/${checkpoint_id}"
_local_stats="${_ckpt_path}/assets/local/norm_stats.json"
if [[ ! -f "${_local_stats}" ]]; then
    _nested_stats="$(find "${_ckpt_path}/assets/local" -mindepth 2 -maxdepth 2 -type f -name norm_stats.json | head -n 1)"
    if [[ -n "${_nested_stats}" ]]; then
        ln -sfn "${_nested_stats}" "${_local_stats}"
        echo "[info] linked norm stats: ${_local_stats} -> ${_nested_stats}"
    fi
fi

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --train_config_name ${train_config_name} \
    --model_name ${model_name} \
    --ckpt_setting ${model_name} \
    --seed ${seed} \
    --expert_check ${expert_check} \
    --checkpoint_id ${checkpoint_id} \
    --policy_name ${policy_name} 