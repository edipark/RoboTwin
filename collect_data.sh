#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}
export LD_PRELOAD=/home/yonsei_meat/miniconda3/envs/RoboTwin/lib/libnvidia_malloc_compat.so${LD_PRELOAD:+:$LD_PRELOAD}
# OIDN CUDA 백엔드가 B200(Blackwell)에서 미지원 → CUDA 플러그인을 .disabled로 rename하여 CPU 폴백 강제
# 대상: sapien/oidn_library/libOpenImageDenoise_device_cuda.so.2.0.1.disabled
# (컨테이너 재생성 시 troubleshooting.md 문제 3 조치 참고)

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config
rm -rf data/${task_name}/${task_config}/.cache
