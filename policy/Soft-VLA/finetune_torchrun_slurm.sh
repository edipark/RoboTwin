#!/bin/bash
#SBATCH --job-name=softvla_robotwin_ft
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/lustre/meat124/Soft-VLA/logs/softvla_robotwin_ft_%j.log
#SBATCH --error=/lustre/meat124/Soft-VLA/logs/softvla_robotwin_ft_%j.log
#
# ── Usage ──────────────────────────────────────────────────────────────────────
# GPU/파티션은 sbatch 실행 시 --partition / --gres 로 지정합니다.
#
#   sbatch -p suma_a6000 --gres=gpu:4 \
#       policy/Soft-VLA/finetune_torchrun_slurm.sh
#
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

mkdir -p /lustre/meat124/Soft-VLA/logs

echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPUs assigned: $CUDA_VISIBLE_DEVICES"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo ""

# ── Repo root (이 스크립트는 RoboTwin root에서 sbatch) ────────────────────────
REPO_ROOT="/lustre/meat124/Soft-VLA"
PYTHON="${REPO_ROOT}/.venv/bin/python"

echo "Python: ${PYTHON}"
echo "Python version: $(${PYTHON} -V 2>&1)"

# ── Environment ───────────────────────────────────────────────────────────────
export PYTHONPATH="${REPO_ROOT}/third_party/RoboTwin/policy/Soft-VLA/src:${REPO_ROOT}/src"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export USE_TF=0
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# ── Training ──────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"

"${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=4 \
    --master_port 29502 \
    third_party/RoboTwin/policy/Soft-VLA/scripts/finetune.py \
    --task_name beat_block_hammer \
    --task_config demo_clean_right \
    --domain_id 8 \
    --stage1_steps 500 \
    --stage2_steps 2000 \
    --stage2_lr 1e-4 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --exp_name softvla_robotwin_finetune \
    --phase2_checkpoint ./checkpoints/p2_xvla_action_decoder_v1/10000 \
    --wandb_enabled

echo ""
echo "End time: $(date)"
