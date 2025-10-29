#!/bin/bash
#SBATCH --job-name=estimate_implicit_reg
#SBATCH --output=logs/slurm/%x-%A_%a.log
#SBATCH --error=logs/slurm/%x-%A_%a.log
#SBATCH --array=0-20%6
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --partition=whartonstat
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/jrudoler/inductive-bias

set -euo pipefail

# --chdir above already places us at the repo root
LOG_DIR="${PWD}/logs"
mkdir -p "${LOG_DIR}/slurm" "${LOG_DIR}/wandb"

SWEEP_ID="${1:-${SWEEP_ID:-}}"
if [[ -z "${SWEEP_ID}" ]]; then
    echo "Usage: $0 <sweep_id_path_or_name>" >&2
    exit 1
fi

export WANDB_DIR="${LOG_DIR}/wandb"
export PYTHONUNBUFFERED=1

JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

echo "$(date --iso-8601=seconds) | Starting sweep job ${JOB_ID} task ${TASK_ID}"

poetry run wandb agent --count 1 "${SWEEP_ID}"

echo "$(date --iso-8601=seconds) | Finished sweep job ${JOB_ID} task ${TASK_ID}"
