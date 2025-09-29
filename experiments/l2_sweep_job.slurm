#!/bin/bash
#SBATCH --job-name=l2_sweep
#SBATCH --output=logs/slurm/l2_sweep_%A_%a.out
#SBATCH --error=logs/slurm/l2_sweep_%A_%a.err
#SBATCH --array=0-269
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=whartonstat
#SBATCH --gres=gpu:1
#SBATCH --chdir=/home/jrudoler/inductive-bias

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(git -C "${WORKDIR}" rev-parse --show-toplevel 2>/dev/null || "${WORKDIR}")"

cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs"
SLURM_LOG_DIR="${LOG_DIR}/slurm"

mkdir -p "${SLURM_LOG_DIR}" "${LOG_DIR}/wandb"

SWEEP_PATH="${1:-}" 
if [[ -z "${SWEEP_PATH}" ]]; then
    echo "Usage: $0 <sweep_id_path>" >&2
    exit 1
fi

export WANDB_DIR="${LOG_DIR}/wandb"
export PYTHONUNBUFFERED=1

echo "$(date --iso-8601=seconds) | Starting sweep job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0}"

poetry run wandb agent --count 1 "${SWEEP_PATH}"

echo "$(date --iso-8601=seconds) | Finished sweep job ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID:-0}"
