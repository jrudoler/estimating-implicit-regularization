#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=kernel_rbf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=<repo-root>
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(git -C "${WORKDIR}" rev-parse --show-toplevel 2>/dev/null || "${WORKDIR}")"

cd "${PROJECT_ROOT}"

LOG_DIR="${PROJECT_ROOT}/logs"
SLURM_LOG_DIR="${LOG_DIR}/slurm"

mkdir -p "${SLURM_LOG_DIR}" "${LOG_DIR}/wandb"

export WANDB_DIR="${LOG_DIR}/wandb"
export PYTHONUNBUFFERED=1

echo "$(date --iso-8601=seconds) | Starting kernel regression job $SLURM_JOB_ID on $(hostname)"

poetry run python experiments/kernel-regression-RBF.py

echo "$(date --iso-8601=seconds) | Finished kernel regression job $SLURM_JOB_ID"
