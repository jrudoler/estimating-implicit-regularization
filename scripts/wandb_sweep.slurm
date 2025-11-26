#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --job-name=wandb_sweep
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log

set -euo pipefail

mkdir -p logs/slurm

SWEEP_ID="${1:-${WANDB_SWEEP_ID:-}}"
if [[ -z "${SWEEP_ID}" ]]; then
  echo "Error: provide a sweep identifier as an argument or set WANDB_SWEEP_ID." >&2
  exit 1
fi

uv run wandb agent --count 1 "${SWEEP_ID}"

