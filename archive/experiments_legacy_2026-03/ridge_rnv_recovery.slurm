#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --job-name=ridge_rnv_recovery
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log

set -euo pipefail

mkdir -p logs/slurm

echo "Starting Ridge + RowNormVariance recovery test"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

cd /home/jrudoler/inductive-bias
uv run python experiments/ridge_rnv_recovery_test.py
