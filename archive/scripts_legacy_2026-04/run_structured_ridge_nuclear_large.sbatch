#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --job-name=structured_rn_large
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/jrudoler/inductive-bias
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log

set -euo pipefail

mkdir -p logs/slurm logs/phase2

uv run python experiments/run_structured_ridge_nuclear_matrix.py \
  --seeds 42,123,456,789 \
  --n-samples 4096 \
  --input-dim 16 \
  --width 16 \
  --batch-size 256 \
  --lr 0.001 \
  --max-epochs 120 \
  --patience 15 \
  --noise-std 0.0 \
  --bias-lr 0.05 \
  --bias-max-epochs 300 \
  --bias-patience 40 \
  --output-csv logs/phase2/structured_ridge_nuclear_matrix_large.csv \
  --output-summary-md logs/phase2/structured_ridge_nuclear_summary_large.md
