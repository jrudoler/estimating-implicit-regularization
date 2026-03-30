#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --job-name=geom_suite_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/jrudoler/inductive-bias
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log

set -euo pipefail

mkdir -p logs/slurm

uv run python experiments/nonlinear_multi_geometry_suite.py \
  --n-samples 256 \
  --input-dim 12 \
  --depth 2 \
  --width 32 \
  --max-epochs 350 \
  --patience 60 \
  --n-replicates 8 \
  --estimation-max-epochs 2500 \
  --estimation-patience 250 \
  --target-gradient-scale 0.3 \
  --output-dir logs/phase2/nonlinear_multi_geometry_suite_gpu_scale03 \
  --figure-dir figures/nonlinear_multi_geometry_suite_gpu_scale03
