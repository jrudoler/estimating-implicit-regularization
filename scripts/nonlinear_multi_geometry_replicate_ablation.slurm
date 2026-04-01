#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --job-name=geom_repl_ablate
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --chdir=/home/jrudoler/inductive-bias
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log

set -euo pipefail

mkdir -p logs/slurm

CASE_NAME="${CASE_NAME:-all}"
OUTPUT_SUFFIX="${OUTPUT_SUFFIX:-gpu_scale03}"
REPLICATE_COUNTS="${REPLICATE_COUNTS:-1,10,50,100,1000}"
SEED="${SEED:-42}"

srun uv run python experiments/nonlinear_multi_geometry_replicate_ablation.py \
  --n-samples 256 \
  --input-dim 12 \
  --depth 2 \
  --width 32 \
  --max-epochs 350 \
  --patience 60 \
  --seed "${SEED}" \
  --cases "${CASE_NAME}" \
  --replicate-counts "${REPLICATE_COUNTS}" \
  --estimation-max-epochs 2500 \
  --estimation-patience 250 \
  --target-gradient-scale 0.3 \
  --output-dir "logs/phase2/nonlinear_multi_geometry_replicate_ablation_${OUTPUT_SUFFIX}" \
  --figure-dir "figures/nonlinear_multi_geometry_replicate_ablation_${OUTPUT_SUFFIX}"
