#!/bin/bash
#SBATCH --partition=standby
#SBATCH --job-name=olmo_ridge_1b_wiki50m
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/jrudoler/inductive-bias
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log

set -euo pipefail

export HF_HOME="${HF_HOME:-/shared_data0/jrudoler/.cache/huggingface}"

PYTHONPATH=src uv run python analysis/olmo_ridge_estimation/run.py \
    --output data/generated/olmo_ridge_estimation/olmo2_1b_stage1_wiki0001_50m.json \
    --stats-output data/generated/olmo_ridge_estimation/olmo2_1b_stage1_wiki0001_50m.pt

