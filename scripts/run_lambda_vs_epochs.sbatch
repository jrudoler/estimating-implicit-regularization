#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=lambda_vs_epochs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log

PYTHONPATH=src uv run python analysis/plot_lambda_vs_epochs/run.py \
    --out results/figures/lambda_vs_epochs.pdf \
    --save-data data/generated/lambda_vs_epochs/results.pt
