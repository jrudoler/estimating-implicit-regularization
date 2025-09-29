#!/bin/bash
#SBATCH --partition=whartonstat
#SBATCH --job-name=kernel_rbf
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --chdir=/home/jrudoler/inductive-bias
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log

poetry run python experiments/kernel-regression-RBF.py