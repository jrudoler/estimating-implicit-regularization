#!/bin/bash
#SBATCH --job-name=elasticnet_sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=0-179
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Check for sweep ID
SWEEP_PATH=$1
if [ -z "$SWEEP_PATH" ]; then
    echo "Usage: $0 <sweep_id_path>"
    exit 1
fi

# Run the W&B agent using poetry environment
poetry run wandb agent --count 1 $SWEEP_PATH
