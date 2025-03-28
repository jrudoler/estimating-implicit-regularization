#!/bin/bash
#SBATCH --job-name=train_models
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --array=0-2   # adjust based on number of configurations
#SBATCH --time=01:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --partition=whartonstat
#SBATCH -G 1

# Use yq to extract parameters for the current array task
CONFIG_FILE="model-runs.yaml"
IDX=$SLURM_ARRAY_TASK_ID

RUN_NAME=$(yq e ".[$IDX].run_name" "$CONFIG_FILE")
DROPOUT_RATE=$(yq e ".[$IDX].dropout_rate" "$CONFIG_FILE")
MAX_EPOCHS=$(yq e ".[$IDX].max_epochs" "$CONFIG_FILE")
EARLY_STOPPING=$(yq e ".[$IDX].early_stopping" "$CONFIG_FILE")
PATIENCE=$(yq e ".[$IDX].patience" "$CONFIG_FILE")

# print the run parameters
echo "Running job with the following parameters:"
echo "Run Name: $RUN_NAME"
echo "Dropout Rate: $DROPOUT_RATE"
echo "Max Epochs: $MAX_EPOCHS"
echo "Early Stopping: $EARLY_STOPPING"
echo "Patience: $PATIENCE"

# Call your training script with these parameters
poetry run python train_networks.py \
  --run_name "$RUN_NAME" \
  --dropout_rate "$DROPOUT_RATE" \
  --max_epochs "$MAX_EPOCHS" \
  --early_stopping "$EARLY_STOPPING" \
  --patience "$PATIENCE"

