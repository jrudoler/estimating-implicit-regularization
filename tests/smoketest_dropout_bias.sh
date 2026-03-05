#!/bin/bash
# Smoketest script for experiments/dropout_bias_estimation.py
# Runs with minimal settings to verify the script works before cluster deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

echo "Running smoketest for experiments/dropout_bias_estimation.py..."
echo "This will use minimal settings (2 epochs training, 5 epochs bias estimation)"
echo ""

uv run python experiments/dropout_bias_estimation.py \
    --config-path tests/smoketest_config.json \
    --disable-wandb \
    --checkpoint-dir tests/smoketest_checkpoints

echo ""
echo "✅ Smoketest completed successfully!"

