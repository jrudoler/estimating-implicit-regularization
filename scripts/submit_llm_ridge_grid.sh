#!/usr/bin/env bash
# Submit independent LLM ridge-estimation jobs on standby H200 GPUs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p logs/slurm data/generated/olmo_ridge_estimation/model_grid

export HF_HOME="${HF_HOME:-/shared_data0/jrudoler/.cache/huggingface}"

token_budget="${TOKEN_BUDGET:-50000000}"
sequence_length="${SEQUENCE_LENGTH:-2048}"
batch_size="${BATCH_SIZE:-1}"
shard_max_params="${GRADIENT_SHARD_MAX_PARAMS:-64000000}"

# Fields: slug|model_id|revision|torch_dtype|extra_args
# Ai2 does not currently expose a public OLMo 2 4B language-model checkpoint;
# qwen3_4b_base is included as the 4B-scale open text-model comparison.
models=(
  "olmo2_1b_stage1|allenai/OLMo-2-0425-1B|stage1-step1907359-tokens4001B|float32|"
  "olmo2_7b_1124|allenai/OLMo-2-1124-7B|main|bfloat16|--gradient-checkpointing"
  "qwen3_4b_base|Qwen/Qwen3-4B-Base|main|bfloat16|"
  "qwen25_05b|Qwen/Qwen2.5-0.5B|main|float32|"
  "qwen25_15b|Qwen/Qwen2.5-1.5B|main|float32|"
)

if [[ "${INCLUDE_GATED_GEMMA:-0}" == "1" ]]; then
  models+=(
    "gemma3_1b_pt|google/gemma-3-1b-pt|main|bfloat16|"
  )
fi

for spec in "${models[@]}"; do
  IFS="|" read -r slug model_id revision torch_dtype extra_args <<< "$spec"
  output="data/generated/olmo_ridge_estimation/model_grid/${slug}_wiki0001_50m.json"
  stats_output="data/generated/olmo_ridge_estimation/model_grid/${slug}_wiki0001_50m.pt"
  gradient_dir="data/generated/olmo_ridge_estimation/model_grid/gradients/${slug}_wiki0001_50m"
  job_name="ridge_${slug}"

  sbatch \
    --partition=standby \
    --job-name="$job_name" \
    --gres=gpu:h200:1 \
    --cpus-per-task=8 \
    --mem=192G \
    --time=16:00:00 \
    --chdir="$repo_root" \
    --output="logs/slurm/%x-%j.log" \
    --error="logs/slurm/%x-%j.log" \
    --wrap="export HF_HOME=\"$HF_HOME\"; PYTHONPATH=src uv run python analysis/olmo_ridge_estimation/run.py --model-id '$model_id' --revision '$revision' --token-budget '$token_budget' --sequence-length '$sequence_length' --batch-size '$batch_size' --torch-dtype '$torch_dtype' --output '$output' --stats-output '$stats_output' --save-gradients --gradient-output-dir '$gradient_dir' --gradient-save-dtype '$torch_dtype' --gradient-shard-max-params '$shard_max_params' $extra_args"
done

