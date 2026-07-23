#!/usr/bin/env bash
# Submit the controlled LLM ridge-estimation extension grid on standby H200 GPUs.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mode="${1:-${MODE:-smoke}}"
case "$mode" in
  smoke)
    default_token_budget=1048576
    suffix="wiki0001_1m_smoke"
    time_limit="${TIME_LIMIT:-04:00:00}"
    ;;
  full)
    default_token_budget=50000000
    suffix="wiki0001_50m"
    time_limit="${TIME_LIMIT:-20:00:00}"
    ;;
  *)
    echo "Usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac

mkdir -p logs/slurm data/generated/olmo_ridge_estimation/model_grid

export HF_HOME="${HF_HOME:-/shared_data0/jrudoler/.cache/huggingface}"
gradient_root="${GRADIENT_ROOT:-/shared_data0/jrudoler/inductive-bias/olmo_ridge_estimation/model_grid/gradients}"
mkdir -p "$gradient_root"

token_budget="${TOKEN_BUDGET:-$default_token_budget}"
sequence_length="${SEQUENCE_LENGTH:-2048}"
batch_size="${BATCH_SIZE:-1}"
shard_max_params="${GRADIENT_SHARD_MAX_PARAMS:-64000000}"

# Fields: slug|model_id|revision|torch_dtype|extra_args
models=(
  "gemma3_270m|google/gemma-3-270m|main|float32|"
  "gemma3_1b_pt|google/gemma-3-1b-pt|main|float32|"
  "qwen25_3b|Qwen/Qwen2.5-3B|main|bfloat16|"
  "qwen25_7b|Qwen/Qwen2.5-7B|main|bfloat16|--gradient-checkpointing"
)

for spec in "${models[@]}"; do
  IFS="|" read -r slug model_id revision torch_dtype extra_args <<< "$spec"
  run_id="${slug}_${suffix}"
  output="data/generated/olmo_ridge_estimation/model_grid/${run_id}.json"
  stats_output="data/generated/olmo_ridge_estimation/model_grid/${run_id}.pt"
  gradient_dir="${gradient_root}/${run_id}"
  job_name="ridge_${slug}_${mode}"

  if [[ "${FORCE:-0}" != "1" && ( -e "$output" || -e "$stats_output" || -e "$gradient_dir/manifest.json" ) ]]; then
    echo "Skipping existing run artifacts for ${run_id}; set FORCE=1 to resubmit." >&2
    continue
  fi

  sbatch \
    --partition=standby \
    --job-name="$job_name" \
    --gres=gpu:h200:1 \
    --cpus-per-task=8 \
    --mem=192G \
    --time="$time_limit" \
    --chdir="$repo_root" \
    --output="logs/slurm/%x-%j.log" \
    --error="logs/slurm/%x-%j.log" \
    --wrap="export HF_HOME=\"$HF_HOME\"; PYTHONPATH=src uv run python analysis/olmo_ridge_estimation/run.py --model-id '$model_id' --revision '$revision' --token-budget '$token_budget' --sequence-length '$sequence_length' --batch-size '$batch_size' --torch-dtype '$torch_dtype' --output '$output' --stats-output '$stats_output' --save-gradients --gradient-output-dir '$gradient_dir' --gradient-save-dtype '$torch_dtype' --gradient-shard-max-params '$shard_max_params' $extra_args"
done
