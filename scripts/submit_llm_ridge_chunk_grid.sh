#!/usr/bin/env bash
# Submit attention/MLP-only LLM ridge jobs with exact per-chunk fit diagnostics.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

mkdir -p logs/slurm data/generated/olmo_ridge_estimation/chunk_grid

export HF_HOME="${HF_HOME:-/shared_data0/jrudoler/.cache/huggingface}"
token_budget="${TOKEN_BUDGET:-0}"
sequence_length="${SEQUENCE_LENGTH:-2048}"
batch_size="${BATCH_SIZE:-1}"
chunk_token_budget="${CHUNK_TOKEN_BUDGET:-10000000}"
suffix="${SUFFIX:-wikitext103_full_chunk10m_fit_v2}"
time_limit="${TIME_LIMIT:-36:00:00}"
partition="${PARTITION:-standby}"
gres="${GRES:-gpu:h200:1}"
cpus="${CPUS_PER_TASK:-8}"

# OLMo 2 combines release series and is retained only as an explicit diagnostic.
# Pythia-deduped is the controlled replacement scaling ladder.
# Fields: slug|model_id|revision|torch_dtype|extra_args
models=(
  "gemma3_270m|google/gemma-3-270m|main|float32|"
  "gemma3_1b_pt|google/gemma-3-1b-pt|main|float32|"
  "gemma3_4b_pt|google/gemma-3-4b-pt|main|bfloat16|"
  "qwen25_05b|Qwen/Qwen2.5-0.5B|main|float32|"
  "qwen25_15b|Qwen/Qwen2.5-1.5B|main|float32|"
  "qwen25_3b|Qwen/Qwen2.5-3B|main|bfloat16|"
  "qwen25_7b|Qwen/Qwen2.5-7B|main|bfloat16|--gradient-checkpointing"
  "qwen3_06b_base|Qwen/Qwen3-0.6B-Base|main|float32|"
  "qwen3_4b_base|Qwen/Qwen3-4B-Base|main|bfloat16|"
  "pythia_1b_deduped|EleutherAI/pythia-1b-deduped|main|float32|"
  "pythia_28b_deduped|EleutherAI/pythia-2.8b-deduped|main|float32|"
  "pythia_69b_deduped|EleutherAI/pythia-6.9b-deduped|main|float32|--gradient-checkpointing"
)

if [[ "${INCLUDE_LARGE_ENDPOINTS:-0}" == "1" ]]; then
  models+=(
    "gemma3_12b_pt|google/gemma-3-12b-pt|main|bfloat16|--gradient-checkpointing"
    "qwen25_14b|Qwen/Qwen2.5-14B|main|bfloat16|--gradient-checkpointing"
    "qwen3_14b_base|Qwen/Qwen3-14B-Base|main|bfloat16|--gradient-checkpointing"
    "pythia_12b_deduped|EleutherAI/pythia-12b-deduped|main|bfloat16|--gradient-checkpointing"
  )
fi

if [[ "${INCLUDE_OLMO_DIAGNOSTICS:-0}" == "1" ]]; then
  models+=(
    "olmo2_1b_stage1|allenai/OLMo-2-0425-1B|stage1-step1907359-tokens4001B|float32|"
    "olmo2_7b_stage1_final|allenai/OLMo-2-1124-7B|stage1-step928646-tokens3896B|float32|--gradient-checkpointing"
    "olmo2_13b_stage1_final|allenai/OLMo-2-1124-13B|stage1-step596057-tokens5001B|bfloat16|--gradient-checkpointing"
  )
fi

selected_slugs="${MODEL_SLUGS:-}"
if [[ -z "$selected_slugs" ]]; then
  echo "Set MODEL_SLUGS explicitly; broad chunk-fit grids are intentionally disabled." >&2
  exit 2
fi

for spec in "${models[@]}"; do
  IFS="|" read -r slug model_id revision torch_dtype extra_args <<< "$spec"
  if [[ -n "$selected_slugs" && " $selected_slugs " != *" $slug "* ]]; then
    continue
  fi

  run_id="${slug}_${suffix}"
  output="data/generated/olmo_ridge_estimation/chunk_grid/${run_id}.json"
  stats_output="data/generated/olmo_ridge_estimation/chunk_grid/${run_id}.pt"
  job_name="ridgechunk_${slug}"

  if [[ "${FORCE:-0}" != "1" && ( -e "$output" || -e "$stats_output" ) ]]; then
    echo "Skipping existing run artifacts for ${run_id}; set FORCE=1 to resubmit." >&2
    continue
  fi

  sbatch \
    --partition="$partition" \
    --job-name="$job_name" \
    --gres="$gres" \
    --cpus-per-task="$cpus" \
    --mem="${MEM:-256G}" \
    --time="$time_limit" \
    --chdir="$repo_root" \
    --output="logs/slurm/%x-%j.log" \
    --error="logs/slurm/%x-%j.log" \
    --wrap="export HF_HOME=\"$HF_HOME\"; PYTHONPATH=src uv run python analysis/olmo_ridge_estimation/run.py --model-id '$model_id' --revision '$revision' --dataset-source wikitext --wikitext-config wikitext-103-raw-v1 --wikitext-split train --token-budget '$token_budget' --sequence-length '$sequence_length' --batch-size '$batch_size' --torch-dtype '$torch_dtype' --parameter-scope attention_mlp --bootstrap-batches 0 --chunk-token-budget '$chunk_token_budget' --output '$output' --stats-output '$stats_output' $extra_args"
done
