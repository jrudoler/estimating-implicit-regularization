#!/usr/bin/env bash
# Compare global ridge, per-layer ridge, an activation-weighted quadratic, and
# squared path norm at the same 8 dropout x 10 seed MNIST endpoints.
#
# Each task trains one d3/w256 model and fits every family by exact NNLS.  The
# same task also performs deterministic two-fold gradient cross-fitting.
#
# Environment controls:
#   DRY_RUN=1        prepare files and show the submission without submitting
#   SMOKE=1          run two corner cases with a two-epoch training budget
#   ONLY_MISSING=1   submit only result files that are absent/empty
#   CONCURRENCY=N    maximum simultaneously running array tasks (default 10)
#   RUN_ROOT=path    alternate output root
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_root="${RUN_ROOT:-data/generated/dropout_structured_regularizers}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  run_root="${RUN_ROOT:-data/generated/dropout_structured_regularizers/smoke}"
fi
config_dir="$run_root/configs"
results_dir="$run_root/results"
mkdir -p "$config_dir" "$results_dir" logs/slurm

dropouts=(0.0 0.05 0.1 0.15 0.2 0.3 0.4 0.5)
seeds=(123 42 666 314 17 111 325 643 432 51)
max_epochs="${MAX_EPOCHS:-500}"
patience="${PATIENCE:-50}"
concurrency="${CONCURRENCY:-10}"
job_name="${JOB_NAME:-structreg}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  max_epochs="${MAX_EPOCHS:-2}"
  patience="${PATIENCE:-2}"
  job_name="${JOB_NAME:-structsmoke}"
fi

manifest="$run_root/manifest.tsv"
: > "$manifest"
index=0
for dropout in "${dropouts[@]}"; do
  for seed in "${seeds[@]}"; do
    tag="d3_w256_do${dropout}_s${seed}"
    output="$results_dir/${tag}.json"
    cat > "$config_dir/${tag}.json" <<JSON
{"depth":3,"width":256,"dropout":${dropout},"seed":${seed},
 "batchnorm":false,"l2_lambda":0.0,"lr":0.05,"momentum":0.9,
 "batch_size":2048,"num_workers":2,"max_epochs":${max_epochs},
 "patience":${patience},"split_seed":2027}
JSON
    printf '%d\t%s\t%s\t%s\t%s\n' \
      "$index" "$tag" "$dropout" "$seed" "$output" >> "$manifest"
    index=$((index + 1))
  done
done
total="$index"

tasklist="$run_root/tasklist.txt"
if [[ "${ONLY_MISSING:-0}" == "1" ]]; then
  : > "$tasklist"
  while IFS=$'\t' read -r task_index _tag _dropout _seed output; do
    [[ -s "$output" ]] || echo "$task_index" >> "$tasklist"
  done < "$manifest"
  missing="$(wc -l < "$tasklist")"
  if [[ "$missing" == "0" ]]; then
    echo "All $total outputs already exist."
    exit 0
  fi
  array_spec="$(paste -sd, "$tasklist")%${concurrency}"
elif [[ "${SMOKE:-0}" == "1" ]]; then
  array_spec="0,$((total - 1))%2"
else
  array_spec="0-$((total - 1))%${concurrency}"
fi

wrap="set -euo pipefail; \
line=\$(awk -F'\\t' -v i=\$SLURM_ARRAY_TASK_ID '\$1==i' '$manifest'); \
tag=\$(echo \"\$line\" | cut -f2); dropout=\$(echo \"\$line\" | cut -f3); \
seed=\$(echo \"\$line\" | cut -f4); output=\$(echo \"\$line\" | cut -f5); \
export TMPDIR=\"\${SLURM_TMPDIR:-/tmp}/structreg_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}\"; \
mkdir -p \"\$TMPDIR\"; \
echo \"task=\$SLURM_ARRAY_TASK_ID tag=\$tag dropout=\$dropout seed=\$seed\"; \
PYTHONPATH=src uv run python analysis/dropout_structured_regularizers/run.py \
--dropout \"\$dropout\" --seed \"\$seed\" --depth 3 --width 256 \
--max-epochs '$max_epochs' --patience '$patience' \
--batch-size 2048 --num-workers 2 --split-seed 2027 \
--data-root '$repo_root/data/raw' --output \"\$output\"; \
rm -rf \"\$TMPDIR\""

echo "Prepared $total runs; manifest=$manifest; array=$array_spec"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: not submitting."
  echo "$wrap"
  exit 0
fi

sbatch \
  --partition=whartonstat \
  --job-name="$job_name" \
  --gres=gpu:1 \
  --cpus-per-task=3 \
  --mem=16G \
  --time=2:00:00 \
  --array="$array_spec" \
  --chdir="$repo_root" \
  --output="logs/slurm/%x-%A_%a.log" \
  --error="logs/slurm/%x-%A_%a.log" \
  --wrap="$wrap"
