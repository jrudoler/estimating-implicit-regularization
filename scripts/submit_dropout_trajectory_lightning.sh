#!/usr/bin/env bash
# Faithfully replay one Figure-5 configuration through the original Lightning
# training path, saving complete checkpoints and exact closed-form estimates at
# 0.5T, 0.8T, and T. T is the per-seed endpoint from the completed corrected
# Figure-5 grid.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_root="${RUN_ROOT:-data/generated/dropout_trajectory_lightning}"
grid_root="${GRID_ROOT:-data/generated/closed_form_dropout_grid/results}"
dropout="${DROPOUT:-0.3}"
depth="${DEPTH:-3}"
width="${WIDTH:-256}"
concurrency="${CONCURRENCY:-10}"
seeds=(17 42 51 111 123 314 325 432 643 666)

mkdir -p "$run_root/results" "$run_root/checkpoints" logs/slurm
manifest="$run_root/manifest.tsv"
: > "$manifest"

idx=0
for seed in "${seeds[@]}"; do
    source_json="$grid_root/d${depth}_w${width}_do${dropout}_s${seed}.json"
    if [[ ! -s "$source_json" ]]; then
        echo "Missing endpoint source: $source_json" >&2
        exit 1
    fi
    endpoint="$(
        uv run python -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["stationarity"]["epochs_run"])' \
            "$source_json"
    )"
    tag="d${depth}_w${width}_do${dropout}_s${seed}"
    output="$run_root/results/${tag}.json"
    checkpoint_dir="$run_root/checkpoints/$tag"
    printf '%d\t%s\t%s\t%s\t%s\t%s\n' \
        "$idx" "$tag" "$seed" "$endpoint" "$output" "$checkpoint_dir" >> "$manifest"
    idx=$((idx + 1))
done

tasklist="$run_root/tasklist.txt"
: > "$tasklist"
while IFS=$'\t' read -r task_index tag seed endpoint output checkpoint_dir; do
    if [[ ! -s "$output" ]]; then
        echo "$task_index" >> "$tasklist"
    fi
done < "$manifest"

missing="$(wc -l < "$tasklist")"
echo "Prepared ${#seeds[@]} runs; missing=$missing; manifest=$manifest"
if [[ "$missing" == "0" ]]; then
    echo "Nothing to submit."
    exit 0
fi

array_spec="$(paste -sd, "$tasklist")%${concurrency}"
wrap="set -euo pipefail; \
line=\$(awk -F'\\t' -v i=\$SLURM_ARRAY_TASK_ID '\$1==i' '$manifest'); \
tag=\$(echo \"\$line\" | cut -f2); seed=\$(echo \"\$line\" | cut -f3); \
endpoint=\$(echo \"\$line\" | cut -f4); output=\$(echo \"\$line\" | cut -f5); \
checkpoint_dir=\$(echo \"\$line\" | cut -f6); \
PYTHONPATH=src uv run python analysis/dropout_trajectory_lightning/run.py \
--dropout '$dropout' --seed \"\$seed\" --depth '$depth' --width '$width' \
--endpoint-epochs \"\$endpoint\" --data-root '$repo_root/data/raw' \
--checkpoint-dir \"\$checkpoint_dir\" --output \"\$output\""

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: array=$array_spec"
    echo "$wrap"
    exit 0
fi

sbatch \
    --partition=whartonstat \
    --job-name=droptraj \
    --gres=gpu:1 \
    --cpus-per-task=3 \
    --mem=16G \
    --time=02:00:00 \
    --array="$array_spec" \
    --chdir="$repo_root" \
    --output="logs/slurm/%x-%A_%a.log" \
    --error="logs/slurm/%x-%A_%a.log" \
    --wrap="$wrap"
