#!/usr/bin/env bash
# A1(c): in/out-of-family detection + with/without-M isolation.
#
# Three conditions sharing one baseline, each fit by two regularizer families:
#   (dropout=0.0, bn=off)  <- shared no-augmentation baseline
#   (dropout=0.3, bn=off)  <- M = dropout      (expected IN-family for ridge)
#   (dropout=0.0, bn=on)   <- M = BatchNorm    (expected OUT-of-family for ridge,
#                                               in-family for a scale-invariant R)
# Families: ridge (L2) vs stable_rank (scale-invariant).
# 3 conditions x 2 families x 20 seeds = 120 runs, submitted as one SLURM array.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_root="${RUN_ROOT:-data/generated/isolation_detector}"
cfg_dir="$run_root/configs"
out_dir="$run_root/results"
mkdir -p "$cfg_dir" "$out_dir" logs/slurm

seeds=(123 42 666 314 17 111 325 643 432 51 7 19 23 31 37 41 53 59 61 67)
manifest="$run_root/manifest.tsv"
: > "$manifest"

idx=0
for cond in "dropout:0.3:false" "batchnorm:0.0:true" "baseline:0.0:false"; do
  IFS=":" read -r cond_name cond_dropout cond_bn <<< "$cond"
  for family in ridge stable_rank; do
    for seed in "${seeds[@]}"; do
      tag="${cond_name}_${family}_s${seed}"
      cfg="$cfg_dir/${tag}.json"
      cat > "$cfg" <<JSON
{"depth":3,"width":256,"dropout":${cond_dropout},"batchnorm":${cond_bn},
 "l2_penalty":0.0,"lr":0.05,"momentum":0.9,"batch_size":2048,
 "num_data_workers":2,"max_epochs":500,"patience":50,
 "bias_lr":0.05,"bias_max_epochs":2000,"bias_patience":100,
 "seed":${seed},"bias_types":"${family}"}
JSON
      printf '%d\t%s\t%s\t%s\t%s\n' "$idx" "$tag" "$cfg" "$out_dir/${tag}.json" "$cond_name" >> "$manifest"
      idx=$((idx + 1))
    done
  done
done

total=$idx
echo "Prepared $total runs; manifest=$manifest"

sbatch \
  --partition=whartonstat \
  --job-name=isodet \
  --gres=gpu:1 \
  --cpus-per-task=3 \
  --mem=16G \
  --time=2:00:00 \
  --array="0-$((total - 1))%20" \
  --chdir="$repo_root" \
  --output="logs/slurm/%x-%A_%a.log" \
  --error="logs/slurm/%x-%A_%a.log" \
  --wrap="line=\$(awk -v i=\$SLURM_ARRAY_TASK_ID '\$1==i' '$manifest'); cfg=\$(echo \"\$line\" | cut -f3); out=\$(echo \"\$line\" | cut -f4); tag=\$(echo \"\$line\" | cut -f2); PYTHONPATH=src uv run python analysis/dropout_bias_estimation/run.py --disable-wandb --config-path \"\$cfg\" --checkpoint-dir '$run_root/ckpt'/\"\$tag\" --output \"\$out\""
