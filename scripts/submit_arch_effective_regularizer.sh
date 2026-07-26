#!/usr/bin/env bash
# A3: effective-regularizer profile of an MLP vs a CNN on the same task.
#
# Reviewers asked why only MLPs were studied.  The estimator is architecture
# agnostic, so we simply swap the predictive module.  An architecture class is
# NOT an ablatable module (there is no "same model but not a CNN" reference), so
# this is a DESCRIPTIVE measurement using the estimator as an instrument, not a
# validation of the method.  The reportable object per architecture is the
# per-family (fitted coefficient, goodness-of-fit) profile.
#
# Grid: arch {mlp, cnn} x dataset {mnist, cifar10}
#       x family {ridge, nuclear_norm, stable_rank, spectral_entropy, spectral_gap}
#       x 5 seeds = 100 runs, submitted as one SLURM array.
#
# Each run fits exactly ONE family: at a trained optimum the candidate families'
# gradients are strongly collinear, so a joint fit is non-identifiable.
#
# Two fitting choices worth knowing about (both explained in run.py):
#   grad_match_reduction=sum   - constant rescaling of the matching objective;
#                                same minimizer, but keeps Adam off its eps floor.
#   bias_init=closed_form      - every family is scale * penalty(theta), so the
#                                optimum is an analytic 1-D least squares; the
#                                iterative fit starts there and the JSON reports
#                                both, which is the convergence check.
#
# Parameter budgets are matched WITHIN each dataset (the MLP-vs-CNN comparison is
# within-dataset), by choosing the MLP width per dataset:
#   MNIST:    mlp depth3 width256 = 335,114   cnn fc96 = 320,938  (ratio 0.96)
#   CIFAR-10: mlp depth3 width128 = 427,658   cnn fc96 = 413,674  (ratio 0.97)
#
# Set DRY_RUN=1 to write configs + manifest and print the sbatch command without
# submitting.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_root="${RUN_ROOT:-data/generated/arch_effective_regularizer}"
cfg_dir="$run_root/configs"
out_dir="$run_root/results"
mkdir -p "$cfg_dir" "$out_dir" logs/slurm

seeds=(123 42 666 314 17)
families=(ridge nuclear_norm stable_rank spectral_entropy spectral_gap)
manifest="$run_root/manifest.tsv"
: > "$manifest"

idx=0
for arch in mlp cnn; do
  for dataset in mnist cifar10; do
    # MLP width chosen per dataset for a matched parameter budget (see header);
    # ignored by the CNN, which uses channels 32,64 and fc_width 96 throughout.
    if [[ "$dataset" == "mnist" ]]; then width=256; else width=128; fi
    for family in "${families[@]}"; do
      for seed in "${seeds[@]}"; do
        tag="${arch}_${dataset}_${family}_s${seed}"
        cfg="$cfg_dir/${tag}.json"
        cat > "$cfg" <<JSON
{"arch":"${arch}","dataset":"${dataset}","family":"${family}","seed":${seed},
 "depth":3,"width":${width},"channels":"32,64","fc_width":96,
 "dropout":0.0,"batchnorm":false,"l2_lambda":0.0,
 "lr":0.05,"momentum":0.9,"batch_size":2048,"num_workers":2,
 "max_epochs":500,"patience":50,"model_monitor":"train/loss",
 "train_max_minutes":40,
 "bias_lr":0.05,"bias_max_epochs":150,"bias_patience":30,
 "bias_lr_patience":25,"bias_max_minutes":30,
 "grad_match_reduction":"sum","bias_init":"closed_form",
 "stationarity_max_batches":0,"stationarity_tol":0.001}
JSON
        printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$idx" "$tag" "$cfg" "$out_dir/${tag}.json" "$arch" "$dataset" "$family" \
          >> "$manifest"
        idx=$((idx + 1))
      done
    done
  done
done

total=$idx
echo "Prepared $total runs; manifest=$manifest"

wrap="line=\$(awk -v i=\$SLURM_ARRAY_TASK_ID '\$1==i' '$manifest'); \
cfg=\$(echo \"\$line\" | cut -f3); out=\$(echo \"\$line\" | cut -f4); \
PYTHONPATH=src uv run python analysis/arch_effective_regularizer/run.py \
--disable-wandb --config-path \"\$cfg\" \
--data-root '$repo_root/data/raw' \
--checkpoint-dir '$run_root/ckpt' --output \"\$out\""

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: not submitting.  Would run:"
  echo "sbatch --partition=whartonstat --job-name=archreg --gres=gpu:1 --cpus-per-task=3 \\"
  echo "  --mem=16G --time=2:00:00 --array=0-$((total - 1))%15 --chdir=$repo_root \\"
  echo "  --output=logs/slurm/%x-%A_%a.log --error=logs/slurm/%x-%A_%a.log --wrap=..."
  echo "--- wrapped command ---"
  echo "$wrap"
  exit 0
fi

sbatch \
  --partition=whartonstat \
  --job-name=archreg \
  --gres=gpu:1 \
  --cpus-per-task=3 \
  --mem=16G \
  --time=2:00:00 \
  --array="0-$((total - 1))%15" \
  --chdir="$repo_root" \
  --output="logs/slurm/%x-%A_%a.log" \
  --error="logs/slurm/%x-%A_%a.log" \
  --wrap="$wrap"
