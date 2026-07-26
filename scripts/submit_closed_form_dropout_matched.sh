#!/usr/bin/env bash
# Dropout implicit-bias sweep with MATCHED stationarity (convergence control for
# analysis/closed_form_dropout_sweep).
#
# The unmatched sweep found that the ridge (L2) explanation of the residual
# full-batch gradient gets worse with dropout while scale-invariant spectral
# families get better -- but higher dropout also left models further from
# stationarity (Spearman(dropout, relative_grad_norm) = +0.914), which can inflate
# the "unexplained" component mechanically.  This launch removes the confound:
#
#   * NO val-loss early stopping.  Every condition gets the identical, long,
#     fixed schedule (3000 epochs; the model's own MultiStepLR to epoch 200 plus a
#     common tail decay), so stopping time cannot depend on the dropout rate.
#   * The full closed-form panel (all 5 families) + both stationarity measures are
#     recorded every 25 epochs along the WHOLE trajectory, which lets the analysis
#     compare every dropout rate at a *selected* common stationarity level
#     (matched-band), not just at the endpoint.
#
# Grid: dropout {0, .05, .1, .15, .2, .3, .4, .5} x 12 seeds = 96 runs.
# The first 10 seeds are the ones used by closed_form_dropout_sweep, so the matched
# and unmatched sweeps are directly comparable run-for-run.
#
# Set DRY_RUN=1 to write the manifest and print the sbatch command without submitting.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_root="${RUN_ROOT:-data/generated/closed_form_dropout_matched}"
out_dir="$run_root/results"
mkdir -p "$out_dir" logs/slurm

dropouts=(0.0 0.05 0.1 0.15 0.2 0.3 0.4 0.5)
seeds=(17 42 51 111 123 314 325 432 643 666 7 19)

max_epochs="${MAX_EPOCHS:-3000}"
probe_every="${PROBE_EVERY:-25}"
probe_every_early="${PROBE_EVERY_EARLY:-5}"
early_until="${EARLY_UNTIL:-200}"
mask_reps="${MASK_REPS:-12}"
tail_start="${TAIL_START:-600}"
tail_every="${TAIL_EVERY:-300}"
tail_gamma="${TAIL_GAMMA:-0.5}"
target="${TARGET_REL_GRAD:-3e-4}"

manifest="$run_root/manifest.tsv"
: > "$manifest"

idx=0
for do_ in "${dropouts[@]}"; do
  for seed in "${seeds[@]}"; do
    tag="do${do_}_s${seed}"
    printf '%d\t%s\t%s\t%s\t%s\n' "$idx" "$tag" "$do_" "$seed" "$out_dir/${tag}.json" >> "$manifest"
    idx=$((idx + 1))
  done
done

total=$idx
echo "Prepared $total runs; manifest=$manifest"

wrap="line=\$(awk -v i=\$SLURM_ARRAY_TASK_ID '\$1==i' '$manifest'); \
do_=\$(echo \"\$line\" | cut -f3); seed=\$(echo \"\$line\" | cut -f4); \
out=\$(echo \"\$line\" | cut -f5); \
PYTHONPATH=src uv run python analysis/closed_form_dropout_matched/run.py \
--dropout \"\$do_\" --seed \"\$seed\" --depth 3 --width 256 \
--max-epochs $max_epochs --probe-every $probe_every \
--probe-every-early $probe_every_early --early-until $early_until --mask-reps $mask_reps \
--tail-decay-start $tail_start --tail-decay-every $tail_every --tail-decay-gamma $tail_gamma \
--target-rel-grad $target --max-minutes 200 \
--data-root '$repo_root/data/raw' --output \"\$out\""

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: not submitting.  Would run:"
  echo "$wrap"
  exit 0
fi

sbatch \
  --partition=whartonstat \
  --job-name=cfdmatch \
  --gres=gpu:1 \
  --cpus-per-task=3 \
  --mem=16G \
  --time=4:00:00 \
  --array="0-$((total - 1))%20" \
  --chdir="$repo_root" \
  --output="logs/slurm/%x-%A_%a.log" \
  --error="logs/slurm/%x-%A_%a.log" \
  --wrap="$wrap"
