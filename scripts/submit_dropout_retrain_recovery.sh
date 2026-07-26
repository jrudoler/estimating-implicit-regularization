#!/usr/bin/env bash
# Success mode 2 for dropout (reviewers iMcK Q4, rnFp Q2, AC):
# estimate dropout's implicit regularizer, ABLATE dropout, retrain with that
# regularizer added EXPLICITLY, and test whether the retrained model reproduces
# the dropout-trained one.
#
# Coefficients are the medians of the closed-form optima
#   s* = <grad P, -grad L> / ||grad P||^2
# over the 10 seeds of data/generated/closed_form_dropout_sweep (MNIST,
# DeepReLUClassifier depth=3 width=256, l2=0, lr=0.05, momentum=0.9, bs=2048,
# val-loss patience 50).  Only the closed form is used: the iterative
# gradient-matching fit overshoots the optimum by 38-61x
# (data/generated/verify_estimator_convergence/FINDINGS.md).
#
#   family        s*(dropout=0.0)   s*(dropout=0.3)   marginal (0.3 - 0.0)
#   ridge          +4.125367e-05     +3.556925e-05     -5.684422e-06  (NEGATIVE)
#   stable_rank    -3.445486e-05     +6.309242e-05     +9.754728e-05
#
# TWO PROTOCOLS, because s* at an endpoint is the TOTAL effective regularization
# there -- it also absorbs the implicit bias of SGD and early stopping, which the
# dropout=0 reference has as well, so it is NOT dropout's marginal contribution.
# (In the paper's early-stopping demonstration the ablated reference is trained to
# convergence, so it has zero early-stopping regularization by construction and no
# double counting arises.)
#   protocol T ("total", literal success mode 2): s = s*(dropout=0.3)
#   protocol M ("marginal"):                     s = s*(0.3) - s*(0.0)
# The ridge marginal is negative, so protocol M is NOT realizable as a positive
# ridge penalty; it is run as a genuine negative coefficient (anti-penalty) and
# reported as such.  It is numerically benign here: the induced growth factor is
# exp(2|s| * lr_eff * n_steps) ~ 1.02 over a full run.
#
# The 10x controls exist to show the comparison is sensitive to the VALUE of the
# estimated coefficient, not merely to adding some penalty.
#
# 8 conditions x 10 seeds = 80 runs, one SLURM array.  Every run gets its own
# Lightning root directory and logger=False (a shared checkpoint dir and a shared
# CSVLogger have both cost us runs before).
#
# DRY_RUN=1 writes the manifest and prints the sbatch command without submitting.
# ONLY_TAGS="a b c" restricts the array to the named tags (used for re-running
# failures).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_root="${RUN_ROOT:-data/generated/dropout_retrain_recovery}"
out_dir="$run_root/results"
ckpt_root="$run_root/ckpt"
mkdir -p "$out_dir" "$ckpt_root" logs/slurm

seeds=(123 42 666 314 17 111 325 643 432 51)

# condition | dropout | family | coefficient | protocol
conditions=(
  "target|0.3|none|0.0|none"
  "ablated|0.0|none|0.0|none"
  "ridge_T|0.0|ridge|3.556925e-05|T"
  "srank_T|0.0|stable_rank|6.309242e-05|T"
  "srank_M|0.0|stable_rank|9.754728e-05|M"
  "ridge_M|0.0|ridge|-5.684422e-06|M_negative"
  "ridge_T10x|0.0|ridge|3.556925e-04|control_10x"
  "srank_T10x|0.0|stable_rank|6.309242e-04|control_10x"
)

manifest="$run_root/manifest.tsv"
: > "$manifest"

idx=0
for spec in "${conditions[@]}"; do
  IFS='|' read -r cond dropout family coef protocol <<< "$spec"
  for seed in "${seeds[@]}"; do
    tag="${cond}_s${seed}"
    if [[ -n "${ONLY_TAGS:-}" ]] && [[ " $ONLY_TAGS " != *" $tag "* ]]; then
      continue
    fi
    printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$idx" "$tag" "$cond" "$dropout" "$family" "$coef" "$protocol" "$seed" \
      >> "$manifest"
    idx=$((idx + 1))
  done
done

total=$idx
echo "Prepared $total runs; manifest=$manifest"

# NOTE: --penalty-coef must be passed in the "--opt=value" form.  argparse's
# negative-number matcher is ^-\d+$|^-\d*\.\d+$, which does NOT match scientific
# notation, so a space-separated "-5.68e-06" is parsed as an unknown option.
wrap="set -euo pipefail; \
line=\$(awk -v i=\$SLURM_ARRAY_TASK_ID -F'\t' '\$1==i' '$manifest'); \
tag=\$(echo \"\$line\" | cut -f2); cond=\$(echo \"\$line\" | cut -f3); \
dropout=\$(echo \"\$line\" | cut -f4); family=\$(echo \"\$line\" | cut -f5); \
coef=\$(echo \"\$line\" | cut -f6); protocol=\$(echo \"\$line\" | cut -f7); \
seed=\$(echo \"\$line\" | cut -f8); \
PYTHONPATH=src uv run python analysis/dropout_retrain_recovery/run.py \
--tag \"\$tag\" --condition \"\$cond\" --protocol \"\$protocol\" \
--dropout \"\$dropout\" --penalty-family \"\$family\" \"--penalty-coef=\$coef\" \
--seed \"\$seed\" --data-root '$repo_root/data/raw' \
--checkpoint-dir '$repo_root/$ckpt_root' --output '$repo_root/$out_dir/'\"\$tag\"'.json'"

# CPU_ONLY=1 drops --gres=gpu:1.  This model is tiny (335k params, bs=2048,
# ~2.5 s/epoch on a CPU core, ~130 epochs), so a full run costs ~6 min on CPU and
# the array does not need GPUs at all.  Use it when standby's GPUs are saturated:
# waiting for a GPU slot costs far more wall clock than the run itself.  Device is
# held constant across ALL conditions within a submission, so every contrast
# (target / ablated / retrained) stays internally consistent.
declare -a sbatch_args=(
  --partition=whartonstat
  --job-name="${JOB_NAME:-droprecov}"
  --cpus-per-task="${CPUS:-4}"
  --mem=16G
  --time=2:00:00
  --array="0-$((total - 1))%${CONCURRENCY:-20}"
  --chdir="$repo_root"
  --output="logs/slurm/%x-%A_%a.log"
  --error="logs/slurm/%x-%A_%a.log"
)
if [[ "${CPU_ONLY:-0}" != "1" ]]; then
  sbatch_args+=(--gres=gpu:1)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: not submitting.  Would run:"
  echo "sbatch ${sbatch_args[*]} --wrap=..."
  echo "--- wrapped command ---"
  echo "$wrap"
  exit 0
fi

sbatch "${sbatch_args[@]}" --wrap="$wrap"
