#!/usr/bin/env bash
# Inject the per-layer activation-weighted quadratic (better fit than ridge /
# stable_rank in dropout_structured_regularizers) and retrain after ablating
# dropout -- the missing reinstate arm of dropout_retrain_recovery.
#
# Stages:
#   STAGE=prepare   target + ablated only (needed for per-seed moments/coefs)
#   STAGE=specs     build injection specs from prepare outputs (local, no sbatch)
#   STAGE=retrain   actquad_T / actquad_M / actquad_M_matched / actquad_10x
#   STAGE=analyze   aggregate FINDINGS.md + summary.json (local)
#
# Configs match dropout_retrain_recovery (MNIST, d3/w256, l2=0, lr=0.05,
# momentum=0.9, bs=2048, patience 50).  TARGET/ABLATED are re-run here because
# the prior experiment did not save moments, test softmax, or structured fits.
#
# Environment:
#   DRY_RUN=1 CPU_ONLY=1 ONLY_TAGS="..." ONLY_MISSING=1 CONCURRENCY=N
#   RUN_ROOT=path JOB_NAME=name MATCHED_BUDGET=N (override; else from specs)
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

stage="${STAGE:-prepare}"
run_root="${RUN_ROOT:-data/generated/dropout_actquad_retrain}"
out_dir="$run_root/results"
ckpt_root="$run_root/ckpt"
art_dir="$run_root/artifacts"
spec_dir="$run_root/specs"
mkdir -p "$out_dir" "$ckpt_root" "$art_dir" "$spec_dir" logs/slurm

seeds=(123 42 666 314 17 111 325 643 432 51)

# condition | dropout | protocol | coef_key | fixed_epochs_flag
# fixed_epochs_flag: 0 = early stop; "matched" = use matched_budget_epochs from specs
prepare_conditions=(
  "target|0.3|none||0"
  "ablated|0.0|none||0"
)
retrain_conditions=(
  "actquad_T|0.0|T|coef_total|0"
  "actquad_M|0.0|M|coef_marginal|0"
  "actquad_M_matched|0.0|M|coef_marginal|matched"
  "actquad_10x|0.0|control_10x|coef_10x|0"
)

case "$stage" in
  prepare) conditions=("${prepare_conditions[@]}") ;;
  retrain) conditions=("${retrain_conditions[@]}") ;;
  specs)
    PYTHONPATH=src uv run python analysis/dropout_actquad_retrain/build_specs.py \
      --results "$out_dir" --artifacts "$art_dir" --out "$spec_dir"
    exit 0
    ;;
  analyze)
    PYTHONPATH=src uv run python analysis/dropout_actquad_retrain/analyze.py \
      --results "$out_dir" --artifacts "$art_dir" \
      --json-out "$run_root/summary.json" --md-out "$run_root/FINDINGS.md"
    exit 0
    ;;
  *)
    echo "Unknown STAGE=$stage (prepare|specs|retrain|analyze)" >&2
    exit 1
    ;;
esac

matched_budget="${MATCHED_BUDGET:-0}"
if [[ "$stage" == "retrain" && "$matched_budget" == "0" ]]; then
  if [[ -f "$spec_dir/summary.json" ]]; then
    matched_budget="$(python3 -c "import json; print(json.load(open('$spec_dir/summary.json'))['matched_budget_epochs'])")"
  else
    echo "Missing $spec_dir/summary.json; run STAGE=specs first." >&2
    exit 1
  fi
fi

manifest="$run_root/manifest_${stage}.tsv"
: > "$manifest"
idx=0
for spec in "${conditions[@]}"; do
  IFS='|' read -r cond dropout protocol coef_key fixed_flag <<< "$spec"
  for seed in "${seeds[@]}"; do
    tag="${cond}_s${seed}"
    if [[ -n "${ONLY_TAGS:-}" ]] && [[ " $ONLY_TAGS " != *" $tag "* ]]; then
      continue
    fi
    if [[ "${ONLY_MISSING:-0}" == "1" && -s "$out_dir/${tag}.json" ]]; then
      continue
    fi
    fixed_epochs=0
    if [[ "$fixed_flag" == "matched" ]]; then
      fixed_epochs="$matched_budget"
    fi
    printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$idx" "$tag" "$cond" "$dropout" "$protocol" "$coef_key" "$fixed_epochs" "$seed" \
      >> "$manifest"
    idx=$((idx + 1))
  done
done

total=$idx
if [[ "$total" -eq 0 ]]; then
  echo "No tasks to submit (manifest empty)."
  exit 0
fi
echo "STAGE=$stage prepared $total runs; manifest=$manifest; matched_budget=$matched_budget"

wrap="set -euo pipefail; \
line=\$(awk -v i=\$SLURM_ARRAY_TASK_ID -F'\t' '\$1==i' '$manifest'); \
tag=\$(echo \"\$line\" | cut -f2); cond=\$(echo \"\$line\" | cut -f3); \
dropout=\$(echo \"\$line\" | cut -f4); protocol=\$(echo \"\$line\" | cut -f5); \
coef_key=\$(echo \"\$line\" | cut -f6); fixed_epochs=\$(echo \"\$line\" | cut -f7); \
seed=\$(echo \"\$line\" | cut -f8); \
spec_args=''; fixed_args=''; \
if [[ -n \"\$coef_key\" ]]; then \
  spec_args=\"--spec $repo_root/$spec_dir/seed\${seed}.json --coef-key \$coef_key\"; \
fi; \
if [[ \"\$fixed_epochs\" != \"0\" ]]; then \
  fixed_args=\"--fixed-epochs \$fixed_epochs\"; \
fi; \
export TMPDIR=\"\${SLURM_TMPDIR:-/tmp}/actquad_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}\"; \
mkdir -p \"\$TMPDIR\"; \
echo \"task=\$SLURM_ARRAY_TASK_ID tag=\$tag cond=\$cond seed=\$seed fixed_epochs=\$fixed_epochs\"; \
PYTHONPATH=src uv run python analysis/dropout_actquad_retrain/run.py \
  --tag \"\$tag\" --condition \"\$cond\" --protocol \"\$protocol\" \
  --dropout \"\$dropout\" --seed \"\$seed\" \
  --data-root '$repo_root/data/raw' \
  --checkpoint-dir '$repo_root/$ckpt_root' \
  --output '$repo_root/$out_dir/'\"\$tag\"'.json' \
  --artifact '$repo_root/$art_dir/'\"\$tag\"'.pt' \
  \$spec_args \$fixed_args; \
rm -rf \"\$TMPDIR\""

declare -a sbatch_args=(
  --partition="${PARTITION:-whartonstat}"
  --job-name="${JOB_NAME:-actquad_${stage}}"
  --cpus-per-task="${CPUS:-4}"
  --mem=16G
  --time="${TIME:-3:00:00}"
  --array="0-$((total - 1))%${CONCURRENCY:-20}"
  --chdir="$repo_root"
  --output="logs/slurm/%x-%A_%a.log"
  --error="logs/slurm/%x-%A_%a.log"
)
if [[ "${CPU_ONLY:-0}" != "1" ]]; then
  sbatch_args+=(--gres=gpu:1)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: not submitting. Would run:"
  echo "sbatch ${sbatch_args[*]} --wrap=..."
  echo "--- wrapped command ---"
  echo "$wrap"
  exit 0
fi

sbatch "${sbatch_args[@]}" --wrap="$wrap"
