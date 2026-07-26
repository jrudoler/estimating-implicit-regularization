#!/usr/bin/env bash
# FULL Figure-5 grid re-estimated with the EXACT closed-form optimum.
#
# Figure 5 of the submission reports that the estimated ridge coefficient rises
# monotonically with the dropout rate, across depth {3,5} x width {128,256,512}.
# That figure was produced by the iterative gradient-matching fit, which does not
# converge (it overshoots the exact 1-D least-squares optimum by 38-61x; see
# data/generated/verify_estimator_convergence/FINDINGS.md).
#
# analysis/closed_form_dropout_sweep re-estimated ONE cell (depth=3, width=256)
# with the exact closed form and found the ridge coefficient essentially FLAT in
# the dropout rate (Spearman -0.338).  But in the ORIGINAL iterative data the
# per-facet trend strength varies a lot (Spearman 0.47 at depth=3/width=512 up to
# 0.93 at depth=5/width=128), so a single cell cannot settle whether the flat
# result is universal or architecture-specific.  This launch closes that gap by
# running the whole grid:
#
#   depth {3,5} x width {128,256,512} x dropout {0,.05,.1,.15,.2,.3,.4,.5}
#   x 10 seeds = 480 runs.
#
# Seeds are exactly the ones used by the original sweep and by
# closed_form_dropout_sweep, so the comparison is run-for-run comparable.
# Model config matches the original sweep: l2_penalty=0, lr=0.05, momentum=0.9,
# batch_size=2048, MNIST, 500 max epochs with val-loss patience 50 (the defaults
# of analysis/closed_form_dropout_sweep/run.py, which is invoked unmodified).
#
# Race-safety: closed_form_dropout_sweep/run.py builds its Trainer with
# logger=False and enable_checkpointing=False, so no run writes a checkpoint or a
# CSV log -- there is no shared checkpoint dir and no shared CSVLogger to race on
# (both of which have cost us runs before).  Each task additionally gets its own
# TMPDIR.
#
# Env knobs:
#   DRY_RUN=1        write manifest/configs, print the sbatch command, do not submit
#   SMOKE=1          submit only SMOKE_N tasks with MAX_EPOCHS=2 (pipeline check)
#   ONLY_MISSING=1   build the array only from runs whose output JSON is absent
#   MAX_EPOCHS, PATIENCE, CONCURRENCY, RUN_ROOT, JOB_NAME
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_root="${RUN_ROOT:-data/generated/closed_form_dropout_grid}"
cfg_dir="$run_root/configs"
out_dir="$run_root/results"
mkdir -p "$cfg_dir" "$out_dir" logs/slurm

depths=(3 5)
widths=(128 256 512)
dropouts=(0.0 0.05 0.1 0.15 0.2 0.3 0.4 0.5)
seeds=(123 42 666 314 17 111 325 643 432 51)

max_epochs="${MAX_EPOCHS:-500}"
patience="${PATIENCE:-50}"
concurrency="${CONCURRENCY:-20}"
job_name="${JOB_NAME:-cfgrid}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  max_epochs="${MAX_EPOCHS:-2}"
  patience="${PATIENCE:-2}"
  job_name="${JOB_NAME:-cfgsmoke}"
  run_root="${RUN_ROOT:-data/generated/closed_form_dropout_grid/smoke}"
  cfg_dir="$run_root/configs"
  out_dir="$run_root/results"
  mkdir -p "$cfg_dir" "$out_dir"
fi

manifest="$run_root/manifest.tsv"
: > "$manifest"

# Full grid manifest: idx  tag  depth  width  dropout  seed  output
idx=0
for depth in "${depths[@]}"; do
  for width in "${widths[@]}"; do
    for do_ in "${dropouts[@]}"; do
      for seed in "${seeds[@]}"; do
        tag="d${depth}_w${width}_do${do_}_s${seed}"
        out="$out_dir/${tag}.json"
        cat > "$cfg_dir/${tag}.json" <<JSON
{"depth":${depth},"width":${width},"dropout":${do_},"batchnorm":false,
 "l2_penalty":0.0,"lr":0.05,"momentum":0.9,"batch_size":2048,
 "num_data_workers":2,"max_epochs":${max_epochs},"patience":${patience},
 "seed":${seed},"estimator":"closed_form","dataset":"mnist"}
JSON
        printf '%d\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$idx" "$tag" "$depth" "$width" "$do_" "$seed" "$out" >> "$manifest"
        idx=$((idx + 1))
      done
    done
  done
done
total=$idx

# The array indexes a task list, which is either the whole manifest or only the
# rows still missing an output file.  Keeping the manifest complete (and stable)
# means re-run rounds do not renumber anything.
tasklist="$run_root/tasklist.txt"
if [[ "${ONLY_MISSING:-0}" == "1" ]]; then
  : > "$tasklist"
  while IFS=$'\t' read -r i tag depth width do_ seed out; do
    [[ -s "$out" ]] || echo "$i" >> "$tasklist"
  done < "$manifest"
  n_missing=$(wc -l < "$tasklist")
  echo "manifest=$manifest total=$total missing=$n_missing"
  if [[ "$n_missing" == "0" ]]; then echo "nothing to do"; exit 0; fi
  array_spec="$(paste -sd, "$tasklist")%${concurrency}"
elif [[ "${SMOKE:-0}" == "1" ]]; then
  smoke_n="${SMOKE_N:-3}"
  # Take the extreme corners of the grid so the smoke test exercises the
  # smallest and largest models: first row, a middle row, and the last row.
  array_spec="0,$(( total / 2 )),$(( total - 1 ))%${concurrency}"
  echo "SMOKE: manifest=$manifest total=$total array=$array_spec (n=$smoke_n)"
else
  echo "manifest=$manifest total=$total"
  array_spec="0-$((total - 1))%${concurrency}"
fi

wrap="set -euo pipefail; \
line=\$(awk -F'\\t' -v i=\$SLURM_ARRAY_TASK_ID '\$1==i' '$manifest'); \
tag=\$(echo \"\$line\" | cut -f2); depth=\$(echo \"\$line\" | cut -f3); \
width=\$(echo \"\$line\" | cut -f4); do_=\$(echo \"\$line\" | cut -f5); \
seed=\$(echo \"\$line\" | cut -f6); out=\$(echo \"\$line\" | cut -f7); \
export TMPDIR=\"\${SLURM_TMPDIR:-/tmp}/cfgrid_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}\"; \
mkdir -p \"\$TMPDIR\"; \
echo \"task \$SLURM_ARRAY_TASK_ID tag=\$tag depth=\$depth width=\$width dropout=\$do_ seed=\$seed\"; \
PYTHONPATH=src uv run python analysis/closed_form_dropout_sweep/run.py \
--depth \"\$depth\" --width \"\$width\" --dropout \"\$do_\" --seed \"\$seed\" \
--max-epochs $max_epochs --patience $patience \
--data-root '$repo_root/data/raw' --output \"\$out\"; \
rm -rf \"\$TMPDIR\""

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: not submitting.  array=$array_spec"
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
