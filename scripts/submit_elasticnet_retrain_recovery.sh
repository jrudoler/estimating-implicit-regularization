#!/usr/bin/env bash
# Explicitly retrain seven accurately recovered elastic-net endpoints under:
# true coefficients, estimated coefficients, and no penalty.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_root="${RUN_ROOT:-data/generated/elasticnet_retrain_recovery}"
if [[ "${SMOKE:-0}" == "1" ]]; then
  run_root="${RUN_ROOT:-data/generated/elasticnet_retrain_recovery/smoke}"
fi
results_dir="$run_root/results"
mkdir -p "$results_dir" logs/slurm

manifest="$run_root/manifest.tsv"
cat > "$manifest" <<'TSV'
0	l1e-1_l2e-1_s0	0	0.1	0.1	0.09239934384822845	0.09955903887748718	checkpoints/epoch=166-step=20875-v1.ckpt
1	l1e-3_l2e0_s0	0	0.001	1.0	0.0011017086217179894	0.9643414616584778	checkpoints/epoch=24-step=3125.ckpt
2	l1e-1_l2e-1_s1	1	0.1	0.1	0.10973823815584183	0.08618584275245667	checkpoints/epoch=131-step=16500-v1.ckpt
3	l1e-3_l2e0_s1	1	0.001	1.0	0.000782977091614157	0.7794128060340881	checkpoints/epoch=34-step=4375-v1.ckpt
4	l1e-1_l2e0_s2	2	0.1	1.0	0.09137012809515	0.9793940186500549	checkpoints/epoch=67-step=8500.ckpt
5	l1e-1_l2e-1_s2	2	0.1	0.1	0.08733551949262619	0.12487634271383286	checkpoints/epoch=120-step=15125-v4.ckpt
6	l1e-1_l2e0_s1	1	0.1	1.0	0.08125452697277069	0.9343676567077637	checkpoints/epoch=124-step=15625-v2.ckpt
TSV

if [[ "${SMOKE:-0}" == "1" ]]; then
  array_spec="0"
else
  array_spec="0-6%7"
fi

wrap="set -euo pipefail; \
line=\$(awk -F'\\t' -v i=\$SLURM_ARRAY_TASK_ID '\$1==i' '$manifest'); \
tag=\$(echo \"\$line\" | cut -f2); seed=\$(echo \"\$line\" | cut -f3); \
true_l1=\$(echo \"\$line\" | cut -f4); true_l2=\$(echo \"\$line\" | cut -f5); \
estimated_l1=\$(echo \"\$line\" | cut -f6); estimated_l2=\$(echo \"\$line\" | cut -f7); \
checkpoint=\$(echo \"\$line\" | cut -f8); \
PYTHONPATH=src uv run python analysis/elasticnet_retrain_recovery/run.py \
--tag \"\$tag\" --seed \"\$seed\" --true-l1 \"\$true_l1\" --true-l2 \"\$true_l2\" \
--estimated-l1 \"\$estimated_l1\" --estimated-l2 \"\$estimated_l2\" \
--smooth 0.001 --batch-size 32 --target-checkpoint \"\$checkpoint\" \
--output '$results_dir'/\"\$tag\".json"

echo "Prepared 7 matched endpoints; manifest=$manifest; array=$array_spec"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: not submitting."
  echo "$wrap"
  exit 0
fi

sbatch \
  --partition=whartonstat \
  --job-name="${JOB_NAME:-enretrain}" \
  --gres=gpu:1 \
  --cpus-per-task=3 \
  --mem=12G \
  --time=00:30:00 \
  --array="$array_spec" \
  --chdir="$repo_root" \
  --output="logs/slurm/%x-%A_%a.log" \
  --error="logs/slurm/%x-%A_%a.log" \
  --wrap="$wrap"
