#!/bin/bash
# Submit from the isolated source checkout on Locust. This fallback still runs
# the declared Snakemake rule; it avoids executor-plugin submission discovery.
#SBATCH --partition=whartonstat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-29%6
#SBATCH --job-name=dropout-controls
#SBATCH --output=dropout-controls-%A_%a.log
set -euo pipefail
seeds=(123 42 666 314 17 111 325 643 432 51)
rates=(0.0 0.3 0.5)
index=${SLURM_ARRAY_TASK_ID:?Submit as an array}
tag="d3_w256_do${rates[index / 10]}_s${seeds[index % 10]}"
export PYTHONPATH=src UV_NO_SYNC=1
export XDG_CACHE_HOME="$PWD/cache" UV_CACHE_DIR="$PWD/cache/uv"
: "${DROPOUT_CODE_REVISION:?Set the source revision before submission}"
: "${DROPOUT_SOURCE_SHA256:?Set the source bundle hash before submission}"
# Specs must be prepared before submission. Each task owns different outputs;
# nolock permits those independent rule instances in a shared source checkout.
test -f "data/generated/dropout_checkpoint_audit/specs/$tag.json"
.venv/bin/snakemake "data/generated/dropout_checkpoint_audit/results/$tag.json" \
    -s workflow/Snakefile --executor local --cores 2 --nolock --rerun-incomplete \
    --config dropout_source_root="${DROPOUT_SOURCE_ROOT:-/home/jrudoler/inductive-bias}" \
    --printshellcmds
