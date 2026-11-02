#!/bin/bash
# Submit in an isolated source snapshot. Pilot: sbatch --array=0,10,20 ...
# Remaining seeds after pilot review: sbatch --array=1-9,11-19,21-29%6 ...
#SBATCH --partition=whartonstat
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --job-name=dropout-conv
#SBATCH --output=logs/dropout-conv-%A_%a.log
set -euo pipefail
seeds=(123 42 666 314 17 111 325 643 432 51)
rates=(0.0 0.3 0.5)
index=${SLURM_ARRAY_TASK_ID:?Submit as an array}
tag="d3_w256_do${rates[index / 10]}_s${seeds[index % 10]}"
export PYTHONPATH=src UV_NO_SYNC=1 OMP_NUM_THREADS=2
: "${DROPOUT_CODE_REVISION:?Set the source revision}"
: "${DROPOUT_SOURCE_SHA256:?Set the source snapshot hash}"
.venv/bin/snakemake "data/generated/dropout_convergence/$tag" \
    -s workflow/Snakefile --executor local --cores 2 --nolock \
    --rerun-incomplete --keep-incomplete --printshellcmds \
    --config dropout_recovered_only=true optimizer_fix_stats_recovered_only=true \
    convergence_source=/shared_data0/jrudoler/dropout-audit.jiQDhK/data/generated/dropout_checkpoint_audit/results
