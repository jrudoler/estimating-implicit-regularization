#!/bin/bash
# Submit with --dependency=afterok:<pilot-array-id>. This never launches training.
#SBATCH --partition=standby
#SBATCH --qos=standby
#SBATCH --job-name=dropout-loss-review
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/jrudoler/inductive-bias
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log
set -euo pipefail
: "${DROPOUT_REVIEW_SOURCE_DIR:?Frozen review source required}"
study=${DROPOUT_STUDY:-dropout_loss_plateau}
[[ "$study" == dropout_loss_plateau || "$study" == dropout_loss_plateau_absolute ]]
export UV_NO_SYNC=1 OMP_NUM_THREADS=2
export XDG_CACHE_HOME=/home/jrudoler/inductive-bias/.snakemake/cache
.venv/bin/snakemake -s "$DROPOUT_REVIEW_SOURCE_DIR/workflow/rules/dropout_loss_plateau.smk" \
    --cores 2 --executor local --nolock --rerun-incomplete --keep-incomplete \
    --config "plateau_study=$study" --allowed-rules review_dropout_loss_plateau_pilots -- \
    "results/data/${study}_pilot_review.json"
