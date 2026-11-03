#!/bin/bash
# Read-only audit of completed training; writes only the verification receipt.
#SBATCH --partition=whartonstat
#SBATCH --qos=normal
#SBATCH --nodelist=node9
#SBATCH --job-name=dropout-loss-verify
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/jrudoler/inductive-bias
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log
set -euo pipefail
: "${DROPOUT_VERIFY_SOURCE_DIR:?Frozen verification source directory required}"
export UV_NO_SYNC=1 OMP_NUM_THREADS=2
export XDG_CACHE_HOME=/home/jrudoler/inductive-bias/.snakemake/cache
.venv/bin/snakemake -s "$DROPOUT_VERIFY_SOURCE_DIR/workflow/rules/verify_dropout_loss_plateau.smk" \
    --cores 2 --executor local --nolock --rerun-incomplete --keep-incomplete \
    --allowed-rules verify_dropout_loss_plateau -- \
    data/generated/dropout_loss_plateau_absolute/full_grid_verification.json
