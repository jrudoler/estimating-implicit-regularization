#!/bin/bash
# Figure-only job: never launches or resumes any training.
#SBATCH --partition=whartonstat
#SBATCH --job-name=dropout-loss-abs-figure
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --chdir=/home/jrudoler/inductive-bias
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log
set -euo pipefail
: "${DROPOUT_SOURCE_DIR:?Frozen source directory required}"
export UV_NO_SYNC=1 OMP_NUM_THREADS=2
export XDG_CACHE_HOME=/home/jrudoler/inductive-bias/.snakemake/cache
export MPLCONFIGDIR=/home/jrudoler/inductive-bias/.snakemake/matplotlib
.venv/bin/snakemake -s "$DROPOUT_SOURCE_DIR/workflow/rules/dropout_loss_plateau.smk" \
    --cores 2 --executor local --nolock --rerun-incomplete --keep-incomplete \
    --config plateau_study=dropout_loss_plateau_absolute \
    --allowed-rules plot_dropout_loss_plateau -- \
    results/figures/dropout_bias_ridge_panel_loss_plateau_absolute.pdf
