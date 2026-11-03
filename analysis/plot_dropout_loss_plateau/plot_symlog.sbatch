#!/bin/bash
# Presentation only; refuses partial grids and never launches training.
#SBATCH --partition=whartonstat
#SBATCH --qos=normal
#SBATCH --nodelist=node9
#SBATCH --job-name=dropout-loss-logfigure
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/home/jrudoler/inductive-bias
#SBATCH --output=logs/slurm/%x-%j.log
#SBATCH --error=logs/slurm/%x-%j.log
set -euo pipefail
export UV_NO_SYNC=1 OMP_NUM_THREADS=2
export XDG_CACHE_HOME=/home/jrudoler/inductive-bias/.snakemake/cache
export MPLCONFIGDIR=/home/jrudoler/inductive-bias/.snakemake/matplotlib
.venv/bin/snakemake -s workflow/rules/plot_dropout_loss_plateau_symlog.smk \
    --cores 2 --executor local --nolock --rerun-incomplete --keep-incomplete \
    --allowed-rules plot_dropout_loss_plateau_symlog -- \
    results/figures/dropout_bias_ridge_panel_loss_plateau_absolute_symlog.pdf
