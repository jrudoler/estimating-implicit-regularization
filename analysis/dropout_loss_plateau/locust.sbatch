#!/bin/bash
# Each array member runs one explicit target in an isolated workflow directory.
#SBATCH --partition=standby
#SBATCH --qos=standby
#SBATCH --job-name=dropout-loss-plateau
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/jrudoler/inductive-bias
#SBATCH --output=logs/slurm/%x-%A_%a.log
#SBATCH --error=logs/slurm/%x-%A_%a.log
set -euo pipefail
: "${DROPOUT_SOURCE_DIR:?Frozen source directory required}"
: "${DROPOUT_SOURCE_SHA256:?Frozen source archive hash required}"
: "${DROPOUT_CODE_REVISION:?Base code revision required}"
: "${DROPOUT_TASK_FILE:?Explicit task list required}"
repo_root=/home/jrudoler/inductive-bias
study=${DROPOUT_STUDY:-dropout_loss_plateau}
[[ "$study" == dropout_loss_plateau || "$study" == dropout_loss_plateau_absolute ]]
tag=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$DROPOUT_TASK_FILE")
[[ "$tag" =~ ^d(3|5)_w(128|256|512)_do0\.(0|05|1|15|2|3|4|5)_s(123|42|666|314|17|111|325|643|432|51)$ ]]
if [[ -n "${DROPOUT_NODE_SCRATCH:-}" ]]; then
    [[ "$DROPOUT_NODE_SCRATCH" == /scratch/jrudoler/dropout_loss_plateau_absolute ]]
    scratch_available_kib=$(df -Pk /scratch | awk 'NR==2 {print $4}')
    [[ "$scratch_available_kib" -gt 52428800 ]]
    task_dir="$DROPOUT_NODE_SCRATCH/work/$tag"
else
    task_dir="$repo_root/data/generated/$study/work/$tag"
fi
mkdir -p "$task_dir"
for name in data results logs .venv; do
    if [[ ! -e "$task_dir/$name" ]]; then
        ln -s "$repo_root/$name" "$task_dir/$name"
    fi
done
export UV_NO_SYNC=1 OMP_NUM_THREADS=2 XDG_CACHE_HOME="$task_dir/.cache"
"$repo_root/.venv/bin/snakemake" "data/generated/$study/$tag/result.json" \
    --directory "$task_dir" -s "$DROPOUT_SOURCE_DIR/workflow/rules/dropout_loss_plateau.smk" \
    --executor local --cores 2 --nolock --rerun-incomplete --keep-incomplete --printshellcmds \
    --config "plateau_study=$study"
