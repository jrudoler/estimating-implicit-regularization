# Experiment Launcher Archive (2026-03)

The files below were moved out of `experiments/` to reduce duplicate launch surfaces.
Use `scripts/wandb_sweep.slurm` as the canonical launcher for sweep agents.

- `experiments/implicit_reg_sweep.slurm` -> `archive/experiments_legacy_2026-03/implicit_reg_sweep.slurm`
  - Reason: duplicate W&B agent launcher using `poetry`.
- `experiments/elasticnet_sweep_job.slurm` -> `archive/experiments_legacy_2026-03/elasticnet_sweep_job.slurm`
  - Reason: duplicate W&B agent launcher using `poetry`.
- `experiments/kernel-regression-RBG.slurm` -> `archive/experiments_legacy_2026-03/kernel-regression-RBG.slurm`
  - Reason: experiment-local launcher superseded by canonical launcher and `uv` workflow.
- `experiments/ridge_rnv_recovery.slurm` -> `archive/experiments_legacy_2026-03/ridge_rnv_recovery.slurm`
  - Reason: one-off launcher superseded by canonical launcher.
