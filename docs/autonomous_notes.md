# Autonomous Notes

Use this file as the default non-manuscript log for autonomous method, implementation, and experiment notes.

## Conventions

- Add dated entries when autonomous work changes methods, assumptions, or experiment execution.
- Prefer logging here or in another `docs/` note instead of writing those updates into `paper/`.
- Promote durable results into a more specific doc in `docs/` when the workstream becomes substantial.
- For analysis plotting, default to a single `PDF` output unless the user explicitly requests an additional export format.

## 2026-03-27

- Added [`SmoothedPowerBias`](/home/jrudoler/inductive-bias/src/core/bias.py) for the family `R_p(theta) = lambda * sum_j (theta_j^2 + epsilon)^(p/2)` with optional trainable `lambda` and trainable global exponent `p`.
- Added [`experiments/nonlinear_power_retrain_geometry.py`](/home/jrudoler/inductive-bias/experiments/nonlinear_power_retrain_geometry.py) to train nonlinear models under a fixed smoothed-power regularizer, retrain on resamples, and then fit one shared `(lambda, p)` to the stacked collection of solutions and task gradients.
- Registered the new experiment in [`experiments/REGISTRY.yaml`](/home/jrudoler/inductive-bias/experiments/REGISTRY.yaml).

## 2026-04-12

- Elastic-net recovery: refactored [`experiments/elasticnet_train_and_recover.py`](/home/jrudoler/inductive-bias/experiments/elasticnet_train_and_recover.py) into `run_elasticnet_recovery()` with per-run `seed`, `recovery/log_mult_*` metrics logged to W\&B, and EarlyStopping on `train_bias/loss`. [`sweeps/elasticnet_sweep_config.yaml`](/home/jrudoler/inductive-bias/sweeps/elasticnet_sweep_config.yaml) now includes ten seeds per $(\lambda_1,\lambda_2,\beta)$. Added [`scripts/run_elasticnet_figure_batch.py`](/home/jrudoler/inductive-bias/scripts/run_elasticnet_figure_batch.py) (local CSV) and [`scripts/plot_elasticnet_recovery.py`](/home/jrudoler/inductive-bias/scripts/plot_elasticnet_recovery.py) (2$\times$2 mean/SE PDF). Optional env: `ELASTICNET_FAST`, `ELASTICNET_ULTRA` for shorter epochs.

## 2026-03-30

- Added [`SmoothedSchattenBias`](/home/jrudoler/inductive-bias/src/core/bias.py) so the same trainable `(lambda, p)` parameterization now covers spectral penalties over singular values, including the smoothed nuclear norm at `p = 1`.
- Added [`experiments/nonlinear_multi_geometry_suite.py`](/home/jrudoler/inductive-bias/experiments/nonlinear_multi_geometry_suite.py) to run a nonlinear retrain suite over single-component (`L2`, `L1`, nuclear) and multi-component (`L1 + L2`, `L2 + nuclear`, `L1 + L2 + nuclear`) geometries, saving JSON/CSV summaries plus per-case and summary PDF figures.
- Added [`scripts/nonlinear_multi_geometry_suite.slurm`](/home/jrudoler/inductive-bias/scripts/nonlinear_multi_geometry_suite.slurm) as a reproducible Slurm launcher for the larger GPU-backed version of the suite.
- Added exact sanity outputs under [`logs/phase2/nonlinear_multi_geometry_exact_sanity_scale03`](/home/jrudoler/inductive-bias/logs/phase2/nonlinear_multi_geometry_exact_sanity_scale03), a local nonlinear baseline under [`logs/phase2/nonlinear_multi_geometry_suite_scale03`](/home/jrudoler/inductive-bias/logs/phase2/nonlinear_multi_geometry_suite_scale03), and a larger GPU-backed suite under [`logs/phase2/nonlinear_multi_geometry_suite_gpu_scale03`](/home/jrudoler/inductive-bias/logs/phase2/nonlinear_multi_geometry_suite_gpu_scale03).
- Refactored [`experiments/nonlinear_multi_geometry_suite.py`](/home/jrudoler/inductive-bias/experiments/nonlinear_multi_geometry_suite.py) to separate replicate-pool collection from geometry fitting so later ablations can reuse one large bootstrap pool instead of retraining separately for every `n_replicates`.
- Added [`experiments/nonlinear_multi_geometry_replicate_ablation.py`](/home/jrudoler/inductive-bias/experiments/nonlinear_multi_geometry_replicate_ablation.py) and [`scripts/nonlinear_multi_geometry_replicate_ablation.slurm`](/home/jrudoler/inductive-bias/scripts/nonlinear_multi_geometry_replicate_ablation.slurm) to study how recovery changes as the number of bootstrap retrains grows.
- Ran the replicate ablation and summarized it in [`docs/nonlinear_multi_geometry_replicate_ablation_2026-03-30.md`](/home/jrudoler/inductive-bias/docs/nonlinear_multi_geometry_replicate_ablation_2026-03-30.md). The main finding was that larger bootstrap pools did not reliably improve identifiability; most cases were flat or slightly worse at high replicate counts, with only `single_l1` showing a modest gain.
