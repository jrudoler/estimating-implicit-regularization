# Cleanup Index (2026-04)

This archive index records the dated move that slimmed the active tree around the current manuscript and retained research trajectory.

## Archived Experiments

- `archive/experiments_legacy_2026-04/`
- Includes legacy pairwise recovery scripts, matrix-spectrum/resampling drivers, older kernel-regression and gradient-penalty experiments, and nonlinear resampling utilities that are no longer in the active experiment surface.

## Archived Scripts

- `archive/scripts_legacy_2026-04/`
- Includes matrix-spectrum plotting utilities, stale helper scripts, and an older structured ridge/nuclear batch launcher that are no longer part of the active workflow.

## Archived Notebooks

- `archive/notebooks_legacy_2026-04/`
- Includes notebooks no longer tied to the current paper or retained documented workflow.

## Archived Core Code

- `archive/src_core_legacy_2026-04/`
- Currently contains `entk.py`, which became unused after the notebook cleanup.

## Archived Generated Results

- `archive/generated_results_legacy_2026-04/`
- Contains legacy root-level CSV outputs associated with archived recovery scripts.

## 2026-05 cleanup: `analysis/` audit against compiled paper

`analysis/` had grown to 37 subdirs; only 16 fed the 8 figures actually `\includegraphics`-ed by `paper/main.tex`. The 17 dirs below moved to `archive/analysis_legacy_2026-05/`. Workflow (`workflow/Snakefile`, `workflow/rules/plot.smk`, `workflow/rules/train.smk`, `workflow/rules/paper.smk`) and `config/sweeps.yaml` were pruned to match.

### Superseded standalone plot wrappers

These produced PDFs (`Lambda_comparison-ols.pdf`, `predictive_weights_comparison_ols.pdf`, `lambda_vs_epochs.pdf`, `ols_full_matrix_recovery.pdf`, `ols_full_matrix_distance_to_theory.pdf`) that were folded into `ols_composite.pdf` via `plot_ols_composite`. The underlying data experiments (`linear_regression_ols`, `lambda_vs_epochs`, `ols_full_matrix_recovery`) remain active.

- `analysis/plot_lambda_vs_epochs/`
- `analysis/plot_linear_regression_ols/`
- `analysis/plot_ols_full_matrix_recovery/`

### Stale / abandoned

- `analysis/barrett_igr_trajectory/` — wired in `train.smk` but never had a paper figure; orphan plot wrapper confirms abandonment.
- `analysis/plot_barrett_igr_trajectory/` — orphan plot wrapper, never wired into any rule.
- `analysis/bootstrap_bias_recovery/` — distinct from the OLS bootstrap experiments that *are* in the appendix; sweep with no plotting rule.

### Identifiability sweeps (no paper figure)

- `analysis/function_class_identifiability/`
- `analysis/l2_train_and_recover/`
- `analysis/l2_orthogonal_train_and_recover/`
- `analysis/l2_nuclear_train_and_recover/`
- `analysis/mixed_bias_recovery/`

### MNIST implicit-bias sweeps (no paper figure)

- `analysis/mnist_deep_relu_bias/`
- `analysis/mnist_implicit_reg/`

### Nonlinear extensions (no paper figure)

Paper currently focuses on linear/OLS. The nonlinear cluster was wired into `train.smk` but did not produce a compiled paper figure.

- `analysis/nonlinear_early_stopping_probe/` (also was not wired into any rule)
- `analysis/nonlinear_multi_geometry_replicate_ablation/`
- `analysis/nonlinear_multi_geometry_suite/`
- `analysis/nonlinear_power_retrain_geometry/`

### Companion paper-submodule cleanup

Removed these orphan PDFs from `paper/figures/` in manuscript commit `8fc494c`: `Lambda_comparison-ols.pdf`, `predictive_weights_comparison_ols.pdf`, `lambda_vs_epochs.pdf`, `ols_full_matrix_recovery.pdf`, `ols_full_matrix_distance_to_theory.pdf`, `OLS_early_stopping_figure.pdf`. The parent repo records that manuscript commit through the `paper/` submodule pointer.
