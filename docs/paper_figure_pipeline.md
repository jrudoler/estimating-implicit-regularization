# Paper Figure Pipeline

This note is the end-to-end map from active experiment or notebook logic to the artifacts under `paper/figures/`.

It is meant to answer three questions clearly:

1. What code or data generates each figure?
2. What intermediate artifacts are expected?
3. What file under `paper/figures/` is the canonical output?

## Conventions

- The manuscript lives in the `paper/` git submodule.
- The canonical destination for manuscript-bound figures is `paper/figures/`.
- The top-level builder is `scripts/build_paper_figures.py`.
- Some figures are generated directly from scripts or notebook wrappers.
- Some figures depend on prior experiment outputs in `results/` or W&B sweeps.
- `paper/figures/OLS_early_stopping_figure.pdf` is a preserved final panel assembled externally, but its component figures are still regenerated automatically.

## Top-Level Entry Points

- Build all paper figures:
  - `uv run python scripts/build_paper_figures.py`
- Build one figure or workflow:
  - `uv run python scripts/build_paper_figures.py --figures <figure_id>`
- Active provenance summary:
  - `docs/paper_figure_inventory.md`

## Figure-by-Figure Pipeline

### 1. `tradeoff-vis.png`

- Figure id:
  - `tradeoff-vis`
- Upstream source:
  - `notebooks/method-vis.ipynb`
- Maintained generator:
  - `scripts/regenerate_method_vis_figures.py`
- Intermediate artifacts:
  - none required
- Canonical output:
  - `paper/figures/tradeoff-vis.png`
- Build command:
  - `uv run python scripts/build_paper_figures.py --figures tradeoff-vis`

### 2. `sgd-vs-full-batch.png`

- Figure id:
  - `sgd-vs-full-batch`
- Upstream source:
  - `notebooks/method-vis.ipynb`
- Maintained generator:
  - `scripts/regenerate_method_vis_figures.py`
- Intermediate artifacts:
  - none required
- Canonical output:
  - `paper/figures/sgd-vs-full-batch.png`
- Build command:
  - `uv run python scripts/build_paper_figures.py --figures sgd-vs-full-batch`

### 3. `elasticnet_recovery_mean_se.pdf`

- Figure id:
  - `elasticnet_recovery_mean_se`
- Upstream experiment:
  - `experiments/elasticnet_train_and_recover.py`
- Active sweep config:
  - `sweeps/elasticnet_sweep_config_beta1e3.yaml`
- Confirmed W&B sweep:
  - `9a7ll8aa`
- Sweep structure:
  - `l1`: `6` values
  - `l2`: `6` values
  - `smooth`: fixed `1e-3`
  - `seed`: `0..9`
- Logged metrics consumed by the figure:
  - `recovery/log_mult_l1`
  - `recovery/log_mult_l2`
  - `recovery/lambda_1_hat`
  - `recovery/lambda_2_hat`
  - `true_l1`
  - `true_l2`
  - `smooth`
  - `seed`
- Maintained figure generator:
  - `scripts/plot_elasticnet_recovery.py`
- Top-level builder path:
  - `scripts/build_paper_figures.py` defaults to sweep `9a7ll8aa`
- Canonical output:
  - `paper/figures/elasticnet_recovery_mean_se.pdf`
- Local fallback:
  - `scripts/run_elasticnet_figure_batch.py`
  - this is only a fallback if explicitly pointed at a suitable CSV
- Important non-source:
  - `archive/notebooks_legacy_2026-04/elasticnet_viz.ipynb`
  - archived sweep `krcszr6z`
  - this older notebook sweep is not the paper `mean/SE` source
- Build command:
  - `uv run python scripts/build_paper_figures.py --figures elasticnet_recovery_mean_se`

### 4. `OLS_early_stopping_figure.pdf`

- Figure id:
  - `OLS_early_stopping_figure`
- Status:
  - preserved final manuscript asset
- Final paper asset:
  - `paper/figures/OLS_early_stopping_figure.pdf`
- Reason preserved:
  - the final annotated panel was assembled externally rather than by one repo-native script
- Notebook lineage for automated component figures:
  - `notebooks/linear-regression.ipynb`
- Maintained component generator:
  - `scripts/regenerate_linear_regression_ols_figures.py`
- Automated component outputs:
  - `paper/figures/Lambda_comparison-ols.pdf`
  - `paper/figures/predictive_weights_comparison_ols.pdf`
- Top-level builder behavior:
  - `scripts/build_paper_figures.py --figures OLS_early_stopping_figure`
  - preserves the final assembled panel
  - regenerates both component figures
- Related follow-on notebooks:
  - `notebooks/linear-regression-trajectory.ipynb`
  - `notebooks/linear-regression-trajectory-selfcontained.ipynb`
  - `notebooks/linear-regression-trajectory-sgd-vs-gd.ipynb`

### 5. `lambda_vs_epochs.pdf`

- Figure id:
  - `lambda_vs_epochs`
- Maintained generator:
  - `scripts/plot_lambda_vs_epochs.py`
- Canonical output:
  - `paper/figures/lambda_vs_epochs.pdf`
- Important note:
  - this script is active and automated, but heavier than the other figure builders because it reruns training internally
- Build command:
  - `uv run python scripts/build_paper_figures.py --figures lambda_vs_epochs`

### 6. `dropout_bias_ridge_panel.png`

- Figure id:
  - `dropout_bias_ridge_panel`
- Upstream experiment:
  - `experiments/dropout_bias_estimation.py`
- Active sweep config:
  - `sweeps/dropout_l2_bias.yaml`
- Confirmed W&B sweep:
  - `chiy2qjz`
- Notebook lineage:
  - `notebooks/l2_estimation_deep_ReLU.ipynb`
- Maintained generator:
  - `scripts/regenerate_dropout_bias_ridge_panel.py`
- Canonical output:
  - `paper/figures/dropout_bias_ridge_panel.png`
- Important note:
  - this requires W&B access at build time
- Build command:
  - `uv run python scripts/build_paper_figures.py --figures dropout_bias_ridge_panel`

### 7. `barrett_igr_figure2.pdf`

- Figure id:
  - `barrett_igr_figure2`
- Upstream experiment:
  - `experiments/barrett_igr_figure2.py`
- Expected intermediate result:
  - `results/barrett_igr_figure2.pt`
- Maintained plotter:
  - `analysis/barrett_igr_figure2_plot.py`
- Canonical output:
  - `paper/figures/barrett_igr_figure2.pdf`
- Build command:
  - `uv run python scripts/build_paper_figures.py --figures barrett_igr_figure2`

### 8. `barrett_igr_long_horizon.pdf`

- Figure id:
  - `barrett_igr_long_horizon`
- Upstream experiment:
  - `experiments/barrett_igr_long_horizon.py`
- Expected intermediate results:
  - `results/barrett_igr_lh_synth_eta001.pt`
  - `results/barrett_igr_lh_synth_eta003.pt`
  - `results/barrett_igr_lh_mnist_tanh_eta001.pt`
  - `results/barrett_igr_lh_mnist_relu_eta001.pt`
- Maintained plotter:
  - `analysis/barrett_igr_long_horizon_plot.py`
- Canonical output:
  - `paper/figures/barrett_igr_long_horizon.pdf`
- Build command:
  - `uv run python scripts/build_paper_figures.py --figures barrett_igr_long_horizon`

## Additional Reproducible Paper-Figure Components

These are not currently direct `paper/main.tex` includes, but they are part of the maintained reproducible figure surface.

### OLS component figures

- `paper/figures/Lambda_comparison-ols.pdf`
- `paper/figures/predictive_weights_comparison_ols.pdf`
- Source:
  - `notebooks/linear-regression.ipynb`
- Maintained generator:
  - `scripts/regenerate_linear_regression_ols_figures.py`

## Build Behavior Summary

### Figures generated directly from wrappers or scripts

- `tradeoff-vis`
- `sgd-vs-full-batch`
- `elasticnet_recovery_mean_se`
- `lambda_vs_epochs`
- `dropout_bias_ridge_panel`
- `barrett_igr_figure2`
- `barrett_igr_long_horizon`

### Preserved manuscript assets with automation around them

- `OLS_early_stopping_figure`
  - final assembled panel preserved
  - component figures regenerated

## Current Source-of-Truth Files

- Figure inventory:
  - `docs/paper_figure_inventory.md`
- End-to-end pipeline note:
  - `docs/paper_figure_pipeline.md`
- Top-level builder:
  - `scripts/build_paper_figures.py`
- Preserved-asset export helper:
  - `scripts/export_notebook_figure.py`

## Practical Rules

- If a figure is in `paper/main.tex`, its canonical output should be under `paper/figures/`.
- If a figure depends on experiment outputs, the experiment should write numeric artifacts to `results/` or W&B, and only the final plot step should write to `paper/figures/`.
- If a final paper panel was assembled externally, keep that final panel preserved, but still automate the reproducible component plots when possible.
