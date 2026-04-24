# Paper Figure Pipeline

This note is the end-to-end map from active experiment or notebook logic to the artifacts under `paper/generated/figures/`.

It is meant to answer three questions clearly:

1. What code or data generates each figure?
2. What intermediate artifacts are expected?
3. What file under `paper/generated/figures/` is the canonical output?

## Conventions

- The manuscript lives in the `paper/` git submodule.
- The canonical destination for manuscript-bound figures is `paper/generated/figures/`.
- The top-level builder is the Snakemake workflow: run `uv run snakemake -s workflow/Snakefile --cores 4 figures` (figures only) or `--profile workflow/profiles/slurm paper` (end-to-end).
- Some figures are generated directly from scripts or notebook wrappers.
- Some figures depend on prior experiment outputs in `data/generated/<analysis>/` (the local snapshot of training outputs) or W&B sweeps pulled into `data/generated/<analysis>/runs.parquet`.
- `paper/generated/figures/OLS_early_stopping_figure.pdf` is a preserved final panel assembled externally, but its component figures are still regenerated automatically.

## Top-Level Entry Points

- Build all paper figures:
  - `uv run snakemake -s workflow/Snakefile --cores 4 figures`
- Build one figure or workflow:
  - `uv run snakemake -s workflow/Snakefile --cores 4 results/figures/<figure_id>.<ext>`
- Active provenance summary:
  - `docs/paper_figure_inventory.md`

## Figure-by-Figure Pipeline

### 1. `tradeoff-vis.png`

- Figure id:
  - `tradeoff-vis`
- Upstream source:
  - `notebooks/method-vis.ipynb`
- Maintained generator:
  - `analysis/plot_method_vis/run.py`
- Intermediate artifacts:
  - none required
- Canonical output:
  - `paper/generated/figures/tradeoff-vis.png`
- Build command:
  - `uv run snakemake -s workflow/Snakefile --cores 4 results/figures/tradeoff-vis.png`

### 2. `sgd-vs-full-batch.png`

- Figure id:
  - `sgd-vs-full-batch`
- Upstream source:
  - `notebooks/method-vis.ipynb`
- Maintained generator:
  - `analysis/plot_method_vis/run.py`
- Intermediate artifacts:
  - none required
- Canonical output:
  - `paper/generated/figures/sgd-vs-full-batch.png`
- Build command:
  - `uv run snakemake -s workflow/Snakefile --cores 4 results/figures/sgd-vs-full-batch.png`

### 3. `elasticnet_recovery_mean_se.pdf`

- Figure id:
  - `elasticnet_recovery_mean_se`
- Upstream experiment:
  - `analysis/elasticnet_train_and_recover/run.py`
- Active sweep config:
  - `data/provided/sweeps/elasticnet_sweep_config_beta1e3.yaml`
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
  - `analysis/plot_elasticnet_recovery/run.py`
- Top-level builder path:
  - `uv run snakemake -s workflow/Snakefile` defaults to sweep `9a7ll8aa`
- Canonical output:
  - `paper/generated/figures/elasticnet_recovery_mean_se.pdf`
- Local fallback:
  - `analysis/elasticnet_train_and_recover/helpers/run_figure_batch.py`
  - this is only a fallback if explicitly pointed at a suitable CSV
- Important non-source:
  - `archive/notebooks_legacy_2026-04/elasticnet_viz.ipynb`
  - archived sweep `krcszr6z`
  - this older notebook sweep is not the paper `mean/SE` source
- Build command:
  - `uv run snakemake -s workflow/Snakefile --cores 4 results/figures/elasticnet_recovery_mean_se.pdf`

### 4. `OLS_early_stopping_figure.pdf`

- Figure id:
  - `OLS_early_stopping_figure`
- Status:
  - preserved final manuscript asset
- Final paper asset:
  - `paper/generated/figures/OLS_early_stopping_figure.pdf`
- Reason preserved:
  - the final annotated panel was assembled externally rather than by one repo-native script
- Notebook lineage for automated component figures:
  - `notebooks/linear-regression.ipynb`
- Maintained component generator:
  - `analysis/plot_linear_regression_ols/run.py`
- Automated component outputs:
  - `paper/generated/figures/Lambda_comparison-ols.pdf`
  - `paper/generated/figures/predictive_weights_comparison_ols.pdf`
- Top-level builder behavior:
  - `uv run snakemake -s workflow/Snakefile --cores 4 results/figures/OLS_early_stopping_figure.pdf`
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
  - `analysis/plot_lambda_vs_epochs/run.py`
- Canonical output:
  - `paper/generated/figures/lambda_vs_epochs.pdf`
- Important note:
  - this script is active and automated, but heavier than the other figure builders because it reruns training internally
- Build command:
  - `uv run snakemake -s workflow/Snakefile --cores 4 results/figures/lambda_vs_epochs.pdf`

### 6. `dropout_bias_ridge_panel.png`

- Figure id:
  - `dropout_bias_ridge_panel`
- Upstream experiment:
  - `analysis/dropout_bias_estimation/run.py`
- Active sweep config:
  - `data/provided/sweeps/dropout_l2_bias.yaml`
- Confirmed W&B sweep:
  - `chiy2qjz`
- Notebook lineage:
  - `notebooks/l2_estimation_deep_ReLU.ipynb`
- Maintained generator:
  - `analysis/plot_dropout_bias_ridge_panel/run.py`
- Canonical output:
  - `paper/generated/figures/dropout_bias_ridge_panel.png`
- Important note:
  - this requires W&B access at build time
- Build command:
  - `uv run snakemake -s workflow/Snakefile --cores 4 results/figures/dropout_bias_ridge_panel.png`

### 7. `barrett_igr_figure2.pdf`

- Figure id:
  - `barrett_igr_figure2`
- Upstream experiment:
  - `analysis/barrett_igr_figure2/run.py`
- Expected intermediate result:
  - `data/generated/barrett_igr_figure2/results.pt`
- Maintained plotter:
  - `analysis/plot_barrett_igr_figure2/run.py`
- Canonical output:
  - `paper/generated/figures/barrett_igr_figure2.pdf`
- Build command:
  - `uv run snakemake -s workflow/Snakefile --cores 4 results/figures/barrett_igr_figure2.pdf`

### 8. `barrett_igr_long_horizon.pdf`

- Figure id:
  - `barrett_igr_long_horizon`
- Upstream experiment:
  - `analysis/barrett_igr_long_horizon/run.py`
- Expected intermediate results:
  - `data/generated/barrett_igr_long_horizon/synth_eta001.pt`
  - `data/generated/barrett_igr_long_horizon/synth_eta003.pt`
  - `data/generated/barrett_igr_long_horizon/mnist_tanh_eta001.pt`
  - `data/generated/barrett_igr_long_horizon/mnist_relu_eta001.pt`
- Maintained plotter:
  - `analysis/plot_barrett_igr_long_horizon/run.py`
- Canonical output:
  - `paper/generated/figures/barrett_igr_long_horizon.pdf`
- Build command:
  - `uv run snakemake -s workflow/Snakefile --cores 4 results/figures/barrett_igr_long_horizon.pdf`

## Additional Reproducible Paper-Figure Components

These are not currently direct `paper/main.tex` includes, but they are part of the maintained reproducible figure surface.

### OLS component figures

- `paper/generated/figures/Lambda_comparison-ols.pdf`
- `paper/generated/figures/predictive_weights_comparison_ols.pdf`
- Source:
  - `notebooks/linear-regression.ipynb`
- Maintained generator:
  - `analysis/plot_linear_regression_ols/run.py`

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
  - `uv run snakemake -s workflow/Snakefile`
- Preserved-asset export helper:
  - `scripts/export_notebook_figure.py`

## Practical Rules

- If a figure is in `paper/main.tex`, its canonical output should be under `paper/generated/figures/`.
- If a figure depends on experiment outputs, the experiment should write numeric artifacts to `results/` or W&B, and only the final plot step should write to `paper/generated/figures/`.
- If a final paper panel was assembled externally, keep that final panel preserved, but still automate the reproducible component plots when possible.
