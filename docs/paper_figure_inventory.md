# Paper Figure Inventory

This file is the active provenance map for figures referenced by [`paper/main.tex`](/home/jrudoler/inductive-bias/paper/main.tex).

| Figure in `paper/main.tex` | Classification | Active source workflow | Canonical output |
| --- | --- | --- | --- |
| `figures/tradeoff-vis.png` | preserved manuscript asset | Notebook-derived construction in [`notebooks/method-vis.ipynb`](/home/jrudoler/inductive-bias/notebooks/method-vis.ipynb); exact current asset is preserved in `paper/figures` during this cleanup pass | [`paper/figures/tradeoff-vis.png`](/home/jrudoler/inductive-bias/paper/figures/tradeoff-vis.png) |
| `figures/sgd-vs-full-batch.png` | notebook-generated | [`notebooks/method-vis.ipynb`](/home/jrudoler/inductive-bias/notebooks/method-vis.ipynb) with CLI export helper [`scripts/export_notebook_figure.py`](/home/jrudoler/inductive-bias/scripts/export_notebook_figure.py) | [`paper/figures/sgd-vs-full-batch.png`](/home/jrudoler/inductive-bias/paper/figures/sgd-vs-full-batch.png) |
| `figures/elasticnet_recovery_mean_se.pdf` | experiment + plot-script generated | [`experiments/elasticnet_train_and_recover.py`](/home/jrudoler/inductive-bias/experiments/elasticnet_train_and_recover.py) -> [`scripts/run_elasticnet_figure_batch.py`](/home/jrudoler/inductive-bias/scripts/run_elasticnet_figure_batch.py) -> [`scripts/plot_elasticnet_recovery.py`](/home/jrudoler/inductive-bias/scripts/plot_elasticnet_recovery.py) | [`paper/figures/elasticnet_recovery_mean_se.pdf`](/home/jrudoler/inductive-bias/paper/figures/elasticnet_recovery_mean_se.pdf) |
| `figures/OLS_early_stopping_figure.pdf` | preserved manuscript asset | Current paper asset is preserved; related OLS early-stopping notebook lineage remains in [`notebooks/linear-regression-trajectory.ipynb`](/home/jrudoler/inductive-bias/notebooks/linear-regression-trajectory.ipynb) and [`notebooks/linear-regression-trajectory-selfcontained.ipynb`](/home/jrudoler/inductive-bias/notebooks/linear-regression-trajectory-selfcontained.ipynb) | [`paper/figures/OLS_early_stopping_figure.pdf`](/home/jrudoler/inductive-bias/paper/figures/OLS_early_stopping_figure.pdf) |
| `figures/lambda_vs_epochs.pdf` | script-generated | [`scripts/plot_lambda_vs_epochs.py`](/home/jrudoler/inductive-bias/scripts/plot_lambda_vs_epochs.py) | [`paper/figures/lambda_vs_epochs.pdf`](/home/jrudoler/inductive-bias/paper/figures/lambda_vs_epochs.pdf) |
| `figures/dropout_bias_ridge_panel.png` | notebook-derived lineage, preserved manuscript asset | Notebook provenance in [`notebooks/l2_estimation_deep_ReLU.ipynb`](/home/jrudoler/inductive-bias/notebooks/l2_estimation_deep_ReLU.ipynb); exact manuscript asset is preserved during this cleanup pass | [`paper/figures/dropout_bias_ridge_panel.png`](/home/jrudoler/inductive-bias/paper/figures/dropout_bias_ridge_panel.png) |
| `figures/barrett_igr_figure2.pdf` | experiment + plot-script generated | [`experiments/barrett_igr_figure2.py`](/home/jrudoler/inductive-bias/experiments/barrett_igr_figure2.py) -> [`analysis/barrett_igr_figure2_plot.py`](/home/jrudoler/inductive-bias/analysis/barrett_igr_figure2_plot.py) | [`paper/figures/barrett_igr_figure2.pdf`](/home/jrudoler/inductive-bias/paper/figures/barrett_igr_figure2.pdf) |
| `figures/barrett_igr_long_horizon.pdf` | experiment + plot-script generated | [`experiments/barrett_igr_long_horizon.py`](/home/jrudoler/inductive-bias/experiments/barrett_igr_long_horizon.py) -> [`analysis/barrett_igr_long_horizon_plot.py`](/home/jrudoler/inductive-bias/analysis/barrett_igr_long_horizon_plot.py) | [`paper/figures/barrett_igr_long_horizon.pdf`](/home/jrudoler/inductive-bias/paper/figures/barrett_igr_long_horizon.pdf) |

## Build Entry Point

- Use [`scripts/build_paper_figures.py`](/home/jrudoler/inductive-bias/scripts/build_paper_figures.py) as the top-level paper figure builder.
- Script-backed figures are regenerated when prerequisites exist.
- Preserved assets are verified or copied into `paper/figures`.
- For SLURM-backed or multi-stage experiments, raw numeric outputs stay under `results/` or `artifacts/`; only the final plotting step writes to `paper/figures`.
