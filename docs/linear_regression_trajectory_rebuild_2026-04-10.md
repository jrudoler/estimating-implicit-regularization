# Linear Regression Trajectory Rebuild (2026-04-10)

- Updated [`notebooks/linear-regression-trajectory.ipynb`](/home/kevin/OneDrive/Documents/UMich/Research/inductive-bias/notebooks/linear-regression-trajectory.ipynb) to keep callback-based stopping with `EarlyStopping(monitor="train/loss", patience=5, min_delta=1e-3, mode="min")`.
- The single-run fit remains the trajectory-wide fit the notebook currently uses.
- Replaced the right-hand weight-vector `imshow` panel with the three-column seaborn heatmap layout from `linear-regression.ipynb`, including a shared horizontal colorbar.
- Figure cells continue to render inline in the notebook with a single visible output per cell.

## Executed notebook results

- Callback-selected stop steps across 10 runs: `[523, 492, 438, 487, 467, 484, 450, 504, 443, 541]`
- Single-run callback-selected stop step: `523`
- Max explicit-vs-trained mismatch at the callback stop step: `6.675720e-06`
- Single-run trajectory diagonal distance at callback-selected `k_*`: `3.403221`
- Single-run trajectory matrix distance to full `Q_{k_*}`: `3.201725`
- 10-run stacked diagonal distance to the weighted theory target: `0.129830`
- 10-run stacked matrix distance to the median full theory target: `0.342909`
- Callback stop-step median `[IQR]`: `485.5 [454.2, 501.0]`
- Median trajectory distance at `m = 1` `[IQR]`: `11112.886719 [8244.173828, 12944.708984]`
- Trajectory distance at each run's callback-selected `k_*` `[IQR]`: `15.253193 [12.985670, 19.170626]`
- Median trajectory distance at final plotted `m` `[IQR]`: `2.032479 [1.283569, 2.595150]`

## Regenerated outputs

- [`figures/linear_regression_trajectory_fig3_single_trajectory.pdf`](/home/kevin/OneDrive/Documents/UMich/Research/inductive-bias/figures/linear_regression_trajectory_fig3_single_trajectory.pdf)
- [`figures/linear_regression_trajectory_fig3_10run_endpoint.pdf`](/home/kevin/OneDrive/Documents/UMich/Research/inductive-bias/figures/linear_regression_trajectory_fig3_10run_endpoint.pdf)
- [`figures/linear_regression_trajectory_distance_to_theory.pdf`](/home/kevin/OneDrive/Documents/UMich/Research/inductive-bias/figures/linear_regression_trajectory_distance_to_theory.pdf)
