# Nonlinear Multi-Geometry Replicate Ablation (2026-03-30)

## Setup

- Script: [experiments/nonlinear_multi_geometry_replicate_ablation.py](/home/jrudoler/inductive-bias/experiments/nonlinear_multi_geometry_replicate_ablation.py)
- Launcher: [scripts/nonlinear_multi_geometry_replicate_ablation.slurm](/home/jrudoler/inductive-bias/scripts/nonlinear_multi_geometry_replicate_ablation.slurm)
- Data/model defaults matched the earlier GPU suite: `n_samples=256`, `input_dim=12`, `depth=2`, `width=32`, `max_epochs=350`, `patience=60`, `target_gradient_scale=0.3`.
- Resampling mode was `bootstrap`. Each fit uses the full-data solution plus `n_replicates` additional bootstrap retrains.
- Requested counts were `1, 10, 50, 100, 1000` for five cases. The `multi_l1_l2_nuclear` case was rerun with `1, 10, 50, 100` only after the `n=1000` solve turned out to be disproportionately expensive relative to the information gained.

## Aggregate Artifacts

- Aggregate summary CSV: [summary.csv](/home/jrudoler/inductive-bias/logs/phase2/nonlinear_multi_geometry_replicate_ablation_gpu_scale03/summary.csv)
- Aggregate component CSV: [component_recovery.csv](/home/jrudoler/inductive-bias/logs/phase2/nonlinear_multi_geometry_replicate_ablation_gpu_scale03/component_recovery.csv)
- Aggregate figure: [summary.pdf](/home/jrudoler/inductive-bias/figures/nonlinear_multi_geometry_replicate_ablation_gpu_scale03/summary.pdf)

Per-case figures:

- [single_l2.pdf](/home/jrudoler/inductive-bias/figures/nonlinear_multi_geometry_replicate_ablation_single_l2_gpu_scale03/single_l2.pdf)
- [single_l1.pdf](/home/jrudoler/inductive-bias/figures/nonlinear_multi_geometry_replicate_ablation_single_l1_gpu_scale03/single_l1.pdf)
- [single_nuclear.pdf](/home/jrudoler/inductive-bias/figures/nonlinear_multi_geometry_replicate_ablation_single_nuclear_gpu_scale03/single_nuclear.pdf)
- [multi_l1_l2.pdf](/home/jrudoler/inductive-bias/figures/nonlinear_multi_geometry_replicate_ablation_multi_l1_l2_gpu_scale03/multi_l1_l2.pdf)
- [multi_l2_nuclear.pdf](/home/jrudoler/inductive-bias/figures/nonlinear_multi_geometry_replicate_ablation_multi_l2_nuclear_gpu_scale03/multi_l2_nuclear.pdf)
- [multi_l1_l2_nuclear.pdf](/home/jrudoler/inductive-bias/figures/nonlinear_multi_geometry_replicate_ablation_multi_l1_l2_nuclear_gpu_scale03/multi_l1_l2_nuclear.pdf)

## Main Result

Increasing the number of bootstrap retrains did **not** reliably improve identifiability in this nonlinear geometry-recovery setting.

- `single_l1` improved slightly with more replicates.
- `single_l2`, `single_nuclear`, `multi_l1_l2`, and `multi_l2_nuclear` were effectively flat or slightly worse at `n=1000` than at `n=1`.
- `multi_l1_l2_nuclear` improved from `n=10` to `n=100`, but its best cosine/residual was still at `n=1`, while component-wise `(lambda, p)` errors remained large.

## Case-Level Summary

- `single_l2`: cosine `0.9477 -> 0.9422`, residual `0.3191 -> 0.3354` from `n=1` to `n=1000`.
- `single_l1`: cosine `0.7830 -> 0.7943`, residual `0.6221 -> 0.6076` from `n=1` to `n=1000`.
- `single_nuclear`: cosine `0.8450 -> 0.8379`, residual `0.5348 -> 0.5459` from `n=1` to `n=1000`.
- `multi_l1_l2`: cosine `0.9377 -> 0.9295`, residual `0.3474 -> 0.3691` from `n=1` to `n=1000`.
- `multi_l2_nuclear`: cosine `0.9536 -> 0.9442`, residual `0.3010 -> 0.3296` from `n=1` to `n=1000`.
- `multi_l1_l2_nuclear`: cosine `0.9556 -> 0.9410`, residual `0.2946 -> 0.3383` from `n=1` to `n=100` only.

## Interpretation

The results suggest that the dominant limitation is not lack of replicate count. Adding more bootstrap retrains appears to reduce the estimator's ability to overfit a small collection of solutions, but it does not remove the mismatch between the fitted regularizer family and the actual geometry induced by nonlinear training. In the mixed-family settings, more replicates can even make that mismatch more visible, producing slightly worse cosine/residual metrics while only modestly helping some component-wise parameter errors.
