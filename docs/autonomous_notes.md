# Autonomous Notes

Use this file as the default non-manuscript log for autonomous method, implementation, and experiment notes.

## Conventions

- Add dated entries when autonomous work changes methods, assumptions, or experiment execution.
- Prefer logging here or in another `docs/` note instead of writing those updates into `paper/`.
- Promote durable results into a more specific doc in `docs/` when the workstream becomes substantial.
- For analysis plotting, default to a single `PDF` output unless the user explicitly requests an additional export format.

## 2026-04-21

- Added empirical reproduction of Barrett & Dherin (2022) "Implicit Gradient Regularization" on branch `barrett-igr-reproduction`. New files:
  - [`src/core/igr_trajectory.py`](/home/jrudoler/inductive-bias/src/core/igr_trajectory.py): closed-form scalar-λ fitter for `R = (λ/p)·||∇L||²` and a `compute_full_batch_grad_and_hvp` helper.
  - [`experiments/barrett_igr_trajectory.py`](/home/jrudoler/inductive-bias/experiments/barrett_igr_trajectory.py): single-run trajectory collector with `--mode flow_ref` (GD at η compared to a k-substep near-flow reference) and `--mode sgd` (mini-batch residual target `-Δθ/η - g_full`). Supports `--dataset {synthetic, mnist}` and `--activation {relu, tanh, gelu}` for MNIST MLPs.
  - [`scripts/run_barrett_igr_sweep.py`](/home/jrudoler/inductive-bias/scripts/run_barrett_igr_sweep.py): local sweep driver covering synthetic (η × p × seed) and MNIST (arch × η × seed).
  - [`analysis/barrett_igr_plot.py`](/home/jrudoler/inductive-bias/analysis/barrett_igr_plot.py): produces [`figures/barrett_igr_reproduction.pdf`](/home/jrudoler/inductive-bias/figures/barrett_igr_reproduction.pdf).
- Target form used is `(Δθ_flow − Δθ_GD)/η`, not `(Δθ_GD − Δθ_flow)/η`. Derivation: a single GD step is `−η·g` exactly; one true-flow step over time η is `−η·g + (η²/2)·Hg + O(η³)`; dividing the gap by η gives `(η/2)·Hg`, which equals `∇R` at Barrett's `λ = η·p/4`. First smoke test had the sign flipped and produced `λ̂ ≈ −η·p/4` — fixed.
- Finite-substep bias: k sub-steps of size η/k accumulate their own `O(h²)` Euler error. The per-unit-time gap becomes `(η/2)·(1 − 1/k)·Hg`, so `λ̂ ≈ (ηp/4)·(1 − 1/k)`. Observed ratios across the synthetic sweep match this precisely (k=100 ⇒ ratio ≈ 0.989; k=10 ⇒ ratio ≈ 0.90). No additional correction applied in code; document the bias and use large k where affordable.
- Empirical findings (synthetic OLS, `n=1000`, `p ∈ {5,10,20}`, `η ∈ {1e-3, 3e-3, 1e-2, 3e-2}`, `k=100`, 3 seeds, 300 GD steps):
  - Flow-ref recovers `λ = ηp/4` with ratio ≈ 0.99 at small η, ≈ 0.97 at η = 0.03 (higher-order `O(η²)` terms). Residual ratio `||target − predicted||/||target|| ≤ 0.003` everywhere. Seed-to-seed variance < 0.001 in the ratio.
  - SGD mode (B ∈ {32, 128}, 400 steps): per-step residual `target = g_batch − g_full` is pure mini-batch sampling noise, orthogonal to `Hg` in expectation. Linear regression of target on `Hg` yields `residual_ratio ≈ 1` and sign-flipping λ̂ across seeds. **Conclusion:** single-step SGD residuals do NOT expose Barrett's IGR directly; they expose batch-sampling noise. To extract Barrett's effect from SGD would require epoch-averaged targets plus a noise-covariance model, out of scope for this reproduction.
- Empirical findings (MNIST, `n=2000 subsample`, hidden `[64,32]`, η ∈ {3e-4, 1e-3, 3e-3}, k=10, 2 seeds, 30 steps):
  - Linear logistic regression: ratio ≈ 0.895, residual_ratio ≤ 0.015.
  - Tanh MLP: ratio ≈ 0.900, residual_ratio ≤ 0.004. Matches 1 − 1/k = 0.9 exactly.
  - ReLU MLP: ratio ranges over 0.90–1.38 with residual_ratio ≈ 0.99 (fit explains ~nothing). The ReLU kinks break the backward-error smoothness assumption: the coarse GD step and the k-substep reference can cross different activation boundaries, and the resulting `(Δθ_flow − Δθ_GD)` is no longer aligned with `(η/2)·Hg`. Moral: Barrett's IGR formula applies cleanly to smooth losses/architectures; piecewise-linear activations violate the assumption.
- Limitations / next steps: (i) MNIST runs used `n=2000` subsample and `k=10` for CPU budget; bump to full 60k and `k ≥ 50` on GPU for a paper-grade figure; (ii) no formal uncertainty quantification on the fit (variance-of-ratio is tiny empirically but not bootstrapped); (iii) SGD residual framing is worth a separate deeper pass — possibly compare epoch-averaged `Δθ_SGD` to a flow reference; (iv) pre-existing `GradientSquaredPenaltyEstimator` in [`src/core/estimators.py`](/home/jrudoler/inductive-bias/src/core/estimators.py) uses the *analytic* `(η/2)·Hg` target; the new trajectory fitter complements it by using the observed `Δθ_flow − Δθ_GD`.
- Registered as `barrett_igr_trajectory` in [`experiments/REGISTRY.yaml`](/home/jrudoler/inductive-bias/experiments/REGISTRY.yaml).

## 2026-04-20

- Updated [`notebooks/method-vis.ipynb`](/home/jrudoler/inductive-bias/notebooks/method-vis.ipynb) with torch-based regression loss visualizations for the gradient-step-deviation section: the contour cell now follows a short minibatch trajectory and, at each iterate, overlays both the recomputed full-batch and minibatch autograd steps on the full-batch MSE landscape; the following 3D surface cell reuses the same multi-step trajectory on the full-batch loss surface.

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

## 2026-04-10

- Added [`notebooks/linear-regression-trajectory.ipynb`](/home/jrudoler/inductive-bias/notebooks/linear-regression-trajectory.ipynb) to implement Appendix B trajectory fitting variants for OLS early stopping.
- Implemented two rendered Figure 3 variants in the notebook: single-trajectory fitting and endpoint-only fitting across 10 runs, with outputs saved to stable PDF paths under [`figures/`](/home/jrudoler/inductive-bias/figures/).
- Updated the notebook plotting layout and y-axis scaling to avoid compressed/squashed visual outputs, and added a configurable burn-in (`BURN_IN = 50`) for the single-trajectory estimator and distance-vs-steps curve.
- Renamed [`notebooks/linear-regression-trajectory.ipynb`](/home/jrudoler/inductive-bias/notebooks/linear-regression-trajectory.ipynb) to [`notebooks/linear-regression-trajectory-selfcontained.ipynb`](/home/jrudoler/inductive-bias/notebooks/linear-regression-trajectory-selfcontained.ipynb), and created a new [`notebooks/linear-regression-trajectory.ipynb`](/home/jrudoler/inductive-bias/notebooks/linear-regression-trajectory.ipynb) that reuses `src/core/utils.py` (`compute_Q_matrix`, `compute_beta_closed_form`) while keeping old Fig.3 hyperparameters and update rule.
- Updated [`notebooks/linear-regression-trajectory.ipynb`](/home/jrudoler/inductive-bias/notebooks/linear-regression-trajectory.ipynb) to reintroduce Lightning-based training with `EarlyStopping(monitor="train/loss", patience=5, mode="min")`, matching the old `linear-regression.ipynb` setup; trajectory states are now collected from Lightning epochs via a callback.
- Updated the trajectory notebook to run the single-trajectory experiment with `SINGLE_MAX_EPOCHS = P * MAX_EPOCHS`, while keeping endpoint-only multi-run fits at `MAX_EPOCHS`; also removed absolute path printing from notebook outputs.
- Reworked the trajectory notebook to restore the 10-trajectory endpoint variant and to align single-trajectory theory with `linear-regression.ipynb` by using `k = trainer.current_epoch` with `max_epochs=500`; added an explicit coupled (no-early-stopping) run check under the same seed/data.
- Extended `core.estimators` with core-level diagonal-ridge fitting utilities (`fit_diag_matrix_ridge_closed_form`, `BiasFromTargets`) and refactored the trajectory notebook to use `core.estimators` + `core.bias` instead of notebook-local fitting code; this also fixed the gradient-scaling convention by fitting against the full MSE gradient (factor 2), matching the theoretical `DiagMatrixRidgeBias` gradient parameterization.
- Added `BiasWithMSETrajectory` in [`src/core/estimators.py`](/home/jrudoler/inductive-bias/src/core/estimators.py), a minimal `BiasWithMSE` extension for fitting diagonal-ridge bias models directly from precomputed trajectory `(theta, target-gradient)` batches.
- Updated [`notebooks/linear-regression-trajectory.ipynb`](/home/jrudoler/inductive-bias/notebooks/linear-regression-trajectory.ipynb) to keep optimizer-based fitting throughout (no closed-form fitting path), use 5000 epochs for the single-trajectory bias fit and 500 epochs for multi-endpoint fits, and stabilize prefix-distance curves with warm-started multi-step Adam updates plus gradient clipping.
- Reworked [`notebooks/linear-regression-trajectory.ipynb`](/home/jrudoler/inductive-bias/notebooks/linear-regression-trajectory.ipynb) to follow Appendix B coupled-trajectory targets: build paired early-stopped and non-early trajectories per run, compute regression targets from their per-step update-direction differences, compute theory `Q` at the early-stop step, and fit both single-trajectory and endpoint-only estimators against those coupled targets.
- Fixed a time-index mismatch in [`notebooks/linear-regression-trajectory.ipynb`](/home/jrudoler/inductive-bias/notebooks/linear-regression-trajectory.ipynb): endpoint fits now use the non-early trajectory at the early-stop index (not the terminal `T` endpoint), and trajectory-match distances are compared to theory over the same post-stop window (`EARLY_STOP_STEP+1..m`) used in fitting.
