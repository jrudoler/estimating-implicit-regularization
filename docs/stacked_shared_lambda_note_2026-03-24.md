# Stacked Shared-Lambda Note (2026-03-24)

## What Was Missing Before
In the nonlinear retrain bootstrap experiment, we previously had:
- `single`: fit one estimator separately on each retrained replicate
- `aggregate`: average the replicate-specific recovered lambdas

This is **not** the same as a true shared-parameter stacked fit.

For replicate `r`, write the local equation as

`b_r ≈ A_r λ`

where:
- `b_r` is the target gradient at replicate `r`
- `A_r` is the replicate-specific bias-gradient design
- `λ` is the shared regularization vector

The previous `aggregate` estimator computed

`λ_hat_agg = (1 / R) Σ_r λ_hat_r`

where each `λ_hat_r` was fit independently.

That does **not** solve the joint problem

`λ_hat_stack = argmin_λ Σ_r ||A_r λ - b_r||^2`

unless the replicate systems are effectively identical.

## What Was Implemented Now
A true stacked shared-`lambda` estimator was added to:
- `experiments/nonlinear_resampling_retrain.py`

The script now explicitly builds the concatenated system

`b_stack = [b_1; ...; b_R]`

`A_stack = [A_1; ...; A_R]`

and fits one shared parameter vector against that stacked system.

Two modes are supported:

### 1. Standard stacked
Shared positive coefficients are fit directly in raw space.

### 2. Normalized stacked
A single global column normalization is computed from the concatenated design:

`S = diag(||A_stack[:,1]||, ..., ||A_stack[:,p]||)`

Then the fit is done in normalized coordinates using the whole stacked system, not replicate-specific normalizations.

This is the mathematically coherent version of normalization for a true stacked shared-parameter estimator.

## Current Experiment
Output:
- `logs/phase2/nonlinear_retrain_stacked_global_norm_comparison.csv`

Setting:
- nonlinear retrain-per-bootstrap experiment
- teacher: ReLU, identity spectrum
- student: depth 3, width 64
- `n_samples=2048`
- `n_replicates=6`
- ground truth: `ridge=0.003`, `nuclear_norm=0.004`
- seeds: `42, 123, 456`

Compared:
- `uniform_standard`
- `mixed_standard`
- `uniform_normalized`
- `mixed_normalized`

## Main Results
Across-seed means:

### Standard mode
- `uniform_standard`
  - aggregate: `3.9690`
  - stacked: `0.9132`
- `mixed_standard`
  - aggregate: `3.9604`
  - stacked: `0.9144`

### Normalized mode
- `uniform_normalized`
  - aggregate: `0.7932`
  - stacked: `1.8315`
- `mixed_normalized`
  - aggregate: `0.8556`
  - stacked: `1.8750`

## Interpretation
1. The true stacked shared-parameter fit matters.
   - In standard mode, moving from `aggregate` to `stacked` reduces mean relative error from about `4.0` to about `0.91`.
   - This is a large and stable gain across all three seeds.

2. The sampling tweak is secondary once stacking is used.
   - `uniform_standard` stacked: `0.9132`
   - `mixed_standard` stacked: `0.9144`
   - So the main gain is from joint shared fitting, not the mixture sampler.

3. Global normalization on the stacked system is coherent but does not help in the current implementation.
   - The normalized stacked estimator is consistently worse than normalized aggregation.
   - So the idea is mathematically clean, but the current optimizer / parameterization is not yet the right practical estimator.

## Current Bottom Line
The important new positive result is:
- a true stacked shared-`lambda` estimator helps substantially in the nonlinear retrain setting

The important negative result is:
- globally normalized stacked fitting, as currently implemented, is worse than normalized aggregation

So the next practical baseline should be:
- `uniform_standard` with true stacking
- `uniform_normalized` with aggregation

Those are currently the two strongest estimators in this experiment family.
