# Targeted Bootstrap Retrain Summary (2026-03-18)

## Scope
This note records the current state of the targeted-bootstrap experiments after moving back to the setting that matters scientifically:
- retrain a new predictive model for each bootstrap replicate
- use the bias estimator classes rather than an exact OLS solve
- allow the learned weights, and therefore the local bias-gradient geometry, to change across replicates

The main script for this path is:
- `experiments/nonlinear_resampling_retrain.py`

## Current Default Protocol
The retrain script now defaults to:
- `sampling_policy=weak_direction`
- `sampling_score_mode=abs`
- `importance_correct=True`
- `estimator_mode=standard`
- `resample_mode=bootstrap`
- `sample_fraction=1.0`

So the current default is:
1. train a full nonlinear model on the original dataset
2. compute the weak direction from the full-model bias-gradient design
3. construct bootstrap sampling probabilities from the absolute directional loss-gradient score
4. sample full-size bootstrap replicates with replacement
5. apply inverse-probability importance weights during both predictive-model retraining and bias-estimator fitting
6. estimate replicate-specific lambdas with the estimator classes and average across replicates

## Fixed-Weight Diagnostic Result
Before returning to retraining, a fixed-weight diagnostic was run to check the sampling rule itself.

Output:
- `logs/phase2/nonlinear_targeted_bootstrap_abs_large.csv`

Setting:
- nonlinear fixed-model experiment
- `n_samples=8192`
- score mode `abs`

Across 3 seeds:
- uniform bootstrap aggregate error: `0.6403`
- targeted `abs` without correction: `13.4100`
- targeted `abs` with importance correction: `0.6223`

Interpretation:
- uncorrected targeting badly changes the target equation
- importance correction is necessary
- with correction, the targeted sampler becomes competitive with uniform bootstrap in the fixed-weight setting

This was only a diagnostic. It does not test the geometry-change mechanism because the predictive weights are frozen.

## Retrain + Estimator Result
Output:
- `logs/phase2/nonlinear_retrain_targeted_importance_comparison.csv`

Setting:
- nonlinear retrain-per-bootstrap experiment
- teacher: ReLU, identity spectrum
- student: depth 3, width 64
- ground truth: `ridge=0.003`, `nuclear_norm=0.004`
- `n_samples=2048`
- `n_replicates=6`
- estimator: `standard`
- comparison: `uniform` vs `targeted_abs_importance`
- seeds: `42, 123, 456`

Across-seed means:
- full-data estimate: `4.1255`
- uniform bootstrap aggregate: `3.9690`
- targeted bootstrap aggregate: `3.9897`

Per seed:
- seed `42`
  - uniform: `4.0308`
  - targeted: `3.8055`
- seed `123`
  - uniform: `3.9567`
  - targeted: `4.0415`
- seed `456`
  - uniform: `3.9196`
  - targeted: `4.1220`

So targeted importance sampling beat uniform on only `1/3` seeds in this run and was slightly worse on average.

## Geometry / ESS Diagnostics
Mean diagnostics across the same 3-seed retrain comparison:

Uniform bootstrap:
- replicate pairwise cosine: `0.2897`
- ESS fraction: `1.0000`
- replicate train MSE: `0.0073`
- replicate test MSE: `0.0243`

Targeted bootstrap with importance correction:
- replicate pairwise cosine: `0.3090`
- ESS fraction: `0.2488`
- replicate train MSE: `0.0381`
- replicate test MSE: `0.0371`

Interpretation:
- the targeted sampler does change the retrained models
- but in this run it did not improve the local geometry in the intended way
- replicate pairwise cosine actually increased slightly on average
- the importance weights reduce effective sample size substantially
- replicate train/test fit is worse under targeted weighted retraining

## What We Have Learned So Far
1. Fixed-weight target-only resampling and retrain-per-bootstrap are different experiments.
   - Fixed-weight resampling isolates target-gradient variance effects.
   - Retrain-per-bootstrap is the correct setting for testing whether geometry changes across retrained models help identifiability.

2. Uncorrected targeted sampling is not acceptable if the estimand is the original learning rule.
   - It changes the objective too aggressively.

3. Importance correction is necessary for the targeted scheme to be coherent.
   - In the fixed-weight diagnostic, this was the difference between failure and near-parity with uniform bootstrap.

4. In the nonlinear retrain setting, targeted importance sampling does not yet help.
   - On the current 3-seed run, it is slightly worse than uniform bootstrap on average.
   - The likely reason is that the targeting rule is not yet producing more complementary replicate designs; instead it is lowering effective sample size and degrading fit.

5. Using the estimator classes instead of exact OLS did not by itself rescue the targeted retrain idea.
   - The current standard estimator remains inaccurate in absolute terms for both uniform and targeted bootstrap.

## Current Bottom Line
The targeted weak-direction bootstrap idea is mathematically coherent only with importance correction, but in the nonlinear retrain setting tested here it does not currently improve bias recovery over ordinary uniform bootstrap.

The main remaining question is not whether targeting can change the replicate models. It can. The question is whether we can design a targeting rule that changes them in a direction that improves identifiability instead of just reducing effective sample size.

## Most Sensible Next Steps
1. Mix targeted and uniform sampling:
   - `pi = alpha / n + (1 - alpha) q_targeted`
   - this should increase ESS while preserving some directional targeting
2. Target a multi-dimensional weak subspace instead of a single weak direction
3. Compare `standard` and `normalized` estimators under the same retrain-targeted protocol
4. Log replicate-specific recovered lambdas directly for the targeted retrain runs to distinguish variance reduction from systematic bias
