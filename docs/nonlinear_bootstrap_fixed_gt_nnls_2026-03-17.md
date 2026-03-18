# Nonlinear Bootstrap With Fixed GT And NNLS (2026-03-17)

This note records the corrected nonlinear bootstrap train-only experiment after fixing two issues in the earlier analysis:

1. ground-truth regularization parameters are now held fixed across seeds
2. coefficient recovery uses nonnegative least squares (NNLS) rather than unconstrained OLS

## Setup

Script:
- `experiments/nonlinear_resampling_retrain.py`

Protocol:
- bootstrap resampling with replacement
- bootstrap sample size equal to the original training split
- recovery uses training-data gradients only
- teacher activation: `relu`
- teacher spectrum: `identity`
- student depth: `3`
- student width: `64`
- bias pair: `ridge,nuclear_norm`
- `n_replicates=6`
- solver: `nnls`

## Important Estimator Note

This corrected retrain-per-bootstrap study does **not** use normalized gradients.

For stacked shared-lambda estimation, each bootstrap replicate has a different local design matrix and different bias-gradient norms. Normalizing each replicate's bias gradients would change the mapping between normalized coefficients and the underlying shared `lambda`, so a single normalized coefficient vector is not the correct shared parameterization across replicates.

Accordingly, the corrected stacked study uses raw-gradient NNLS.

## Fixed GT Settings

Base pair:
- ridge: `0.003`
- nuclear_norm: `0.004`

Scales tested:
- `0.5`: `(0.0015, 0.0020)`
- `1.0`: `(0.0030, 0.0040)`
- `2.0`: `(0.0060, 0.0080)`

Each setting was averaged across 5 seeds.

## Relative Error Summary

Scale `0.5`:
- full: `1.3239`
- single: `1.1425`
- aggregate: `0.8480`
- stacked: `0.7293`
- stacked beats full on `5/5` seeds

Scale `1.0`:
- full: `1.2022`
- single: `1.0497`
- aggregate: `0.8487`
- stacked: `0.8099`
- stacked beats full on `4/5` seeds

Scale `2.0`:
- full: `0.8739`
- single: `0.9702`
- aggregate: `0.8146`
- stacked: `0.8190`
- stacked beats full on `2/5` seeds

Interpretation:
- bootstrap aggregation helps most at smaller GT scales
- the improvement weakens as the true regularization scale grows
- `aggregate` and `stacked` are close; most of the gain comes from averaging over bootstrap retrains rather than a large extra benefit from the joint stacked solve

## Recovered Coefficients

A smaller companion run averaged coefficient estimates across 3 seeds.

Scale `0.5`, true `(ridge, nuclear) = (0.0015, 0.0020)`:
- full: `(0.00197, 0.00001)`
- aggregate: `(0.00229, 0.00003)`
- stacked: `(0.00184, 0.00002)`

Scale `1.0`, true `(0.0030, 0.0040)`:
- full: `(0.00426, 0.00009)`
- aggregate: `(0.00498, 0.00011)`
- stacked: `(0.00491, 0.00009)`

Scale `2.0`, true `(0.0060, 0.0080)`:
- full: `(0.00975, 0.00039)`
- aggregate: `(0.01024, 0.00042)`
- stacked: `(0.01023, 0.00042)`

Interpretation:
- the dominant failure mode is still systematic under-recovery of `nuclear_norm`
- bootstrap does not recover the nuclear coefficient accurately; it mainly stabilizes the ridge estimate and slightly improves overall relative error
- the nuclear estimate increases with GT scale, but remains far too small in absolute terms

## Bottom Line

After fixing GT drift and enforcing positivity, the nonlinear bootstrap train-only story is cleaner:
- yes, bootstrap retraining can improve coefficient recovery relative to a single full-data solve, especially at smaller GT scales
- no, it does not fix the main structural failure in this nonlinear setup
- the method still assigns most of the recovered signal to `ridge`, with `nuclear_norm` substantially suppressed
