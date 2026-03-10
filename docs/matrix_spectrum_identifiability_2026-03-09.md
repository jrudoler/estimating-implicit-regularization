# Matrix Spectrum Identifiability Summary

Date: 2026-03-09

## Scope

This note summarizes the synthetic identifiability work completed after the structured ReLU study, focusing on a cleaner multi-output linear control where local bias-gradient geometry can be interpreted directly.

## What Was Added

- Structured synthetic ReLU identifiability study for norm-based regularizers.
- Multi-output linear matrix-spectrum control experiment:
  - [matrix_spectrum_identifiability.py](/home/jrudoler/inductive-bias/experiments/matrix_spectrum_identifiability.py)
  - [run_matrix_spectrum_pairs.py](/home/jrudoler/inductive-bias/experiments/run_matrix_spectrum_pairs.py)
- Additional bias families exposed through the shared bias factory:
  - `stable_rank`
  - `spectral_entropy`

Relevant commits:

- `bdcede0` `Add structured ridge-vs-nuclear identifiability study`
- `ab31690` `Add matrix-spectrum identifiability controls`
- `9396e37` `Fix normalized MSE scaling and extend matrix controls`

## Main Experimental Progression

### 1. Structured ReLU synthetic study

The first controlled study used a one-hidden-layer ReLU teacher/student setup with configurable input covariance spectrum and teacher first-layer singular spectrum.

What it showed:

- The raw empirical geometry matched the theory:
  - flatter singular spectra made `ridge` and `nuclear_norm` gradients more collinear
- But end-to-end lambda recovery in the ReLU student did not track that trend cleanly
- Predictive fit and gradient-matching loss were both reasonable, so the failure was not just optimization collapse

Interpretation:

- The ReLU parameterization and training dynamics blur the direct matrix-spectrum effect
- Pairwise cosine alone was not enough to explain recovery error in the nonlinear setup

### 2. Linear and 2-layer linear student controls

These were tried and then removed because they did not answer the right question.

What happened:

- Scalar-output pure linear student was degenerate for spectral questions
  - the effective weight is `1 x d`, so it has only one singular value
  - `ridge` and `nuclear_norm` become trivially collinear
- 2-layer linear factorization still washed out the teacher-spectrum effect

Conclusion:

- The right clean control is not scalar-output linear regression
- It should be a multi-output matrix problem where the analyzed parameter really has a nontrivial singular spectrum

### 3. Multi-output linear matrix-spectrum control

The current clean control uses a multi-output linear student `Y = XW` with:

- configurable input covariance spectrum
- configurable teacher singular spectrum
- empirical bias gradients computed with autograd on the full train split
- full-batch `LBFGS` training so the regularized objective is near stationarity

This experiment also logs:

- pairwise cosine between candidate bias gradients
- local condition number of the selected bias-gradient system
- exact OLS coefficient recovery from empirical gradients
- noisy OLS recovery under small perturbations to the target gradient

## Main Findings

### Ridge vs nuclear norm

This pair now shows the expected identifiability trend in the multi-output linear control.

From [matrix_spectrum_pairs_summary.md](/home/jrudoler/inductive-bias/logs/phase2/matrix_spectrum_pairs_summary.md):

- `ridge__nuclear_norm`, `identity`
  - cosine `1.0000`
  - normalized estimator mean relative error `0.1857`
  - noisy OLS mean relative error `0.0160`
  - condition number about `5.25e4`
- `ridge__nuclear_norm`, `spiked`
  - cosine `0.6034`
  - normalized estimator mean relative error `0.0002`
  - noisy OLS mean relative error `0.0001`
  - condition number about `4.04`
- `ridge__nuclear_norm`, `power_law`
  - cosine `0.5761`
  - normalized estimator mean relative error `0.0001`
  - noisy OLS mean relative error `0.0002`
  - condition number about `3.72`

Interpretation:

- Flat singular spectra make `ridge` and `nuclear_norm` gradients nearly identical
- That does not prevent exact noiseless recovery at a true optimum
- It does make recovery much more sensitive to small perturbations
- In this setting, higher collinearity does correspond to worse local identifiability

### Ridge vs stable rank

This pair is robust across all tested spectra.

- cosine is essentially `0`
- condition number is `1`
- normalized estimator and OLS recovery are both near exact

Interpretation:

- `stable_rank` is scale-invariant, so its gradient is orthogonal to the ridge direction
- This makes the pair cleanly distinguishable in the local linear system

### Ridge vs spectral entropy

This pair is also mostly robust, but it shows a useful nuance.

- cosine is essentially `0` in all tested regimes
- `spiked` and `power_law` are recovered almost exactly
- `identity` is slightly less stable than `stable_rank`
  - noisy OLS mean relative error `0.0045`

Interpretation:

- Low collinearity is not the whole story
- `spectral_entropy` is also scale-invariant, so it stays orthogonal to ridge
- But near a flat spectrum the entropy gradient becomes small because entropy is near its maximum
- So identifiability depends on conditioning of the whole bias-gradient system, not only pairwise cosine

## Normalized Gradient Matching: What Was Wrong

The normalized estimator fits coefficients in a normalized basis:

- `g_i = ∇R_i`
- `g_i / ||g_i||`

Then it maps the fitted normalized coefficients back to the original lambdas by dividing by `||g_i||`.

This idea is fine, but it only works if the target loss gradient uses the exact same reduction as the predictive loss.

Bug that was found:

- For vector-output MSE, the normalized estimator was averaging over samples but not over output coordinates
- That made the target gradient too large by a factor equal to `output_dim`
- In the matrix control, `output_dim = 8`, so recovered lambdas were about `8x` too large
- This is why the mean relative error was previously around `7`

Fix:

- Updated MSE target-gradient scaling in:
  - [estimators.py](/home/jrudoler/inductive-bias/src/core/estimators.py)
  - [function_class_identifiability.py](/home/jrudoler/inductive-bias/experiments/function_class_identifiability.py)

Effect of the fix:

- In the matrix control, the normalized estimator now agrees closely with exact OLS in all well-conditioned regimes
- The remaining error is concentrated where the recovery problem is actually ill-conditioned

## Best Current Interpretation

The current evidence supports the following:

1. Raw bias-gradient collinearity behaves as expected under controlled singular spectra.
2. In nonlinear ReLU students, that geometry does not cleanly translate into coefficient identifiability.
3. In the clean multi-output linear control, the expected trend does appear:
   - more collinear `ridge`/`nuclear_norm` gradients lead to less identifiable coefficients
4. Exact recovery in noiseless arithmetic is not the right test by itself.
   - the better diagnostic is sensitivity to perturbations
5. Pairwise cosine is useful but incomplete.
   - full conditioning also depends on gradient magnitudes and weak directions

## Files and Outputs

Primary code:

- [matrix_spectrum_identifiability.py](/home/jrudoler/inductive-bias/experiments/matrix_spectrum_identifiability.py)
- [run_matrix_spectrum_pairs.py](/home/jrudoler/inductive-bias/experiments/run_matrix_spectrum_pairs.py)
- [estimators.py](/home/jrudoler/inductive-bias/src/core/estimators.py)
- [function_class_identifiability.py](/home/jrudoler/inductive-bias/experiments/function_class_identifiability.py)

Primary outputs:

- [matrix_spectrum_pairs.csv](/home/jrudoler/inductive-bias/logs/phase2/matrix_spectrum_pairs.csv)
- [matrix_spectrum_pairs_summary.md](/home/jrudoler/inductive-bias/logs/phase2/matrix_spectrum_pairs_summary.md)
- [structured_ridge_nuclear_matrix_large.csv](/home/jrudoler/inductive-bias/logs/phase2/structured_ridge_nuclear_matrix_large.csv)
- [structured_ridge_nuclear_summary_large.md](/home/jrudoler/inductive-bias/logs/phase2/structured_ridge_nuclear_summary_large.md)

## Suggested Next Steps

1. Extend the matrix control to a few more regularizer pairs with interpretable geometry.
   - Good candidates: `ridge` vs `orthogonal`, `ridge` vs `spectral_gap`
2. Add explicit diagnostics beyond pairwise cosine to all identifiability summaries.
   - smallest singular value of the selected bias-gradient system
   - per-bias gradient norm
   - target alignment with the weakest identifiable direction
3. Decide whether the normalized estimator should be benchmarked against closed-form OLS routinely.
   - For small candidate sets, OLS is the right reference implementation
4. If the goal is to return to nonlinear students, use the matrix control as the baseline.
   - Any ReLU result should be compared against the clean matrix benchmark, not interpreted in isolation
5. Consider adding figures to the paper for:
   - pairwise cosine vs noisy recovery error
   - condition number vs noisy recovery error
   - comparison of `ridge/nuclear_norm`, `ridge/stable_rank`, and `ridge/spectral_entropy`
