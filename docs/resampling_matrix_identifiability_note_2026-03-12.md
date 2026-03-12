# Resampling, Shared-Lambda Recovery, and Changing Gradient Geometry

This note summarizes what the recent matrix-spectrum resampling experiments do and do not show, with emphasis on the distinction between:

1. resampling the target gradient while keeping the regularizer-gradient design fixed, and
2. retraining on each resample so that both the target and the design change.

The target audience here is the local identifiability problem

\[
g_{\mathrm{loss}}(w) \approx \sum_{j=1}^p \lambda_j g_j(w),
\]

where:

- \(w\) denotes the predictive model parameters,
- \(g_{\mathrm{loss}}(w)\) is the negative data-loss gradient,
- \(g_j(w)\) is the gradient of regularizer \(R_j\),
- \(\lambda \in \mathbb{R}^p\) is the shared vector of explicit or implicit regularization strengths to be recovered.

In the matrix-spectrum experiments we focused on \(p=2\), usually `ridge` and `nuclear_norm`.

## 1. Local Linear System View

At a fixed parameter vector \(w\), define the design matrix

\[
A(w) = \begin{bmatrix}
g_1(w) & g_2(w) & \cdots & g_p(w)
\end{bmatrix} \in \mathbb{R}^{d \times p},
\]

and the target vector

\[
b(w; D) = g_{\mathrm{loss}}(w; D) \in \mathbb{R}^{d},
\]

where \(D\) is the dataset used to compute the loss gradient.

Recovery of \(\lambda\) is then the regression problem

\[
b(w; D) \approx A(w)\lambda.
\]

The usual local-identifiability quantities are then properties of \(A(w)\):

- pairwise cosine of the columns,
- condition number of \(A(w)^\top A(w)\),
- smallest singular value of \(A(w)\).

When the columns of \(A(w)\) are nearly collinear, small perturbations in \(b\) induce large perturbations in \(\hat\lambda\).

## 2. What The Fixed-Weight Resampling Experiment Actually Tests

File:

- [matrix_spectrum_resampling.py](/home/jrudoler/inductive-bias/experiments/matrix_spectrum_resampling.py)

In that experiment:

1. train a single predictive model once on the full training split,
2. freeze its weights \(w\),
3. resample datasets \(D_1,\dots,D_B\),
4. recompute only the target gradients \(b_r = b(w; D_r)\),
5. solve for a shared \(\lambda\) using either:
   - per-resample fits plus averaging, or
   - one stacked fit over all resamples.

Since \(w\) is fixed, the regularizer gradients are fixed:

\[
A_r = A(w) = A \qquad \text{for all } r.
\]

The replicate systems are therefore

\[
b_r = A\lambda^\star + \varepsilon_r,
\]

where \(\varepsilon_r\) is the error induced by using a finite resample to estimate the loss gradient at fixed \(w\).

This is exactly the classical repeated-response linear model with common design:

\[
\hat\lambda_r = (A^\top A)^{-1}A^\top b_r.
\]

Hence

\[
\mathbb{E}[\hat\lambda_r] = \lambda^\star + (A^\top A)^{-1}A^\top \mathbb{E}[\varepsilon_r].
\]

If \(\varepsilon_r\) is approximately mean-zero, then averaging helps by reducing its variance:

\[
\bar b = \frac{1}{B}\sum_{r=1}^B b_r
      = A\lambda^\star + \bar\varepsilon,
\qquad
\mathrm{Var}(\bar\varepsilon) = \frac{1}{B}\mathrm{Var}(\varepsilon_r)
\]

under independence.

The stacked estimator solves

\[
\hat\lambda_{\mathrm{stack}}
= \arg\min_\lambda \sum_{r=1}^B \|b_r - A\lambda\|_2^2
= \arg\min_\lambda \|\tilde b - \tilde A \lambda\|_2^2,
\]

with

\[
\tilde b =
\begin{bmatrix}
b_1 \\ \vdots \\ b_B
\end{bmatrix},
\qquad
\tilde A =
\begin{bmatrix}
A \\ \vdots \\ A
\end{bmatrix}.
\]

But then

\[
\tilde A^\top \tilde A = B A^\top A,
\qquad
\tilde A^\top \tilde b = A^\top \sum_{r=1}^B b_r,
\]

so

\[
\hat\lambda_{\mathrm{stack}}
= (A^\top A)^{-1} A^\top \bar b.
\]

Therefore, in the fixed-weight experiment:

- stacked OLS is equivalent to OLS on the averaged target gradient,
- and, because OLS is linear in \(b\) when \(A\) is fixed, it is also equivalent to averaging the per-resample OLS estimates.

This confirms the prior belief that, without changing the regularizer gradients, resampling is fundamentally just a target-denoising exercise.

### Consequence

If the only issue is noisy estimation of \(b\), and \(A\) is fixed but ill-conditioned rather than singular, resampling can materially improve recovery.

If \(A\) is nearly singular, resampling cannot solve the problem; it only reduces variance in the projected target, while the inverse map \((A^\top A)^{-1}\) remains unstable.

That is exactly what we observed in the fixed-weight matrix-spectrum experiment:

- in hard but nondegenerate regimes, stacking helped,
- in the flat-spectrum nearly singular regime, it did not meaningfully rescue recovery.

## 3. Why Retraining On Each Resample Is A Different Statistical Problem

The original motivation for retraining was different:

> perhaps resampling the data changes the learned weights enough that the regularizer gradients become less collinear in some replicates, making the overall shared-\(\lambda\) problem easier.

That corresponds to replicate-specific systems

\[
b_r = b(w_r; D_r), \qquad A_r = A(w_r),
\]

where each \(w_r\) is obtained by training on the resample \(D_r\).

The stacked estimator then becomes

\[
\hat\lambda_{\mathrm{stack}}
= \arg\min_\lambda \sum_{r=1}^B \|b_r - A_r \lambda\|_2^2.
\]

This is no longer repeated-response regression with common design. Instead it is a regression with replicate-specific design matrices.

Two effects now matter:

1. the targets \(b_r\) vary across replicates,
2. the geometry \(A_r^\top A_r\) also varies across replicates.

The normal equations are

\[
\left(\sum_{r=1}^B A_r^\top A_r\right)\hat\lambda_{\mathrm{stack}}
= \sum_{r=1}^B A_r^\top b_r.
\]

This makes clear what the user originally hoped for:

- if some replicates have better-conditioned designs,
- and these designs point in complementary directions,
- then \(\sum_r A_r^\top A_r\) might be substantially better conditioned than any individual \(A_r^\top A_r\).

That is the correct mathematical mechanism by which retraining on resamples could help beyond simple target averaging.

## 4. Why The First Retrain-Per-Resample Experiment Was Too Easy

File:

- [matrix_spectrum_resampling_retrain.py](/home/jrudoler/inductive-bias/experiments/matrix_spectrum_resampling_retrain.py)

The first version trained each replicate model on a resample \(D_r\), then evaluated the gradient equation on that same \(D_r\).

Thus each replicate approximately satisfied its own first-order stationarity condition:

\[
b_r \approx A_r \lambda^\star.
\]

If optimization is good and the model is well specified, then every replicate is already close to exactly consistent with the same \(\lambda^\star\).

In that case the stacked problem is trivial:

- each replicate equation is individually almost exact,
- the shared-\(\lambda\) fit is therefore almost exact as well,
- even when pairwise collinearity is high.

This is why the retrain-per-resample same-data experiment produced near-zero recovery error even in flat-spectrum settings.

That result does **not** mean resampling solved the identifiability problem.
It means the experiment was too close to an exact structural identity.

## 5. Why Retrain + Held-Out Gradient Evaluation Became Hard Again

To break that triviality, we added a version where:

- \(w_r\) is trained on resample \(D_r\),
- but \(b_r\) is evaluated on a held-out split \(D_{\mathrm{val}}\),
- while \(A_r\) still comes from \(w_r\).

Now the replicate equations are

\[
b_r^{\mathrm{val}} \approx A_r \lambda^\star + \delta_r,
\]

where \(\delta_r\) is not just finite-sample noise. It includes:

- train/validation mismatch,
- imperfect optimization,
- finite-sample movement of the fitted weights,
- model mismatch from fitting on smaller resamples.

This is no longer a simple mean-zero noise model.

The key point is that stacking helps when replicate variation induces *useful geometric diversity* in the designs \(A_r\), not when it merely induces extra structured discrepancy \(\delta_r\).

What we observed was:

- replicate-specific pairwise cosines did vary somewhat,
- but not enough, and not in a systematically complementary way,
- so \(\sum_r A_r^\top A_r\) did not improve enough to offset the added mismatch.

Hence stacked retrain-per-resample recovery was not better than the per-replicate alternatives in the held-out-gradient version.

## 6. Why The User's Prior Was Reasonable

The prior was:

> if we only resample targets while keeping weights fixed, this is basically the same as denoising the target side of the same linear system; what we really want is variability in training so we may observe replicate models with less collinear regularizer gradients.

That prior is correct.

More precisely:

- fixed-weight resampling can only reduce variance in \(b\),
- it cannot change the intrinsic geometry of \(A\),
- so it cannot create identifiability that was absent in the local model.

Retraining on resamples is exactly the right way to seek geometric diversification, because it changes \(A_r\).

However, for that strategy to help, the resample-to-resample variation in \(A_r\) must satisfy a strong condition:

\[
\sum_{r=1}^B A_r^\top A_r
\]

must be materially better conditioned than the individual

\[
A_r^\top A_r.
\]

In the present matrix-spectrum experiments, that did not happen in a useful way:

- same-data evaluation made the problem nearly exact,
- held-out evaluation introduced mismatch without enough compensating geometric gain.

## 7. Statistical Interpretation

There are three distinct regimes.

### A. Fixed design, noisy target

\[
b_r = A\lambda^\star + \varepsilon_r.
\]

This is the regime where bootstrap/subsampling can help by variance reduction.

### B. Random design, exact replicate equations

\[
b_r \approx A_r \lambda^\star.
\]

If each replicate is already nearly exact, shared-\(\lambda\) recovery is trivial.
This is not informative about whether resampling resolves identifiability.

### C. Random design, random mismatch

\[
b_r = A_r \lambda^\star + \delta_r.
\]

Now resampling helps only if design diversity improves conditioning faster than mismatch accumulates.
That did not occur in the present held-out-gradient version.

## 8. Practical Conclusion

The current experiments support the following statements.

1. Fixed-weight resampling helps only through target denoising.
2. Retraining on each resample is the right way to test whether resampling can improve gradient geometry.
3. In the current matrix-spectrum control, retraining did **not** produce enough useful design variation to improve shared-\(\lambda\) recovery once the trivial same-data identity was removed.

So the bootstrap idea is not wrong, but its success depends on a stronger property than "weights change with the data":

> the resample-induced designs must become jointly more informative in aggregate.

That is a much sharper requirement than simple target-noise reduction.

## 9. Most Relevant Files

- Fixed-weight target-resampling experiment:
  - [matrix_spectrum_resampling.py](/home/jrudoler/inductive-bias/experiments/matrix_spectrum_resampling.py)
- Retrain-per-resample experiment:
  - [matrix_spectrum_resampling_retrain.py](/home/jrudoler/inductive-bias/experiments/matrix_spectrum_resampling_retrain.py)
- Fixed-weight spikiness sweep:
  - [matrix_resampling_spikiness_sweep_small.csv](/home/jrudoler/inductive-bias/logs/phase2/matrix_resampling_spikiness_sweep_small.csv)
- Retrain-per-resample pilot results were run ad hoc from the CLI and are currently reflected in terminal logs rather than a dedicated CSV artifact.

## 10. Recommended Next Experiment

The next experiment should target a setting where:

1. replicate-specific models move enough to change \(A_r\) materially,
2. the replicate equations are not exact by construction,
3. but the mismatch term \(\delta_r\) is still small enough that geometric gains can matter.

The cleanest candidate is likely:

- multiple independently trained models from the same learning rule,
- each on a finite dataset drawn from the same distribution,
- with shared true \(\lambda\),
- and stacked recovery across those trained instances.

That is closer to the intended use case than bootstrap resampling of a single synthetic dataset.
