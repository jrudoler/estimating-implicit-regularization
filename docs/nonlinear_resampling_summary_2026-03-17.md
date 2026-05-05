# Nonlinear Resampling Summary

Date: 2026-03-17

## Scope

This note summarizes the first resampling experiment run on the structured nonlinear teacher-student pipeline in:

- [function_class_identifiability.py](experiments/function_class_identifiability.py)

The main question was whether the fixed-weight stacked shared-lambda estimator helps in nonlinear settings where single-dataset recovery is already poor.

## What Was Added

- Nonlinear fixed-weight resampling experiment:
  - [nonlinear_resampling_identifiability.py](experiments/nonlinear_resampling_identifiability.py)

This mirrors the matrix fixed-weight resampling study:

1. train one predictive model once,
2. freeze the learned weights,
3. resample the training split,
4. recompute the loss-gradient targets on the resamples,
5. recover one shared `lambda` by either:
   - per-resample OLS plus averaging, or
   - one stacked OLS solve across all resamples.

As in the matrix fixed-weight case, the regularizer-gradient design is held fixed by the frozen weights, so stacked OLS is effectively target-denoising rather than geometry-changing.

## Teacher And Student

Teacher:

- structured synthetic `teacher_relu` target generator
- controlled input spectrum and teacher first-layer singular spectrum
- configurable teacher activation

Student:

- `DeepReLURegressor`
- one or more hidden ReLU layers
- explicit `ridge` and `nuclear_norm` penalties during training

## Regimes Run

All runs used:

- `n_samples=2048`
- `input_dim=16`
- zero observation noise
- `gt_bias_types=ridge,nuclear_norm`
- auto-balanced ground-truth lambdas
- shared-lambda recovery on the single original training split plus resampled fixed-weight targets

Representative settings:

1. shallow ReLU student, flat teacher spectrum
2. shallow ReLU student, spiky teacher spectrum
3. deep ReLU student, flat teacher spectrum
4. deep ReLU student, spiky teacher spectrum
5. deep ReLU student, flat teacher spectrum with bootstrap instead of subsample

## Main Findings

### 1. Single-dataset recovery is already poor in the nonlinear setup

Representative full-data OLS errors:

- shallow ReLU, flat teacher:
  - cosine `0.6789`
  - full OLS mean relative error `0.9840`
- deep ReLU, flat teacher:
  - cosine `0.2902`
  - full OLS mean relative error `1.0288`
- deep ReLU, spiky teacher:
  - cosine `0.2776`
  - full OLS mean relative error `1.7983`

So, unlike the clean linear matrix control, the nonlinear local equation is already a poor approximation even on the full original training set.

### 2. Fixed-weight resampling does not rescue the nonlinear cases

Representative results:

- shallow ReLU, flat teacher, subsample:
  - single-resample mean relative error `1.1287`
  - stacked mean relative error `1.1016`
- deep ReLU, flat teacher, subsample:
  - single `1.2838`
  - stacked `1.2553`
- deep ReLU, spiky teacher, subsample:
  - single `1.7458`
  - stacked `1.7458`
- deep ReLU, flat teacher, bootstrap:
  - single `0.9259`
  - stacked `0.8984`

These changes are small relative to the absolute error scale.

### 3. The nonlinear failure is not primarily target-noise variance

In the linear matrix control, full-data OLS was nearly exact, and stacked resampling helped in hard-but-plausible regimes by averaging noisy targets against a fixed design.

Here, full-data OLS itself is already inaccurate. That means:

- the local training-gradient equation is already badly misspecified or badly conditioned at the learned nonlinear weights,
- so averaging more target gradients does not fix the main problem.

## Interpretation

The nonlinear results support the following:

1. The favorable resampling result from the matrix fixed-weight experiment was specific to a setting where the underlying local system was basically correct and only the target side was noisy.
2. In the nonlinear teacher-student setting, the dominant issue is not simply noisy estimation of the target gradient.
3. Therefore, fixed-weight stacked resampling is not enough to restore accurate recovery.

In short:

- linear matrix control: fixed-weight resampling can help
- nonlinear ReLU setting: fixed-weight resampling does not materially help

## Most Relevant Next Step

The next experiment should move to retrain-per-resample in the nonlinear setting.

That is the only version that can test the stronger hypothesis:

> resampling may change the learned nonlinear weights enough that the replicate-specific regularizer gradients become jointly more informative.

If that also fails, then the evidence will point away from bootstrap/resampling as a practical solution for nonlinear implicit-bias recovery.
