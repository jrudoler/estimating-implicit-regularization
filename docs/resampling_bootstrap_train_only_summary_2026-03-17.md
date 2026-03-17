# Resampling Update: Bootstrap + Training-Data Gradient Evaluation (2026-03-17)

This note records two decisions for the resampling / shared-lambda experiments.

## Protocol Going Forward

1. Resampling should default to bootstrap rather than subsampling.
2. Each bootstrap replicate should have the same sample size as the original training split.
3. Bias-parameter recovery should be evaluated on the training data used to fit the corresponding model.
4. Held-out validation or test gradients are not the target of these experiments.

Concretely, the resampling experiment defaults are now:
- `resample_mode=bootstrap`
- `sample_fraction=1.0`
- retrain-per-resample recovery uses the replicate training set for each replicate equation
- full-data recovery uses the original full training split

## Reason For The Change

The prior exploratory settings mixed two effects that should be kept separate:
- changing the training sample size
- changing the resampling mechanism

Using small-fraction subsamples made `single` resample runs incomparable to the original full-data run. A bootstrap replicate of full training size is the correct control if the goal is to compare:
- one original dataset equation
- many retrained bootstrap equations with a shared regularization vector

Restricting gradient evaluation to training data also aligns the experiment with the intended question: whether the learning rule's shared implicit/explicit bias can be recovered from the data used to fit the model.

## Current Interpretation

In the nonlinear retrain-per-resample setting, the right baseline comparison is now:
- `full`: one full-dataset training equation
- `single`: one bootstrap-retrained model at a time
- `aggregate`: average of the per-bootstrap estimates
- `stacked`: one shared-lambda least-squares fit across all bootstrap-retrained equations

The next runs should therefore be interpreted as asking whether full-size bootstrap retraining improves recovery relative to the single original training equation, while keeping the recovery target on training data throughout.
