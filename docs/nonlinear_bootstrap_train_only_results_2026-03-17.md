# Nonlinear Bootstrap Train-Only Results (2026-03-17)

This note records the first rerun of the nonlinear retrain-per-resample experiment after aligning the protocol with the intended method:

- resampling uses bootstrap with replacement
- each bootstrap replicate has the same size as the original training split
- bias recovery is evaluated on the training data used to fit the corresponding model

## Experiment

Script:
- `experiments/nonlinear_resampling_retrain.py`

Configuration:
- structured nonlinear teacher-student setup
- teacher activation: `relu`
- teacher spectrum: `identity`
- student depth: `3`
- student width: `64`
- bias pair: `ridge,nuclear_norm`
- `n_replicates=6`
- `resample_mode=bootstrap`
- `sample_fraction=1.0`
- gradient evaluation: training data only

## Results

Seed `42`
- `full_ols_mean_rel_error = 1.0288`
- `single_mean_rel_error = 1.0452`
- `aggregate_mean_rel_error = 0.8791`
- `stacked_mean_rel_error = 0.8821`
- `full_train_mse = 0.00613`
- `replicate_train_mse_mean = 0.00718`

Seed `123`
- `full_ols_mean_rel_error = 0.6296`
- `single_mean_rel_error = 1.2146`
- `aggregate_mean_rel_error = 0.8375`
- `stacked_mean_rel_error = 0.8313`
- `full_train_mse = 0.00780`
- `replicate_train_mse_mean = 0.00820`

Seed `456`
- `full_ols_mean_rel_error = 0.8933`
- `single_mean_rel_error = 0.8637`
- `aggregate_mean_rel_error = 0.7019`
- `stacked_mean_rel_error = 0.7041`
- `full_train_mse = 0.00675`
- `replicate_train_mse_mean = 0.00674`

## Interpretation

The protocol is now correctly matched to the intended bootstrap estimator.

The bootstrap-retrained models fit their own training data at roughly the same MSE scale as the original full-data model. However, shared-lambda recovery is still only partially improved:

- `single` bootstrap replicates are not consistently better than the original full-data solve
- `aggregate` and `stacked` improve over `single`
- relative to `full`, bootstrap aggregation helps on some seeds but not all

So in this nonlinear train-only setting, bootstrap retraining changes the local geometry and can modestly improve recovery, but it is not yet a reliable fix.
