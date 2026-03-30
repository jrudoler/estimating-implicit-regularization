# Nonlinear Power Retrain Geometry Note (2026-03-27)

## Goal

Replace basis-coefficient recovery with direct estimation of a single effective regularizer family

`R_p(theta) = lambda * sum_j (theta_j^2 + epsilon)^(p / 2)`

by fitting one global `lambda` and one global `p` from a collection of retrained nonlinear solutions and their task gradients.

## What Was Added

- `SmoothedPowerBias` in [`src/core/bias.py`](/home/jrudoler/inductive-bias/src/core/bias.py)
  - supports fixed or trainable `lambda`
  - supports fixed or trainable global exponent `p`
  - exposes an analytic parameter-gradient formula for efficient fitting
- [`experiments/nonlinear_power_retrain_geometry.py`](/home/jrudoler/inductive-bias/experiments/nonlinear_power_retrain_geometry.py)
  - trains nonlinear ReLU regressors with a fixed smoothed-power regularizer
  - retrains on bootstrap resamples
  - stacks the resulting solutions and target gradients
  - fits one shared `(lambda, p)` to that collection

## Sanity Check

Before using trained networks, I fit the estimator on synthetic solutions with exact gradients generated from the same family.

Setup:
- `8` random solution vectors
- dimension `200`
- true `lambda = 0.03`
- true `p = 1.5`
- `epsilon = 1e-6`

Recovered:
- `lambda_hat = 0.03000023`
- `p_hat = 1.5000037`
- relative residual `1.07e-5`
- cosine `0.99999993`

Interpretation:
- the joint `(lambda, p)` estimator is numerically correct on exact in-family data

## Nonlinear Retrain Run

Command:

```bash
uv run python experiments/nonlinear_power_retrain_geometry.py \
  --n-samples 256 \
  --input-dim 12 \
  --depth 2 \
  --width 24 \
  --max-epochs 300 \
  --patience 50 \
  --n-replicates 6 \
  --estimation-max-epochs 1500 \
  --estimation-patience 150 \
  --gt-p 1.5 \
  --gt-lambda 0.05 \
  --lambda-init 0.02 \
  --p-init 2.0 \
  --output-json logs/phase2/nonlinear_power_retrain_geometry_p15_r6.json
```

Result file:
- `logs/phase2/nonlinear_power_retrain_geometry_p15_r6.json`

Recovered from the full trained solution:
- true `lambda = 0.05`
- true `p = 1.5`
- `lambda_hat = 0.0902`
- `p_hat = 1.9021`
- residual `0.3603`
- cosine `0.9329`

Recovered from the stacked 6-replicate collection:
- `lambda_hat = 0.0810`
- `p_hat = 1.8720`
- residual `0.3890`
- cosine `0.9212`

## Interpretation

- The estimator works in principle: it recovers `(lambda, p)` almost exactly on exact in-family synthetic data.
- In the nonlinear retrained network setting, recovery is only partial.
- The fitted `p` moves in the right direction but is biased upward relative to the ground truth `p = 1.5`.
- Using a collection of retrained solutions helps slightly on `p` versus the single full-solution fit, but does not remove the bias.
- The main bottleneck now appears to be the quality of the stationarity approximation in the trained nonlinear models, not the mechanics of the `(lambda, p)` estimator itself.
