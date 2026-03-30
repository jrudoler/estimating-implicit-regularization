# Autonomous Notes

Use this file as the default non-manuscript log for autonomous method, implementation, and experiment notes.

## Conventions

- Add dated entries when autonomous work changes methods, assumptions, or experiment execution.
- Prefer logging here or in another `docs/` note instead of writing those updates into `paper/`.
- Promote durable results into a more specific doc in `docs/` when the workstream becomes substantial.
- For analysis plotting, default to a single `PDF` output unless the user explicitly requests an additional export format.

## 2026-03-27

- Added [`SmoothedPowerBias`](/home/jrudoler/inductive-bias/src/core/bias.py) for the family `R_p(theta) = lambda * sum_j (theta_j^2 + epsilon)^(p/2)` with optional trainable `lambda` and trainable global exponent `p`.
- Added [`experiments/nonlinear_power_retrain_geometry.py`](/home/jrudoler/inductive-bias/experiments/nonlinear_power_retrain_geometry.py) to train nonlinear models under a fixed smoothed-power regularizer, retrain on resamples, and then fit one shared `(lambda, p)` to the stacked collection of solutions and task gradients.
- Registered the new experiment in [`experiments/REGISTRY.yaml`](/home/jrudoler/inductive-bias/experiments/REGISTRY.yaml).
