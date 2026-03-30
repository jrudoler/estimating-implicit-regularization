# Nonlinear Multi-Geometry Suite

Date: 2026-03-30

## Scope

This note records a nonlinear retrain study for estimating multiple trainable power-family regularizers of the form

- elementwise: `lambda * sum_j (theta_j^2 + epsilon)^(p / 2)`
- spectral: `lambda * sum_layers sum_i (sigma_i(W)^2 + epsilon)^(p / 2)`

The study fits the geometry directly by recovering one `(lambda, p)` pair per active component from a collection of retrained solutions and corresponding task gradients.

## Artifacts

- Preferred nonlinear retrain outputs: [`logs/phase2/nonlinear_multi_geometry_suite_gpu_scale03`](/home/jrudoler/inductive-bias/logs/phase2/nonlinear_multi_geometry_suite_gpu_scale03)
- Preferred nonlinear figures: [`figures/nonlinear_multi_geometry_suite_gpu_scale03`](/home/jrudoler/inductive-bias/figures/nonlinear_multi_geometry_suite_gpu_scale03)
- Local baseline retrain outputs: [`logs/phase2/nonlinear_multi_geometry_suite_scale03`](/home/jrudoler/inductive-bias/logs/phase2/nonlinear_multi_geometry_suite_scale03)
- Local baseline figures: [`figures/nonlinear_multi_geometry_suite_scale03`](/home/jrudoler/inductive-bias/figures/nonlinear_multi_geometry_suite_scale03)
- Exact sanity outputs: [`logs/phase2/nonlinear_multi_geometry_exact_sanity_scale03`](/home/jrudoler/inductive-bias/logs/phase2/nonlinear_multi_geometry_exact_sanity_scale03)
- Exact sanity figures: [`figures/nonlinear_multi_geometry_exact_sanity_scale03`](/home/jrudoler/inductive-bias/figures/nonlinear_multi_geometry_exact_sanity_scale03)
- Slurm launcher: [`scripts/nonlinear_multi_geometry_suite.slurm`](/home/jrudoler/inductive-bias/scripts/nonlinear_multi_geometry_suite.slurm)

## Nonlinear Suite

Preferred command:

```bash
sbatch --wait scripts/nonlinear_multi_geometry_suite.slurm
```

Equivalent direct command:

```bash
uv run python experiments/nonlinear_multi_geometry_suite.py \
  --n-samples 256 \
  --input-dim 12 \
  --depth 2 \
  --width 32 \
  --max-epochs 350 \
  --patience 60 \
  --n-replicates 8 \
  --estimation-max-epochs 2500 \
  --estimation-patience 250 \
  --target-gradient-scale 0.3 \
  --output-dir logs/phase2/nonlinear_multi_geometry_suite_gpu_scale03 \
  --figure-dir figures/nonlinear_multi_geometry_suite_gpu_scale03
```

Configuration:

- `n_samples=256`
- `input_dim=12`
- `depth=2`
- `width=32`
- `max_epochs=350`
- `n_replicates=8`
- `target_gradient_scale=0.3`

Case summary from [`component_recovery.csv`](/home/jrudoler/inductive-bias/logs/phase2/nonlinear_multi_geometry_suite_gpu_scale03/component_recovery.csv):

- `single_l2`: cosine `0.9393`, residual `0.3431`, mean lambda relative error `0.2582`, mean `p` absolute error `0.1524`
- `single_l1`: cosine `0.7900`, residual `0.6132`, mean lambda relative error `0.1831`, mean `p` absolute error `0.1330`
- `single_nuclear`: cosine `0.8541`, residual `0.5200`, mean lambda relative error `0.1432`, mean `p` absolute error `0.2093`
- `multi_l1_l2`: cosine `0.9255`, residual `0.3787`, mean lambda relative error `0.5879`, mean `p` absolute error `0.2980`
- `multi_l2_nuclear`: cosine `0.9457`, residual `0.3250`, mean lambda relative error `0.4309`, mean `p` absolute error `0.2354`
- `multi_l1_l2_nuclear`: cosine `0.9405`, residual `0.3398`, mean lambda relative error `1.3614`, mean `p` absolute error `0.4609`

Interpretation:

- The gradient geometry is recovered well in the aggregate across all six cases, and the GPU-backed run improved every single-component case relative to the local baseline.
- Single-component `L2` is the cleanest nonlinear case, but `L1` and nuclear also recover the correct geometry family with moderate scale and exponent error.
- Both two-component mixtures maintain cosine above `0.92`, which means the summed effective geometry is being fit well even when the individual components are not perfectly separated.
- `L1 + L2` remains partially exchangeable inside the elementwise family: the `L1` term is tight, while the `L2` term absorbs more of the residual scale and shape mismatch.
- `L2 + nuclear` is the best-behaved mixed-family case in this suite.
- The three-component fit still reaches high cosine, but it shows the clearest identifiability breakdown: the overall geometry is matched while the elementwise and spectral components can trade off against each other.

## Exact Sanity Sweep

The exact sanity sweep uses random parameter vectors with target gradients produced directly by the same component family, then fits the estimator back to those exact gradients.

Case summary from [`component_recovery.csv`](/home/jrudoler/inductive-bias/logs/phase2/nonlinear_multi_geometry_exact_sanity_scale03/component_recovery.csv):

- `single_l2`: cosine `1.0000`, residual `6.9e-06`
- `single_l1`: cosine `1.0000`, residual `2.6e-04`
- `single_nuclear`: cosine `1.0000`, residual `5.0e-05`
- `multi_l1_l2`: cosine `1.0000`, residual `3.7e-05`
- `multi_l2_nuclear`: cosine `1.0000`, residual `3.9e-05`
- `multi_l1_l2_nuclear`: cosine `1.0000`, residual `2.9e-05`

Interpretation:

- The estimator itself is not the bottleneck. On in-family gradients it recovers both scales and exponents essentially exactly, including the multi-component cases.
- The discrepancy in the nonlinear retrain study comes from the learned solutions only approximately obeying the assumed implicit-regularizer geometry, not from a failure of the fitting routine.

## Slurm Submission

The GPU-backed suite completed successfully via:

```bash
sbatch --wait scripts/nonlinear_multi_geometry_suite.slurm
```

Completed job id: `49602`
