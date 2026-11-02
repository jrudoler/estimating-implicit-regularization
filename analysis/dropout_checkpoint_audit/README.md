# Dropout estimator correction and checkpoint controls

The production ridge-only runner now defaults to exact nonnegative scalar least
squares against the **dropout-free, full-training-set mean CE gradient**. It
records the unconstrained coefficient separately. `--estimator legacy_iterative`
retains the old Adam fit for audits; `--estimator closed_form` rejects joint-family
runs rather than silently fitting each family independently.

New training saves `estimation_endpoint.ckpt` containing the weights actually
estimated. Previously the runner estimated from the final model but reported the
best-validation checkpoint path. An old `predictive_model_checkpoint` path is
therefore not sufficient evidence that the original weights have been recovered.

## Existing evidence recovered from Locust

The July 2026 `verify_estimator_convergence` outputs compare two estimators on the
same endpoint. Baseline and dropout=0.3 iterative/closed-form ratios are 38.5 and
61.4. Their full-gradient residual ratios are respectively 8.929/0.9716 and
6.695/0.9940; the iterative fits are worse than the zero-penalty baseline.

The 480-run `closed_form_dropout_grid` is a **retrained grid**, not a same-weight
comparison to the original W&B sweep. Its checkpointing was disabled. Its signed
ridge Spearman correlations with dropout are:

| Depth | Width 128 | Width 256 | Width 512 |
|---|---:|---:|---:|
| 3 | -0.787 | -0.309 | -0.455 |
| 5 | +0.452 | +0.739 | +0.539 |

These summaries do not establish monotonicity, isolate dropout from incomplete
convergence, or imply that ridge adequately describes dropout. In particular,
the original universal increasing trend is not supported by the corrected grid.

The saved short optimizer ablations do **not** isolate epsilon as the cause:
sum reduction and smaller epsilon barely change their unconverged estimates.
Describe the established failure as iterative nonconvergence. The new audit
compares equal update budgets and varies one optimizer setting at a time.

Historical float32 norms have discrepancies up to 5.04e-4 in
`residual_ratio**2 = 1 - max(cosine, 0)**2`. Historical import explicitly allows
1e-3 and records the measured discrepancy. Fresh results default to 1e-5. The
figure colors use squared positive basis cosine, one global scale, and mark
undefined diagnostics in gray. The figure labels the historical provenance.

## Reproducible workflow

Regenerate the validated standard figure, without staging manuscript changes:

```bash
uv run snakemake -s workflow/Snakefile --cores 2 results/figures/dropout_bias_ridge_panel.pdf
```

All 480 depth/width/dropout/seed combinations must exist exactly once. Validation
records input hashes; plotting rejects changed data or a mismatched receipt.
The original `data/generated/dropout_bias_estimation/runs.parquet` is preserved;
its plotting entrypoint is now `analysis/plot_dropout_bias_ridge_panel/legacy.py`.

Run the bounded GPU controls on Locust:

```bash
uv run snakemake -s workflow/Snakefile --profile workflow/profiles/slurm \
  results/data/dropout_checkpoint_controls.csv \
  --config dropout_source_root=/home/jrudoler/inductive-bias
```

The design is depth 3, width 256; dropout 0, 0.3, 0.5; the original ten seeds;
checkpoints at 0.5T, 0.8T, and T. Existing trajectory checkpoints are reused.
Missing trajectories are explicitly identified replacement replays to the
stored corrected-grid stopping epoch. The spec never launches a full 480-model
retraining grid. Saved checkpoints, source/runtime provenance and dataset-split
hashes accompany every result. Spec generation and diagnostic evaluation are
independent of the original manuscript and remote checkout.

At every checkpoint, 32 independent full-dataset mask draws estimate
`g_train = E[grad L_dropout]`. The direct dropout gradient is
`g_train - g_eval`; the endpoint proxy is `-g_eval`. Their signed ridge
coefficients differ by `(2 theta @ g_train) / ||2 theta||^2`, measuring the
component of nonstationarity that contaminates this coefficient. Both
nonnegative projections and their fit diagnostics are retained.

Independent-mask cross products estimate `||g_train||^2` without the positive
Monte Carlo norm bias. Report its jackknife uncertainty and retain negative
estimates; intervals crossing zero are inconclusive. Do not match on
`||g_eval||`, and do not interpret clipped negative estimates as convergence.
The empirical-bootstrap coefficient intervals quantify mask uncertainty;
paired seed bootstrap contrasts quantify variability across training seeds.
They are reported separately. BatchNorm is deliberately rejected by this
dropout-only objective control.

The optimizer audit uses cached, deterministic minibatch sufficient statistics
on a fixed checkpoint, equal update counts, identical initialization, and the
legacy optimizer/scheduler settings. It is labeled as such, not as an exact
reproduction of historical shuffled batches. A float64 `lstsq` reference and
full-dataset objective gaps accompany every variant.

Trajectory training and checkpoint diagnostics are separate rules. Changing the
estimator does not invalidate saved training weights. To summarize imported
results without enabling any GPU producer rules:

```bash
uv run snakemake -s workflow/Snakefile --cores 2 \
  results/data/dropout_checkpoint_controls.csv --config dropout_recovered_only=true
```

## Completed Locust controls (2026-09-29)

All 30 runs, 90 checkpoints, and 2,880 mask-gradient evaluations completed.
Ten dropout=0.3 trajectories were recovered; the 20 dropout=0/0.5 trajectories
were explicitly labeled replacement replays. All 90 checkpoint hashes were
verified after local recovery. Execution metadata and the exact source archive
are in `data/generated/dropout_checkpoint_audit/execution_manifest.json` and
`source/source-v3.tar.gz`; aggregate outputs are `summary.json` and
`results/data/dropout_checkpoint_controls.csv`. The archived execution precedes
the subsequent workflow split and a roundoff-only centering improvement; raw
results are preserved. Zero-dropout direct-gradient roundoff was below 7e-17;
the summary checks this and uses the analytic zero, with undefined relative fit.

Final-checkpoint means over ten seeds (coefficients in units of **1e-5**):

| Dropout | Endpoint proxy, nonnegative | Direct dropout, nonnegative | Direct dropout, signed |
|---|---:|---:|---:|
| 0 | 4.096 | 0 | 0 |
| 0.3 | 3.566 | 0 | -1.927 |
| 0.5 | 3.625 | 0 | -2.076 |

The direct signed coefficient's seed-bootstrap 95% intervals are
[-1.971, -1.884]e-5 at dropout=0.3 and [-2.146, -2.016]e-5 at dropout=0.5.
Every nonzero-dropout checkpoint's 32-draw Monte Carlo coefficient interval
also lies below zero. Thus **positive ridge explains none of the directly
measured dropout contribution** at these checkpoints. Even permitting a
negative ridge coefficient explains only 0.266% and 0.125% of its gradient
energy on average at the two final endpoints; these are MC point estimates,
not noise-corrected population fit statistics.

All 90 training-gradient squared-norm intervals have positive lower bounds.
The endpoint proxy is therefore not a stationary-objective identity here.
At dropout=0.3/0.5 the projected nonstationarity contribution
`direct - endpoint` averages -5.492e-5/-5.701e-5, larger in magnitude than the
positive endpoint proxy itself. The no-dropout control still has a positive
endpoint proxy (~4.10e-5) despite exactly zero dropout contribution. This is
strong evidence that interpreting that proxy as a dropout ridge penalty is
not justified for these finite-training checkpoints. It does **not** establish
what fully converged models would do or that dropout has no regularizing effect.

Paired endpoint-proxy differences from dropout=0 are -5.30e-6 (95% seed CI
[-5.93, -4.67]e-6) at 0.3 and -4.70e-6 ([-5.45, -4.00]e-6) at 0.5. The
numerical correction does not support a universal increasing dropout/ridge
trend; the broader historical grid remains architecture dependent.

On the three seed-123 final checkpoints, float64 closed form agrees with
`torch.linalg.lstsq` within **3.7e-19** absolute coefficient error. With 54,000
matched optimizer updates, the cached legacy baseline remains 1.85x, 1.85x,
and 2.30x above the exact coefficient (dropout 0/0.3/0.5). Removing scheduling
reduces these to 1.021x/1.047x/1.045x; direct-coefficient fitting gives
1.003x/1.005x/1.015x. Sum reduction and smaller epsilon give similar partial
improvements; float64 alone barely changes the baseline. Objective gaps and
coefficient errors for every variant are retained in the raw results. These
longer fixed-batch audits are not an exact replay of the historical 38–61x fits.

## Direct-coefficient optimizer confirmation

The follow-up audit reuses the **same 30 saved final checkpoints** (depth 3,
width 256, dropout 0/0.3/0.5, ten seeds). On Locust, it cached the deterministic
dropout-free full-training-set scalar quadratic and all 27 minibatch quadratics
for each checkpoint. It then replayed 2,000 epochs / 54,000 float32 Adam
coefficient updates per variant locally. The replay of the previous
direct-coefficient/plateau audit on the three seed-123 checkpoints agrees to
within 9.9e-8 relative error. No network was trained in this comparison.

| Direct-lambda variant | Median relative error to full-dataset closed form | Worst error |
|---|---:|---:|
| Direct lambda with original plateau scheduler | 0.470% | 1.885% |
| No scheduler, LR 0.05 | 100% | 2462% |
| No scheduler, LR 0.001 | 77.3% | 183% |
| No scheduler, LR 0.0001 | 17.1% | 48.7% |
| Planned 10x decays at epochs 25/100/400/1000/1500, fixed minibatches | 0.248% | 0.928% |
| Same decay, deterministic full-dataset scalar gradient | 3.59e-6% | 1.02e-5% |

Thus **optimizing lambda directly plus a suitable LR decay can recover the
closed-form coefficient when both minimize the identical full-dataset
objective**. Merely removing the scheduler at these constant rates does not.
All 30 full-dataset replay coefficients agree with closed form to at most
1.03e-7 relative error. On the fixed-minibatch replay, all 30 planned-decay
fits are within 1%, but exact equality is not expected: equal weighting of the
27 cached minibatches gives a scalar optimum 0.493% away from the
sample-weighted full-dataset optimum at the median checkpoint, and 1.928% at
the worst. This fixed-batch weighting difference is an audit property, **not
evidence that the original shuffled minibatch fit had the same bias**.

Inputs and per-checkpoint outputs are
`data/generated/dropout_checkpoint_audit/optimizer_fix_stats.json` and
`results/data/dropout_optimizer_fix_confirmation.json`; the remote source
overlay and job provenance are in `source/optimizer-fix-all-seeds.tar.gz` and
`optimizer_fix_execution.json`. Run the local replay with:

```bash
uv run snakemake -s workflow/Snakefile --cores 2 \
  results/data/dropout_optimizer_fix_confirmation.json \
  --config dropout_recovered_only=true optimizer_fix_stats_recovered_only=true
```

The full-dataset comparison isolates numerical fitting, not the scientific
interpretation of the positive endpoint coefficient. The nonstationarity and
direct-dropout-gradient findings above remain unchanged.

**Remaining limitation:** the original 480-cell corrected sweep did not save
its checkpoints. The standard figure uses the recovered, validated historical
retrained float32 grid, explicitly labeled as such. A fresh float64 same-weight
recomputation of all 480 cells cannot be claimed. No full-grid retraining was
launched, and neither manuscript text nor manuscript build artifacts changed.

## Validation

```bash
PYTHONPATH=src:. uv run pytest tests/test_scalar_estimation.py tests/test_predictive_loss_grad.py
```

If the installed Slurm executor stalls before submission, prepare specs first,
then submit `analysis/dropout_checkpoint_audit/locust.sbatch` from an isolated
source checkout. Set `DROPOUT_CODE_REVISION` and `DROPOUT_SOURCE_SHA256` to the
actual revision and transferred source archive hash. The array runs the same
Snakemake rules, limits concurrency to six GPUs, and has exactly 30 entries;
`sbatch --array=11 ...` is the saved-checkpoint seed-42 pilot. Do not overwrite
source files during an active array. The original Locust checkout remains
read-only, and no manuscript output is a dependency or target.

Tests cover scalar recovery and boundary optima, undefined diagnostics, unequal
batch weighting and batch-size invariance, preservation of model state, the
Lightning evaluation-mode invariant, production checkpoint provenance, and
missing/duplicate/stale grid inputs. Generated data and completed controls are
required before claiming new scientific validation; unit tests alone do not
verify the experimental conclusions.
