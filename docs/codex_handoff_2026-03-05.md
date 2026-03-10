# Codex Handoff (2026-03-05)

## Goal

This project workstream had three linked goals:

1. Clean up the experiment surface so active vs legacy scripts are explicit and reproducible.
2. Run a controlled identifiability study where bias-gradient geometry is deliberately high-collinear vs low-noncollinear.
3. Run bootstrap/subsampling as a separate track in an isolated worktree and decide whether it should remain optional analysis or enter the main workflow.

This file is meant to preserve context for follow-on work in Codex CLI.

## Current Repository State

- Main repo: `/home/jrudoler/inductive-bias`
- Main branch: `main`
- Main repo status at handoff: clean
- Bootstrap worktree: `/home/jrudoler/inductive-bias-bootstrap`
- Bootstrap branch: `bootstrap-resampling`
- Latest bootstrap commit seen during handoff: `34f4591` (`Enable gradient-based identifiability metrics in bootstrap recovery.`)

There is also an active Slurm array sweep:

- Sweep ID: `jhrudoler-penn/inductive-bias-bootstrap/bvx8c05t`
- W&B URL: <https://wandb.ai/jhrudoler-penn/inductive-bias-bootstrap/sweeps/bvx8c05t>
- Slurm array job: `46011`
- Concurrency cap: `%6` GPU tasks at a time

At handoff time, the array was still running.

## What Was Already In Place

Before the bootstrap worktree work, the repo already had a good amount of Phase 1 and Phase 2 work implemented:

- `experiments/REGISTRY.yaml` exists and is part of the cleanup / canonicalization effort.
- `scripts/wandb_sweep.slurm` is the canonical Slurm launcher.
- `experiments/function_class_identifiability.py` is the strongest existing synthetic identifiability experiment.
- `experiments/run_function_class_matrix.py` already compares:
  - `high_collinear = ridge,nuclear_norm`
  - `low_noncollinear = ridge,weight_coherence`
- `analysis/function_class_matrix_summary.md` already summarizes the function-class matrix results.

That means the repo already had a credible synthetic identifiability path before the bootstrap-specific work.

## Bootstrap Worktree Work Completed

The bootstrap track was intentionally separated into its own worktree so it could evolve without polluting `main`.

### Files of interest

- `../inductive-bias-bootstrap/experiments/bootstrap_bias_recovery.py`
- `../inductive-bias-bootstrap/sweeps/bootstrap_bias_recovery.yaml`

### What the bootstrap script does

`experiments/bootstrap_bias_recovery.py`:

- trains one MNIST classifier with known explicit regularization:
  - `Ridge`
  - `WeightCoherence`
- then freezes that trained predictive model
- then runs repeated bias-estimation replicates with one of:
  - `full`
  - `subsample`
  - `bootstrap`

Important clarification:

- resampling is **not** used for model fitting
- resampling is used **only for the bias-estimation phase**

So the question being tested is:

> Given one trained model, does changing the data used to estimate the loss-gradient target improve the stability of recovered bias coefficients?

### Bias models being compared

There are two levels of comparison:

1. **Ground-truth regularizers in model training**
   - `RidgeBias`
   - `WeightCoherenceBias`

2. **Estimator variants in bias recovery**
   - standard gradient matching
   - normalized gradient matching

The normalized estimator is meant to mitigate gradient-magnitude imbalance between regularizers before recovering final lambda values.

### Metrics logged

Per replicate:

- recovered ridge lambda
- recovered coherence lambda

Aggregate:

- mean
- std
- 95% CI
- CV
- sign consistency
- relative error vs ground truth

Geometry diagnostics:

- `identifiability/gram_condition_number`
- `identifiability/cosine_similarity`

## Existing Bootstrap Summary Artifact

There is already a small write-up in:

- `analysis/bootstrap_subsample_study.md`

That summary used a smaller setting than the full W&B sweep and reached this practical conclusion:

- ridge recovery was fairly stable
- coherence recovery remained unreliable
- bootstrap/subsample reduced variance in some cases but increased bias
- bootstrap looked more useful as a diagnostic than as a default estimation path

This is consistent with the overall working hypothesis:

> bootstrap can help measure instability, but it is not a reliable fix for identifiability problems.

## Why The Current Sweep Is Slow

The currently launched sweep is expensive for two reasons.

### 1. The sweep grid is large

Current grid in `../inductive-bias-bootstrap/sweeps/bootstrap_bias_recovery.yaml`:

- `gt_ridge_lambda`: 3 values
- `gt_coherence_lambda`: 2 values
- `resample_mode`: 3 values
- `sample_fraction`: 3 values
- `depth`: 2 values
- `use_normalized`: 2 values
- `seed`: 3 values

Total: `648` runs

### 2. Each run contains 10 sequential replicate fits

Each run does:

1. one predictive model fit
2. then `n_replicates = 10` separate bias-estimator fits

This makes runtime dominated by the repeated bias-estimation loop rather than the predictive model training.

## Recommended Sweep Simplification

If Codex is asked to tighten the experiment rather than preserve the current large sweep, the strongest cuts are:

1. Reduce `n_replicates` from `10` to `5`
2. Drop `sample_fraction = 1.0` for `subsample` and `bootstrap`
3. Drop one of the depth values for the bootstrap study
4. Potentially reduce seed count from `3` to `2`
5. Potentially reduce `bias_max_epochs` if early stopping is consistently much earlier

These changes should be considered before launching another large resampling sweep.

## Synthetic Task / Teacher-Student Notes

The repo already contains a synthetic-style path that is likely better for studying bootstrap than MNIST:

- `experiments/function_class_identifiability.py`

It generates synthetic regression targets from a known function class:

- `linear`
- `polynomial`
- `sine`

In discussion, "teacher" meant the hidden data-generating process used to create targets from inputs. In this repo, that corresponds to the synthetic target-generation logic, while the trainable deep ReLU model is the "student."

This synthetic path is probably the best place to move the bootstrap question next, because:

- it is much faster than MNIST
- it already has controlled geometry regimes
- it isolates identifiability better than image classification

## Controlled Collinearity Framing

The most important conceptual distinction for future work is:

- **fundamental identifiability failure**
  - bias gradients are nearly collinear
  - many lambda combinations explain the same target gradient
  - bootstrap will not rescue this

- **finite-sample estimation noise**
  - bias gradients are well separated
  - target-gradient estimation is noisy
  - bootstrap/subsampling may stabilize estimates

The existing function-class matrix already operationalizes this:

- `high_collinear`: `ridge,nuclear_norm`
- `low_noncollinear`: `ridge,weight_coherence`

This is likely the cleanest place to continue the bootstrap question.

## Cleanup Notes

The bootstrap worktree currently has runtime-generated artifacts while the Slurm array is active, including an untracked nested directory:

- `../inductive-bias-bootstrap/inductive-bias-bootstrap/`

It appears to contain run-specific checkpoint directories. Do **not** delete this while the active sweep is still running.

After the sweep is complete, Codex should decide whether to:

1. remove those generated artifacts,
2. move them somewhere more intentional, or
3. expand `.gitignore` in the bootstrap worktree / branch so future runs do not clutter `git status`.

## Suggested Next Actions For Codex

If continuing from here, the best next steps are:

1. Check the status and partial results of Slurm job `46011` and the W&B sweep.
2. Decide whether to let the large 648-run sweep finish or stop it and relaunch a tighter sweep.
3. If the goal is understanding rather than scale, move the bootstrap/subsample experiment onto the synthetic function-class pipeline instead of MNIST.
4. Produce one short go/no-go summary:
   - Does bootstrap reduce variance?
   - Does it worsen bias?
   - Does the answer depend on high-collinear vs low-noncollinear geometry?
5. If bootstrap remains useful only as a diagnostic, keep it out of the default estimation workflow and document it as optional analysis.

## Handy References

- Plan file: `.cursor/plans/identifiability_experiment_cleanup_7076a220.plan.md`
- Synthetic identifiability entrypoint: `experiments/function_class_identifiability.py`
- Matrix driver: `experiments/run_function_class_matrix.py`
- Matrix summary: `analysis/function_class_matrix_summary.md`
- Bootstrap summary: `analysis/bootstrap_subsample_study.md`
- Bootstrap worktree script: `../inductive-bias-bootstrap/experiments/bootstrap_bias_recovery.py`
- Bootstrap worktree sweep: `../inductive-bias-bootstrap/sweeps/bootstrap_bias_recovery.yaml`
- Canonical Slurm launcher: `scripts/wandb_sweep.slurm`
