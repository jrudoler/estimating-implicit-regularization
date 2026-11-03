# Dropout Figure 5 with training-loss stopping

This is a new experiment, not a relabeling of the gradient-gated continuation
studies. All previous results and their uncertainty criteria remain unchanged.
The user authorized the **full 480-cell Figure 5 grid**, after a three-cell pilot,
on 2026-11-03. Manuscript files are not outputs of this workflow.

## Fixed candidate protocol (version 1)

- Fresh seeded MLPs, same architectures, ten seeds, eight dropout rates, exact
  original 54,000-example MNIST training split, batch size 2,048, SGD momentum
  0.9, initial LR 0.05. No extra penalty, no validation/test stopping or selection.
- Average sample-weighted dropout-on training CE in nonoverlapping blocks of
  100 epochs. A new best requires a **0.1% relative decrease**.
- After 500 epochs without that improvement, reduce LR by ten, down to 0.0005.
  Reset the best-loss tracker and patience after each reduction, allowing a
  full new opportunity to improve at the new rate. There are at most two drops.
- Stop after **1,000 epochs without improvement** (after LR reductions have
  had their opportunity). Hard cap: 100,000 training epochs. Hitting the cap is
  `budget_exhausted`, not a loss plateau or a convergence claim.
- Fit the exact signed and nonnegative scalar ridge projection at the final
  weights, using sample-weighted full-training-data gradients accumulated in
  float64. Retain initial, 1,000-epoch, and final checkpoints and every 100-epoch
  loss record. No gradient estimate influences the training stopping decision.

This uses the existing model and objective with an in-memory training loop;
it is not an exact replay of historical Lightning shuffling or initial states.
Code, dataset split, checkpoint, and source-archive hashes record provenance.

## Pilot and interpretation checks

First run depth 3 / width 256 / seed 123 at dropout 0, .3, and .5. After the
loss-based stop, continue each pilot for `max(stop_epoch, 5000)` extra epochs at
the terminal LR, with early stopping disabled. This at least doubles training.
Keep the original stopping endpoint as the figure's endpoint, not the extension.

Before expansion, require all three to stop on a loss plateau and compare signed
coefficients: either both values lie in the existing practical-zero band
`[-1e-7, 1e-7]`, or relative change is at most 5%. Always report the actual absolute
and relative changes, even within that band. This is a practical sensitivity
check, not proof of stationarity or relative accuracy for arbitrarily tiny fits.

At the two pilot endpoints only, preserve a separate 512-mask direct-dropout
gradient comparison. It does not control training or change the original
gradient-gate definitions. A stable endpoint projection is not automatically
dropout's causal penalty; disagreement in this comparison must remain visible.
If the pilot fails, preserve it and revise in a separately named study rather
than changing criteria after seeing the result. Do not silently expand.

## Reproducibility and layout

- Shared stopping controller: `src/core/loss_plateau.py`.
- Entrypoint: `analysis/dropout_loss_plateau/run.py`.
- Workflow: `workflow/rules/dropout_loss_plateau.smk` (standalone default: pilot).
- Intermediates: `data/generated/dropout_loss_plateau/`.
- Final figures and tables: `results/figures/`, `results/data/` through a plot rule.
- `locust.sbatch` takes an explicit task list and frozen source directory. Its
  per-cell working directories avoid shared Snakemake cache races. Combined logs
  are under `logs/slurm/`. Standby was explicitly approved by the user.
- An existing completed result or recovery checkpoint with a different signature
  is rejected rather than overwritten. Recovery restores model, optimizer,
  stopping state, trajectory, and training RNG.

No full-grid array is submitted until the pilot has been inspected. No manuscript
staging rule is part of this standalone workflow.
