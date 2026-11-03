# Execution status and continuation instructions

2026-11-03, approximately 10:09 America/New_York. Active goal: run the proposed
training-loss stopping experiment, regenerate the **full 480-model Figure 5**,
and show the user. The user explicitly selected the full grid, not an 80-model
single-architecture figure. **The goal is not complete.**

## Live jobs (recheck before acting)

- Update at 10:46: all seven revised absolute-loss pilots passed their declared
  practical extension checks. Full-grid array **111699**, 473 remaining cells,
  is now running on node9 with seven concurrent GPUs; figure job **111707**
  depends on it. Together with preserved original task **111667_0**, all eight
  GPUs are in use. The original task hit its 100k cap and is extending to 200k;
  it remains `budget_exhausted`. Current authoritative instructions are in
  `analysis/dropout_loss_plateau_absolute/STATUS.md`. Older next steps below
  describe the superseded relative-loss candidate, not the current grid.
- Update at 10:32: the .3 and .5 relative-tolerance pilots completed, both
  passing only the practical-near-zero branch of coefficient stability. Their
  relative coefficient drifts were **17.66%** and **6.00%**, respectively; do not
  call these <=5% relative stability. Original zero-dropout task **111667_0** is
  still running unchanged. Review job 111669 awaits it.
- A separate absolute-loss-tolerance revision now runs as **111681**, seven
  scratch-backed pilots on node9/whartonstat. Review job **111688** depends on it.
  See `analysis/dropout_loss_plateau_absolute/README.md` and its execution manifest.
  This revision follows the user's request to use node9's free GPUs and protect
  the 50-GiB home quota. All eight GPUs are now allocated: seven revised pilots
  plus the preserved original zero-dropout run. No full grid is submitted yet.
- Working source has been extended for the new study; it no longer matches the
  original frozen training snapshot. **Continue old runs only from their frozen
  sources.** No submitted archive or existing uncertainty definition was changed.
- Update at 10:14: **all three pilot tasks are RUNNING on node9**. The earlier
  late-evening queue estimate is obsolete; resources became available. Progress
  is around 8,000–9,000 epochs, with loss-triggered LR reductions already occurring
  for positive dropout. The zero-dropout loss is still decreasing. No pilot has
  a completed scientific result yet, and job 111669 still waits on the pilots.
- **111667**, array 0–2, three pilots, submitted to standby with user permission.
  At last check all were `PENDING (Resources)`, not running or failed. Slurm's
  current start estimate was November 3 at 22:40:57, not a guarantee. All GPUs
  in standby and whartonstat were allocated. Do not submit duplicate pilots.
- **111669**, a CPU-only pilot-review job, `PENDING (Dependency)`, automatically
  runs after successful completion of all three pilot tasks (`afterok:111667`).
  It writes `results/data/dropout_loss_plateau_pilot_review.json`; it cannot
  launch training or the full grid.
- No full-grid array is submitted yet. No app automation tool was available.

## Frozen sources

- Base branch `agent/neurips-rebuttal-experiments`, revision
  `b2318833a600030a631da996958210ac504bdcc8`. New code remains uncommitted.
- Training: `data/generated/dropout_loss_plateau/source-v1/` and
  `source-v1.tar.gz`, SHA256
  `fe32bbe8ba682ca110d4a5c900df5fbc5b33b494561af5f6862699f6afacb5db`.
  This snapshot includes the original training rule and all runtime helpers.
  The working training files still match this snapshot exactly.
- Review/figure tooling: `data/generated/dropout_loss_plateau/review-source-v1/`
  and `review-source-v1.tar.gz`, SHA256
  `b38248d473b4da662550a8532058a3d0739927b34a2a11fa30793249f64411b3`.
  Its training implementation matches the training snapshot; it additionally
  includes the pilot-review and full-grid plotting rules and tests.
- `data/generated/dropout_loss_plateau/execution_manifest.json` records jobs.
  Do not change the already submitted snapshots.

## What is implemented

Read README.md for the exact stopping and pilot-extension protocol. Model
initialization is fresh and seeded, not a continuation from the previously
gradient-gated endpoints. Every pilot's stopping endpoint is retained separately
from its extension. Intermediate checkpoints and every 100-epoch loss mean are
saved. A large relative coefficient drift inside the practical-zero band must
still be reported, not described as small relative error.

- `src/core/loss_plateau.py`: loss-only stopping and practical coefficient
  comparison. No gradient norm controls stopping.
- `analysis/dropout_loss_plateau/run.py`: resumable training and optional pilot
  extension/direct-dropout diagnostic. Final data go under
  `data/generated/dropout_loss_plateau/<tag>/result.json`.
- Standalone `workflow/rules/dropout_loss_plateau.smk` defaults to three pilots;
  it is also included from `workflow/Snakefile`.
- `locust.sbatch` executes a single explicit target with a per-cell working
  directory, avoiding the shared Snakemake cache race seen in the earlier study.
- `review_pilot.py` validates the three stopped and extended checkpoint hashes,
  recomputes coefficient drift, and reports `ready_for_grid_sensitivity_study`.
  It also exposes direct-dropout comparisons; stability alone is not stationarity.
- `analysis/plot_dropout_loss_plateau/run.py` requires all 480 cells, checks
  checkpoint hashes, common training protocol/source, and projection algebra.
  It does not discard budget-exhausted runs. Outputs:
  `results/figures/dropout_bias_ridge_panel_loss_plateau.pdf`,
  `results/data/dropout_loss_plateau_grid.csv`, and
  `data/generated/dropout_loss_plateau/grid_validation.json`.

## Tests already passed

26 targeted tests: stopping controller, existing convergence/scalar routines,
checkpoint/RNG/momentum replay, full runner interruption during extension,
completed-result preservation, budget-exhaustion labeling, and synthetic
480-cell figure generation/validation. Synthetic data exist only in pytest temp
directories, never as experiment outputs. Pilot, main-workflow single-cell, and
full-figure dry runs all passed. The latter contains exactly 480 training jobs
and one plotting job. No actual new scientific figure has been produced yet.

## Next actions

1. Recheck `squeue`/`sacct` for 111667 and 111669 with network permission. Pending
   resources is a verified wait, not an experiment failure. Inspect each task's
   `logs/slurm/dropout-loss-plateau-111667_<0,1,2>.log` once it starts.
2. If runtime errors occur, preserve diagnostics and recover the existing run;
   do not restart merely because a polling call times out. If code changes are
   necessary, preserve the old snapshot and use explicit new provenance.
3. Once the pilot review exists, inspect both the recorded stopping decisions
   and the extension changes. If any pilot caps or fails the declared stability
   check, do not launch 477 other cells blindly. Preserve this result, diagnose,
   and design a separate version if needed. A purely relative loss tolerance
   may be demanding for a nearly zero, continually decreasing training loss;
   do not silently add an absolute tolerance to the running experiment.
4. If the pilot is satisfactory, run the remaining **477** cells using the same
   frozen training source (at most six concurrent GPUs). Generate an explicit
   task list from depths (3,5), widths (128,256,512), rates
   (0,.05,.1,.15,.2,.3,.4,.5), seeds (123,42,666,314,17,111,325,643,432,51), excluding
   the three existing pilot tags. The sbatch script requires absolute
   `DROPOUT_SOURCE_DIR`, `DROPOUT_SOURCE_SHA256`, `DROPOUT_CODE_REVISION`, and
   `DROPOUT_TASK_FILE`. Record the submitted array and task-list hash.
5. Once all 480 results exist, use the plotting rule with
   `--allowed-rules plot_dropout_loss_plateau -- results/figures/dropout_bias_ridge_panel_loss_plateau.pdf`
   so plotting cannot silently retrain anything. Set `XDG_CACHE_HOME` to the
   repo's `.snakemake/cache`, `MPLCONFIGDIR` to `.snakemake/matplotlib`, and use
   `.venv/bin/snakemake`. Validate and visually inspect the real PDF, then show
   it to the user with the scientific qualifications supported by actual data.
6. Only mark the active goal complete after the full requested experiment and
   figure have been checked. Queue submission, code, tests, or a pilot figure
   alone do not satisfy it.

All older generated results and uncertainty definitions are intact. `paper/`
is untouched, including its preexisting modified submodule pointer. Unrelated
untracked CIFAR data are untouched. No commits, pushes, merges, dependency
updates, or manuscript changes were made in this goal turn.
