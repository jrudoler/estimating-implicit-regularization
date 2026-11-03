# Absolute training-loss plateau criterion

Separate pilot revision, declared on 2026-11-03 before these runs are launched.
The original relative-loss study remains under `dropout_loss_plateau`; its
running jobs, source snapshots, outputs, and stopping criteria are not changed.

## Motivation from the first pilot

The .3 and .5 runs stopped after 10,600 and 12,500 epochs respectively. The .3
extension to 21,200 epochs changed its fitted endpoint coefficient from
5.9360e-10 to 4.8878e-10: 17.66% relatively, but only 1.0482e-10 absolutely,
within the already declared practical-zero band. This must not be described as
small relative error. Its separate direct-dropout coefficient is -1.4034e-7
at stopping and -1.2236e-7 after extension; the endpoint projection is not an
accurate relative estimate of that signed direct effect.

Meanwhile the no-dropout loss keeps decreasing at more than 0.1% in relative
terms, even after 30,000 epochs with an already tiny absolute loss. This exposes
the drawback of a purely relative plateau rule near zero. We do not retroactively
stop or relabel that original run. It will finish under its original protocol.

## New fixed candidate

**Stop when the 100-epoch mean training cross-entropy has not decreased by more
than 1e-7 for 1,000 epochs.** This uses an absolute minimum improvement, the usual
patience-based early-stopping convention. Relative minimum improvement is zero.
The 1e-7 is a loss-unit practical tolerance, not a theorem or a coefficient bound.

All other choices are unchanged: fresh seeds, same 54,000 training examples,
dropout enabled during training, batch 2,048, SGD momentum .9, initial LR .05,
tenfold LR reductions after 500 plateau epochs down to .0005, resetting patience
after each reduction, maximum 100,000 epochs. No validation/test metric and no
gradient diagnostic enters stopping. All old gradient uncertainty gates remain
unchanged in their original studies.

Validate seven pilot configurations before expanding to the full 480:
the original depth 3, width 256, seed 123, dropout 0/.3/.5, plus dropout .5,
seed 123 at (depth,width) (3,128), (3,512), (5,128), (5,512). This uses the seven
free node9 GPUs while the preserved original zero-dropout run uses the eighth,
and adds cross-architecture sensitivity checks. Extend training by at least its
stopping duration at the terminal LR. Use exactly the earlier coefficient
comparison: both signed estimates inside [-1e-7,1e-7], or relative change <=5%.
Keep and report the raw changes and separate 512-mask direct-dropout comparisons.
Failure is preserved, not silently reclassified. Positive ridge is not a target.

## Shared implementation, separate outputs

The entrypoint, controller, Slurm scripts and plotting code are shared with the
original study, but use `--study-name dropout_loss_plateau_absolute` and the
standalone workflow config `plateau_study=dropout_loss_plateau_absolute`.
Frozen code for already running jobs is untouched. Outputs are isolated under
`data/generated/dropout_loss_plateau_absolute/`.

After successful pilot review, the intended final artifacts are
`results/figures/dropout_bias_ridge_panel_loss_plateau_absolute.pdf` and
`results/data/dropout_loss_plateau_absolute_grid.csv`, covering all 480 cells.
The figure remains an endpoint ridge-projection study, not automatically a
direct-dropout penalty estimate. Manuscript files remain untouched.

## Storage and compute constraints (user update, 2026-11-03)

`/opt/group` is reported extremely slow; no outputs, caches or work directories
for this study go there. Home usage was about 43 GiB of a 50 GiB quota. Node9 has
about 566 GiB free on local `/scratch`. Run on node9 via the approved whartonstat
partition, using `/scratch/jrudoler/dropout_loss_plateau_absolute/<tag>` for all
training outputs and per-job Snakemake/cache files. Local scratch is not a backup
and is tied to node9; retain it for future trajectory analyses.

After completion, copy only the stopping endpoint, extended endpoint (pilots),
and result JSON into the home study directory. Keep all intermediate/recovery
checkpoints on scratch. `storage.json` records their node and location. Copies
are hash-verified, serialized, and guarded by the actual home quota: leave at
least 2 GiB below the soft quota, plus a 256 MiB allowance. If quota cannot be
verified or capacity is unsafe, preserve the complete scratch output and fail
publication rather than filling home. The downstream figure reads only the
published endpoints and summaries. Existing home files are not deleted.

## W&B storage policy

This experiment does not initialize W&B runs, configure a W&B logger, sync run
files, or upload checkpoint artifacts. Training uses a plain SGD loop and local
JSON/checkpoint outputs. The shared model module's lazy W&B artifact-download
helper is not called by this workflow. Keep that distinction when reusing the
code: do not add automatic checkpoint uploads to this 480-model grid.

An audit on 2026-11-03 checked the executed frozen code, experiment logs,
node9 W&B-service processes, and W&B run-file patterns under this study's scratch
directory. No active logging/upload path or generated W&B run files were found.
Existing W&B account artifacts and unrelated local caches were not modified.

## Intermediate picture (not the final 480-model figure)

The user requested an intermediate picture while the full grid runs. The separate
`plot_dropout_loss_plateau_interim` rule reads only already-published results,
verifies endpoint hashes and projection identities, and records missing cells.
It never starts training. Its snapshot receipt freezes the input list on reruns;
use a new UTC timestamp for a newer picture. For example, reproduce the first:

```bash
PYTHONPATH=src UV_NO_SYNC=1 .venv/bin/snakemake \
  -s workflow/rules/plot_dropout_loss_plateau_interim.smk --cores 1 \
  --allowed-rules plot_dropout_loss_plateau_interim \
  --config preview_snapshot=20261103T165406Z
```

The first snapshot has 163/480 stopping endpoints, all `loss_plateau`:
depth-5/width-512 80/80, depth-3/width-512 78/80, and five other pilot endpoints.
Outputs have the distinct stem
`dropout_loss_plateau_absolute_interim_20261103T165406Z`: the PDF is under
`results/figures/`, the CSV under `results/data/`, and the provenance receipt
under `data/generated/dropout_loss_plateau_absolute/previews/20261103T165406Z.json`.
Gray bands and per-cell counts show missing seeds. Median lines are restricted
to complete ten-seed groups. The shared y-axis is logarithmic above 1e-10 and
linear near zero, retaining zero-clamped coefficients. Point colors report the
fraction of the squared clean-gradient norm explained by nonnegative ridge.

Preliminary: width-512/depth-3 medians decrease from 8.416e-9 at dropout .05 to
2.100e-10 at .5 (all ten seeds per positive-rate cell); explained fractions fall
from 1.431% to .159%. Width-512/depth-5 medians at positive dropout range from
9.053e-10 to 2.272e-9, explaining .509% to .802%. These are endpoint projections,
not direct causal dropout penalties or proof of stationarity. Narrower models
are not yet adequately sampled. Do not generalize these incomplete results.

The reporting validator was factored into `src/core/loss_plateau_reporting.py`
for reuse by the working full and interim plot entrypoints. The final collector
still requires all 480 outputs; tests enforce this. Running training and the
queued final-figure job continue to use the untouched frozen `source-v1` code.

## Full-grid verification

`analysis/verify_dropout_loss_plateau/run.py` is a separate, CPU-only workflow
step using `src/core/loss_plateau_verification.py`. It requires every one of the
480 declared results; there is no partial success option. It verifies:

- Exact grid/task/pilot coverage and the original frozen training-source hash.
- The fixed training protocol, split, model/batch settings and unchanged pilot
  robustness criterion (including raw relative changes in the review).
- Independent replay of every saved 100-epoch training-loss block, LR reduction,
  patience count and stopping decision; budget-limited runs retain that label.
- Expected initial/periodic/endpoint/extension epochs, node-local paths and SHA256
  for every trajectory checkpoint; published endpoint copies; recovery-file
  presence. Recovery contents are not reloaded by this file-retention audit.

This is provenance, stopping-rule and storage verification, not a new scientific
stationarity test. It never changes training, drops budget-limited runs, requires
positive coefficients, uploads W&B artifacts or touches the manuscript.

After tests and three real-model smoke checks, its code was frozen separately at
`data/generated/dropout_loss_plateau_absolute/verification-source-v1/`, with
archive SHA256 `c49c15ea6805ee48ea92956cbfb6fd40a614237814e170431caff35fa76f0fdc`.
Do not alter that snapshot. Job **112031**, after `111699`, executes its standalone
Snakemake rule on node9. Only the small receipt
`data/generated/dropout_loss_plateau_absolute/full_grid_verification.json` is
written. The final figure remains job **111707** with its original frozen code.
Inspect both successful jobs, their receipts, and the rendered full figure
before marking the active goal complete.

## Full-grid log-scale presentation

The completed width-256/512 models and width-128 pilots already span several
orders of magnitude. A separate presentation rule makes these coefficients
visible on one shared axis without overwriting the frozen linear-scale figure:

```bash
XDG_CACHE_HOME="$PWD/.snakemake/cache" PYTHONPATH=src UV_NO_SYNC=1 \
  .venv/bin/snakemake -s workflow/rules/plot_dropout_loss_plateau_symlog.smk \
  --cores 1 --allowed-rules plot_dropout_loss_plateau_symlog
```

This is a plotting-only variant of `analysis/plot_dropout_loss_plateau/run.py`,
using `--yscale symlog`. It still requires all 480 declared results and validates
their checkpoint hashes and projection identities. It cannot launch training.
The axis is logarithmic above 1e-10 and linear near zero, preserving zero-clamped
estimates. Points show all seeds, lines show medians, and capped runs remain
visible as crosses. Linear-space violins are omitted in this view to avoid
distorting a density estimate through the nonlinear axis transformation.

Outputs have distinct names:

- `results/figures/dropout_bias_ridge_panel_loss_plateau_absolute_symlog.pdf`
- `results/data/dropout_loss_plateau_absolute_grid_symlog.csv`
- `data/generated/dropout_loss_plateau_absolute/grid_validation_symlog.json`

The receipt records the renderer, validator, style, figure and table hashes,
axis setting and the full source-result list. It does not replace the separate
node-local trajectory verification. No training protocol, frozen source,
existing figure, manuscript file or W&B artifact is changed.

## Completed full-grid results (2026-11-03)

All **480/480** models met the predeclared absolute training-loss stopping rule;
none reached the 100,000-epoch cap. Stopping took 4,800–24,200 epochs (median
10,100). These are stopping endpoints, including the original stopping endpoints
of the extended pilots, not a mixture of stopping and extended checkpoints.
The full audit verified 50,558 loss blocks and 5,853 retained trajectory files.
The linear and log-scale tables contain exactly the same 480 unique models in
48 ten-seed cells, with identical input-result hashes. Actual PDF previews were
visually inspected; the log-scale version is the recommended presentation.

At dropout .5, the ten-seed medians are:

| Depth | Width | Nonnegative ridge coefficient | Clean-gradient squared norm explained |
| ---: | ---: | ---: | ---: |
| 3 | 128 | 1.17840e-7 | 0.02760% |
| 3 | 256 | 3.01276e-9 | 0.11626% |
| 3 | 512 | 2.10013e-10 | 0.15879% |
| 5 | 128 | 8.08465e-7 | 0.06993% |
| 5 | 256 | 3.09614e-8 | 0.29429% |
| 5 | 512 | 1.37146e-9 | 0.60058% |

Across all 420 positive-dropout models, the median explained fraction is
**0.53610%**, with individual values ranging from **0.00209% to 2.47217%**.
Every signed coefficient happened to be positive; positivity was not a gate or
an assumption, and signed values remain available in the tables.

The coefficient trend is architecture-dependent: width-128 models rise at high
dropout, depth-3/width-512 decreases across positive dropout, and
depth-5/width-512 stays approximately flat near 1e-9. Zero-dropout controls also
have nonzero fitted coefficients. These observations do not establish a causal
dropout-to-ridge equivalence. A clear coefficient trend can coexist with very
weak projection fit; high training accuracy or a passed loss-patience rule is
not proof of exact stationarity. Pilot extensions support only the predeclared
practical robustness criterion, with its near-zero/relative-error qualifications.

All intermediate checkpoints remain on node9-local scratch, with published
endpoint copies and locators under `data/generated/dropout_loss_plateau_absolute`.
Scratch is not a backup; retain it for trajectory analysis. Home use was about
46.08 GiB of a 50 GiB quota at completion. No W&B uploads, manuscript edits,
commits, pushes, merges or deletions of existing experiment outputs were made.
