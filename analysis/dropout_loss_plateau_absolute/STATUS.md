# Absolute-loss experiment execution status

Completed and verified 2026-11-03. The explicitly authorized **480-model**
Figure 5 rerun is complete: all models met the training-loss stopping rule,
the full checkpoint audit passed, and the actual final PDFs were rendered and
visually inspected. Use the log-scale figure for interpretation. The original
relative-loss study and earlier gradient-gated studies are preserved.
Manuscript files remain untouched.

## Pilot execution

- Seven pilots: array **111681**, whartonstat / normal, node9, one GPU each.
- All seven pilots and automatic review **111688** completed successfully.
  Each reached the declared training-loss plateau and passed practical
  coefficient robustness after doubling training. Stopping decisions were
  independently replayed from all seven saved loss histories.
- Six pairs lie inside the predeclared +/-1e-7 practical-zero band. Their
  relative changes span 0.019% to 17.66%; this is **not** evidence that every
  coefficient has <=5% relative error. The depth-5/width-128 model is outside
  that band: 6.16652e-7 at stopping versus 5.90119e-7 after doubling (4.3027%
  relative drift). Direct dropout effects remain separate from endpoint fits.
- The pilot review receipt is `results/data/dropout_loss_plateau_absolute_pilot_review.json`,
  SHA256 `3f529a71683944d74de2335f26e916927ea5f383eec4d60c5c25095dc737223d`.
- The preserved original relative-loss run **111667_0** completed its extension
  to 200,000 epochs. Its stopping endpoint hit the 100,000-epoch cap, so its
  scientific status remains `budget_exhausted`. Review **111669** completed and
  correctly reports that the original relative-loss pilot did not pass its
  expansion gate. Its zero-dropout coefficient changed from 7.26130e-9 to
  4.95202e-9 (31.80% relatively, still inside the practical-zero band).

## Frozen source and storage

Use `data/generated/dropout_loss_plateau_absolute/source-v1/` for all runs.
Archive SHA256:
`15fb6023d4d207bf5eafb51e68998958528767a4c74790a57ae7c107ac18e278`.
Do not alter the frozen source or mix in the original relative-loss snapshot.

All new training outputs, intermediate/recovery checkpoints, and job caches
live on **node9-local** `/scratch/jrudoler/dropout_loss_plateau_absolute/`.
Published endpoints, summary JSON and `storage.json` locators live under
`data/generated/dropout_loss_plateau_absolute/`. No new writes to `/opt/group`.
Scratch is not a backup. Do not delete it: it contains the trajectory evidence.

At 14:22: home usage was approximately 47,182 MiB (~46.08 GiB), against a 50 GiB soft
quota; scratch had about 545 GiB free at 13:07. Publication serializes copies, verifies
hashes and refuses to cross 48 GiB minus a 256 MiB safety allowance. A copy
failure preserves complete outputs on scratch. No user files were deleted.

## Full grid completed and verified

`grid_tasks.txt` contains exactly 473 distinct configurations excluding the
seven pilots, ordered with wider/deeper models first. Its SHA256 is
`f1f5f78d8c63812d6f521b1aec0988f00a3de6cec0bdc8acf3ccea3b53b3a33c`.
Combined with pilots it covers all 6 architectures x 8 rates x 10 seeds.

1. **111699**, indices `0-472`: all 473 grid tasks completed with exit 0, plus
   the seven completed pilots. All 480 results have status `loss_plateau`; none
   exhausted the 100,000-epoch cap. Stopping epochs range from 4,800 to 24,200,
   median 10,100. No runs were excluded or rerun to obtain a preferred sign.
2. CPU-only throttle adjustment **111708** historically exited 1 while reporting
   already-finished tasks, but the update to eight concurrent GPU jobs applied.
   This was verified at the time; no retry is needed. The preserved original
   relative-loss run completed without preemption and retains its failed gate.
3. Linear figure job **111707**, log-scale figure job **112200**, and full audit
   **112031** all completed successfully after the training array. The figures
   have exactly equal numerical tables after sorting by configuration; all
   three receipts contain the same 480 unchanged source-result hashes.
4. The full audit replayed **50,558** saved 100-epoch loss blocks and verified
   **5,853** trajectory checkpoint files (23,283,897,041 bytes), including
   expected epoch coverage and all published endpoint copies. Recovery files
   were checked for presence, not reloaded. Pilot robustness, training split,
   source identity and stopping decisions passed. Receipt:
   `data/generated/dropout_loss_plateau_absolute/full_grid_verification.json`,
   SHA256 `74b05ef077b728f07418358e12a897f0d4849c9de8e79f306fb00ff8ce4a11b4`.
   This verifies the declared protocol and retention, not exact stationarity.
5. A separate final check compared every one of the 53 frozen training files
   and 10 frozen verification files byte-for-byte with their archived versions.
   Both archives, executed verifier dependencies, renderer/style hashes, actual
   PDF/table hashes, pilot review and task-list hashes matched. The original
   partial-audit receipts remain preserved but are superseded by the full audit.
6. Preferred figure:
   `results/figures/dropout_bias_ridge_panel_loss_plateau_absolute_symlog.pdf`.
   Its receipt is `data/generated/dropout_loss_plateau_absolute/grid_validation_symlog.json`.
   All 480 seeds are shown, with no missing cells; the scale retains zeros and
   the plotting code retains capped runs (none occurred). The actual PDF was
   rendered with Ghostscript and visually inspected. Its numerical table is
   `results/data/dropout_loss_plateau_absolute_grid_symlog.csv`.
7. The frozen linear figure is preserved at
   `results/figures/dropout_bias_ridge_panel_loss_plateau_absolute.pdf`, with its
   original `grid_validation.json` and `results/data/dropout_loss_plateau_absolute_grid.csv`.
   Visual inspection found that its shared linear scale compresses most panels
   and crowds the top-left header; use the readable log-scale version above.

Twenty-one relevant tests passed again at 13:15, including storage quota guards,
loss-controller behavior, interruption replay, publication, and synthetic
480-cell/interim plotting and full-grid verification. Six verifier tests cover
missing blocks, incorrect patience/LR/status, corrupted or missing checkpoints,
wrong node, short extensions, incomplete grids, a complete synthetic 480-model
grid, changed frozen source and a failed pilot gate. The reusable history and
trajectory checks also passed on three real models (two extended pilots and
one depth-5/width-256 grid run). Completed pilots' loss stopping decisions were independently
replayed from their saved 100-epoch means. New log-scale assertions also verify
all 480 plotted points, preservation of zero/capped estimates, exact CLI output
hashes and axis settings. The standalone rule's dry-run refused the incomplete
grid as intended; it has no training rule. Both frozen source archives still
match their recorded SHA256 values. No commits, pushes or merges made.

The user requested and was shown a separate interim picture with 163 completed
models, snapshot `20261103T165406Z`. Its PDF/CSV/receipt have explicit interim
names; see README for the reproducible rule and qualifications. The frozen
training/final-figure source remains untouched. The final full-grid figures now
supersede that interim picture: all six architectures have 80/80 results. No
experiment or follow-up jobs remain active. See README for the final numerical
summary and the distinction between a coefficient trend and explanatory fit.

## Training-accuracy snapshot

At the user's request, CPU-only checkpoint inference evaluated a fixed snapshot
of 263 completed stopping endpoints on the same 54,000 training examples. No
model weights were changed, no training GPUs were taken, no validation/test data
were evaluated, and no W&B activity was involved. Results, model/checkpoint and
split hashes, and deterministic dropout-mask seeds are preserved in
`data/generated/dropout_loss_plateau_absolute/accuracy/20261103T173935Z.json`.

261/263 models had exactly 100% clean (dropout-off) training accuracy. The two
width-128, dropout-.5, seed-123 pilots made 1 error (depth 3) and 6 errors
(depth 5), respectively. Dropout-on evaluation used one complete pass with one
independent mask per example, not a many-draw uncertainty estimate.

For dropout .5, the ten-seed matched averages of original logged training
accuracy versus new dropout-on checkpoint accuracy were:

| Depth / width | Original logged accuracy | New checkpoint accuracy |
| --- | ---: | ---: |
| 3 / 256 | 98.50815% | 99.97722% |
| 3 / 512 | 99.20944% | 99.99685% |
| 5 / 256 | 98.46185% | 99.95611% |
| 5 / 512 | 99.31926% | 99.99389% |

Important limitations: none of the 480 recorded original manuscript checkpoint
paths exists locally. Original values above are archived final-epoch dropout-on
training logs from `data/generated/dropout_bias_estimation/runs.parquet`, not
reevaluation of original manuscript weights. Thirty separately saved old-protocol
replays were also evaluated and are explicitly labeled as replays in the JSON.
The old protocol used validation-loss patience 50 and a 500-epoch cap; the new
protocol also changes the learning-rate schedule. Differences cannot be
attributed solely to the stopping tolerance. This remains an incomplete-grid
accuracy snapshot; accurate classification does not establish stationarity of
loss gradients or stability of the fitted coefficient.

Live files created by cluster jobs may appear stale/missing in ordinary command
reads; escalated read-only checks see the authoritative shared home filesystem.
This was observed for the review receipt. Verify before diagnosing a failure.

The user additionally cautioned against W&B storage consumption. Audit at 11:02
confirmed no W&B initialization/logger/upload path in the executed workflow,
no W&B service processes on node9 and no W&B run-file patterns in study scratch.
All experiment outputs remain local. The lazy model artifact-download helper
is not called. No account artifacts or unrelated caches were deleted or changed.
