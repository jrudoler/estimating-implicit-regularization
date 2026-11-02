# Resume on Locust after the convergence pilot

Prepared 2026-11-02 for the existing **NeurIPS rebuttals** Codex chat,
thread `019f9eae-a944-7630-98d7-602bd869137c` on `locust-login`.
That chat's working directory is `/home/jrudoler/inductive-bias`.

## Current user intent

Continue training the dropout networks far enough that the endpoint ridge
coefficient can be interpreted using stationarity. The user then requested:
finish the pilot, preserve all work, and hand execution back to the existing
Locust chat. **No remaining seeds or longer continuation have been submitted.**
Resume the scientific experiment there; do not repeat the completed pilot or
launch the same short schedule across all seeds without reviewing its failure.

## Git and runtime

- Branch: `agent/neurips-rebuttal-experiments`. The commit containing this file
  preserves the estimator correction, previous audit, corrected figure, and new
  convergence implementation. Local and Locust checkouts are synchronized using
  a git bundle based on `d0dd4da305bd1b7a71d13ccef06bf1ab10554c3d`.
- Locust checkout: `/home/jrudoler/inductive-bias`.
- Isolated pilot workspace: `/shared_data0/jrudoler/dropout-convergence.ptSLQ3`.
- Frozen runtime: `/home/jrudoler/.venvs/dropout-convergence-ptSLQ3`, linked as
  `.venv` in both the pilot workspace and the main Locust checkout. Dependencies
  are from `uv.lock`: torch 2.11.0+cu128, Lightning 2.6.1.
- The executed snapshot is `continuation-v1.tar.gz` in the pilot workspace,
  SHA256 `05a1610fc5bae845fe10fd2acfa86b9b509cbccb525d06d0792f6515f663e43c`.
  It predates the later summary entrypoint and this handoff document. Preserve it.
- Use `PYTHONPATH=src UV_NO_SYNC=1 .venv/bin/python ...` or the Snakemake rules.
  The environment is shared with the frozen pilot workspace; do not mutate its
  dependencies while any future jobs are using it.
- Parent repo and `paper/` are independent. Do not edit, update, stage, or reset
  the manuscript. The remote paper checkout was already at `7e48ecf` while the
  parent records `dd1fd1f`; that existing difference is preserved. The Mac also
  retains its modified manuscript build artifacts and unrelated figure files.

## Completed pilot and results

SLURM array **110657**, entries **0,10,20**, ran seed 123 at dropout 0/0.3/0.5.
It initially queued in `whartonstat`, then moved to the permitted `standby` QOS
on the same Locust cluster because the primary queue was full. All three jobs
completed; all three scientific statuses are **`budget_exhausted`**, not
converged. No follow-on array or monitoring automation was created.

Each run restored the verified original endpoint's weights and SGD momentum,
kept the exact original 54,000-example MNIST split, and trained the same dropout
cross-entropy objective, without weight decay. Training used batch size 2048,
4000 additional epochs, LR 0.01 decayed by 0.2 after epochs 1000/2500/3500.
The stochastic stream was explicitly restarted with a recorded seed.

Final **signed** coefficients and approximate upper gradient-norm bounds:

| Dropout | Endpoint coefficient | Direct dropout coefficient | Remaining coefficient (direct minus endpoint) | Training-gradient norm upper bound |
|---|---:|---:|---:|---:|
| 0 | 2.83606e-6 | 0 | -2.83606e-6 | 2.15173e-4 |
| 0.3 | 5.71301e-7 | -6.34741e-6 | -6.91871e-6 | 2.19526e-3 |
| 0.5 | 1.03109e-6 | -9.43483e-6 | -1.04659e-5 | 4.12077e-3 |

All final gradient-norm limits were 1e-4. All final coefficient and norm tests
failed; coefficient limits were about 1e-7. These are three single-seed pilot
results, not ten-seed conclusions. The direct *nonnegative* ridge coefficients
are zero. Lower endpoint estimates alone do not establish stationarity.

For each tag `d3_w256_do{0.0,0.3,0.5}_s123`, results live under
`/shared_data0/jrudoler/dropout-convergence.ptSLQ3/data/generated/dropout_convergence/<tag>/`:

- `result.json`: complete trajectory, status, protocol, source hashes, and
  checkpoint provenance. `progress.json` is the last live status and may still
  say running; **result.json is authoritative after job completion**.
- `epoch_00000.ckpt` through `epoch_04000.ckpt` at 250-epoch intervals.
- `latest.ckpt`: model, optimizer, RNG state, history, and protocol signature.
- Logs: `logs/dropout-conv-110657_{0,10,20}.log` in the pilot workspace.

The main Locust checkout's `data/generated/dropout_convergence` is linked to
these outputs. Do not overwrite them when changing the continuation protocol.
The implementation deliberately rejects resuming `latest.ckpt` with a changed
protocol signature. A longer/new schedule needs a separately named study and
explicit checkpoint import, or a carefully recorded extension mechanism.
Also, the current LR milestones are fixed: merely raising `--max-epochs` leaves
the LR at 8e-5 after epoch 3500. Review the schedule as well as the epoch budget.

## Convergence definition and next work

Read `README.md`, `src/core/dropout_convergence.py`, and the three result files.
The declared gate requires a small *upper uncertainty bound* on the expected
dropout training-gradient norm, plus a small interval for its ridge projection.
At zero dropout the norm gate is absolute; otherwise it includes a relative gate
and at least a tenfold norm reduction from the initial audit. Coefficient bias
must fit inside `±max(1e-7, 0.05*abs(endpoint_signed_lambda))`.

Diagnostics start with 32 independent full-dataset mask draws. Near candidates
can refine to 128/512; two passing checkpoints need an independent 512-draw
confirmation. A negative squared-norm estimate or an interval crossing zero is
not evidence of convergence. Do not weaken tolerances simply to obtain a pass.

The next experiment should test a better sustained optimization schedule or a
larger budget on these pilot seeds, recording any optimizer change. Only then
expand the validated protocol to the original ten seeds at all three rates.
Assess whether the ridge contribution remains negative/poorly aligned or tends
toward zero; convergence does not guarantee positive ridge regularization.
Retain finite-budget failures rather than excluding them from the results.

`analysis/dropout_convergence/summarize.py` and its Snakemake rule require all
30 distinct endpoints; they intentionally cannot summarize this three-run pilot
as if it were the full experiment. A normal request for the aggregate workflow
target would launch the other 27 runs, so do not run that target yet.

## Prior evidence that supersedes older chat conclusions

- Production ridge estimation now uses sample-weighted, float64 accumulated
  full-dataset gradients and exact nonnegative scalar least squares. Signed
  coefficients and fit diagnostics are retained; legacy iteration is selectable.
- Direct-lambda fitting plus appropriate LR decay agrees with closed form to
  1.03e-7 relative error on the identical full-dataset objective at 30 saved
  endpoints. Removing the scheduler at the tested constant rates failed.
- The September audit directly measured `g_train = E_mask[grad L_dropout]` with
  32 full-dataset draws at 90 checkpoints. The dropout contribution is
  `g_train - g_eval`; the endpoint proxy is `-g_eval`. Their ridge projection
  difference is the remaining training gradient's projection.
- All 90 original audit checkpoints had positive lower squared-norm bounds.
  Comparing 0.8T with T does NOT establish proximity to a stationary network.
  The older Locust chat's claim that nonstationarity was within about 4% at
  0.8T is superseded by the direct gradient measurements.
- The 480-cell corrected figure is a recovered historical retrained grid.
  Its original endpoints were not saved, so it is not a same-weight comparison
  against the original W&B sweep. Do not launch a new 480-run grid silently.
- Read `analysis/dropout_checkpoint_audit/README.md` for the complete audit.

## Preserved data and checks

- Previous remote audit: `/shared_data0/jrudoler/dropout-audit.jiQDhK`.
- Original trajectory checkpoints: the main checkout's
  `data/generated/dropout_trajectory_lightning/checkpoints`.
- Full local dropout artifacts, including recovered checkpoints and all audit
  outputs, were backed up as `local-handoff-data.tar.gz` in the pilot workspace,
  SHA256 `9c4b10c6c4f24515588d716b82bfa6862a82ff31f11cb3f34770e0f0f706c698`.
  Extracted copy: `handoff/recovered-local/` under that workspace. The main
  checkout's `data/generated/dropout_checkpoint_audit` links to this recovered
  copy. Historical main-checkout datasets and sweep outputs were preserved.
- Pilot checkpoints/logs have a separate archive at
  `handoff/pilot-110657.tar.gz` under the pilot workspace.
- `data/generated/dropout_convergence/execution_manifest.json` records run and
  handoff metadata; generated data are intentionally not tracked by git.
- Nineteen targeted tests passed, including the negative-norm and coefficient
  uncertainty gates, dropout-disabled estimation, and deterministic replay from
  restored RNG and optimizer state. Pilot and full-design workflow dry runs passed.
- Nothing was merged or published; the git bundle transfers the branch directly
  between the Mac and Locust. Existing unrelated files remain untouched.
