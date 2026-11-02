# Dropout training continuation

Continue the 30 verified final endpoints (depth 3, width 256, dropout 0/0.3/0.5,
original ten seeds). Keep the original 54,000-example training split and dropout
cross-entropy objective, with no added weight decay. Restore SGD momentum; start
a new, explicitly recorded RNG stream and LR schedule. This is a continuation
from saved weights, not an exact replay of the historical stochastic stream.

The first stage allows 4,000 additional epochs, batch size 2,048, starting LR
0.01, multiplied by 0.2 after 1,000/2,500/3,500 additional epochs. Evaluate every
250 epochs, beginning at the original endpoint. Retain exact model/optimizer/RNG
states and all diagnostic checkpoints under `data/generated/dropout_convergence/`.
`latest.ckpt` allows interruption recovery with identical subsequent updates.

Practical stationarity requires both:

- The approximate 95% upper bound on the dropout-averaged training-gradient norm
  is below `min(0.1 * initial_norm, max(1e-4, 0.1 * current_eval_gradient_norm))`.
  At zero dropout the second bound is simply `1e-4`, since train and eval
  gradients are identical. Initial norm is the original audit's MC mean norm.
- The complete MC interval for the remaining signed ridge coefficient
  `direct_dropout - endpoint` is inside
  `±max(1e-7, 0.05 * abs(endpoint_signed_lambda))`.

Screen with 32 independent full-dataset mask draws. When point estimates meet
the tolerances but uncertainty does not, refine to 128 then 512 draws. Require
two consecutive passing checkpoints and an independent 512-draw confirmation.
The jackknife/bootstrap intervals are approximate and do not provide a formal
sequential-testing guarantee. A negative norm estimate or an interval crossing
zero alone never qualifies as convergence. Record `budget_exhausted` if the
declared tolerances are not met; do not label such networks converged. Positive
ridge alignment is an outcome, not a stopping requirement.

Run the seed-123 pilot first using `sbatch --array=0,10,20` with `locust.sbatch`,
then review diagnostics before launching the other 27 entries with at most six
concurrent GPUs. The batch script runs the explicit Snakemake rule. Set
`DROPOUT_CODE_REVISION` and `DROPOUT_SOURCE_SHA256` to the transferred snapshot.
Dependencies are installed from the frozen lock in an isolated Locust workspace.

Local workflow check (does not train):

```bash
uv run snakemake -s workflow/Snakefile --dry-run --cores 2 \
  data/generated/dropout_convergence/d3_w256_do0.3_s123 \
  --config dropout_recovered_only=true optimizer_fix_stats_recovered_only=true
```
