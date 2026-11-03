# Matched replacement of final Figure 5 dropout endpoints

The fixed scope is ten depth-3, width-256, dropout-0.3 seeds from the completed
`dropout_loss_plateau_absolute` grid. For each seed, train two replacements from
that target's archived initial weights and RNG state: dropout disabled with no
explicit penalty, and dropout disabled with the seed's exact nonnegative fitted
ridge coefficient. No baseline subtraction, pooled coefficients, or coefficient
search is used. The no-penalty replay must reproduce the original final-grid
zero-dropout baseline's parameter vector within relative tolerance 1e-6.

All shared model/data/estimation/controller code is imported from the original
hash-checked source snapshot. SGD, initialization, split, minibatch size, precision,
100-epoch training cross-entropy blocks, 1e-7 absolute improvement, LR reductions,
minimum LR, stopping patience, and 100,000-epoch cap match the final grid. The
explicit penalty adds `2 * lambda * theta` to each predictive gradient, including
biases. Cross-entropy alone still controls stopping. The target and replacements
need not stop at identical epochs because this is an adaptive stopping rule.
The two replacement conditions share their initial RNG stream; the original
positive-dropout trajectory consumes additional randomness for dropout masks.

Train and test metrics are evaluated only at the stopping endpoints. Primary
comparisons are test accuracy and prediction disagreement with the paired
original dropout target. We also retain cross-entropy, probability differences,
parameter distance, and full loss/stopping history. Summaries report all ten
seeds and paired accuracy differences, with descriptive percentile bootstrap
intervals across seeds (20,000 resamples, fixed seed). No selection by accuracy,
coefficient, or desired interpretation is allowed.

Reproduction requires the final grid's initial checkpoints at the paths recorded
in its result JSON, the published target and zero-dropout endpoints, the frozen
source bundle, and MNIST under `data/raw/`. The initial checkpoints currently
reside on node9, so the training rule requests that node. Restore those archives
before running this branch on another machine; result JSON alone is insufficient.
Replacement checkpoint and recovery files live under node9 scratch; only stopping
endpoints and JSON records are copied home through the existing quota-checked publisher.
All training is orchestrated by Snakemake:

```bash
uv run snakemake -s workflow/rules/matched_dropout_replacement.smk \
  --profile workflow/profiles/slurm --jobs 8 matched_dropout_replacement_all
uv run snakemake -s workflow/rules/matched_dropout_replacement.smk \
  --cores 2 --allowed-rules summarize_matched_dropout_replacement \
  -- results/data/dropout_matched_replacement.json
```

No W&B runs or uploads are used. Historical validation-stopped replacement
results are preserved in their separate directory and will not be presented as
matched results.
