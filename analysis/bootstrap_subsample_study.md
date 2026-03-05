# Bootstrap/Subsample Worktree Study

This report summarizes the isolated resampling study run in a separate git worktree on branch `bootstrap-resampling`, using:

- script: `experiments/bootstrap_bias_recovery.py`
- command family: `uv run python experiments/bootstrap_bias_recovery.py ...`
- modes compared: `full`, `subsample` (`sample_fraction=0.5`), `bootstrap` (`sample_fraction=0.5`)
- shared settings: MNIST, depth=2, width=64, `gt_ridge_lambda=0.05`, `gt_coherence_lambda=0.01`, `n_replicates=3`, normalized estimator enabled

## Aggregate Results

| Mode | Ridge mean | Ridge rel_err | Ridge CV | Coherence mean | Coherence rel_err | Coherence CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.048923 | 0.0215 | 0.066 | -0.007430 | 1.7430 | 98.425 |
| subsample | 0.049532 | 0.0094 | 0.038 | 0.277784 | 26.7784 | 2.957 |
| bootstrap | 0.050183 | 0.0037 | 0.025 | -0.461959 | 47.1959 | 1.881 |

Identifiability diagnostics were stable across modes for this setup:

- `Gram kappa`: 1.00
- `cosine_similarity` (ridge vs coherence penalty gradients): 0.0000

## Interpretation

- Ridge estimation is stable in all three modes, and replicate variability decreases from full -> subsample -> bootstrap.
- Coherence estimation remains unreliable; resampling decreases coefficient variance but substantially increases mean bias away from the ground truth.
- For this two-bias setting, bootstrap/subsample does **not** improve the practical identifiability of the coherence component, even when gradient geometry looks favorable.

## Practical Takeaway

Resampling appears useful as a variance diagnostic, but not as a universal fix for bias-strength recovery. It should remain an optional analysis mode rather than the default estimation pathway.
