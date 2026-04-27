# Function-Class Identifiability Matrix (Phase 2)

Matrix runs were executed with:

- entrypoint: `experiments/function_class_identifiability.py`
- driver: `experiments/run_function_class_matrix.py`
- regime definitions:
  - `high_collinear`: `ridge,nuclear_norm`
  - `low_noncollinear`: `ridge,weight_coherence`
- grid: function class in `{linear, polynomial, sine}`, depth in `{1,3}`, width in `{64,256}`, seed `42`
- training settings: `n_samples=1024`, `max_epochs=20`, `bias_max_epochs=120`, normalized estimator enabled, ground-truth lambda auto-balancing enabled

## Aggregated Results

| Regime | Function | Mean(gt_mean_rel_error) | Mean(selected_max_abs_cos) | Mean(selected_cond) |
| --- | --- | ---: | ---: | ---: |
| high_collinear | linear | 10.8944 | 0.7931 | 11.5990 |
| high_collinear | polynomial | 17.8927 | 0.7893 | 13.6376 |
| high_collinear | sine | 1.2557 | 0.7020 | 14.8202 |
| low_noncollinear | linear | 5.9283 | 0.0000 | 1.0000 |
| low_noncollinear | polynomial | 21.3913 | 0.0000 | 1.0000 |
| low_noncollinear | sine | 0.6779 | 0.0000 | 1.0000 |

## Notes

- Geometry separation is clear: the non-collinear regime consistently yields near-orthogonal basis geometry (`selected_max_abs_cos=0`, condition number `~1`).
- Recovery quality is function-class dependent; `sine` behaves better than `polynomial` in both regimes in this run.
- This matrix supports the geometric identifiability argument, while also showing that good geometry alone does not guarantee low estimation error in every setting.
