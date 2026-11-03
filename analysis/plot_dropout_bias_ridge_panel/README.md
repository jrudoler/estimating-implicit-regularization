# Publication Figure 5

The standard paper target now uses the completed 480-model
`dropout_loss_plateau_absolute` study, not the historical short-training grid.
The previous iterative and closed-form renderers remain available in
`legacy.py` and `analysis/plot_closed_form_dropout_grid/run.py`. Frozen training
sources, exploratory PDFs, and their receipts are unchanged.

Rebuild only the publication figure and stage it into the manuscript:

```bash
XDG_CACHE_HOME="$PWD/.snakemake/cache" PYTHONPATH=src UV_NO_SYNC=1 \
  .venv/bin/snakemake -s workflow/Snakefile --cores 1 \
  paper/figures/dropout_bias_ridge_panel.pdf \
  --allowed-rules plot_dropout_bias_ridge_panel stage_paper_figure \
  --config dropout_recovered_only=true optimizer_fix_stats_recovered_only=true
```

This command cannot train or contact W&B. Follow the manuscript refresh and
commit/push rules in `AGENTS.md` before staging manuscript changes.

## Inputs and provenance

The renderer requires all 480 result JSONs, verifies their published endpoint
checkpoint hashes and scalar-projection identities, and matches them against
`data/generated/dropout_loss_plateau_absolute/full_grid_verification.json`.
The latter audit independently replayed the loss stopping decisions and checked
the retained trajectory files. Neither check proves exact stationarity.

The figure is `results/figures/dropout_bias_ridge_panel.pdf`. Its exact numerical
table and provenance receipt are
`data/generated/plot_dropout_bias_ridge_panel/{grid.csv,validation.json}`.
The receipt records all source hashes, the audit hash, rendering code/style,
axis/color settings, and output hashes. The staging rule copies this same PDF
to `paper/figures/dropout_bias_ridge_panel.pdf`.

## Presentation

The original 2-by-3 layout, project style, gray violins, individual seed points,
gray width/depth labels, and `crameri:imola` palette are retained. No explanatory
titles, footers, or inset commentary appear on the figure; qualifications belong
in the caption and accompanying text. All 480 points, including 60 zero-dropout
controls, are shown with deterministic seed jitter.

The coefficient axis is logarithmic above 1e-10 and linear near zero, so zero
estimates cannot disappear. Violins are estimated in these displayed coordinates
and truncated to observed seed ranges, rather than stretching a linear-space
density through a log transformation. No points are removed or baseline-subtracted.

Color is the manuscript's uncentered gradient-fit diagnostic:

`R_grad² = 1 - ||g + 2 λ_hat θ||² / ||g||²`, where `g` is the full clean-training
loss gradient and `λ_hat = max(0, -<g,2θ>/||2θ||²)`.

For this one-dimensional nonnegative projection, this equals
`max(cos(-g,2θ), 0)²`. A single fixed 0-to-1 color scale is shared by all panels.
The exploratory plot used the same diagnostic but normalized its color scale
to the observed maximum (about 0.609). The old manuscript figure instead colored
by raw fitting loss, which is not this normalized diagnostic.

## Zero-dropout interpretation

Zero dropout removes the dropout contribution, not finite-training residuals.
A loss-patience stop does not require a zero loss gradient. A tiny gradient can
still have substantial relative alignment with `-2θ`, giving positive endpoint
ridge and high R² without any causal dropout penalty.

For depth 3 / width 256 / seed 123, at the stopping epoch 11,900:
`λ_hat = 5.117617184355603e-8`, `R_grad² = 0.481761547268034`, and the clean
gradient norm is `3.6737197782694404e-6`. The separately computed direct-dropout
coefficient is exactly zero. Doubling training to epoch 23,800 gives
`λ_hat = 5.11665401063686e-8`, while the direct coefficient remains zero.
This passes the predeclared practical robustness check; it does not turn the
endpoint projection into a dropout-specific estimate. Across architectures,
zero-dropout median coefficients range from 1.83e-8 to 6.41e-8 and median R²
from 0.137 to 0.597. Among positive-dropout runs the median R² is only 0.005361.

These controls must remain visible. Subtracting their coefficients from other
models would not isolate dropout causally, because their fitted parameter vectors
and gradient geometry differ. No such subtraction is performed.
