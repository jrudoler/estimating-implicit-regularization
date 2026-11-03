"""Validate and plot the complete 480-model loss-stopped Figure-5 rerun."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from core.dropout_continuation import atomic_json
from core.loss_plateau_reporting import DEPTHS, WIDTHS, RATES, SEEDS, collect
from core.scalar_estimation import file_sha256


def build_figure(frame: pd.DataFrame, *, yscale: str = "linear") -> plt.Figure:
    """Retain every seed, including zero-clamped and epoch-capped estimates."""
    if yscale not in {"linear", "symlog"}:
        raise ValueError("yscale must be linear or symlog")
    logarithmic = yscale == "symlog"
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.4 if logarithmic else 5.0),
                             sharex=True, sharey=True)
    values = frame.projection_r2.dropna()
    norm = Normalize(0., max(float(values.max()), 1e-12) if len(values) else 1.)
    cmap = plt.get_cmap("viridis")
    jitter = {seed: offset for seed, offset in zip(SEEDS, np.linspace(-.19, .19, len(SEEDS)))}
    for i, depth in enumerate(DEPTHS):
        for j, width in enumerate(WIDTHS):
            ax = axes[i, j]
            subset = frame[(frame.depth == depth) & (frame.width == width)]
            for k, rate in enumerate(RATES):
                cell = subset[subset.dropout == rate]
                ys = cell.lambda_nonnegative.to_numpy()
                # A linear-space density should not be stretched onto a log axis.
                # Individual points and medians suffice for the log-scale view.
                if not logarithmic and np.ptp(ys) > 0:
                    violin = ax.violinplot([ys], positions=[k], widths=.7, showextrema=False)
                    for body in violin["bodies"]:
                        body.set_facecolor("#d7d7d7")
                        body.set_edgecolor("none")
                        body.set_alpha(.65)
                for row in cell.itertuples():
                    color = cmap(norm(row.projection_r2)) if pd.notna(row.projection_r2) else "gray"
                    ax.scatter(k + jitter[row.seed], row.lambda_nonnegative, color=color,
                               marker="o" if row.status == "loss_plateau" else "x", s=15,
                               linewidths=.65, zorder=3, alpha=.85)
            medians = subset.groupby("dropout").lambda_nonnegative.median().reindex(RATES)
            ax.plot(range(8), medians, color="black", lw=.9, marker="o", ms=2, zorder=4)
            ax.set_title(f"Depth {depth}, width {width}", fontsize=10)
            ax.set_xticks(range(8), [f"{p:g}" for p in RATES], rotation=45)
            ax.tick_params(labelsize=8)
            if logarithmic:
                ymax = max(1e-10, float(frame.lambda_nonnegative.max())) * 3.
                ax.set_yscale("symlog", linthresh=1e-10, linscale=.5)
                ax.set_yticks([0., *[10. ** k for k in range(-10, int(np.ceil(np.log10(ymax))) + 1)]])
                ax.set_ylim(-1e-11, ymax)
            else:
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
                ax.yaxis.get_offset_text().set_fontsize(8)
            ax.grid(axis="y", alpha=.4)
    fig.subplots_adjust(left=.11, right=.86, bottom=.23 if logarithmic else .19,
                        top=.86, wspace=.18, hspace=.34)
    fig.suptitle("Dropout: training-loss early stopping", fontsize=12, y=.98)
    fig.text(.47, .91, "Exact nonnegative ridge projection · 10 seeds per cell", ha="center", fontsize=9)
    fig.text(.48, .11 if logarithmic else .075, "Dropout probability", ha="center", fontsize=10)
    fig.text(.015, .52, r"Fitted ridge coefficient $\hat\lambda$", rotation=90, va="center", fontsize=10)
    cax = fig.add_axes([.9, .28, .02, .42])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label("Clean-gradient squared norm explained", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1., decimals=0))
    cbar.ax.tick_params(labelsize=8)
    exhausted = int((frame.status == "budget_exhausted").sum())
    if logarithmic:
        fig.text(.5, .063, r"Dots: seeds; line: median. Y-axis logarithmic above $10^{-10}$, linear near zero.",
                 ha="center", fontsize=8)
    fig.text(.5, .018, f"Endpoint projection, not direct dropout effect. × = epoch cap ({exhausted}/480 models).",
             ha="center", fontsize=8)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    parser.add_argument("--study-name", default="dropout_loss_plateau")
    parser.add_argument("--yscale", choices=("linear", "symlog"), default="linear")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    plt.style.use(args.style)
    frame, receipt = collect(args.results_dir, study=args.study_name)
    for path in (args.figure, args.table, args.receipt):
        path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.table, index=False)
    figure = build_figure(frame, yscale=args.yscale)
    figure.savefig(args.figure, format="pdf")
    plt.close(figure)
    receipt.update({"yscale": args.yscale,
                    "symlog_linthresh": 1e-10 if args.yscale == "symlog" else None,
                    "renderer_sha256": file_sha256(Path(__file__)),
                    "reporting_sha256": file_sha256(Path(collect.__code__.co_filename)),
                    "style_sha256": file_sha256(args.style),
                    "figure_sha256": file_sha256(args.figure),
                    "table_sha256": file_sha256(args.table)})
    atomic_json(args.receipt, receipt)
    logging.info("Saved %s after validating all %d models", args.figure, len(frame))


if __name__ == "__main__":
    main()
