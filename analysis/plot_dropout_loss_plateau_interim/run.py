"""Plot a fixed, explicitly incomplete snapshot of the loss-stopped dropout grid."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from core.dropout_continuation import atomic_json
from core.loss_plateau_reporting import DEPTHS, WIDTHS, RATES, SEEDS, collect_paths
from core.scalar_estimation import file_sha256


def build_figure(frame: pd.DataFrame, *, snapshot: str) -> plt.Figure:
    """Show all architectures, retaining empty cells and zero projections."""
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.7), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    max_r2 = frame.projection_r2.max()
    norm = Normalize(0., max(.01, float(max_r2)) if pd.notna(max_r2) else 1.)
    jitter = dict(zip(SEEDS, np.linspace(-.22, .22, len(SEEDS))))
    ymax = max(1e-7, float(frame.lambda_nonnegative.max())) * 3.
    for i, depth in enumerate(DEPTHS):
        for j, width in enumerate(WIDTHS):
            ax = axes[i, j]
            subset = frame[(frame.depth == depth) & (frame.width == width)]
            medians = []
            for k, rate in enumerate(RATES):
                cell = subset[subset.dropout == rate]
                n = len(cell)
                if n < len(SEEDS):
                    ax.axvspan(k - .47, k + .47, color="#eeeeee", zorder=0)
                ax.text(k, .965, f"{n}/10", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7, color="#444444")
                for row in cell.itertuples():
                    color = cmap(norm(row.projection_r2)) if pd.notna(row.projection_r2) else "gray"
                    ax.scatter(k + jitter[row.seed], row.lambda_nonnegative, color=color,
                               s=24, alpha=.9, zorder=3, linewidths=.5,
                               marker="o" if row.status == "loss_plateau" else "x")
                # Do not connect or summarize partial seed groups as if complete.
                medians.append(cell.lambda_nonnegative.median() if n == len(SEEDS) else np.nan)
            ax.plot(range(len(RATES)), medians, color="#111111", lw=1.1,
                    marker="_", ms=9, zorder=4)
            ax.set_title(f"Depth {depth} · width {width}  ({len(subset)}/80)", fontsize=11)
            ax.set_xticks(range(len(RATES)), [f"{p:g}" for p in RATES])
            ax.set_xlim(-.55, len(RATES) - .45)
            ax.set_yscale("symlog", linthresh=1e-10, linscale=.5)
            ax.set_ylim(-1e-11, ymax)
            ax.set_yticks([0., 1e-10, 1e-9, 1e-8, 1e-7, 1e-6])
            ax.set_ylim(-1e-11, ymax)
            ax.tick_params(labelsize=9)
            ax.grid(axis="y", alpha=.7, zorder=0)
            if subset.empty:
                ax.text(.5, .55, "No completed models yet", transform=ax.transAxes,
                        ha="center", va="center", color="#666666", fontsize=10)
    fig.subplots_adjust(left=.10, right=.85, bottom=.23, top=.80, wspace=.13, hspace=.31)
    fig.suptitle(f"INTERIM · Dropout ridge estimates · {len(frame)}/480 models", y=.97, fontsize=15)
    fig.text(.48, .915, "Training-loss stopping; exact nonnegative projection at the stopping checkpoint",
             ha="center", fontsize=10)
    fig.text(.48, .875, f"Snapshot: {snapshot} · gray bands / counts indicate incomplete seed groups",
             ha="center", fontsize=9, color="#444444")
    fig.text(.475, .165, "Dropout probability", ha="center", fontsize=11)
    fig.text(.018, .52, r"Fitted ridge coefficient $\hat\lambda$", rotation=90, va="center", fontsize=11)
    cax = fig.add_axes([.89, .31, .018, .40])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f"{value:.0%}"))
    cbar.set_label("Clean-gradient squared norm explained", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    exhausted = int((frame.status == "budget_exhausted").sum())
    fig.text(.5, .10, "Dots: individual seeds. Black line: median only where all 10 seeds are complete.",
             ha="center", fontsize=9)
    fig.text(.5, .065, r"Shared y-axis: logarithmic above $10^{-10}$, linear near zero. "
             f"× = epoch cap ({exhausted} shown).", ha="center", fontsize=9)
    fig.text(.5, .025, "Endpoint projection is not a direct estimate of dropout's causal penalty. Full grid still running.",
             ha="center", fontsize=9, color="#555555")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--study-name", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--style", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    plt.style.use(args.style)
    frame, receipt = collect_paths(args.runs, study=args.study_name)
    receipt.update({"snapshot": args.snapshot, "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "interim": True, "plot_sha256": file_sha256(Path(__file__)),
                    "style_sha256": file_sha256(args.style)})
    for path in (args.figure, args.table, args.receipt):
        path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.table, index=False)
    figure = build_figure(frame, snapshot=args.snapshot)
    figure.savefig(args.figure, format="pdf")
    plt.close(figure)
    atomic_json(args.receipt, receipt)
    logging.info("Saved intermediate picture for %d/480 models: %s", len(frame), args.figure)


if __name__ == "__main__":
    main()
