"""Publication Figure 5: audited loss-stopped endpoints in the original style."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import NullLocator
from cmap import Colormap
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from core.dropout_continuation import atomic_json
from core.loss_plateau_reporting import DEPTHS, WIDTHS, RATES, SEEDS, collect
from core.scalar_estimation import file_sha256

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY = "dropout_loss_plateau_absolute"
LINTHRESH = 1e-10


def validate_audit(receipt: dict[str, Any], audit: dict[str, Any]) -> None:
    """Bind this presentation to the completed audit, without redoing training."""
    if not audit.get("full_grid_verification_passed"):
        raise ValueError("Full-grid verification has not passed")
    if receipt["n"] != 480 or receipt["missing_cells"]:
        raise ValueError("Publication figure requires the complete 480-model grid")
    for key in ("n", "missing_cells", "protocol", "code_sha256", "statuses"):
        if receipt[key] != audit[key]:
            raise ValueError(f"Grid differs from completed audit: {key}")
    def hashes(payload: dict[str, Any]) -> dict[tuple, tuple[str, str]]:
        return {tuple(s["cell"]): (s["sha256"], s["checkpoint_sha256"])
                for s in payload["sources"]}
    if len(audit["sources"]) != 480 or hashes(receipt) != hashes(audit):
        raise ValueError("Source hashes differ from completed audit")


def build_figure(frame: pd.DataFrame) -> plt.Figure:
    """Preserve all seeds and zero controls; fit violins in displayed coordinates."""
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.), sharex=True, sharey=True)
    cmap = Colormap("crameri:imola").to_mpl()
    # A fixed, shared diagnostic scale: colors do not change with data extrema.
    norm = Normalize(0., 1.)
    jitter = dict(zip(SEEDS, np.linspace(-.15, .15, len(SEEDS))))
    ymax = max(float(frame.lambda_nonnegative.max()) * 1.7, 1e-8)
    ticks = [0., *[10. ** k for k in range(-10, int(np.floor(np.log10(ymax))) + 1)]]
    for i, depth in enumerate(DEPTHS):
        for j, width in enumerate(WIDTHS):
            ax = axes[i, j]
            ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=.5)
            transform = ax.yaxis.get_transform()
            subset = frame[(frame.depth == depth) & (frame.width == width)]
            for k, rate in enumerate(RATES):
                cell = subset[subset.dropout == rate]
                ys = cell.lambda_nonnegative.to_numpy()
                transformed = transform.transform(ys)
                if len(ys) > 1 and np.ptp(transformed) > 0:
                    support = np.linspace(transformed.min(), transformed.max(), 100)
                    density = gaussian_kde(transformed)(support)
                    halfwidth = .32 * density / density.max()
                    ax.fill_betweenx(transform.inverted().transform(support),
                                     k - halfwidth, k + halfwidth,
                                     facecolor="lightgray", edgecolor="0.72",
                                     linewidth=.6, alpha=.5, zorder=2)
                for status, marker in (("loss_plateau", "o"), ("budget_exhausted", "x")):
                    points = cell[cell.status == status]
                    if points.empty:
                        continue
                    colors = np.array([cmap(norm(r)) if pd.notna(r) else (.5, .5, .5, 1.)
                                       for r in points.projection_r2])
                    ax.scatter([k + jitter[s] for s in points.seed], points.lambda_nonnegative,
                               c=colors, marker=marker, s=20, linewidths=.4,
                               zorder=3, **({"edgecolors": "0.4"} if marker == "o" else {}))
            ax.set_ylim(-1e-11, ymax)
            ax.set_yticks(ticks)
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_xlim(-.5, 7.5)
            ax.set_xticks(range(len(RATES)), [f"{p:g}" for p in RATES], rotation=45)
            ax.tick_params(labelsize=10)
            ax.grid(axis="y", color="lightgray", linestyle="--", linewidth=.5)
    fig.subplots_adjust(left=.13, right=.82, bottom=.17, top=.90,
                        hspace=.14, wspace=.16)
    for ax, width in zip(axes[0], WIDTHS):
        ax.set_title(f"Width {width}", fontsize=13, color="gray", alpha=.9, pad=6)
    for i, depth in enumerate(DEPTHS):
        axes[i, -1].text(.98, .90, f"Depth {depth}", transform=axes[i, -1].transAxes,
                         ha="right", va="top", fontsize=13, color="gray", alpha=.9)
    cax = fig.add_axes([.90, .18, .025, .72])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                            ticks=np.linspace(0., 1., 6))
    colorbar.set_label(r"$R_{\mathrm{grad}}^2$", fontsize=15)
    colorbar.ax.tick_params(labelsize=10)
    fig.supxlabel("Dropout rate", y=.025, fontsize=15)
    fig.supylabel(r"Estimated ridge penalty $\hat{\lambda}$", x=.015, fontsize=15)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--style", type=Path, default=REPO_ROOT / "clean_fig.mplstyle")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.style.use(args.style)
    frame, receipt = collect(args.results_dir, study=STUDY)
    validate_audit(receipt, json.loads(args.validation.read_text()))
    for path in (args.output, args.table, args.receipt):
        path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.table, index=False)
    figure = build_figure(frame)
    figure.savefig(args.output, format="pdf", bbox_inches="tight", pad_inches=.02)
    plt.close(figure)
    receipt.update({"study": STUDY, "yscale": "symlog", "symlog_linthresh": LINTHRESH,
                    "violin_coordinates": "symlog", "color_diagnostic": "projection_r2_clamped",
                    "color_limits": [0., 1.], "colormap": "crameri:imola",
                    "validation_sha256": file_sha256(args.validation),
                    "renderer_sha256": file_sha256(Path(__file__)),
                    "reporting_sha256": file_sha256(Path(collect.__code__.co_filename)),
                    "style_sha256": file_sha256(args.style),
                    "figure_sha256": file_sha256(args.output),
                    "table_sha256": file_sha256(args.table)})
    atomic_json(args.receipt, receipt)
    logging.info("Saved publication figure %s with all %d audited endpoints", args.output, len(frame))


if __name__ == "__main__":
    main()
