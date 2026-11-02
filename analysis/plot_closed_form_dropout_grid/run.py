#!/usr/bin/env python3
"""Plot the Figure-5 dropout grid using the exact closed-form ridge estimator."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cmap import Colormap
from core.dropout_grid import load_grid
from core.scalar_estimation import file_sha256


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
DEFAULT_RESULTS_DIR = (
    REPO_ROOT / "data" / "generated" / "closed_form_dropout_grid" / "results"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "figures" / "dropout_bias_ridge_panel_closed_form.pdf"
)
EXPECTED_RUNS = 480
WIDTH_ORDER = (128, 256, 512)
DEPTH_ORDER = (3, 5)
DROPOUT_ORDER = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
FACET_LABEL_SIZE = 13
SHARED_AXIS_LABEL_SIZE = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation", type=Path, required=True,
                        help="Provenance/coverage receipt from validate_dropout_grid")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_results(results_dir: Path, validation: Path) -> pd.DataFrame:
    """Load the corrected Figure-5 grid into one plotting table."""
    rows: list[dict[str, float | int | str]] = []
    receipt = json.loads(validation.read_text())
    hashes = {tuple(s["cell"]): s["sha256"] for s in receipt["sources"]}
    records = load_grid(results_dir, diagnostic_tolerance=receipt.get("diagnostic_tolerance", 1e-5))
    if set(hashes) != set(records) or len(receipt["sources"]) != EXPECTED_RUNS:
        raise ValueError("Validation receipt does not cover this grid")
    for key, (path, payload) in sorted(records.items()):
        if hashes[key] != file_sha256(path):
            raise ValueError(f"Stale validation receipt: {path}")
        config = payload["config"]
        ridge = payload["closed_form"]["ridge"]
        rows.append(
            {
                "file": path.name,
                "depth": int(config["depth"]),
                "width": int(config["width"]),
                "dropout": float(config["dropout"]),
                "seed": int(config["seed"]),
                "estimated_ridge": float(ridge["scale_star_clamped"]),
                "estimated_ridge_signed": float(ridge["scale_star"]),
                "residual_ratio": ridge["residual_ratio_clamped"],
                "projection_r2": max(ridge["grad_cosine"], 0.)**2 if ridge["grad_cosine"] is not None else float("nan"),
                "grad_cosine": ridge["grad_cosine"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["historical_source"] = receipt.get("historical_source")
    if len(frame) != EXPECTED_RUNS:
        raise RuntimeError(
            f"Expected {EXPECTED_RUNS} completed runs under {results_dir}, found {len(frame)}."
        )
    return frame


def build_figure(frame: pd.DataFrame) -> plt.Figure:
    """Build a layout-compatible closed-form replacement for Figure 5."""
    cmap = Colormap("crameri:imola").to_mpl()
    norm = mpl.colors.Normalize(
        vmin=0.0,
        vmax=max(float(frame["projection_r2"].max()), 1e-12) if frame["projection_r2"].notna().any() else 1.0,
    )

    grid = sns.FacetGrid(
        frame,
        col="width",
        row="depth",
        col_order=WIDTH_ORDER,
        row_order=DEPTH_ORDER,
        sharex=True,
        sharey=True,
        height=2.5,
        aspect=1,
        margin_titles=False,
    )

    for row_index, depth in enumerate(DEPTH_ORDER):
        for column_index, width in enumerate(WIDTH_ORDER):
            axis = grid.axes[row_index, column_index]
            subset = frame[(frame["depth"] == depth) & (frame["width"] == width)]
            sns.violinplot(
                data=subset,
                x="dropout",
                y="estimated_ridge",
                order=DROPOUT_ORDER,
                inner=None,
                cut=0,
                density_norm="width",
                color="lightgray",
                alpha=0.5,
                ax=axis,
            )
            sns.stripplot(
                data=subset[subset["projection_r2"].notna()],
                x="dropout",
                y="estimated_ridge",
                order=DROPOUT_ORDER,
                hue="projection_r2",
                hue_norm=norm,
                palette=cmap,
                size=4.5,
                linewidth=0.4,
                alpha=0.65,
                edgecolor="auto",
                legend=False,
                ax=axis,
            )
            undefined = subset[subset["projection_r2"].isna()]
            if not undefined.empty:
                sns.stripplot(data=undefined, x="dropout", y="estimated_ridge",
                              order=DROPOUT_ORDER, color="gray", marker="x", ax=axis)

            medians = (
                subset.groupby("dropout", observed=False)["estimated_ridge"]
                .median()
                .reindex(DROPOUT_ORDER)
            )
            axis.plot(
                range(len(DROPOUT_ORDER)),
                medians.to_numpy(),
                color="black",
                marker="o",
                markersize=2.8,
                linewidth=1.1,
                zorder=5,
            )
            axis.grid(True, color="lightgray", linestyle="--", linewidth=0.5)
            axis.set_title("")
            axis.set_xlabel("")
            axis.set_ylabel("")
            axis.tick_params(axis="both", labelsize=10)
            axis.tick_params(axis="x", rotation=45)

    for axis, width in zip(grid.axes[0], WIDTH_ORDER):
        axis.set_title(
            f"Width {width}",
            pad=6,
            y=1.06,
            fontsize=FACET_LABEL_SIZE,
            color="gray",
            alpha=0.9,
        )

    grid.figure.subplots_adjust(
        left=0.13,
        bottom=0.19,
        top=0.92,
        right=0.83,
        hspace=0.18,
        wspace=0.16,
    )
    for row_index, depth in enumerate(DEPTH_ORDER):
        last_position = grid.axes[row_index, -1].get_position()
        grid.figure.text(
            0.735,
            0.5 * (last_position.y0 + last_position.y1),
            f"Depth {depth}",
            ha="left",
            va="center",
            fontsize=FACET_LABEL_SIZE,
            color="gray",
            alpha=0.9,
        )

    colorbar_axis = grid.figure.add_axes([0.90, 0.18, 0.025, 0.72])
    scalar_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    colorbar = grid.figure.colorbar(scalar_mappable, cax=colorbar_axis)
    colorbar.set_label("Gradient energy explained", fontsize=11)
    colorbar.ax.tick_params(labelsize=10)

    grid.figure.supxlabel("Dropout rate", y=0.06, fontsize=SHARED_AXIS_LABEL_SIZE)
    if frame.attrs.get("historical_source"):
        grid.figure.text(.5, .01, "Historical retrained grid (float32); checkpoint controls reported separately",
                         ha="center", fontsize=8, color="dimgray")
    grid.figure.supylabel(
        r"Closed-form ridge penalty $\hat{\lambda}$",
        fontsize=SHARED_AXIS_LABEL_SIZE,
        x=0.015,
    )
    return grid.figure


def main() -> None:
    configure_logging()
    args = parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    frame = load_results(args.results_dir, args.validation)
    LOGGER.info("Loaded %d closed-form Figure-5 runs.", len(frame))
    figure = build_figure(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    LOGGER.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
