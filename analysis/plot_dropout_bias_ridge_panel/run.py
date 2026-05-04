#!/usr/bin/env python3
"""Regenerate the dropout ridge panel figure from a local W&B run snapshot."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from cmap import Colormap

from core.plotting import panelplot
from core.wandb_utils import get_sweep_runs, wandb_summary_df


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FIGURES_DIR = REPO_ROOT / "results" / "figures"
STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
DEFAULT_OUTPUT = RESULTS_FIGURES_DIR / "dropout_bias_ridge_panel.pdf"


def _format_facet_value(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--runs-parquet",
        type=Path,
        help="Local W&B sweep snapshot produced by pull_wandb_sweep.",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        help="Manual fallback: live W&B sweep id (use with --entity-project).",
    )
    parser.add_argument(
        "--entity-project",
        type=str,
        help=(
            "Manual fallback: W&B '<entity>/<project>'. If omitted, uses "
            "WANDB_ENTITY_PROJECT or WANDB_ENTITY plus WANDB_PROJECT."
        ),
    )
    parser.add_argument(
        "--state",
        type=str,
        default="finished",
        help="Optional W&B run state filter. Use 'all' to keep every run.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="W&B API timeout in seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination figure path. Defaults to results/figures/dropout_bias_ridge_panel.pdf.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster export DPI.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def configure_style() -> None:
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))


def load_dropout_summary_from_parquet(path: Path):
    summary_df = pd.read_parquet(path)
    return filter_dropout_summary(summary_df)


def load_dropout_summary_from_wandb(
    *,
    sweep_id: str,
    entity_project: str | None,
    state: str,
    timeout: int,
):
    runs = get_sweep_runs(
        sweep_id=sweep_id,
        entity_project=entity_project,
        state=None if state == "all" else state,
        timeout=timeout,
    )
    if not runs:
        raise RuntimeError(
            f"No W&B runs found for sweep {sweep_id} with state={state!r}."
        )

    summary_df = wandb_summary_df(runs, include_system_metrics=False)
    return filter_dropout_summary(summary_df)


def filter_dropout_summary(summary_df):
    required_columns = {"width", "depth", "dropout", "estimated_ridge", "train_bias/loss"}
    missing_columns = sorted(required_columns - set(summary_df.columns))
    if missing_columns:
        raise KeyError(
            "Missing required columns in W&B summary: " + ", ".join(missing_columns)
        )

    summary_df = summary_df.dropna(subset=list(required_columns)).copy()
    summary_df = summary_df[summary_df["depth"] != 1].copy()
    if summary_df.empty:
        raise RuntimeError(
            "Sweep summary is empty after dropping incomplete rows and depth == 1 runs."
        )
    return summary_df


def build_figure(summary_df) -> plt.Figure:
    grid = sns.FacetGrid(
        summary_df,
        col="width",
        row="depth",
        sharex=True,
        sharey=True,
        margin_titles=False,
        height=2.5,
        aspect=1,
    )
    grid.map_dataframe(
        panelplot,
        x="dropout",
        y="estimated_ridge",
        hue="train_bias/loss",
    )

    grid.set_titles("")
    grid.figure.subplots_adjust(
        bottom=0.18,
        top=0.90,
        right=0.84,
        hspace=0.14,
        wspace=0.16,
    )
    for ax, width in zip(grid.axes[0], grid.col_names):
        ax.set_title(f"Width {_format_facet_value(width)}", pad=8)
    for row_index, depth in enumerate(grid.row_names):
        grid.axes[row_index, -1].text(
            1.06,
            0.5,
            f"Depth {_format_facet_value(depth)}",
            transform=grid.axes[row_index, -1].transAxes,
            rotation=0,
            ha="left",
            va="center",
            fontsize=mpl.rcParams["axes.titlesize"],
        )

    colorbar_axis = grid.figure.add_axes([0.95, 0.18, 0.025, 0.72])
    norm = mpl.colors.Normalize(
        vmin=summary_df["train_bias/loss"].min(),
        vmax=summary_df["train_bias/loss"].max(),
    )
    scalar_mappable = plt.cm.ScalarMappable(
        cmap=Colormap("crameri:imola").to_mpl(), norm=norm
    )
    scalar_mappable.set_array([])
    colorbar = grid.figure.colorbar(scalar_mappable, cax=colorbar_axis)
    colorbar.set_label("Loss from fitting regularizer")

    grid.set_axis_labels("", "")
    grid.figure.supxlabel("Dropout rate", y=0.025)
    grid.figure.supylabel(r"Estimated ridge penalty $\hat{\lambda}$")
    grid.tick_params(axis="x", rotation=45)
    return grid.figure


def main() -> None:
    configure_logging()
    args = parse_args()
    configure_style()

    if args.runs_parquet:
        LOGGER.info("Loading local W&B snapshot %s", args.runs_parquet)
        summary_df = load_dropout_summary_from_parquet(args.runs_parquet)
    elif args.sweep_id:
        LOGGER.info("Loading live W&B sweep %s", args.sweep_id)
        summary_df = load_dropout_summary_from_wandb(
            sweep_id=args.sweep_id,
            entity_project=args.entity_project,
            state=args.state,
            timeout=args.timeout,
        )
    else:
        raise SystemExit("Provide --runs-parquet or --sweep-id")
    LOGGER.info("Loaded %d runs for the dropout ridge panel.", len(summary_df))

    figure = build_figure(summary_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    LOGGER.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
