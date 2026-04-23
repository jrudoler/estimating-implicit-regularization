#!/usr/bin/env python3
"""Regenerate the dropout ridge panel figure from the W&B sweep used in the paper."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from core.plotting import panelplot
from core.wandb_utils import get_sweep_runs, wandb_summary_df


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES_DIR = REPO_ROOT / "paper" / "figures"
STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
DEFAULT_OUTPUT = PAPER_FIGURES_DIR / "dropout_bias_ridge_panel.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--sweep-id",
        type=str,
        default="chiy2qjz",
        help="W&B sweep id used by notebooks/l2_estimation_deep_ReLU.ipynb.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="jhrudoler-penn",
        help="W&B entity or organization.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="inductive-bias",
        help="W&B project containing the target sweep.",
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
        help="Destination figure path. Defaults to paper/figures/dropout_bias_ridge_panel.png.",
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


def load_dropout_summary(
    *,
    sweep_id: str,
    entity: str,
    project: str,
    state: str,
    timeout: int,
):
    runs = get_sweep_runs(
        sweep_id=sweep_id,
        entity=entity,
        project=project,
        state=None if state == "all" else state,
        timeout=timeout,
    )
    if not runs:
        raise RuntimeError(
            f"No W&B runs found for sweep {entity}/{project}/{sweep_id} with state={state!r}."
        )

    summary_df = wandb_summary_df(runs, include_system_metrics=False)
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
        margin_titles=True,
        height=2.5,
        aspect=1,
    )
    grid.map_dataframe(
        panelplot,
        x="dropout",
        y="estimated_ridge",
        hue="train_bias/loss",
    )

    colorbar_axis = grid.figure.add_axes([1.05, 0.1, 0.02, 0.8])
    norm = mpl.colors.Normalize(
        vmin=summary_df["train_bias/loss"].min(),
        vmax=summary_df["train_bias/loss"].max(),
    )
    scalar_mappable = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    scalar_mappable.set_array([])
    colorbar = grid.figure.colorbar(scalar_mappable, cax=colorbar_axis)
    colorbar.set_label("train_bias/loss")

    grid.set_axis_labels("dropout", "estimated_ridge")
    grid.tick_params(axis="x", rotation=45)
    return grid.figure


def main() -> None:
    configure_logging()
    args = parse_args()
    configure_style()

    LOGGER.info(
        "Loading W&B sweep %s/%s/%s",
        args.entity,
        args.project,
        args.sweep_id,
    )
    summary_df = load_dropout_summary(
        sweep_id=args.sweep_id,
        entity=args.entity,
        project=args.project,
        state=args.state,
        timeout=args.timeout,
    )
    LOGGER.info("Loaded %d runs for the dropout ridge panel.", len(summary_df))

    figure = build_figure(summary_df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    LOGGER.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
