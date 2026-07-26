#!/usr/bin/env python3
"""Plot closed-form dropout lambda along the original Lightning trajectory."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
DEFAULT_SUMMARY = (
    REPO_ROOT / "data" / "generated" / "dropout_trajectory_lightning" / "summary.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "results" / "figures" / "dropout_lambda_trajectory_stability.pdf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_figure(summary: dict[str, object]) -> plt.Figure:
    rows = pd.DataFrame(summary["rows"])
    figure, axis = plt.subplots(figsize=(4.8, 3.5))
    for _, seed_rows in rows.groupby("seed"):
        seed_rows = seed_rows.sort_values("fraction")
        axis.plot(
            seed_rows["fraction"],
            seed_rows["lambda"],
            color="0.72",
            linewidth=0.8,
            alpha=0.65,
            zorder=1,
        )

    grouped = rows.groupby("fraction")["lambda"]
    fractions = np.asarray(sorted(rows["fraction"].unique()), dtype=float)
    medians = grouped.median().reindex(fractions).to_numpy()
    lower = grouped.quantile(0.25).reindex(fractions).to_numpy()
    upper = grouped.quantile(0.75).reindex(fractions).to_numpy()
    axis.fill_between(fractions, lower, upper, color="#3B6FB6", alpha=0.22, linewidth=0)
    axis.plot(
        fractions,
        medians,
        color="#24518A",
        marker="o",
        linewidth=2.0,
        markersize=5,
        zorder=3,
        label="Median and IQR",
    )
    axis.set_xticks(fractions)
    axis.set_xticklabels([r"$0.5T$", r"$0.8T$", r"$T$"])
    axis.set_xlabel("Training checkpoint")
    axis.set_ylabel(r"Closed-form ridge penalty $\hat{\lambda}$")
    config = summary["config"]
    axis.set_title(
        f"Depth {config['depth']}, width {config['width']}, dropout {config['dropout']}"
    )
    axis.grid(True, color="lightgray", linestyle="--", linewidth=0.5)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    return figure


def main() -> None:
    configure_logging()
    args = parse_args()
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
    summary = json.loads(args.summary.read_text())
    figure = build_figure(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    LOGGER.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
