#!/usr/bin/env python3
"""Plot summary figures for the matrix-spectrum identifiability experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOGGER = logging.getLogger(__name__)

PAIR_LABELS: dict[str, str] = {
    "ridge__nuclear_norm": "Ridge vs Nuclear Norm",
    "ridge__stable_rank": "Ridge vs Stable Rank",
    "ridge__spectral_entropy": "Ridge vs Spectral Entropy",
}

SPECTRUM_LABELS: dict[str, str] = {
    "identity": "Uniform spectrum",
    "spiked": "Spiked spectrum",
    "power_law": "Power-law spectrum",
}

PAIR_COLORS: dict[str, str] = {
    "ridge__nuclear_norm": "#1b4965",
    "ridge__stable_rank": "#c1121f",
    "ridge__spectral_entropy": "#588157",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("logs/phase2/matrix_spectrum_pairs.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures/phase2"),
    )
    parser.add_argument(
        "--style",
        type=Path,
        default=Path("clean_fig.mplstyle"),
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "pair_cosine",
        "gt_mean_rel_error",
        "noisy_ols_mean_rel_error",
        "selected_cond",
    ]
    grouped = (
        results.groupby(["pair", "teacher_spectrum"], as_index=False)[numeric_columns]
        .agg(["mean", "std"])
    )
    grouped.columns = [
        "pair",
        "teacher_spectrum",
        "pair_cosine_mean",
        "pair_cosine_std",
        "gt_mean_rel_error_mean",
        "gt_mean_rel_error_std",
        "noisy_ols_mean_rel_error_mean",
        "noisy_ols_mean_rel_error_std",
        "selected_cond_mean",
        "selected_cond_std",
    ]
    return grouped.sort_values(["pair", "pair_cosine_mean"])


def annotate_points(
    ax: plt.Axes,
    x_values: pd.Series,
    y_values: pd.Series,
    labels: pd.Series,
) -> None:
    for x_value, y_value, label in zip(x_values, y_values, labels):
        ax.annotate(
            SPECTRUM_LABELS.get(label, label),
            xy=(x_value, y_value),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )


def plot_ridge_nuclear(summary: pd.DataFrame, output_dir: Path) -> None:
    ridge_nuclear = summary.loc[summary["pair"] == "ridge__nuclear_norm"].copy()
    ridge_nuclear = ridge_nuclear.sort_values("pair_cosine_mean")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    color = PAIR_COLORS["ridge__nuclear_norm"]

    axes[0].errorbar(
        ridge_nuclear["pair_cosine_mean"],
        ridge_nuclear["gt_mean_rel_error_mean"],
        xerr=ridge_nuclear["pair_cosine_std"],
        yerr=ridge_nuclear["gt_mean_rel_error_std"],
        color=color,
        marker="o",
        linewidth=2,
        capsize=4,
    )
    annotate_points(
        axes[0],
        ridge_nuclear["pair_cosine_mean"],
        ridge_nuclear["gt_mean_rel_error_mean"],
        ridge_nuclear["teacher_spectrum"],
    )
    axes[0].set_title("Normalized Estimator")
    axes[0].set_xlabel("Gradient collinearity (cosine)")
    axes[0].set_ylabel("Bias recovery mean relative error")

    axes[1].errorbar(
        ridge_nuclear["pair_cosine_mean"],
        ridge_nuclear["noisy_ols_mean_rel_error_mean"],
        xerr=ridge_nuclear["pair_cosine_std"],
        yerr=ridge_nuclear["noisy_ols_mean_rel_error_std"],
        color=color,
        marker="o",
        linewidth=2,
        capsize=4,
    )
    annotate_points(
        axes[1],
        ridge_nuclear["pair_cosine_mean"],
        ridge_nuclear["noisy_ols_mean_rel_error_mean"],
        ridge_nuclear["teacher_spectrum"],
    )
    axes[1].set_title("Noisy OLS Probe")
    axes[1].set_xlabel("Gradient collinearity (cosine)")
    axes[1].set_ylabel("Mean relative error")

    fig.suptitle("Ridge vs Nuclear Norm: Higher Collinearity Increases Recovery Error")

    output_path = output_dir / "ridge_nuclear_collinearity_vs_recovery.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    LOGGER.info("Wrote %s", output_path)
    plt.close(fig)


def plot_pair_comparison(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    for pair, pair_df in summary.groupby("pair"):
        pair_df = pair_df.sort_values("pair_cosine_mean")
        label = PAIR_LABELS.get(pair, pair)
        color = PAIR_COLORS.get(pair, "#333333")
        axes[0].errorbar(
            pair_df["pair_cosine_mean"],
            pair_df["gt_mean_rel_error_mean"],
            xerr=pair_df["pair_cosine_std"],
            yerr=pair_df["gt_mean_rel_error_std"],
            marker="o",
            linewidth=2,
            capsize=4,
            color=color,
            label=label,
        )
        axes[1].errorbar(
            pair_df["pair_cosine_mean"],
            pair_df["noisy_ols_mean_rel_error_mean"],
            xerr=pair_df["pair_cosine_std"],
            yerr=pair_df["noisy_ols_mean_rel_error_std"],
            marker="o",
            linewidth=2,
            capsize=4,
            color=color,
            label=label,
        )

    axes[0].set_title("Normalized Estimator")
    axes[0].set_xlabel("Gradient collinearity (cosine)")
    axes[0].set_ylabel("Bias recovery mean relative error")
    axes[0].set_yscale("log")

    axes[1].set_title("Noisy OLS Probe")
    axes[1].set_xlabel("Gradient collinearity (cosine)")
    axes[1].set_ylabel("Mean relative error")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, loc="upper left")

    fig.suptitle("Conditioning Depends on Both Collinearity and Gradient Strength")

    output_path = output_dir / "matrix_pair_comparison.pdf"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    LOGGER.info("Wrote %s", output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_logging()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use(args.style)

    LOGGER.info("Loading %s", args.input_csv)
    results = pd.read_csv(args.input_csv)
    summary = summarize_results(results)

    plot_ridge_nuclear(summary, args.output_dir)
    plot_pair_comparison(summary, args.output_dir)


if __name__ == "__main__":
    main()
