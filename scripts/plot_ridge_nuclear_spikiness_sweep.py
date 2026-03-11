#!/usr/bin/env python3
"""Plot ridge-vs-nuclear recovery against a continuous spectrum-spikiness axis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("fontTools").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("logs/phase2/ridge_nuclear_spikiness_sweep.csv"),
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


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        results.groupby("teacher_spectrum_decay", as_index=False)[
            [
                "spectral_spikiness",
                "teacher_effective_rank",
                "teacher_condition_number",
                "pair_cosine",
                "gt_mean_rel_error",
                "noisy_ols_mean_rel_error",
                "ols_relative_residual",
            ]
        ]
        .agg(["mean", "std"])
    )
    grouped.columns = [
        "teacher_spectrum_decay",
        "spectral_spikiness_mean",
        "spectral_spikiness_std",
        "teacher_effective_rank_mean",
        "teacher_effective_rank_std",
        "teacher_condition_number_mean",
        "teacher_condition_number_std",
        "pair_cosine_mean",
        "pair_cosine_std",
        "gt_mean_rel_error_mean",
        "gt_mean_rel_error_std",
        "noisy_ols_mean_rel_error_mean",
        "noisy_ols_mean_rel_error_std",
        "ols_relative_residual_mean",
        "ols_relative_residual_std",
    ]
    return grouped.sort_values("spectral_spikiness_mean")


def plot_sweep(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax_left = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    ax_right = ax_left.twinx()

    x_values = summary["spectral_spikiness_mean"]
    converged_mask = summary["ols_relative_residual_mean"] < 1e-2

    left_color = "#1b4965"
    right_color = "#c1121f"

    converged = summary.loc[converged_mask].copy()
    non_converged = summary.loc[~converged_mask].copy()

    ax_left.plot(
        converged["spectral_spikiness_mean"],
        converged["gt_mean_rel_error_mean"],
        color=left_color,
        marker="o",
        linewidth=2.25,
        label="Normalized estimator",
    )
    ax_left.fill_between(
        converged["spectral_spikiness_mean"],
        (converged["gt_mean_rel_error_mean"] - converged["gt_mean_rel_error_std"].fillna(0.0)).clip(lower=1e-6),
        (converged["gt_mean_rel_error_mean"] + converged["gt_mean_rel_error_std"].fillna(0.0)).clip(lower=1e-6),
        color=left_color,
        alpha=0.18,
    )

    ax_right.plot(
        converged["spectral_spikiness_mean"],
        converged["noisy_ols_mean_rel_error_mean"],
        color=right_color,
        marker="s",
        linewidth=2.25,
        label="Noisy OLS probe",
    )
    ax_right.fill_between(
        converged["spectral_spikiness_mean"],
        (converged["noisy_ols_mean_rel_error_mean"] - converged["noisy_ols_mean_rel_error_std"].fillna(0.0)).clip(lower=1e-6),
        (converged["noisy_ols_mean_rel_error_mean"] + converged["noisy_ols_mean_rel_error_std"].fillna(0.0)).clip(lower=1e-6),
        color=right_color,
        alpha=0.18,
    )

    if not non_converged.empty:
        ax_left.plot(
            non_converged["spectral_spikiness_mean"],
            non_converged["gt_mean_rel_error_mean"],
            color=left_color,
            marker="o",
            linestyle="--",
            linewidth=1.75,
            alpha=0.65,
        )
        ax_right.plot(
            non_converged["spectral_spikiness_mean"],
            non_converged["noisy_ols_mean_rel_error_mean"],
            color=right_color,
            marker="s",
            linestyle="--",
            linewidth=1.75,
            alpha=0.65,
        )
        ax_left.axvspan(
            float(non_converged["spectral_spikiness_mean"].min()),
            float(non_converged["spectral_spikiness_mean"].max()),
            color="#bdbdbd",
            alpha=0.12,
        )

    for row in summary.itertuples(index=False):
        ax_left.annotate(
            f"{row.pair_cosine_mean:.2f}",
            xy=(row.spectral_spikiness_mean, row.gt_mean_rel_error_mean),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=left_color,
        )

    ax_left.set_xlabel("Teacher spectral spikiness (1 - effective rank / rank)")
    ax_left.set_ylabel("Normalized estimator mean relative error", color=left_color)
    ax_right.set_ylabel("Noisy OLS mean relative error", color=right_color)
    ax_left.tick_params(axis="y", colors=left_color)
    ax_right.tick_params(axis="y", colors=right_color)
    ax_left.set_yscale("log")
    ax_right.set_yscale("log")
    ax_left.set_title("Ridge vs Nuclear Norm Recovery Across a Continuous Spectrum Sweep")

    left_handles, left_labels = ax_left.get_legend_handles_labels()
    right_handles, right_labels = ax_right.get_legend_handles_labels()
    ax_left.legend(
        left_handles + right_handles,
        left_labels + right_labels,
        loc="upper left",
        frameon=False,
    )

    cosine_note = (
        f"pairwise cosine ranges from {summary['pair_cosine_mean'].min():.2f} "
        f"to {summary['pair_cosine_mean'].max():.2f}"
    )
    ax_left.text(
        0.99,
        0.02,
        cosine_note,
        transform=ax_left.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )
    if not non_converged.empty:
        ax_left.text(
            0.99,
            0.10,
            "shaded tail: OLS residual > 1e-2",
            transform=ax_left.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )

    for suffix in ("png", "pdf"):
        output_path = output_dir / f"ridge_nuclear_spikiness_twin_axes.{suffix}"
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
    plot_sweep(summary, args.output_dir)


if __name__ == "__main__":
    main()
