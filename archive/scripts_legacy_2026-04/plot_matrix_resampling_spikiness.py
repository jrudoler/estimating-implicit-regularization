#!/usr/bin/env python3
"""Plot stacked vs single-resample recovery across spectral spikiness."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
INPUT_CSV = REPO_ROOT / "logs/phase2/matrix_resampling_spikiness_sweep_small.csv"
OUTPUT_PDF = REPO_ROOT / "figures/phase2/matrix_resampling_spikiness.pdf"


def load_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float | str] = dict(row)
            for key in [
                "teacher_spectrum_decay",
                "spectral_spikiness",
                "full_cos",
                "single_mean_rel_error",
                "aggregate_mean_rel_error",
                "stacked_mean_rel_error",
                "train_mse",
                "test_mse",
            ]:
                parsed[key] = float(row[key])
            rows.append(parsed)
    return rows


def mean_by_spikiness(rows: list[dict[str, float | str]], value_key: str) -> tuple[list[float], list[float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        grouped[float(row["spectral_spikiness"])].append(float(row[value_key]))
    xs = sorted(grouped)
    ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
    return xs, ys


def main() -> None:
    plt.style.use(str(STYLE_PATH))
    rows = load_rows(INPUT_CSV)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    modes = ["subsample", "bootstrap"]
    titles = {"subsample": "Subsample", "bootstrap": "Bootstrap"}
    colors = {
        "single_mean_rel_error": "#B04632",
        "stacked_mean_rel_error": "#244D7A",
    }
    labels = {
        "single_mean_rel_error": "Single resample",
        "stacked_mean_rel_error": "Stacked shared lambda",
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)

    for ax, mode in zip(axes, modes, strict=True):
        mode_rows = [row for row in rows if row["resample_mode"] == mode]
        xs, single = mean_by_spikiness(mode_rows, "single_mean_rel_error")
        _, stacked = mean_by_spikiness(mode_rows, "stacked_mean_rel_error")

        ax.plot(xs, single, marker="o", linewidth=2.0, color=colors["single_mean_rel_error"], label=labels["single_mean_rel_error"])
        ax.plot(xs, stacked, marker="o", linewidth=2.0, color=colors["stacked_mean_rel_error"], label=labels["stacked_mean_rel_error"])
        ax.axhline(1e-4, color="0.35", linestyle="--", linewidth=1.2, label="Full-data OLS (~0)")
        ax.set_title(titles[mode])
        ax.set_xlabel("Spectral Spikiness")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean Relative Recovery Error")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
