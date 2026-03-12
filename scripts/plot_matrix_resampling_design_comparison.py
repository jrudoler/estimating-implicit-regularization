#!/usr/bin/env python3
"""Compare fixed-design and changing-design resampling across spectral spikiness."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
FIXED_CSV = REPO_ROOT / "logs/phase2/matrix_resampling_spikiness_matched.csv"
RETRAIN_CSV = REPO_ROOT / "logs/phase2/matrix_resampling_retrain_sweep.csv"
OUTPUT_PDF = REPO_ROOT / "figures/phase2/matrix_resampling_design_comparison.pdf"


def read_csv(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float | str] = dict(row)
            for key, value in row.items():
                if key in {"resample_mode", "gradient_dataset", "teacher_spectrum", "pair", "mode"}:
                    continue
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


def mean_curve(rows: list[dict[str, float | str]], y_key: str) -> tuple[list[float], list[float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        grouped[float(row["spectral_spikiness"])].append(float(row[y_key]))
    xs = sorted(grouped)
    ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]
    return xs, ys


def main() -> None:
    plt.style.use(str(STYLE_PATH))
    fixed_rows = read_csv(FIXED_CSV)
    retrain_rows = read_csv(RETRAIN_CSV)
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    mode_titles = {"subsample": "Subsample", "bootstrap": "Bootstrap"}
    styles = [
        ("Fixed Design", "#244D7A", fixed_rows, None),
        ("Retrain, Same Split", "#3B7C3B", retrain_rows, "train"),
        ("Retrain, Held-Out Val", "#B04632", retrain_rows, "val"),
    ]

    for ax, mode in zip(axes, ["subsample", "bootstrap"], strict=True):
        for label, color, rows, gradient_dataset in styles:
            if gradient_dataset is None:
                selected = [row for row in rows if row["resample_mode"] == mode]
            else:
                selected = [
                    row
                    for row in rows
                    if row["resample_mode"] == mode and row["gradient_dataset"] == gradient_dataset
                ]
            xs, ys = mean_curve(selected, "stacked_mean_rel_error")
            ax.plot(xs, ys, marker="o", linewidth=2.0, color=color, label=label)
        ax.set_title(mode_titles[mode])
        ax.set_xlabel("Spectral Spikiness")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Stacked Mean Relative Error")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
