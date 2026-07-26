"""Plot the SWA effective-regularizer identification and recovery study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEEDS = (123, 42, 666, 314, 17, 111, 325, 643, 432, 51)
FAMILIES = (
    "ridge",
    "nuclear_norm",
    "stable_rank",
    "spectral_entropy",
    "spectral_gap",
    "burnin_centered_ridge",
)
LABELS = (
    "Origin\n$L_2$",
    "Nuclear",
    "Stable\nrank",
    "Spec.\nentropy",
    "Spec.\ngap",
    "Checkpoint\n$L_2$",
)


def mean_se(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values)
    return float(array.mean()), float(array.std(ddof=1) / np.sqrt(array.size))


def load_runs(results_dir: Path, seeds: set[int]) -> list[dict[str, Any]]:
    runs = [
        json.loads(path.read_text())
        for path in sorted(results_dir.glob("seed*.json"))
    ]
    runs = [run for run in runs if int(run["config"]["seed"]) in seeds]
    if len(runs) != len(seeds):
        found = {int(run["config"]["seed"]) for run in runs}
        raise SystemExit(f"Missing seeds: {sorted(seeds - found)}")
    return runs


def bar_with_se(
    axis: plt.Axes,
    means: list[float],
    ses: list[float],
    labels: tuple[str, ...],
    colors: list[str],
) -> None:
    positions = np.arange(len(means))
    axis.bar(
        positions,
        means,
        yerr=ses,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        capsize=3,
    )
    axis.set_xticks(positions, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/generated/swa_effective_regularizer/results"),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--style",
        type=Path,
        default=REPO_ROOT / "clean_fig.mplstyle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/swa_effective_regularizer.pdf"),
    )
    args = parser.parse_args()
    seeds = {int(seed) for seed in args.seeds.split(",")}
    runs = load_runs(args.results_dir, seeds)
    plt.style.use(args.style)

    endpoint_r2: list[list[float]] = []
    displacement_r2: list[list[float]] = []
    for family in FAMILIES:
        endpoint_r2.append(
            [
                run["endpoints"]["swa_average"]["closed_form"][family][
                    "projection_r2_positive"
                ]
                for run in runs
            ]
        )
        displacement_r2.append(
            [
                run["displacements"]["averaging_only"]["candidates"][family][
                    "projection_r2_positive"
                ]
                for run in runs
            ]
        )

    endpoint_summary = [mean_se(values) for values in endpoint_r2]
    displacement_summary = [mean_se(values) for values in displacement_r2]

    control_distances: list[float] = []
    high_lr_distances: list[float] = []
    explicit_distances: list[float] = []
    agreements: list[float] = []
    for run in runs:
        recovery = run["explicit_recovery"]
        swa_effect_norm = (
            recovery["parameter_distance"]
            / recovery["parameter_distance_over_swa_effect"]
        )
        control_distances.append(
            run["displacements"]["full_swa"]["displacement_norm"]
            / swa_effect_norm
        )
        high_lr_distances.append(
            run["displacements"]["averaging_only"]["displacement_norm"]
            / swa_effect_norm
        )
        explicit_distances.append(recovery["parameter_distance_over_swa_effect"])
        agreements.append(recovery["prediction_agreement"])
    recovery_summary = [
        mean_se(values)
        for values in (control_distances, high_lr_distances, explicit_distances)
    ]

    blue = "#4C78A8"
    highlight = "#E45756"
    gray = "#9D9D9D"
    colors = [blue] * (len(FAMILIES) - 1) + [highlight]
    figure, axes = plt.subplots(1, 3, figsize=(15.0, 3.8))

    bar_with_se(
        axes[0],
        [value[0] for value in endpoint_summary],
        [value[1] for value in endpoint_summary],
        LABELS,
        colors,
    )
    axes[0].set_ylabel("Positive projection $R^2$")
    axes[0].set_ylim(0, 1.03)
    axes[0].tick_params(axis="x", labelsize=9)
    axes[0].set_title("A   Fit at the SWA endpoint", loc="left")

    bar_with_se(
        axes[1],
        [value[0] for value in displacement_summary],
        [value[1] for value in displacement_summary],
        LABELS,
        colors,
    )
    axes[1].set_ylabel("Positive projection $R^2$")
    axes[1].set_ylim(0, 1.03)
    axes[1].tick_params(axis="x", labelsize=9)
    axes[1].set_title("B   Fit to averaging displacement", loc="left")

    recovery_labels = ("Decayed\nSGD", "High-LR\nlast", "Explicit\npenalty")
    bar_with_se(
        axes[2],
        [value[0] for value in recovery_summary],
        [value[1] for value in recovery_summary],
        recovery_labels,
        [gray, blue, highlight],
    )
    axes[2].set_ylabel("Distance to SWA / SWA effect size")
    axes[2].set_ylim(0, 1.03)
    axes[2].set_title("C   Explicit-penalty recovery", loc="left")
    axes[2].text(
        0.98,
        0.96,
        f"prediction agreement: {np.mean(agreements):.1%}",
        ha="right",
        va="top",
        transform=axes[2].transAxes,
        fontsize=11,
    )

    figure.suptitle(
        "SWA averaging acts like a quadratic penalty centered at the pre-SWA checkpoint",
        y=1.03,
    )
    figure.tight_layout(w_pad=2.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output)
    plt.close(figure)


if __name__ == "__main__":
    main()
