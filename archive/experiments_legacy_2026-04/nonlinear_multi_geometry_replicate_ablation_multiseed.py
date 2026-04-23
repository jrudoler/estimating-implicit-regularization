#!/usr/bin/env python3
"""Aggregate replicate-ablation runs across seeds and plot variability bands."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))


STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
if STYLE_PATH.exists():
    plt.style.use(str(STYLE_PATH))

# Okabe-Ito colorblind-safe palette.
COLORBLIND_CYCLE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

COMPONENT_COLOR_MAP = {
    "l1": "#009E73",
    "l2": "#0072B2",
    "nuclear": "#D55E00",
}


SUMMARY_NUMERIC_FIELDS = [
    "gradient_cosine",
    "relative_residual",
    "full_train_mse",
    "full_test_mse",
    "replicate_train_mse_mean",
    "replicate_test_mse_mean",
    "mean_lambda_rel_error",
    "mean_p_abs_error",
]

COMPONENT_NUMERIC_FIELDS = [
    "gradient_cosine",
    "relative_residual",
    "true_lambda",
    "estimated_lambda",
    "true_p",
    "estimated_p",
    "lambda_rel_error",
    "p_abs_error",
]


def parse_int_list(raw_value: str) -> list[int]:
    values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_path_list(raw_value: str) -> list[Path]:
    paths = [Path(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not paths:
        raise ValueError("Expected at least one path.")
    return paths


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def attach_seed(rows: Iterable[dict[str, str]], seed: int) -> list[dict[str, str]]:
    return [{**row, "seed": str(seed)} for row in rows]


def aggregate_rows(
    rows: Sequence[dict[str, str]],
    *,
    group_fields: list[str],
    numeric_fields: list[str],
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped[key].append(row)

    aggregated_rows: list[dict[str, str]] = []
    for key in sorted(grouped):
        bucket = grouped[key]
        base = {field: value for field, value in zip(group_fields, key)}
        base["n_seeds"] = str(len(bucket))
        for numeric_field in numeric_fields:
            values = np.array([float(row[numeric_field]) for row in bucket], dtype=float)
            base[f"{numeric_field}_mean"] = f"{float(values.mean()):.12g}"
            base[f"{numeric_field}_std"] = f"{float(values.std(ddof=0)):.12g}"
            base[f"{numeric_field}_min"] = f"{float(values.min()):.12g}"
            base[f"{numeric_field}_max"] = f"{float(values.max()):.12g}"
        aggregated_rows.append(base)
    return aggregated_rows


def write_rows(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def case_seed_metric_matrix(
    summary_rows: Sequence[dict[str, str]],
    *,
    case_name: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    filtered_rows = [row for row in summary_rows if row["case_name"] == case_name]
    counts = sorted({int(row["n_replicates"]) for row in filtered_rows})
    seeds = sorted({int(row["seed"]) for row in filtered_rows})
    matrix = np.zeros((len(seeds), len(counts)), dtype=float)
    for seed_index, seed in enumerate(seeds):
        rows_by_count = {
            int(row["n_replicates"]): row
            for row in filtered_rows
            if int(row["seed"]) == seed
        }
        for count_index, count in enumerate(counts):
            matrix[seed_index, count_index] = float(rows_by_count[count][metric])
    return np.array(counts, dtype=float), matrix, seeds


def plot_case_variability(
    summary_rows: Sequence[dict[str, str]],
    *,
    case_name: str,
    output_path: Path,
) -> None:
    filtered_rows = [row for row in summary_rows if row["case_name"] == case_name]
    case_title = filtered_rows[0]["case_title"]
    metric_specs = [
        ("gradient_cosine", "cosine", "Gradient Alignment", "#0072B2"),
        ("relative_residual", "relative residual", "Relative Residual", "#D55E00"),
        ("mean_lambda_rel_error", "mean lambda rel. error", "Scale Error", "#009E73"),
        ("mean_p_abs_error", "mean |p error|", "Exponent Error", "#CC79A7"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    flat_axes = axes.reshape(-1)
    for axis, (metric, ylabel, title, color) in zip(flat_axes, metric_specs):
        counts, matrix, seeds = case_seed_metric_matrix(summary_rows, case_name=case_name, metric=metric)
        for seed_values in matrix:
            axis.plot(counts, seed_values, color=color, alpha=0.18, linewidth=1.0)
        mean_values = matrix.mean(axis=0)
        std_values = matrix.std(axis=0, ddof=0)
        axis.plot(counts, mean_values, color=color, marker="o", linewidth=2.0)
        axis.fill_between(
            counts,
            mean_values - std_values,
            mean_values + std_values,
            color=color,
            alpha=0.2,
        )
        axis.set_xscale("log")
        axis.set_xlabel("bootstrap retrains")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    fig.suptitle(f"{case_title}\nMean ± 1 SD across {len(seeds)} seeds", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_global_summary(
    aggregated_summary_rows: Sequence[dict[str, str]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    case_names = sorted({row["case_name"] for row in aggregated_summary_rows})
    for case_index, case_name in enumerate(case_names):
        color = COLORBLIND_CYCLE[case_index % len(COLORBLIND_CYCLE)]
        case_rows = sorted(
            (row for row in aggregated_summary_rows if row["case_name"] == case_name),
            key=lambda row: int(row["n_replicates"]),
        )
        counts = np.array([int(row["n_replicates"]) for row in case_rows], dtype=float)
        cosine_mean = np.array([float(row["gradient_cosine_mean"]) for row in case_rows], dtype=float)
        cosine_std = np.array([float(row["gradient_cosine_std"]) for row in case_rows], dtype=float)
        residual_mean = np.array([float(row["relative_residual_mean"]) for row in case_rows], dtype=float)
        residual_std = np.array([float(row["relative_residual_std"]) for row in case_rows], dtype=float)

        axes[0].plot(counts, cosine_mean, color=color, marker="o", label=case_name)
        axes[0].fill_between(counts, cosine_mean - cosine_std, cosine_mean + cosine_std, color=color, alpha=0.12)
        axes[1].plot(counts, residual_mean, color=color, marker="o", label=case_name)
        axes[1].fill_between(
            counts,
            residual_mean - residual_std,
            residual_mean + residual_std,
            color=color,
            alpha=0.12,
        )

    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("bootstrap retrains")
    axes[0].set_ylabel("cosine")
    axes[0].set_title("Gradient Alignment")
    axes[1].set_ylabel("relative residual")
    axes[1].set_title("Relative Residual")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Replicate Ablation with Seed Variability", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def component_case_label_matrix(
    component_rows: Sequence[dict[str, str]],
    *,
    case_name: str,
    label: str,
    metric: str,
    truth_metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    filtered_rows = [
        row
        for row in component_rows
        if row["case_name"] == case_name and row["label"] == label
    ]
    counts = sorted({int(row["n_replicates"]) for row in filtered_rows})
    seeds = sorted({int(row["seed"]) for row in filtered_rows})
    estimate_matrix = np.zeros((len(seeds), len(counts)), dtype=float)
    truth_matrix = np.zeros((len(seeds), len(counts)), dtype=float)
    for seed_index, seed in enumerate(seeds):
        rows_by_count = {
            int(row["n_replicates"]): row
            for row in filtered_rows
            if int(row["seed"]) == seed
        }
        for count_index, count in enumerate(counts):
            row = rows_by_count[count]
            estimate_matrix[seed_index, count_index] = float(row[metric])
            truth_matrix[seed_index, count_index] = float(row[truth_metric])
    truth_values = truth_matrix.mean(axis=0)
    return np.array(counts, dtype=float), estimate_matrix, truth_values, seeds


def plot_parameter_facets_by_case(
    component_rows: Sequence[dict[str, str]],
    *,
    metric: str,
    truth_metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    case_names = sorted({row["case_name"] for row in component_rows})
    case_title_lookup = {
        row["case_name"]: row["case_title"]
        for row in component_rows
    }
    component_labels = sorted({row["label"] for row in component_rows})
    label_colors = {
        label: COMPONENT_COLOR_MAP.get(label, COLORBLIND_CYCLE[index % len(COLORBLIND_CYCLE)])
        for index, label in enumerate(component_labels)
    }

    n_cols = 3
    n_rows = int(np.ceil(len(case_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 3.4 * n_rows))
    flat_axes = np.asarray(axes).reshape(-1)

    for axis_index, case_name in enumerate(case_names):
        axis = flat_axes[axis_index]
        labels_in_case = sorted(
            {
                row["label"]
                for row in component_rows
                if row["case_name"] == case_name
            }
        )
        for label in labels_in_case:
            counts, estimate_matrix, truth_values, _ = component_case_label_matrix(
                component_rows,
                case_name=case_name,
                label=label,
                metric=metric,
                truth_metric=truth_metric,
            )
            color = label_colors[label]
            for seed_values in estimate_matrix:
                axis.plot(counts, seed_values, color=color, alpha=0.15, linewidth=1.0)
            estimate_mean = estimate_matrix.mean(axis=0)
            estimate_std = estimate_matrix.std(axis=0, ddof=0)
            axis.plot(
                counts,
                estimate_mean,
                color=color,
                linewidth=2.0,
                marker="o",
            )
            axis.fill_between(
                counts,
                estimate_mean - estimate_std,
                estimate_mean + estimate_std,
                color=color,
                alpha=0.2,
            )
            axis.plot(
                counts,
                truth_values,
                color=color,
                linestyle="--",
                linewidth=1.4,
            )

        axis.set_xscale("log")
        axis.set_xlabel("bootstrap retrains")
        axis.set_ylabel(ylabel)
        axis.set_title(case_title_lookup[case_name])

    for axis in flat_axes[len(case_names):]:
        axis.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], color=label_colors[label], linewidth=2.0, label=label)
        for label in component_labels
    ]
    fig.legend(
        handles=legend_handles,
        labels=[handle.get_label() for handle in legend_handles],
        frameon=False,
        loc="upper center",
        ncol=max(1, min(4, len(legend_handles))),
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        f"{title}\nSolid: estimated mean ± 1 SD across seeds, dashed: ground truth",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=str, required=True)
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/phase2/nonlinear_multi_geometry_replicate_ablation_multiseed"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures/nonlinear_multi_geometry_replicate_ablation_multiseed"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = parse_path_list(args.input_dirs)
    seeds = parse_int_list(args.seeds)
    if len(input_dirs) != len(seeds):
        raise ValueError("input_dirs and seeds must have matching lengths.")

    summary_by_seed_rows: list[dict[str, str]] = []
    component_by_seed_rows: list[dict[str, str]] = []
    for input_dir, seed in zip(input_dirs, seeds):
        summary_by_seed_rows.extend(attach_seed(read_csv_rows(input_dir / "summary.csv"), seed))
        component_by_seed_rows.extend(attach_seed(read_csv_rows(input_dir / "component_recovery.csv"), seed))

    summary_by_seed_rows.sort(key=lambda row: (row["case_name"], int(row["seed"]), int(row["n_replicates"])))
    component_by_seed_rows.sort(
        key=lambda row: (row["case_name"], int(row["seed"]), int(row["n_replicates"]), row["label"])
    )

    aggregated_summary_rows = aggregate_rows(
        summary_by_seed_rows,
        group_fields=["case_name", "case_title", "n_components", "n_replicates", "n_total_solutions"],
        numeric_fields=SUMMARY_NUMERIC_FIELDS,
    )
    aggregated_component_rows = aggregate_rows(
        component_by_seed_rows,
        group_fields=["case_name", "case_title", "n_replicates", "n_total_solutions", "label", "family"],
        numeric_fields=COMPONENT_NUMERIC_FIELDS,
    )

    write_rows(summary_by_seed_rows, args.output_dir / "summary_by_seed.csv")
    write_rows(aggregated_summary_rows, args.output_dir / "summary_aggregated.csv")
    write_rows(component_by_seed_rows, args.output_dir / "component_recovery_by_seed.csv")
    write_rows(aggregated_component_rows, args.output_dir / "component_recovery_aggregated.csv")

    case_names = sorted({row["case_name"] for row in summary_by_seed_rows})
    for case_name in case_names:
        plot_case_variability(
            summary_by_seed_rows,
            case_name=case_name,
            output_path=args.figure_dir / f"{case_name}.pdf",
        )
    plot_global_summary(aggregated_summary_rows, args.figure_dir / "summary.pdf")
    plot_parameter_facets_by_case(
        component_by_seed_rows,
        metric="estimated_lambda",
        truth_metric="true_lambda",
        ylabel="lambda",
        title="Estimated Lambda by Ground-Truth Case",
        output_path=args.figure_dir / "facet_lambda_by_case.pdf",
    )
    plot_parameter_facets_by_case(
        component_by_seed_rows,
        metric="estimated_p",
        truth_metric="true_p",
        ylabel="p",
        title="Estimated Exponent by Ground-Truth Case",
        output_path=args.figure_dir / "facet_p_by_case.pdf",
    )


if __name__ == "__main__":
    main()
