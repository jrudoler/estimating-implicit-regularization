#!/usr/bin/env python3
"""Measure geometry-recovery identifiability as bootstrap retrain count increases."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import json
import logging
from pathlib import Path
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from experiments.nonlinear_multi_geometry_suite import (  # noqa: E402
    CaseRecovery,
    GeometryCase,
    build_cases,
    collect_case_replicate_pool,
    generate_dataset,
    recover_case_from_pool,
    set_seed,
)


STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
if STYLE_PATH.exists():
    plt.style.use(str(STYLE_PATH))

# Okabe-Ito colorblind-safe palette.
COLORBLIND_CYCLE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]
COMPONENT_COLOR_MAP = {
    "l1": "#009E73",
    "l2": "#0072B2",
    "nuclear": "#D55E00",
}

LOGGER = logging.getLogger(__name__)


def parse_replicate_counts(raw_value: str) -> list[int]:
    counts = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not counts:
        raise ValueError("At least one replicate count is required.")
    if any(count < 0 for count in counts):
        raise ValueError("Replicate counts must be non-negative.")
    return sorted(set(counts))


def select_cases(
    cases: Sequence[GeometryCase],
    *,
    suite: str,
    case_names: str,
) -> list[GeometryCase]:
    selected_cases = list(cases)
    if suite == "single":
        selected_cases = [case for case in selected_cases if len(case.components) == 1]
    elif suite == "multi":
        selected_cases = [case for case in selected_cases if len(case.components) > 1]

    if case_names != "all":
        requested_names = {name.strip() for name in case_names.split(",") if name.strip()}
        selected_cases = [case for case in selected_cases if case.name in requested_names]
        missing = requested_names.difference({case.name for case in selected_cases})
        if missing:
            raise ValueError(f"Unknown case names: {sorted(missing)}")
    if not selected_cases:
        raise ValueError("No cases selected for replicate ablation.")
    return selected_cases


def build_datasets(args: argparse.Namespace) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    dataset, _ = generate_dataset(
        n_samples=args.n_samples,
        input_dim=args.input_dim,
        function_class="teacher_relu",
        noise_std=args.noise_std,
        seed=args.seed,
        data_mode="structured",
        input_rank=args.input_rank if args.input_rank > 0 else args.input_dim,
        input_spectrum=args.input_spectrum,
        input_spectrum_decay=args.input_spectrum_decay,
        teacher_rank=args.teacher_rank if args.teacher_rank > 0 else min(args.input_dim, args.teacher_hidden_dim),
        teacher_spectrum=args.teacher_spectrum,
        teacher_spectrum_decay=args.teacher_spectrum_decay,
        teacher_hidden_dim=args.teacher_hidden_dim,
        teacher_activation=args.teacher_activation,
        target_scale=args.target_scale,
    )
    test_size = int(len(dataset) * args.test_fraction)
    val_size = int(len(dataset) * args.val_fraction)
    train_size = len(dataset) - val_size - test_size
    train_split, val_split, test_split = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    full_inputs, full_targets = dataset.tensors
    train_indices = torch.as_tensor(train_split.indices, dtype=torch.long)
    val_indices = torch.as_tensor(val_split.indices, dtype=torch.long)
    test_indices = torch.as_tensor(test_split.indices, dtype=torch.long)
    train_dataset = TensorDataset(full_inputs[train_indices], full_targets[train_indices])
    val_dataset = TensorDataset(full_inputs[val_indices], full_targets[val_indices])
    test_dataset = TensorDataset(full_inputs[test_indices], full_targets[test_indices])
    return train_dataset, val_dataset, test_dataset


def mean_component_lambda_error(result: CaseRecovery) -> float:
    return float(np.mean([component.lambda_rel_error for component in result.components]))


def mean_component_p_error(result: CaseRecovery) -> float:
    return float(np.mean([component.p_abs_error for component in result.components]))


def plot_case_replicate_ablation(
    case: GeometryCase,
    count_results: Sequence[tuple[int, CaseRecovery]],
    output_path: Path,
) -> None:
    replicate_counts = np.array([count for count, _ in count_results], dtype=float)
    first_result = count_results[0][1]
    component_labels = [component.label for component in first_result.components]
    label_colors = {
        label: COMPONENT_COLOR_MAP.get(label, COLORBLIND_CYCLE[index % len(COLORBLIND_CYCLE)])
        for index, label in enumerate(component_labels)
    }

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))

    axes[0, 0].plot(
        replicate_counts,
        [result.gradient_cosine for _, result in count_results],
        marker="o",
        color="#0072B2",
    )
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("bootstrap retrains")
    axes[0, 0].set_ylabel("cosine")
    axes[0, 0].set_title("Gradient Alignment")

    axes[0, 1].plot(
        replicate_counts,
        [result.relative_residual for _, result in count_results],
        marker="o",
        color="#D55E00",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("bootstrap retrains")
    axes[0, 1].set_ylabel("relative residual")
    axes[0, 1].set_title("Relative Residual")

    for component_index, label in enumerate(component_labels):
        color = label_colors[label]
        true_lambda = first_result.components[component_index].true_lambda
        estimated_lambdas = [
            result.components[component_index].estimated_lambda
            for _, result in count_results
        ]
        axes[1, 0].plot(
            replicate_counts,
            estimated_lambdas,
            marker="o",
            color=color,
            label=f"{label} estimate",
        )
        axes[1, 0].axhline(true_lambda, color=color, linestyle="--", linewidth=1.0)

        true_p = first_result.components[component_index].true_p
        estimated_ps = [
            result.components[component_index].estimated_p
            for _, result in count_results
        ]
        axes[1, 1].plot(
            replicate_counts,
            estimated_ps,
            marker="o",
            color=color,
            label=f"{label} estimate",
        )
        axes[1, 1].axhline(true_p, color=color, linestyle="--", linewidth=1.0)

    for axis in axes[1]:
        axis.set_xscale("log")
        axis.set_xlabel("bootstrap retrains")
    axes[1, 0].set_ylabel("lambda")
    axes[1, 0].set_title("Scale Trajectories")
    axes[1, 1].set_ylabel("p")
    axes[1, 1].set_title("Exponent Trajectories")
    axes[1, 1].legend(frameon=False, loc="best")

    fig.suptitle(
        (
            f"{case.title}\n"
            "Each fit uses the full-data solution plus the indicated number of bootstrap retrains."
        ),
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary(
    all_results: dict[str, list[tuple[int, CaseRecovery]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for case_index, (case_name, count_results) in enumerate(all_results.items()):
        replicate_counts = np.array([count for count, _ in count_results], dtype=float)
        color = COLORBLIND_CYCLE[case_index % len(COLORBLIND_CYCLE)]
        axes[0].plot(
            replicate_counts,
            [result.gradient_cosine for _, result in count_results],
            marker="o",
            color=color,
            label=case_name,
        )
        axes[1].plot(
            replicate_counts,
            [result.relative_residual for _, result in count_results],
            marker="o",
            color=color,
            label=case_name,
        )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("bootstrap retrains")
    axes[0].set_ylabel("cosine")
    axes[0].set_title("Gradient Alignment")
    axes[1].set_ylabel("relative residual")
    axes[1].set_title("Relative Residual")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Geometry Recovery vs Bootstrap Retrain Count", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_summary_csv(
    all_results: dict[str, list[tuple[int, CaseRecovery]]],
    output_path: Path,
) -> None:
    rows = [
        {
            "case_name": result.case_name,
            "case_title": result.case_title,
            "n_components": result.n_components,
            "n_replicates": n_replicates,
            "n_total_solutions": n_replicates + 1,
            "gradient_cosine": result.gradient_cosine,
            "relative_residual": result.relative_residual,
            "full_train_mse": result.full_train_mse,
            "full_test_mse": result.full_test_mse,
            "replicate_train_mse_mean": result.replicate_train_mse_mean,
            "replicate_test_mse_mean": result.replicate_test_mse_mean,
            "stationarity_residual_mean": result.stationarity_residual_mean,
            "mean_lambda_rel_error": mean_component_lambda_error(result),
            "mean_p_abs_error": mean_component_p_error(result),
        }
        for count_results in all_results.values()
        for n_replicates, result in count_results
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_component_csv(
    all_results: dict[str, list[tuple[int, CaseRecovery]]],
    output_path: Path,
) -> None:
    rows = [
        {
            "case_name": result.case_name,
            "case_title": result.case_title,
            "n_replicates": n_replicates,
            "n_total_solutions": n_replicates + 1,
            "gradient_cosine": result.gradient_cosine,
            "relative_residual": result.relative_residual,
            "label": component.label,
            "family": component.family,
            "true_lambda": component.true_lambda,
            "estimated_lambda": component.estimated_lambda,
            "true_p": component.true_p,
            "estimated_p": component.estimated_p,
            "lambda_rel_error": component.lambda_rel_error,
            "p_abs_error": component.p_abs_error,
        }
        for count_results in all_results.values()
        for n_replicates, result in count_results
        for component in result.components
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=12)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=350)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-spectrum", type=str, default="identity")
    parser.add_argument("--input-rank", type=int, default=0)
    parser.add_argument("--input-spectrum-decay", type=float, default=1.5)
    parser.add_argument("--teacher-spectrum", type=str, default="identity")
    parser.add_argument("--teacher-rank", type=int, default=0)
    parser.add_argument("--teacher-spectrum-decay", type=float, default=0.0)
    parser.add_argument("--teacher-hidden-dim", type=int, default=16)
    parser.add_argument("--teacher-activation", type=str, default="relu", choices=["relu", "tanh", "linear"])
    parser.add_argument("--target-scale", type=float, default=1.0)

    parser.add_argument("--regularizer-epsilon", type=float, default=1e-6)
    parser.add_argument("--target-gradient-scale", type=float, default=0.3)
    parser.add_argument("--suite", type=str, default="all", choices=["single", "multi", "all"])
    parser.add_argument("--cases", type=str, default="all")
    parser.add_argument("--resample-mode", type=str, default="bootstrap", choices=["subsample", "bootstrap"])
    parser.add_argument("--replicate-counts", type=str, default="1,10,50,100,1000")
    parser.add_argument("--sample-fraction", type=float, default=1.0)

    parser.add_argument("--estimation-lr", type=float, default=5e-2)
    parser.add_argument("--estimation-max-epochs", type=int, default=2500)
    parser.add_argument("--estimation-patience", type=int, default=250)
    parser.add_argument("--p-min", type=float, default=0.5)
    parser.add_argument("--p-max", type=float, default=3.0)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/phase2/nonlinear_multi_geometry_replicate_ablation"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures/nonlinear_multi_geometry_replicate_ablation"),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    replicate_counts = parse_replicate_counts(args.replicate_counts)
    max_replicates = max(replicate_counts)
    train_dataset, val_dataset, test_dataset = build_datasets(args)

    cases = select_cases(
        build_cases(
            base_scale=args.target_gradient_scale,
            epsilon=args.regularizer_epsilon,
        ),
        suite=args.suite,
        case_names=args.cases,
    )
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    LOGGER.info(
        "Running replicate ablation for %d cases with counts=%s on %s",
        len(cases),
        replicate_counts,
        accelerator,
    )

    all_results: dict[str, list[tuple[int, CaseRecovery]]] = {}
    for case_index, case in enumerate(cases):
        LOGGER.info(
            "Collecting replicate pool for %s (%d/%d) with max_replicates=%d",
            case.name,
            case_index + 1,
            len(cases),
            max_replicates,
        )
        pool = collect_case_replicate_pool(
            case,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            input_dim=args.input_dim,
            depth=args.depth,
            width=args.width,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            accelerator=accelerator,
            resample_mode=args.resample_mode,
            n_replicates=max_replicates,
            sample_fraction=args.sample_fraction,
            seed=args.seed + 1_000 * case_index,
        )
        count_results: list[tuple[int, CaseRecovery]] = []
        for n_replicates in replicate_counts:
            result = recover_case_from_pool(
                pool,
                n_replicates=n_replicates,
                estimation_lr=args.estimation_lr,
                estimation_max_epochs=args.estimation_max_epochs,
                estimation_patience=args.estimation_patience,
                p_min=args.p_min,
                p_max=args.p_max,
            )
            count_results.append((n_replicates, result))
            LOGGER.info(
                "Case %s | n_replicates=%d | cosine=%.4f residual=%.4f mean_lambda_err=%.4f mean_p_err=%.4f",
                case.name,
                n_replicates,
                result.gradient_cosine,
                result.relative_residual,
                mean_component_lambda_error(result),
                mean_component_p_error(result),
            )
        all_results[case.name] = count_results
        plot_case_replicate_ablation(
            case,
            count_results,
            args.figure_dir / f"{case.name}.pdf",
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sweep_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                case_name: [
                    {
                        "n_replicates": n_replicates,
                        "result": asdict(result),
                    }
                    for n_replicates, result in count_results
                ]
                for case_name, count_results in all_results.items()
            },
            handle,
            indent=2,
        )
    write_summary_csv(all_results, args.output_dir / "summary.csv")
    write_component_csv(all_results, args.output_dir / "component_recovery.csv")
    plot_metric_summary(all_results, args.figure_dir / "summary.pdf")
    LOGGER.info("Saved replicate ablation outputs to %s and %s", args.output_dir, args.figure_dir)


if __name__ == "__main__":
    main()
