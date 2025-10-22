#!/usr/bin/env python3
"""Plot implicit gradient regularisation results for MNIST sweeps."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STYLE = REPO_ROOT / "clean_fig.mplstyle"
LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="JSON result files produced by experiments/mnist_implicit_reg.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "figures" / "mnist_implicit_reg.png",
        help="Path to save the generated figure.",
    )
    parser.add_argument(
        "--style",
        type=Path,
        default=DEFAULT_STYLE,
        help="Matplotlib style sheet to apply.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively.",
    )
    return parser.parse_args()


def load_record(path: Path) -> Dict[str, float]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    best = payload.get("best_metrics") or {}
    if not best:
        raise ValueError(f"No best_metrics found in {path}.")

    lambda_theoretical = payload.get("lambda_theoretical")
    if lambda_theoretical is None:
        raise ValueError(f"Missing lambda_theoretical in {path}.")

    num_params = payload.get("num_params")
    if num_params is None:
        raise ValueError(f"Missing num_params in {path}.")

    rig = best.get("rig")
    test_acc = best.get("test_accuracy")
    if rig is None or test_acc is None:
        raise ValueError(f"Incomplete best_metrics in {path}.")

    return {
        "lambda": float(lambda_theoretical),
        "rig": float(rig),
        "test_accuracy": float(test_acc) * 100.0,
        "num_params": int(num_params),
        "source": str(path),
    }


def group_records(records: Sequence[Dict[str, float]]) -> Dict[int, List[Dict[str, float]]]:
    grouped: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for record in records:
        grouped[int(record["num_params"])].append(record)
    for values in grouped.values():
        values.sort(key=lambda entry: entry["lambda"])
    return dict(grouped)


def plot_records(grouped: Dict[int, List[Dict[str, float]]], output: Path, show: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    ax_reg, ax_acc = axes

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, (num_params, entries) in enumerate(sorted(grouped.items())):
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        lambdas = [entry["lambda"] for entry in entries]
        rig_values = [entry["rig"] for entry in entries]
        accuracies = [entry["test_accuracy"] for entry in entries]

        label = f"{num_params:,}"
        ax_reg.plot(
            lambdas,
            rig_values,
            marker="o",
            linestyle="--",
            color=color,
            label=label,
        )
        ax_acc.plot(
            lambdas,
            accuracies,
            marker="o",
            linestyle="--",
            color=color,
            label=label,
        )

    xlabel = r"$\lambda = \text{learning rate} \times \text{network size} / 4$"
    ax_reg.set_xlabel(xlabel)
    ax_acc.set_xlabel(xlabel)

    ax_reg.set_ylabel(r"$\text{Regularisation } R_{IG}$")
    ax_acc.set_ylabel("Test Accuracy (%)")

    ax_reg.set_xscale("log")
    ax_reg.set_yscale("log")
    ax_acc.set_xscale("log")

    for ax in axes:
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

    handles, labels = ax_reg.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            title="# parameters",
            loc="lower center",
            ncol=min(len(handles), 3),
            frameon=False,
        )

    ax_reg.set_title("(a)")
    ax_acc.set_title("(b)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.style.exists():
        plt.style.use(str(args.style))

    records: List[Dict[str, float]] = []
    for path in args.inputs:
        if path.is_dir():
            for candidate in sorted(path.glob("*.json")):
                try:
                    records.append(load_record(candidate))
                except ValueError as exc:
                    LOGGER.warning("%s", exc)
        else:
            try:
                records.append(load_record(path))
            except ValueError as exc:
                LOGGER.warning("%s", exc)

    if not records:
        raise SystemExit("No valid result files were provided.")

    grouped = group_records(records)
    plot_records(grouped, args.output, args.show)


if __name__ == "__main__":
    main()
