#!/usr/bin/env python3
"""Build or verify the figures referenced by paper/main.tex."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES_DIR = REPO_ROOT / "paper" / "figures"
DEFAULT_BARRETT_LONG_HORIZON_RESULTS = [
    REPO_ROOT / "results" / "barrett_igr_lh_synth_eta001.pt",
    REPO_ROOT / "results" / "barrett_igr_lh_synth_eta003.pt",
    REPO_ROOT / "results" / "barrett_igr_lh_mnist_tanh_eta001.pt",
    REPO_ROOT / "results" / "barrett_igr_lh_mnist_relu_eta001.pt",
]


@dataclass(frozen=True)
class FigureTask:
    figure_id: str
    description: str


FIGURE_TASKS: tuple[FigureTask, ...] = (
    FigureTask("tradeoff-vis", "Preserved manuscript asset with notebook provenance."),
    FigureTask("sgd-vs-full-batch", "Notebook-derived asset exported from notebooks/method-vis.ipynb."),
    FigureTask("elasticnet_recovery_mean_se", "Experiment + plot script figure."),
    FigureTask("OLS_early_stopping_figure", "Preserved manuscript asset."),
    FigureTask("lambda_vs_epochs", "Script-generated OLS early-stopping figure."),
    FigureTask("dropout_bias_ridge_panel", "Notebook-lineage figure preserved as a manuscript asset."),
    FigureTask("barrett_igr_figure2", "Experiment + plot script figure."),
    FigureTask("barrett_igr_long_horizon", "Experiment + plot script figure."),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        nargs="*",
        choices=[task.figure_id for task in FIGURE_TASKS],
        default=[task.figure_id for task in FIGURE_TASKS],
        help="Subset of paper figures to build or verify.",
    )
    parser.add_argument(
        "--elasticnet-csv",
        type=Path,
        default=REPO_ROOT / "artifacts" / "elasticnet_runs.csv",
        help="CSV prerequisite for scripts/plot_elasticnet_recovery.py.",
    )
    parser.add_argument(
        "--barrett-igr-figure2-results",
        type=Path,
        default=REPO_ROOT / "results" / "barrett_igr_figure2.pt",
        help="Results .pt produced by experiments/barrett_igr_figure2.py.",
    )
    parser.add_argument(
        "--barrett-long-horizon-results",
        type=Path,
        nargs="*",
        default=None,
        help="One or more .pt files from experiments/barrett_igr_long_horizon.py.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    rendered = " ".join(command)
    print(rendered)
    if not dry_run:
        subprocess.run(command, check=True, cwd=REPO_ROOT)


def export_preserved_asset(figure_id: str, dry_run: bool) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "export_notebook_figure.py"),
        figure_id,
    ]
    run_command(command, dry_run)


def build_elasticnet(csv_path: Path, dry_run: bool) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing prerequisite CSV for elasticnet figure: {csv_path}"
        )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plot_elasticnet_recovery.py"),
        "--csv",
        str(csv_path),
        "--output",
        str(PAPER_FIGURES_DIR / "elasticnet_recovery_mean_se.pdf"),
    ]
    run_command(command, dry_run)


def build_lambda_vs_epochs(dry_run: bool) -> None:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "plot_lambda_vs_epochs.py"),
        "--out",
        str(PAPER_FIGURES_DIR / "lambda_vs_epochs.pdf"),
    ]
    run_command(command, dry_run)


def build_barrett_igr_figure2(results_path: Path, dry_run: bool) -> None:
    if not results_path.exists():
        raise FileNotFoundError(
            f"Missing prerequisite results for barrett_igr_figure2: {results_path}"
        )
    command = [
        sys.executable,
        str(REPO_ROOT / "analysis" / "barrett_igr_figure2_plot.py"),
        "--results",
        str(results_path),
        "--out",
        str(PAPER_FIGURES_DIR / "barrett_igr_figure2.pdf"),
    ]
    run_command(command, dry_run)


def build_barrett_long_horizon(
    results_paths: list[Path] | None,
    dry_run: bool,
) -> None:
    resolved = results_paths or DEFAULT_BARRETT_LONG_HORIZON_RESULTS
    missing = [path for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing prerequisite results for barrett_igr_long_horizon: "
            + ", ".join(str(path) for path in missing)
        )
    command = [
        sys.executable,
        str(REPO_ROOT / "analysis" / "barrett_igr_long_horizon_plot.py"),
        "--results",
        *[str(path) for path in resolved],
        "--out",
        str(PAPER_FIGURES_DIR / "barrett_igr_long_horizon.pdf"),
    ]
    run_command(command, dry_run)


def main() -> None:
    args = parse_args()
    failures: list[str] = []

    for figure_id in args.figures:
        try:
            if figure_id in {
                "tradeoff-vis",
                "sgd-vs-full-batch",
                "OLS_early_stopping_figure",
                "dropout_bias_ridge_panel",
            }:
                export_preserved_asset(figure_id, args.dry_run)
            elif figure_id == "elasticnet_recovery_mean_se":
                build_elasticnet(args.elasticnet_csv, args.dry_run)
            elif figure_id == "lambda_vs_epochs":
                build_lambda_vs_epochs(args.dry_run)
            elif figure_id == "barrett_igr_figure2":
                build_barrett_igr_figure2(args.barrett_igr_figure2_results, args.dry_run)
            elif figure_id == "barrett_igr_long_horizon":
                build_barrett_long_horizon(args.barrett_long_horizon_results, args.dry_run)
            else:
                raise ValueError(f"Unhandled figure id: {figure_id}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{figure_id}: {exc}")

    if failures:
        print("\nPaper figure build failed for:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print("\nPaper figure build completed successfully.")


if __name__ == "__main__":
    main()
