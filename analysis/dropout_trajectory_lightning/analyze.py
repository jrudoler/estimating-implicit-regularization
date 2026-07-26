#!/usr/bin/env python3
"""Aggregate the checkpointed Figure-5 Lightning trajectory reruns."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = (
    REPO_ROOT / "data" / "generated" / "dropout_trajectory_lightning" / "results"
)
DEFAULT_GRID_DIR = (
    REPO_ROOT / "data" / "generated" / "closed_form_dropout_grid" / "results"
)
DEFAULT_SUMMARY = (
    REPO_ROOT / "data" / "generated" / "dropout_trajectory_lightning" / "summary.json"
)
DEFAULT_FINDINGS = (
    REPO_ROOT / "data" / "generated" / "dropout_trajectory_lightning" / "FINDINGS.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--findings", type=Path, default=DEFAULT_FINDINGS)
    parser.add_argument("--expected-runs", type=int, default=10)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def load_json_files(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("*.json"))]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize(
    runs: list[dict[str, Any]],
    grid_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, float | int]] = []
    final_comparisons: list[float] = []
    for run in runs:
        config = run["config"]
        seed = int(config["seed"])
        for record in run["trajectory"]:
            ridge = record["closed_form"]["ridge"]
            rows.append(
                {
                    "seed": seed,
                    "fraction": float(record["fraction"]),
                    "epoch": int(record["epoch"]),
                    "lambda": float(ridge["scale_star"]),
                    "cosine": float(ridge["grad_cosine"]),
                    "residual_ratio": float(ridge["residual_ratio"]),
                    "relative_grad_norm": float(
                        record["stationarity"]["relative_grad_norm"]
                    ),
                }
            )

        tag = (
            f"d{config['depth']}_w{config['width']}_do{config['dropout']}_s{seed}.json"
        )
        original = json.loads((grid_dir / tag).read_text())
        original_lambda = float(original["closed_form"]["ridge"]["scale_star"])
        rerun_lambda = float(run["trajectory"][-1]["closed_form"]["ridge"]["scale_star"])
        final_comparisons.append(abs(rerun_lambda - original_lambda) / abs(original_lambda))

    fractions = sorted({float(row["fraction"]) for row in rows})
    by_fraction: dict[str, Any] = {}
    final_by_seed = {
        int(row["seed"]): float(row["lambda"])
        for row in rows
        if float(row["fraction"]) == 1.0
    }
    for fraction in fractions:
        subset = [row for row in rows if float(row["fraction"]) == fraction]
        lambdas = [float(row["lambda"]) for row in subset]
        paired_ratios = [
            float(row["lambda"]) / final_by_seed[int(row["seed"])] for row in subset
        ]
        by_fraction[str(fraction)] = {
            "n": len(subset),
            "median_epoch": statistics.median(int(row["epoch"]) for row in subset),
            "median_lambda": statistics.median(lambdas),
            "lambda_q25": percentile(lambdas, 25),
            "lambda_q75": percentile(lambdas, 75),
            "median_ratio_to_endpoint": statistics.median(paired_ratios),
            "ratio_q25": percentile(paired_ratios, 25),
            "ratio_q75": percentile(paired_ratios, 75),
            "median_cosine": statistics.median(
                float(row["cosine"]) for row in subset
            ),
            "median_residual_ratio": statistics.median(
                float(row["residual_ratio"]) for row in subset
            ),
            "median_relative_grad_norm": statistics.median(
                float(row["relative_grad_norm"]) for row in subset
            ),
        }

    first_config = runs[0]["config"]
    return {
        "n_runs": len(runs),
        "config": {
            "depth": first_config["depth"],
            "width": first_config["width"],
            "dropout": first_config["dropout"],
        },
        "by_fraction": by_fraction,
        "endpoint_reproduction": {
            "median_absolute_relative_error": statistics.median(final_comparisons),
            "max_absolute_relative_error": max(final_comparisons),
        },
        "rows": rows,
    }


def findings_markdown(summary: dict[str, Any]) -> str:
    config = summary["config"]
    lines = [
        "# Original-Lightning dropout trajectory stability",
        "",
        (
            f"{summary['n_runs']} seeds; depth={config['depth']}, width={config['width']}, "
            f"dropout={config['dropout']}. Each run was replayed through the original "
            "Lightning model/data/optimizer path to its stored Figure-5 endpoint."
        ),
        "",
        "| fraction | median epoch | median lambda | paired lambda/lambda(T) | "
        "median cosine | median residual ratio |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for fraction_string, values in summary["by_fraction"].items():
        lines.append(
            f"| {float(fraction_string):.1f} | {values['median_epoch']:.1f} | "
            f"{values['median_lambda']:.6e} | "
            f"{values['median_ratio_to_endpoint']:.3f} "
            f"[{values['ratio_q25']:.3f}, {values['ratio_q75']:.3f}] | "
            f"{values['median_cosine']:.4f} | "
            f"{values['median_residual_ratio']:.4f} |"
        )
    reproduction = summary["endpoint_reproduction"]
    lines.extend(
        [
            "",
            (
                "Endpoint reproduction against the completed corrected Figure-5 grid: "
                f"median absolute relative error "
                f"{100 * reproduction['median_absolute_relative_error']:.2f}%; "
                f"maximum {100 * reproduction['max_absolute_relative_error']:.2f}%."
            ),
            "",
            "Full Lightning checkpoints are retained for every seed at all three fractions.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    configure_logging()
    args = parse_args()
    runs = load_json_files(args.results_dir)
    if len(runs) != args.expected_runs:
        raise RuntimeError(
            f"Expected {args.expected_runs} completed runs, found {len(runs)}."
        )
    summary = summarize(runs, args.grid_dir)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2))
    args.findings.write_text(findings_markdown(summary))
    LOGGER.info("Wrote %s and %s", args.summary, args.findings)


if __name__ == "__main__":
    main()
