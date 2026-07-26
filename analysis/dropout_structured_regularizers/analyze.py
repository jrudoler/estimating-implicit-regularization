"""Aggregate the structured-regularizer dropout sweep into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

LOGGER = logging.getLogger(__name__)
FAMILIES = (
    "global_ridge",
    "layerwise_ridge",
    "activation_quadratic",
    "squared_path_norm",
)


def load_runs(results_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            runs.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, KeyError) as error:
            LOGGER.warning("skipping unreadable result %s: %s", path, error)
    return runs


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def mean_crossfit_r2(run: dict[str, Any], family: str) -> float:
    crossfit = run["candidates"][family]["crossfit"]
    return (
        crossfit["a_to_b"]["heldout"]["projection_r2"]
        + crossfit["b_to_a"]["heldout"]["projection_r2"]
    ) / 2.0


def cluster_bootstrap_median_ci(
    runs: list[dict[str, Any]],
    family: str,
    draws: int = 10_000,
) -> list[float]:
    """Seed-cluster bootstrap CI for paired held-out improvement over ridge."""
    seeds = sorted({run["config"]["seed"] for run in runs})
    by_seed = {
        seed: [run for run in runs if run["config"]["seed"] == seed] for seed in seeds
    }
    generator = np.random.default_rng(2027)
    estimates = np.empty(draws)
    for draw in range(draws):
        sampled_seeds = generator.choice(seeds, size=len(seeds), replace=True)
        improvements = [
            mean_crossfit_r2(run, family)
            - mean_crossfit_r2(run, "global_ridge")
            for seed in sampled_seeds
            for run in by_seed[int(seed)]
        ]
        estimates[draw] = np.median(improvements)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return [float(low), float(high)]


def scope_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    dropout_values = [run["config"]["dropout"] for run in runs]
    for family in FAMILIES:
        full = [run["candidates"][family]["full"]["projection_r2"] for run in runs]
        crossfit = [mean_crossfit_r2(run, family) for run in runs]
        rho_full, p_full = stats.spearmanr(dropout_values, full)
        rho_crossfit, p_crossfit = stats.spearmanr(dropout_values, crossfit)
        improvement_full = [
            run["candidates"][family]["full"]["projection_r2"]
            - run["candidates"]["global_ridge"]["full"]["projection_r2"]
            for run in runs
        ]
        improvement_crossfit = [
            mean_crossfit_r2(run, family)
            - mean_crossfit_r2(run, "global_ridge")
            for run in runs
        ]
        output[family] = {
            "median_full_projection_r2": median(full),
            "median_crossfit_projection_r2": median(crossfit),
            "median_full_improvement_over_global_ridge": median(improvement_full),
            "median_crossfit_improvement_over_global_ridge": median(
                improvement_crossfit
            ),
            "crossfit_improvement_seed_cluster_bootstrap_95ci": (
                cluster_bootstrap_median_ci(runs, family)
            ),
            "crossfit_improvement_positive_runs": sum(
                value > 0 for value in improvement_crossfit
            ),
            "crossfit_improvement_total_runs": len(improvement_crossfit),
            "spearman_dropout_full_r2": {
                "rho": float(rho_full),
                "p": float(p_full),
            },
            "spearman_dropout_crossfit_r2": {
                "rho": float(rho_crossfit),
                "p": float(p_crossfit),
            },
        }
    return output


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    dropouts = sorted({run["config"]["dropout"] for run in runs})
    by_dropout: dict[str, Any] = {}
    for dropout in dropouts:
        subset = [run for run in runs if run["config"]["dropout"] == dropout]
        family_summary: dict[str, Any] = {}
        for family in FAMILIES:
            full_r2 = [
                run["candidates"][family]["full"]["projection_r2"] for run in subset
            ]
            heldout_r2 = [mean_crossfit_r2(run, family) for run in subset]
            active_counts = [
                len(run["candidates"][family]["full"]["active"]) for run in subset
            ]
            family_summary[family] = {
                "median_full_projection_r2": median(full_r2),
                "median_crossfit_projection_r2": median(heldout_r2),
                "median_active_coefficients": median(active_counts),
            }
        winners_full = Counter(
            max(
                FAMILIES,
                key=lambda family: run["candidates"][family]["full"]["projection_r2"],
            )
            for run in subset
        )
        winners_crossfit = Counter(
            max(FAMILIES, key=lambda family: mean_crossfit_r2(run, family))
            for run in subset
        )
        by_dropout[f"{dropout:g}"] = {
            "n": len(subset),
            "families": family_summary,
            "winner_counts_full": dict(winners_full),
            "winner_counts_crossfit": dict(winners_crossfit),
        }

    positive_runs = [run for run in runs if run["config"]["dropout"] > 0]
    return {
        "n_runs": len(runs),
        "dropouts": dropouts,
        "overall": scope_summary(runs),
        "dropout_positive": scope_summary(positive_runs),
        "by_dropout": by_dropout,
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Structured regularizers for the Figure 5 dropout endpoint",
        "",
        f"Runs analyzed: {summary['n_runs']}. Coefficients use exact nonnegative "
        "least squares. Cross-fit values average A→B and B→A held-out gradient "
        "evaluations; zero is the held-out baseline, so negative held-out "
        "$R^2_{\\mathrm{proj}}$ means the fitted direction hurts out of sample.",
        "",
        "## Dropout-positive endpoints (primary comparison)",
        "",
        "| family | median full projection R² | median cross-fit projection R² | "
        "cross-fit improvement over global ridge [seed-cluster 95% CI] | "
        "paired wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        record = summary["dropout_positive"][family]
        low, high = record["crossfit_improvement_seed_cluster_bootstrap_95ci"]
        lines.append(
            f"| {family} | {record['median_full_projection_r2']:.5f} | "
            f"{record['median_crossfit_projection_r2']:.5f} | "
            f"{record['median_crossfit_improvement_over_global_ridge']:+.5f} "
            f"[{low:+.5f}, {high:+.5f}] | "
            f"{record['crossfit_improvement_positive_runs']}/"
            f"{record['crossfit_improvement_total_runs']} |"
        )

    lines.extend(
        [
            "",
            "## By dropout rate",
            "",
            "| dropout | family | median full projection R² | "
            "median cross-fit projection R² |",
            "|---:|---|---:|---:|",
        ]
    )
    for dropout, dropout_record in summary["by_dropout"].items():
        for family in FAMILIES:
            record = dropout_record["families"][family]
            lines.append(
                f"| {dropout} | {family} | "
                f"{record['median_full_projection_r2']:.5f} | "
                f"{record['median_crossfit_projection_r2']:.5f} |"
            )

    best_crossfit = max(
        FAMILIES,
        key=lambda family: summary["dropout_positive"][family][
            "median_crossfit_projection_r2"
        ],
    )
    global_score = summary["dropout_positive"]["global_ridge"][
        "median_crossfit_projection_r2"
    ]
    best_score = summary["dropout_positive"][best_crossfit][
        "median_crossfit_projection_r2"
    ]
    path_improvement = summary["dropout_positive"]["squared_path_norm"][
        "median_crossfit_improvement_over_global_ridge"
    ]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"The strongest median held-out fit is **{best_crossfit}** "
            f"({best_score:.5f}, versus {global_score:.5f} for global ridge). "
            f"The squared path norm changes held-out projection R² by "
            f"{path_improvement:+.5f} relative to global ridge at dropout-positive "
            "endpoints.",
            "",
            "Interpret the full-data numbers as descriptive projection capacity "
            "and the cross-fit numbers as the guard against extra-coefficient "
            "overfitting. A richer family should only replace ridge in the paper "
            "if its held-out improvement is positive and consistent across "
            "dropout rates/seeds. Dropout = 0 is retained above as a negative-control "
            "diagnostic: any apparent fit there is optimizer non-stationarity or "
            "finite-sample gradient structure, not a dropout regularizer.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/generated/dropout_structured_regularizers/results"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/generated/dropout_structured_regularizers/summary.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("data/generated/dropout_structured_regularizers/FINDINGS.md"),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    runs = load_runs(args.results_dir)
    if not runs:
        raise SystemExit(f"no JSON results found in {args.results_dir}")
    summary = summarize(runs)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2))
    args.report_output.write_text(markdown_report(summary))
    LOGGER.info(
        "analyzed %d runs; wrote %s and %s",
        len(runs),
        args.summary_output,
        args.report_output,
    )


if __name__ == "__main__":
    main()
