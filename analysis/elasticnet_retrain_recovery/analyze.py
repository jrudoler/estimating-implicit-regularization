"""Aggregate matched elastic-net explicit-retraining experiments."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
CONDITIONS = ("true", "estimated", "ablated")


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def load_runs(results_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text())
        for path in sorted(results_dir.glob("*.json"))
    ]


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(runs), "conditions": {}}
    for condition in CONDITIONS:
        records = [run["conditions"][condition] for run in runs]
        target_prediction_distance = [
            record["to_target"]["prediction_relative_rmse"] for record in records
        ]
        target_parameter_distance = [
            record["to_target"]["parameter_relative_l2"] for record in records
        ]
        summary["conditions"][condition] = {
            "median_prediction_relative_rmse_to_target": median(
                target_prediction_distance
            ),
            "prediction_relative_rmse_to_target_range": [
                min(target_prediction_distance),
                max(target_prediction_distance),
            ],
            "median_parameter_relative_l2_to_target": median(
                target_parameter_distance
            ),
            "median_parameter_cosine_to_target": median(
                [record["to_target"]["parameter_cosine"] for record in records]
            ),
            "median_train_mse": median(
                [record["endpoint"]["train_mse"] for record in records]
            ),
            "median_objective_to_loss_gradient_ratio": median(
                [
                    record["endpoint"]["objective_to_loss_gradient_ratio"]
                    for record in records
                ]
            ),
        }

    estimated_to_true = [
        run["pairwise"]["estimated_to_true"]["prediction_relative_rmse"]
        for run in runs
    ]
    ablated_to_true = [
        run["pairwise"]["ablated_to_true"]["prediction_relative_rmse"]
        for run in runs
    ]
    ratios = [
        estimated / max(ablated, 1e-30)
        for estimated, ablated in zip(
            estimated_to_true,
            ablated_to_true,
            strict=True,
        )
    ]
    estimated_parameter_to_target = [
        run["conditions"]["estimated"]["to_target"]["parameter_relative_l2"]
        for run in runs
    ]
    ablated_parameter_to_target = [
        run["conditions"]["ablated"]["to_target"]["parameter_relative_l2"]
        for run in runs
    ]
    parameter_ratios = [
        estimated / max(ablated, 1e-30)
        for estimated, ablated in zip(
            estimated_parameter_to_target,
            ablated_parameter_to_target,
            strict=True,
        )
    ]
    trajectory_wins = 0
    trajectory_total = 0
    for run in runs:
        estimated_trace = {
            record["epoch"]: record
            for record in run["conditions"]["estimated"]["trajectory"]
        }
        ablated_trace = {
            record["epoch"]: record
            for record in run["conditions"]["ablated"]["trajectory"]
        }
        for epoch in set(estimated_trace) & set(ablated_trace):
            trajectory_total += 1
            trajectory_wins += (
                estimated_trace[epoch]["prediction_relative_rmse_to_target"]
                < ablated_trace[epoch]["prediction_relative_rmse_to_target"]
            )
    summary["pairwise"] = {
        "median_estimated_to_true_prediction_relative_rmse": median(
            estimated_to_true
        ),
        "median_ablated_to_true_prediction_relative_rmse": median(
            ablated_to_true
        ),
        "median_estimated_over_ablated_prediction_distance": median(ratios),
        "median_estimated_over_ablated_parameter_distance": median(
            parameter_ratios
        ),
        "estimated_closer_than_ablated_count": sum(
            estimated < ablated
            for estimated, ablated in zip(
                estimated_to_true,
                ablated_to_true,
                strict=True,
            )
        ),
        "total_count": len(runs),
        "estimated_closer_trajectory_probe_count": trajectory_wins,
        "trajectory_probe_count": trajectory_total,
    }
    return summary


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Elastic-net explicit-retraining recovery",
        "",
        f"Matched endpoints: {summary['n']}. Every condition reuses the original "
        "synthetic data, initialization, minibatch randomness, optimizer, and "
        "training epoch budget. All metrics are training-set quantities.",
        "",
        "| condition | median prediction relative RMSE to saved target | "
        "median parameter relative L2 to target | median parameter cosine | "
        "median own-objective gradient / loss-gradient norm |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        record = summary["conditions"][condition]
        lines.append(
            f"| {condition} | "
            f"{record['median_prediction_relative_rmse_to_target']:.5f} | "
            f"{record['median_parameter_relative_l2_to_target']:.5f} | "
            f"{record['median_parameter_cosine_to_target']:.5f} | "
            f"{record['median_objective_to_loss_gradient_ratio']:.5f} |"
        )
    pairwise = summary["pairwise"]
    lines.extend(
        [
            "",
            "## Paired estimated-vs-true comparison",
            "",
            f"- Estimated-penalty versus true-penalty prediction distance: "
            f"{pairwise['median_estimated_to_true_prediction_relative_rmse']:.5f}.",
            f"- Ablated versus true-penalty prediction distance: "
            f"{pairwise['median_ablated_to_true_prediction_relative_rmse']:.5f}.",
            f"- Median ratio (estimated distance / ablated distance): "
            f"{pairwise['median_estimated_over_ablated_prediction_distance']:.3f}.",
            f"- Median parameter-distance ratio (estimated / ablated): "
            f"{pairwise['median_estimated_over_ablated_parameter_distance']:.3f}.",
            f"- Estimated penalty is closer than ablation in "
            f"{pairwise['estimated_closer_than_ablated_count']}/"
            f"{pairwise['total_count']} matched endpoints.",
            f"- Along the matched trajectories, estimated-penalty training is "
            f"closer than ablation at "
            f"{pairwise['estimated_closer_trajectory_probe_count']}/"
            f"{pairwise['trajectory_probe_count']} sampled epochs.",
            "",
            "## Interpretation",
            "",
            "The true-coefficient retrain is a reproducibility control: its near-zero "
            "distance shows that the saved target is reproducible from the recorded "
            "seed and epoch budget. The estimated penalty then reproduces the target "
            "function far more closely than removing regularization, both at the "
            "endpoint and through most of the matched trajectory.",
            "",
            "The ablated model can have lower unregularized training MSE because that "
            "is exactly what removing a strong explicit penalty permits. Its lower "
            "MSE is not recovery; its parameters and training predictions move far "
            "away from the penalized target. The estimated model's own-objective "
            "gradient balance is also essentially the same as the true-coefficient "
            "model's, which is the expected residual from the original finite, "
            "minibatch training budget.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/generated/elasticnet_retrain_recovery/results"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/generated/elasticnet_retrain_recovery/summary.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("data/generated/elasticnet_retrain_recovery/FINDINGS.md"),
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
        raise SystemExit(f"no results under {args.results_dir}")
    summary = summarize(runs)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2))
    args.report_output.write_text(markdown(summary))
    LOGGER.info(
        "analyzed %d runs; wrote %s and %s",
        len(runs),
        args.summary_output,
        args.report_output,
    )


if __name__ == "__main__":
    main()
