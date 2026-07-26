"""Aggregate the multi-seed SWA effective-regularizer experiment."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def mean_se(values: list[float | None]) -> tuple[float, float]:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    if array.size == 1:
        return float(array[0]), 0.0
    return float(array.mean()), float(array.std(ddof=1) / np.sqrt(array.size))


def format_mean_se(values: list[float | None], digits: int = 3) -> str:
    mean, se = mean_se(values)
    return f"{mean:.{digits}f} ± {se:.{digits}f}"


def load_runs(results_dir: Path) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            runs.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            LOGGER.warning("Skipping unreadable result %s", path)
    if not runs:
        raise SystemExit(f"No JSON results found under {results_dir}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("data/generated/swa_effective_regularizer/results"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated/swa_effective_regularizer/summary.json"),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated seed allowlist.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    runs = load_runs(args.results_dir)
    if args.seeds:
        seed_allowlist = {int(seed) for seed in args.seeds.split(",")}
        runs = [run for run in runs if run["config"]["seed"] in seed_allowlist]
        if not runs:
            raise SystemExit("No runs remain after applying --seeds")

    endpoint_names = list(runs[0]["endpoints"])
    family_names = list(runs[0]["endpoints"]["swa_average"]["closed_form"])
    displacement_names = list(runs[0]["displacements"])

    summary: dict[str, Any] = {
        "n_runs": len(runs),
        "seeds": [run["config"]["seed"] for run in runs],
        "endpoints": {},
        "displacements": {},
        "explicit_recovery": {},
    }
    print(f"Loaded {len(runs)} runs")
    print()
    print("TRAINING ENDPOINTS")
    print(
        f"{'endpoint':>24} {'train loss':>18} {'train acc':>18} "
        f"{'||grad L||/||theta||':>24}"
    )
    for endpoint in endpoint_names:
        losses = [run["endpoints"][endpoint]["train"]["loss"] for run in runs]
        accuracies = [
            run["endpoints"][endpoint]["train"]["accuracy"] for run in runs
        ]
        relative_gradients = [
            run["endpoints"][endpoint]["stationarity"]["relative_grad_norm"]
            for run in runs
        ]
        summary["endpoints"][endpoint] = {
            "train_loss": dict(zip(("mean", "se"), mean_se(losses))),
            "train_accuracy": dict(zip(("mean", "se"), mean_se(accuracies))),
            "relative_grad_norm": dict(
                zip(("mean", "se"), mean_se(relative_gradients))
            ),
            "closed_form": {},
        }
        for family in family_names:
            records = [
                run["endpoints"][endpoint]["closed_form"][family] for run in runs
            ]
            summary["endpoints"][endpoint]["closed_form"][family] = {
                statistic: dict(
                    zip(
                        ("mean", "se"),
                        mean_se([record[statistic] for record in records]),
                    )
                )
                for statistic in (
                    "scale_star",
                    "grad_cosine",
                    "projection_r2_positive",
                    "residual_ratio_clamped",
                )
            }
        print(
            f"{endpoint:>24} {format_mean_se(losses, 5):>18} "
            f"{format_mean_se(accuracies, 4):>18} "
            f"{format_mean_se(relative_gradients, 6):>24}"
        )

    print()
    if "explicit_recovery" in runs[0]:
        print("EXPLICIT CHECKPOINT-CENTERED RETRAINING")
        for metric in (
            "parameter_distance_over_swa_effect",
            "burnin_displacement_cosine",
            "prediction_agreement",
            "logit_mse",
        ):
            values = [run["explicit_recovery"][metric] for run in runs]
            summary["explicit_recovery"][metric] = dict(
                zip(("mean", "se"), mean_se(values))
            )
            print(f"{metric:>40}: {format_mean_se(values, 5)}")
        swa_effect_norms = [
            run["explicit_recovery"]["parameter_distance"]
            / run["explicit_recovery"]["parameter_distance_over_swa_effect"]
            for run in runs
        ]
        for metric, displacement_name in (
            ("control_distance_over_swa_effect", "full_swa"),
            ("high_lr_last_distance_over_swa_effect", "averaging_only"),
        ):
            values = [
                run["displacements"][displacement_name]["displacement_norm"]
                / effect_norm
                for run, effect_norm in zip(runs, swa_effect_norms)
            ]
            summary["explicit_recovery"][metric] = dict(
                zip(("mean", "se"), mean_se(values))
            )
            print(f"{metric:>40}: {format_mean_se(values, 5)}")
        print()

    print()
    print("KEY ENDPOINT FITS")
    print(
        f"{'endpoint':>24} {'ridge R2+':>18} {'centered R2+':>18} "
        f"{'centered cosine':>18}"
    )
    for endpoint in endpoint_names:
        ridge = [
            run["endpoints"][endpoint]["closed_form"]["ridge"][
                "projection_r2_positive"
            ]
            for run in runs
        ]
        centered_records = [
            run["endpoints"][endpoint]["closed_form"]["burnin_centered_ridge"]
            for run in runs
        ]
        centered_r2 = [
            record["projection_r2_positive"] for record in centered_records
        ]
        centered_cosine = [record["grad_cosine"] for record in centered_records]
        print(
            f"{endpoint:>24} {format_mean_se(ridge):>18} "
            f"{format_mean_se(centered_r2):>18} "
            f"{format_mean_se(centered_cosine):>18}"
        )

    print()
    print("ENDPOINT GRADIENT MATCHING AT THE SWA AVERAGE")
    print(
        f"{'family':>24} {'cosine':>18} {'positive R2':>18} "
        f"{'coefficient':>18}"
    )
    for family in family_names:
        records = [
            run["endpoints"]["swa_average"]["closed_form"][family] for run in runs
        ]
        cosines = [record["grad_cosine"] for record in records]
        positive_r2 = [record["projection_r2_positive"] for record in records]
        coefficients = [record["scale_star"] for record in records]
        print(
            f"{family:>24} {format_mean_se(cosines):>18} "
            f"{format_mean_se(positive_r2):>18} "
            f"{format_mean_se(coefficients, 6):>18}"
        )

    print()
    print("ABLATION DISPLACEMENT MATCHING")
    for displacement in displacement_names:
        print()
        print(displacement)
        print(
            f"{'family':>24} {'cos(-grad P, delta)':>22} "
            f"{'positive R2':>18}"
        )
        displacement_summary: dict[str, Any] = {}
        for family in family_names:
            records = [
                run["displacements"][displacement]["candidates"][family]
                for run in runs
            ]
            cosines = [record["grad_cosine"] for record in records]
            positive_r2 = [record["projection_r2_positive"] for record in records]
            displacement_summary[family] = {
                "cosine": dict(zip(("mean", "se"), mean_se(cosines))),
                "projection_r2_positive": dict(
                    zip(("mean", "se"), mean_se(positive_r2))
                ),
            }
            print(
                f"{family:>24} {format_mean_se(cosines):>22} "
                f"{format_mean_se(positive_r2):>18}"
            )
        joint_r2 = [
            run["displacements"][displacement]["joint_unconstrained"][
                "projection_r2"
            ]
            for run in runs
        ]
        displacement_summary["joint_unconstrained_projection_r2"] = dict(
            zip(("mean", "se"), mean_se(joint_r2))
        )
        summary["displacements"][displacement] = displacement_summary
        print(f"{'joint unconstrained R2':>24} {format_mean_se(joint_r2):>22}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
