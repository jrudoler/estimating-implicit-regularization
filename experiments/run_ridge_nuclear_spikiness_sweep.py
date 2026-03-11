#!/usr/bin/env python3
"""Sweep ridge-vs-nuclear identifiability across a continuous teacher spectrum."""

from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from function_class_identifiability import (  # noqa: E402
    build_spectrum_values,
    participation_ratio,
    positive_condition_number,
)


LOGGER = logging.getLogger(__name__)

FIT_RE = re.compile(
    r"Fit \| pair=(?P<pair>[a-z_]+__[a-z_]+) teacher_spectrum=(?P<teacher_spectrum>[a-z_]+) "
    r"train_mse=(?P<train_mse>[-+0-9.eE]+) val_mse=(?P<val_mse>[-+0-9.eE]+) "
    r"test_mse=(?P<test_mse>[-+0-9.eE]+) train_loss=(?P<train_loss>[-+0-9.eE]+) "
    r"val_loss=(?P<val_loss>[-+0-9.eE]+) test_loss=(?P<test_loss>[-+0-9.eE]+)"
)

GEOMETRY_RE = re.compile(
    r"Geometry \| pair=(?P<pair>[a-z_]+__[a-z_]+) cosine=(?P<cosine>[-+0-9.eE]+) "
    r"input_spectrum=(?P<input_spectrum>[a-z_]+) teacher_spectrum=(?P<teacher_spectrum>[a-z_]+) "
    r"input_rank=(?P<input_rank>\d+) teacher_rank=(?P<teacher_rank>\d+)"
)

RESULT_RE = re.compile(
    r"Result \| pair=(?P<pair>[a-z_]+__[a-z_]+) estimator_mode=(?P<estimator_mode>[a-z_]+) "
    r"gt_mean_rel_error=(?P<gt_mean_rel_error>[-+0-9.eE]+) "
    r"max_rel_error=(?P<max_rel_error>[-+0-9.eE]+) "
    r"ols_mean_rel_error=(?P<ols_mean_rel_error>[-+0-9.eE]+) "
    r"noisy_ols_mean_rel_error=(?P<noisy_ols_mean_rel_error>[-+0-9.eE]+|nan) "
    r"gt_relative_residual=(?P<gt_relative_residual>[-+0-9.eE]+) "
    r"ols_relative_residual=(?P<ols_relative_residual>[-+0-9.eE]+) "
    r"noisy_ols_relative_residual=(?P<noisy_ols_relative_residual>[-+0-9.eE]+|nan) "
    r"selected_max_abs_cos=(?P<selected_max_abs_cos>[-+0-9.eE]+) "
    r"selected_cond=(?P<selected_cond>[-+0-9.eE]+)"
)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_csv_floats(raw_value: str) -> list[float]:
    return [float(part.strip()) for part in raw_value.split(",") if part.strip()]


def parse_csv_ints(raw_value: str) -> list[int]:
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-spectrum-decays", type=str, default="0.0,0.1,0.2,0.35,0.5,0.75,1.0,1.25,1.5,2.0,2.5")
    parser.add_argument("--seeds", type=str, default="42,123,456,789")
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--bias-lr", type=float, default=0.05)
    parser.add_argument("--bias-max-epochs", type=int, default=800)
    parser.add_argument("--bias-patience", type=int, default=100)
    parser.add_argument("--recovery-noise-std", type=float, default=1e-3)
    parser.add_argument("--recovery-noise-draws", type=int, default=32)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("logs/phase2/ridge_nuclear_spikiness_sweep.csv"),
    )
    return parser.parse_args()


def summarize_teacher_spectrum(
    output_dim: int,
    input_dim: int,
    teacher_rank: int,
    decay: float,
) -> dict[str, float]:
    max_rank = min(output_dim, input_dim)
    active_rank = max(1, min(teacher_rank, max_rank))
    singular_values = build_spectrum_values(
        size=max_rank,
        active_rank=active_rank,
        family="power_law",
        decay=decay,
        target_sum=float(active_rank),
    )
    effective_rank = participation_ratio(singular_values)
    spikiness = 1.0 - (effective_rank / float(active_rank))
    entropy = singular_values[:active_rank]
    entropy = entropy / max(float(entropy.sum().item()), 1e-12)
    spectral_entropy = float(-(entropy * torch.log(entropy.clamp(min=1e-12))).sum().item())
    normalized_entropy = spectral_entropy / max(float(torch.log(torch.tensor(float(active_rank))).item()), 1e-12)
    return {
        "teacher_effective_rank": effective_rank,
        "teacher_condition_number": positive_condition_number(singular_values),
        "spectral_spikiness": spikiness,
        "normalized_spectral_entropy": normalized_entropy,
    }


def parse_run_output(output_text: str) -> dict[str, str]:
    fit_match = FIT_RE.search(output_text)
    geometry_match = GEOMETRY_RE.search(output_text)
    result_match = RESULT_RE.search(output_text)
    if not fit_match or not geometry_match or not result_match:
        raise RuntimeError("Failed to parse fit/geometry/result lines from run output.")

    metrics = result_match.groupdict()
    metrics["train_mse"] = fit_match.group("train_mse")
    metrics["val_mse"] = fit_match.group("val_mse")
    metrics["test_mse"] = fit_match.group("test_mse")
    metrics["train_loss"] = fit_match.group("train_loss")
    metrics["val_loss"] = fit_match.group("val_loss")
    metrics["test_loss"] = fit_match.group("test_loss")
    metrics["pair_cosine"] = geometry_match.group("cosine")
    return metrics


def write_csv(rows: Sequence[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    configure_logging()
    args = parse_args()
    decays = parse_csv_floats(args.teacher_spectrum_decays)
    seeds = parse_csv_ints(args.seeds)

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    teacher_rank = min(args.input_dim, args.output_dim)
    total_runs = len(decays) * len(seeds)
    rows: list[dict[str, str]] = []
    run_index = 0

    for decay in decays:
        spectrum_stats = summarize_teacher_spectrum(
            output_dim=args.output_dim,
            input_dim=args.input_dim,
            teacher_rank=teacher_rank,
            decay=decay,
        )
        for seed in seeds:
            run_index += 1
            LOGGER.info(
                "[%d/%d] decay=%.3f spikiness=%.4f seed=%d",
                run_index,
                total_runs,
                decay,
                spectrum_stats["spectral_spikiness"],
                seed,
            )
            command = [
                sys.executable,
                "experiments/matrix_spectrum_identifiability.py",
                "--gt-bias-types",
                "ridge,nuclear_norm",
                "--estimation-bias-types",
                "ridge,nuclear_norm",
                "--candidate-bias-types",
                "ridge,nuclear_norm,stable_rank,spectral_entropy",
                "--teacher-spectrum",
                "power_law",
                "--teacher-spectrum-decay",
                str(decay),
                "--teacher-rank",
                str(teacher_rank),
                "--input-spectrum",
                "identity",
                "--input-rank",
                str(args.input_dim),
                "--auto-balance-gt-lambdas",
                "--n-samples",
                str(args.n_samples),
                "--input-dim",
                str(args.input_dim),
                "--output-dim",
                str(args.output_dim),
                "--batch-size",
                str(min(args.batch_size, args.n_samples)),
                "--lr",
                str(args.lr),
                "--max-epochs",
                str(args.max_epochs),
                "--patience",
                str(args.patience),
                "--bias-lr",
                str(args.bias_lr),
                "--bias-max-epochs",
                str(args.bias_max_epochs),
                "--bias-patience",
                str(args.bias_patience),
                "--recovery-noise-std",
                str(args.recovery_noise_std),
                "--recovery-noise-draws",
                str(args.recovery_noise_draws),
                "--seed",
                str(seed),
            ]
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            combined_output = f"{completed.stdout}\n{completed.stderr}"
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Run failed for decay={decay} seed={seed} with return code {completed.returncode}.\n"
                    f"{combined_output}"
                )
            metrics = parse_run_output(combined_output)
            row: dict[str, str] = {
                "seed": str(seed),
                "teacher_spectrum": "power_law",
                "teacher_spectrum_decay": f"{decay:.6f}",
                "return_code": str(completed.returncode),
                "teacher_rank": str(teacher_rank),
            }
            row.update({key: f"{value:.6f}" for key, value in spectrum_stats.items()})
            row.update(metrics)
            rows.append(row)

    write_csv(rows, args.output_csv)
    LOGGER.info("Wrote %s", args.output_csv)


if __name__ == "__main__":
    main()
