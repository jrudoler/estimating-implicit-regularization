#!/usr/bin/env python3
"""Sweep stacked resampling recovery across matrix-spectrum spikiness."""

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

RESAMPLE_RE = re.compile(
    r"Resample \| pair=(?P<pair>[a-z_]+__[a-z_]+) mode=(?P<mode>[a-z_]+) "
    r"full_cos=(?P<full_cos>[-+0-9.eE]+) full_cond=(?P<full_cond>[-+0-9.eE]+) "
    r"single_mean_rel_error=(?P<single_mean_rel_error>[-+0-9.eE]+) "
    r"aggregate_mean_rel_error=(?P<aggregate_mean_rel_error>[-+0-9.eE]+) "
    r"stacked_mean_rel_error=(?P<stacked_mean_rel_error>[-+0-9.eE]+) "
    r"stacked_cond=(?P<stacked_cond>[-+0-9.eE]+) "
    r"stacked_relative_residual=(?P<stacked_relative_residual>[-+0-9.eE]+) "
    r"replicate_cos_mean=(?P<replicate_cos_mean>[-+0-9.eE]+)"
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
    parser.add_argument("--teacher-spectrum-decays", type=str, default="0.0,0.1,0.2,0.35,0.5,0.75,1.0,1.25,1.5")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--resample-modes", type=str, default="bootstrap")
    parser.add_argument("--n-samples", type=int, default=4096)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--n-replicates", type=int, default=16)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("logs/phase2/matrix_resampling_spikiness_sweep.csv"),
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
    resample_match = RESAMPLE_RE.search(output_text)
    if not fit_match or not resample_match:
        raise RuntimeError("Failed to parse fit/resample lines from run output.")
    metrics = resample_match.groupdict()
    metrics["train_mse"] = fit_match.group("train_mse")
    metrics["val_mse"] = fit_match.group("val_mse")
    metrics["test_mse"] = fit_match.group("test_mse")
    metrics["train_loss"] = fit_match.group("train_loss")
    metrics["val_loss"] = fit_match.group("val_loss")
    metrics["test_loss"] = fit_match.group("test_loss")
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
    resample_modes = [part.strip() for part in args.resample_modes.split(",") if part.strip()]

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    teacher_rank = min(args.input_dim, args.output_dim)
    total_runs = len(decays) * len(seeds) * len(resample_modes)
    rows: list[dict[str, str]] = []
    run_index = 0

    for decay in decays:
        spectrum_stats = summarize_teacher_spectrum(
            output_dim=args.output_dim,
            input_dim=args.input_dim,
            teacher_rank=teacher_rank,
            decay=decay,
        )
        for resample_mode in resample_modes:
            for seed in seeds:
                run_index += 1
                LOGGER.info(
                    "[%d/%d] mode=%s decay=%.3f spikiness=%.4f seed=%d",
                    run_index,
                    total_runs,
                    resample_mode,
                    decay,
                    spectrum_stats["spectral_spikiness"],
                    seed,
                )
                command = [
                    sys.executable,
                    "experiments/matrix_spectrum_resampling.py",
                    "--gt-bias-types",
                    "ridge,nuclear_norm",
                    "--gt-lambdas",
                    "0.05,0.05",
                    "--auto-balance-gt-lambdas",
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
                    "--resample-mode",
                    resample_mode,
                    "--n-replicates",
                    str(args.n_replicates),
                    "--sample-fraction",
                    str(args.sample_fraction),
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
                        f"Run failed for mode={resample_mode} decay={decay} seed={seed} "
                        f"with return code {completed.returncode}.\n{combined_output}"
                    )
                metrics = parse_run_output(combined_output)
                row: dict[str, str] = {
                    "seed": str(seed),
                    "resample_mode": resample_mode,
                    "teacher_spectrum": "power_law",
                    "teacher_spectrum_decay": f"{decay:.6f}",
                    "teacher_rank": str(teacher_rank),
                    "n_replicates": str(args.n_replicates),
                    "sample_fraction": f"{args.sample_fraction:.6f}",
                }
                row.update({key: f"{value:.6f}" for key, value in spectrum_stats.items()})
                row.update(metrics)
                rows.append(row)

    write_csv(rows, args.output_csv)
    LOGGER.info("Wrote %d rows to %s", len(rows), args.output_csv)


if __name__ == "__main__":
    main()
