#!/usr/bin/env python3
"""Run fixed pairwise matrix-spectrum identifiability checks."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, List, Sequence


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


def parse_csv_ints(raw_value: str) -> List[int]:
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="42,123,456")
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
        default=Path("logs/phase2/matrix_spectrum_pairs.csv"),
    )
    parser.add_argument(
        "--output-summary-md",
        type=Path,
        default=Path("logs/phase2/matrix_spectrum_pairs_summary.md"),
    )
    return parser.parse_args()


def parse_run_output(output_text: str) -> Dict[str, str]:
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


def write_csv(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(f"{row['pair']}|{row['teacher_spectrum']}", []).append(row)

    lines = ["# Matrix Spectrum Pair Summary", ""]
    lines.append("| Pair | Teacher Spectrum | Mean(pair_cosine) | Mean(gt_mean_rel_error) | Mean(ols_mean_rel_error) | Mean(noisy_ols_mean_rel_error) | Mean(gt_relative_residual) | Mean(test_mse) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in sorted(grouped):
        pair, teacher_spectrum = key.split("|", 1)
        bucket = grouped[key]
        mean_cos = sum(float(item["pair_cosine"]) for item in bucket) / len(bucket)
        mean_err = sum(float(item["gt_mean_rel_error"]) for item in bucket) / len(bucket)
        mean_ols_err = sum(float(item["ols_mean_rel_error"]) for item in bucket) / len(bucket)
        mean_noisy_ols_err = sum(float(item["noisy_ols_mean_rel_error"]) for item in bucket) / len(bucket)
        mean_gt_resid = sum(float(item["gt_relative_residual"]) for item in bucket) / len(bucket)
        mean_test_mse = sum(float(item["test_mse"]) for item in bucket) / len(bucket)
        lines.append(
            f"| {pair} | {teacher_spectrum} | {mean_cos:.4f} | {mean_err:.4f} | {mean_ols_err:.4f} | {mean_noisy_ols_err:.4f} | {mean_gt_resid:.4f} | {mean_test_mse:.4f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = parse_csv_ints(args.seeds)

    pair_configs = [
        ("ridge,nuclear_norm", "identity", "1.0"),
        ("ridge,nuclear_norm", "spiked", "2.0"),
        ("ridge,nuclear_norm", "power_law", "1.5"),
        ("ridge,stable_rank", "identity", "1.0"),
        ("ridge,stable_rank", "spiked", "2.0"),
        ("ridge,stable_rank", "power_law", "1.5"),
        ("ridge,spectral_entropy", "identity", "1.0"),
        ("ridge,spectral_entropy", "spiked", "2.0"),
        ("ridge,spectral_entropy", "power_law", "1.5"),
    ]

    rows: List[Dict[str, str]] = []
    total_runs = len(pair_configs) * len(seeds)
    run_counter = 0
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"

    for pair_csv, teacher_spectrum, teacher_decay in pair_configs:
        for seed in seeds:
            run_counter += 1
            print(
                f"[{run_counter}/{total_runs}] pair={pair_csv} teacher={teacher_spectrum} seed={seed}",
                flush=True,
            )
            command = [
                sys.executable,
                "experiments/matrix_spectrum_identifiability.py",
                "--gt-bias-types",
                pair_csv,
                "--estimation-bias-types",
                pair_csv,
                "--candidate-bias-types",
                "ridge,nuclear_norm,stable_rank,spectral_entropy",
                "--teacher-spectrum",
                teacher_spectrum,
                "--teacher-spectrum-decay",
                teacher_decay,
                "--teacher-rank",
                str(min(args.input_dim, args.output_dim)),
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
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            output_text = completed.stdout + "\n" + completed.stderr
            row: Dict[str, str] = {
                "pair": pair_csv.replace(",", "__"),
                "teacher_spectrum": teacher_spectrum,
                "seed": str(seed),
                "return_code": str(completed.returncode),
            }
            if completed.returncode == 0:
                metrics = parse_run_output(output_text)
                row.update(metrics)
                print(
                    f"  cosine={float(metrics['pair_cosine']):.4f} "
                    f"gt_err={float(metrics['gt_mean_rel_error']):.4f} "
                    f"test_mse={float(metrics['test_mse']):.4f}",
                    flush=True,
                )
            else:
                row["error_json"] = json.dumps(
                    {"stdout_stderr_tail": output_text[-4000:]},
                    sort_keys=True,
                )
                print(f"  failed with return code {completed.returncode}", flush=True)
            rows.append(row)

    write_csv(rows, args.output_csv)
    write_summary(rows, args.output_summary_md)
    print(f"Wrote {args.output_csv} and {args.output_summary_md}", flush=True)


if __name__ == "__main__":
    main()
