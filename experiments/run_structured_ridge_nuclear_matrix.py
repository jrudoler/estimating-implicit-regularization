#!/usr/bin/env python3
"""Run a controlled structured synthetic matrix for ridge-vs-nuclear identifiability."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Dict, List, Sequence


RESULT_RE = re.compile(
    r"Result \| function_class=(?P<function_class>[a-z_]+) "
    r"selection_mode=(?P<selection_mode>[a-z_]+) "
    r"estimator_mode=(?P<estimator_mode>[a-z_]+) "
    r"mean_rel_error=(?P<mean_rel_error>[-+0-9.eE]+) "
    r"gt_mean_rel_error=(?P<gt_mean_rel_error>[-+0-9.eE]+) "
    r"max_rel_error=(?P<max_rel_error>[-+0-9.eE]+) "
    r"support_f1=(?P<support_f1>[-+0-9.eE]+) "
    r"selected_max_abs_cos=(?P<selected_max_abs_cos>[-+0-9.eE]+) "
    r"selected_cond=(?P<selected_cond>[-+0-9.eE]+)"
)

FIT_RE = re.compile(
    r"Fit \| function_class=(?P<function_class>[a-z_]+) "
    r"data_mode=(?P<data_mode>[a-z_]+) "
    r"test_mse=(?P<test_mse>[-+0-9.eE]+) "
    r"test_loss=(?P<test_loss>[-+0-9.eE]+)"
)


def parse_csv_ints(raw_value: str) -> List[int]:
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="42,123")
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--bias-lr", type=float, default=0.05)
    parser.add_argument("--bias-max-epochs", type=int, default=600)
    parser.add_argument("--bias-patience", type=int, default=60)
    parser.add_argument("--gt-lambdas", type=str, default="0.05,0.05")
    parser.add_argument("--gt-lambda-scale", type=float, default=0.05)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("logs/phase2/structured_ridge_nuclear_matrix.csv"),
    )
    parser.add_argument(
        "--output-summary-md",
        type=Path,
        default=Path("logs/phase2/structured_ridge_nuclear_summary.md"),
    )
    return parser.parse_args()


def parse_run_output(output_text: str) -> Dict[str, str]:
    result_match = RESULT_RE.search(output_text)
    fit_match = FIT_RE.search(output_text)
    if not result_match or not fit_match:
        raise RuntimeError("Failed to parse fit/result lines from run output.")

    metrics = result_match.groupdict()
    metrics["data_mode"] = fit_match.group("data_mode")
    metrics["test_mse"] = fit_match.group("test_mse")
    metrics["test_loss"] = fit_match.group("test_loss")
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
    if not rows:
        output_path.write_text("# Structured Ridge/Nuclear Summary\n\nNo rows collected.\n", encoding="utf-8")
        return

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["regime_name"], []).append(row)

    lines = ["# Structured Ridge/Nuclear Summary", ""]
    lines.append(
        "| Regime | Expected Geometry | Mean(test_mse) | Mean(gt_mean_rel_error) | Mean(selected_max_abs_cos) | Mean(selected_cond) |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for regime_name in sorted(grouped):
        bucket = grouped[regime_name]
        expected_geometry = bucket[0]["expected_geometry"]
        mean_test_mse = sum(float(item["test_mse"]) for item in bucket) / len(bucket)
        mean_gt_error = sum(float(item["gt_mean_rel_error"]) for item in bucket) / len(bucket)
        mean_max_cos = sum(float(item["selected_max_abs_cos"]) for item in bucket) / len(bucket)
        mean_cond = sum(float(item["selected_cond"]) for item in bucket) / len(bucket)
        lines.append(
            f"| {regime_name} | {expected_geometry} | {mean_test_mse:.4f} | {mean_gt_error:.4f} | {mean_max_cos:.4f} | {mean_cond:.4f} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_command(
    args: argparse.Namespace,
    regime: Dict[str, str],
    seed: int,
) -> List[str]:
    return [
        sys.executable,
        "experiments/function_class_identifiability.py",
        "--function-class",
        "teacher_relu",
        "--data-mode",
        "structured",
        "--selection-mode",
        "fixed",
        "--estimator-mode",
        "normalized",
        "--gt-bias-types",
        "ridge,nuclear_norm",
        "--estimation-bias-types",
        "ridge,nuclear_norm",
        "--gt-lambdas",
        args.gt_lambdas,
        "--auto-balance-gt-lambdas",
        "--gt-lambda-scale",
        str(args.gt_lambda_scale),
        "--candidate-bias-types",
        "ridge,nuclear_norm",
        "--n-samples",
        str(args.n_samples),
        "--input-dim",
        str(args.input_dim),
        "--noise-std",
        str(args.noise_std),
        "--depth",
        "1",
        "--width",
        str(args.width),
        "--teacher-hidden-dim",
        str(args.width),
        "--teacher-activation",
        "relu",
        "--input-spectrum",
        regime["input_spectrum"],
        "--input-rank",
        regime["input_rank"],
        "--input-spectrum-decay",
        regime["input_spectrum_decay"],
        "--teacher-spectrum",
        regime["teacher_spectrum"],
        "--teacher-rank",
        regime["teacher_rank"],
        "--teacher-spectrum-decay",
        regime["teacher_spectrum_decay"],
        "--batch-size",
        str(args.batch_size),
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
        "--seed",
        str(seed),
    ]


def main() -> None:
    args = parse_args()
    seeds = parse_csv_ints(args.seeds)
    full_rank = min(args.input_dim, args.width)
    low_rank = max(4, full_rank // 2)

    regimes: List[Dict[str, str]] = [
        {
            "regime_name": "uniform_full_rank",
            "expected_geometry": "high_collinear",
            "input_spectrum": "uniform",
            "input_rank": str(args.input_dim),
            "input_spectrum_decay": "1.0",
            "teacher_spectrum": "uniform",
            "teacher_rank": str(full_rank),
            "teacher_spectrum_decay": "1.0",
        },
        {
            "regime_name": "uniform_low_rank_input",
            "expected_geometry": "high_collinear",
            "input_spectrum": "uniform",
            "input_rank": str(low_rank),
            "input_spectrum_decay": "1.0",
            "teacher_spectrum": "uniform",
            "teacher_rank": str(full_rank),
            "teacher_spectrum_decay": "1.0",
        },
        {
            "regime_name": "spiked_teacher_full_rank",
            "expected_geometry": "lower_collinear",
            "input_spectrum": "uniform",
            "input_rank": str(args.input_dim),
            "input_spectrum_decay": "1.0",
            "teacher_spectrum": "spiked",
            "teacher_rank": str(full_rank),
            "teacher_spectrum_decay": "2.0",
        },
        {
            "regime_name": "power_law_teacher_full_rank",
            "expected_geometry": "lower_collinear",
            "input_spectrum": "uniform",
            "input_rank": str(args.input_dim),
            "input_spectrum_decay": "1.0",
            "teacher_spectrum": "power_law",
            "teacher_rank": str(full_rank),
            "teacher_spectrum_decay": "1.5",
        },
        {
            "regime_name": "spiked_both_low_rank",
            "expected_geometry": "mixed",
            "input_spectrum": "spiked",
            "input_rank": str(low_rank),
            "input_spectrum_decay": "2.0",
            "teacher_spectrum": "spiked",
            "teacher_rank": str(low_rank),
            "teacher_spectrum_decay": "2.0",
        },
    ]

    rows: List[Dict[str, str]] = []
    total_runs = len(regimes) * len(seeds)
    run_counter = 0
    started_at = time.time()

    for regime in regimes:
        for seed in seeds:
            run_counter += 1
            command = build_command(args=args, regime=regime, seed=seed)
            print(
                f"[{run_counter}/{total_runs}] regime={regime['regime_name']} "
                f"seed={seed} input={regime['input_spectrum']}/{regime['input_rank']} "
                f"teacher={regime['teacher_spectrum']}/{regime['teacher_rank']}"
            )
            env = os.environ.copy()
            env["WANDB_MODE"] = "disabled"
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            output_text = completed.stdout + "\n" + completed.stderr

            row: Dict[str, str] = {
                "regime_name": regime["regime_name"],
                "expected_geometry": regime["expected_geometry"],
                "seed": str(seed),
                "return_code": str(completed.returncode),
                **regime,
            }
            if completed.returncode == 0:
                metrics = parse_run_output(output_text)
                row.update(metrics)
                print(
                    "  "
                    + f"test_mse={float(metrics['test_mse']):.4f} "
                    + f"gt_mean_rel_error={float(metrics['gt_mean_rel_error']):.4f} "
                    + f"selected_max_abs_cos={float(metrics['selected_max_abs_cos']):.4f}"
                )
            else:
                row["error_json"] = json.dumps(
                    {"stdout_stderr_tail": output_text[-4000:]},
                    sort_keys=True,
                )
                print(f"  failed with return code {completed.returncode}")
            rows.append(row)

    write_csv(rows, args.output_csv)
    write_summary(rows, args.output_summary_md)
    elapsed = time.time() - started_at
    print(
        f"Wrote {len(rows)} rows to {args.output_csv} and summary to {args.output_summary_md} in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
