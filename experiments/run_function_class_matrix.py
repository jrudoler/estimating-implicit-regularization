#!/usr/bin/env python3
"""Run a fixed function-class identifiability matrix and save structured results."""

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

BIAS_RE = re.compile(
    r"\|\s+(?P<bias_name>[a-z_]+)\s+\|\s+true=(?P<true>[-+0-9.eE]+)\s+estimated=(?P<estimated>[-+0-9.eE]+)"
)


def parse_csv_list(raw_value: str) -> List[str]:
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def parse_csv_ints(raw_value: str) -> List[int]:
    return [int(part.strip()) for part in raw_value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function-classes", type=str, default="linear,polynomial,sine")
    parser.add_argument("--depths", type=str, default="1,3")
    parser.add_argument("--widths", type=str, default="64,256")
    parser.add_argument("--seeds", type=str, default="42")

    parser.add_argument("--n-samples", type=int, default=1024)
    parser.add_argument("--input-dim", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bias-lr", type=float, default=0.05)
    parser.add_argument("--bias-max-epochs", type=int, default=120)
    parser.add_argument("--bias-patience", type=int, default=20)
    parser.add_argument("--gt-lambdas", type=str, default="0.05,0.05")
    parser.add_argument("--gt-lambda-scale", type=float, default=0.05)

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("logs/phase2/function_class_matrix.csv"),
    )
    parser.add_argument(
        "--output-summary-md",
        type=Path,
        default=Path("logs/phase2/function_class_matrix_summary.md"),
    )
    return parser.parse_args()


def build_command(
    function_class: str,
    depth: int,
    width: int,
    seed: int,
    regime_name: str,
    gt_bias_types: str,
    args: argparse.Namespace,
) -> List[str]:
    return [
        sys.executable,
        "experiments/function_class_identifiability.py",
        "--function-class",
        function_class,
        "--selection-mode",
        "fixed",
        "--estimator-mode",
        "normalized",
        "--gt-bias-types",
        gt_bias_types,
        "--gt-lambdas",
        args.gt_lambdas,
        "--auto-balance-gt-lambdas",
        "--gt-lambda-scale",
        str(args.gt_lambda_scale),
        "--estimation-bias-types",
        gt_bias_types,
        "--n-samples",
        str(args.n_samples),
        "--input-dim",
        str(args.input_dim),
        "--noise-std",
        str(args.noise_std),
        "--depth",
        str(depth),
        "--width",
        str(width),
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


def parse_run_output(output_text: str) -> Dict[str, str]:
    result_match = RESULT_RE.search(output_text)
    if not result_match:
        raise RuntimeError("Failed to parse result line from run output.")

    metrics = result_match.groupdict()
    bias_rows = []
    for bias_match in BIAS_RE.finditer(output_text):
        bias_rows.append(
            {
                "bias_name": bias_match.group("bias_name"),
                "true": float(bias_match.group("true")),
                "estimated": float(bias_match.group("estimated")),
            }
        )
    metrics["bias_estimates_json"] = json.dumps(bias_rows, sort_keys=True)
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
        output_path.write_text("# Function Class Matrix Summary\n\nNo rows collected.\n", encoding="utf-8")
        return

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        key = f"{row['regime_name']}|{row['function_class']}"
        grouped.setdefault(key, []).append(row)

    lines = ["# Function Class Matrix Summary", ""]
    lines.append("| Regime | Function | Mean(gt_mean_rel_error) | Mean(selected_max_abs_cos) | Mean(selected_cond) |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for key in sorted(grouped):
        regime_name, function_class = key.split("|", 1)
        bucket = grouped[key]
        mean_gt_error = sum(float(item["gt_mean_rel_error"]) for item in bucket) / len(bucket)
        mean_max_cos = sum(float(item["selected_max_abs_cos"]) for item in bucket) / len(bucket)
        mean_cond = sum(float(item["selected_cond"]) for item in bucket) / len(bucket)
        lines.append(
            f"| {regime_name} | {function_class} | {mean_gt_error:.4f} | {mean_max_cos:.4f} | {mean_cond:.4f} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    function_classes = parse_csv_list(args.function_classes)
    depths = parse_csv_ints(args.depths)
    widths = parse_csv_ints(args.widths)
    seeds = parse_csv_ints(args.seeds)

    regimes = [
        ("high_collinear", "ridge,nuclear_norm"),
        ("low_noncollinear", "ridge,weight_coherence"),
    ]

    rows: List[Dict[str, str]] = []
    total_runs = (
        len(regimes) * len(function_classes) * len(depths) * len(widths) * len(seeds)
    )
    run_counter = 0
    started_at = time.time()

    for regime_name, gt_bias_types in regimes:
        for function_class in function_classes:
            for depth in depths:
                for width in widths:
                    for seed in seeds:
                        run_counter += 1
                        command = build_command(
                            function_class=function_class,
                            depth=depth,
                            width=width,
                            seed=seed,
                            regime_name=regime_name,
                            gt_bias_types=gt_bias_types,
                            args=args,
                        )
                        print(
                            f"[{run_counter}/{total_runs}] regime={regime_name} "
                            f"fn={function_class} depth={depth} width={width} seed={seed}"
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
                            "regime_name": regime_name,
                            "function_class": function_class,
                            "depth": str(depth),
                            "width": str(width),
                            "seed": str(seed),
                            "gt_bias_types": gt_bias_types,
                            "return_code": str(completed.returncode),
                        }

                        if completed.returncode == 0:
                            try:
                                metrics = parse_run_output(output_text)
                                row.update(metrics)
                                print(
                                    "  "
                                    + f"gt_mean_rel_error={float(metrics['gt_mean_rel_error']):.4f} "
                                    + f"selected_max_abs_cos={float(metrics['selected_max_abs_cos']):.4f}"
                                )
                            except RuntimeError as error:
                                row["parse_error"] = str(error)
                                row["output_tail"] = output_text[-2000:]
                                print(f"  parse_error={error}")
                        else:
                            row["output_tail"] = output_text[-2000:]
                            print("  run_failed")

                        rows.append(row)

    write_csv(rows, args.output_csv)
    write_summary(rows, args.output_summary_md)
    elapsed = time.time() - started_at
    print(f"Completed {len(rows)} runs in {elapsed:.1f}s")
    print(f"CSV: {args.output_csv}")
    print(f"Summary: {args.output_summary_md}")


if __name__ == "__main__":
    main()
