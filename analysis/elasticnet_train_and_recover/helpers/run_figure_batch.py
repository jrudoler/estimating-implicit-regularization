#!/usr/bin/env python3
"""
Run elastic-net recovery over a grid without W&B; write CSV for scripts/plot_elasticnet_recovery.py.

Example (fast smoke test):
  ELASTICNET_FAST=1 uv run python scripts/run_elasticnet_figure_batch.py --out artifacts/elasticnet_runs.csv

Full paper grid (slow):
  uv run python scripts/run_elasticnet_figure_batch.py --out artifacts/elasticnet_runs.csv --full-grid
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.elasticnet_train_and_recover.run import (
    _fast_settings_from_env,
    run_elasticnet_recovery,
)

DEFAULT_L1 = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
DEFAULT_L2 = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
DEFAULT_SEEDS = list(range(10))
SMOOTH_PAPER = 1e-3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/elasticnet_runs.csv"))
    ap.add_argument(
        "--full-grid",
        action="store_true",
        help="Use full 6x6 lambda grid and 10 seeds (same as sweep); omit for a small default grid",
    )
    ap.add_argument(
        "--l1-values",
        type=float,
        nargs="*",
        help="Override true L1 grid (e.g. --l1-values 0.01 1.0)",
    )
    ap.add_argument(
        "--l2-values",
        type=float,
        nargs="*",
        help="Override true L2 grid",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        help="Override random seeds / replicates",
    )
    ap.add_argument(
        "--accelerator", type=str, default="auto", help="Lightning accelerator (cpu, cuda, auto)"
    )
    args = ap.parse_args()

    if args.full_grid:
        l1s, l2s, seeds = DEFAULT_L1, DEFAULT_L2, DEFAULT_SEEDS
    else:
        # Small default grid for local figure generation (override with --full-grid for paper sweep scale)
        l1s = [0.0001, 0.01, 1.0]
        l2s = [0.0001, 0.01, 1.0]
        seeds = [0, 1, 2, 3, 4]
    if args.l1_values:
        l1s = args.l1_values
    if args.l2_values:
        l2s = args.l2_values
    if args.seeds:
        seeds = args.seeds

    fast = _fast_settings_from_env()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "true_l1",
        "true_l2",
        "smooth",
        "seed",
        "recovery/lambda_1_hat",
        "recovery/lambda_2_hat",
        "recovery/log_mult_l1",
        "recovery/log_mult_l2",
    ]

    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for seed in seeds:
            for l1 in l1s:
                for l2 in l2s:
                    m = run_elasticnet_recovery(
                        l1,
                        l2,
                        SMOOTH_PAPER,
                        seed,
                        accelerator=args.accelerator,
                        wandb_run=None,
                        **fast,
                    )
                    row = {
                        "true_l1": m["true_l1"],
                        "true_l2": m["true_l2"],
                        "smooth": m["smooth"],
                        "seed": int(m["seed"]),
                        "recovery/lambda_1_hat": m["recovery/lambda_1_hat"],
                        "recovery/lambda_2_hat": m["recovery/lambda_2_hat"],
                        "recovery/log_mult_l1": m["recovery/log_mult_l1"],
                        "recovery/log_mult_l2": m["recovery/log_mult_l2"],
                    }
                    w.writerow(row)
                    f.flush()
                    print(f"wrote l1={l1} l2={l2} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
