#!/usr/bin/env python3
"""Diagnostic partial correlations for the Barrett known-coefficient run.

This helper is intentionally descriptive. The paper-bound Barrett experiment
is now a calibration check for recovering the analytic coefficient
lambda = eta * p / 4, not an attempt to establish an independent causal
relationship between lambda_hat, R_IG, and test accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "data" / "generated" / "barrett_igr_figure2" / "results.pt",
    )
    return parser.parse_args()


def residualize_ranks(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return y minus its linear regression on x (on rank scale)."""
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)


def main() -> None:
    args = parse_args()
    payload = torch.load(args.results, weights_only=False)
    results = [
        r
        for r in payload["results"]
        if r.get("included", True) and "lambda_hat" in r and "r_ig" in r
    ]
    if not results:
        raise SystemExit("No included fitted results in file.")
    lam = np.array([r["lambda_hat"] for r in results], dtype=float)
    rig = np.array([r["r_ig"] for r in results], dtype=float)
    test_acc = np.array([r["best_test_acc"] for r in results], dtype=float)
    prod = lam * rig

    n = len(results)
    print(f"Partial correlation analysis for Barrett Figure 2 (n={n} runs)")
    print("=" * 70)
    print()
    rho_lam_marginal = float(spearmanr(np.log(lam), test_acc).statistic)
    rho_rig_marginal = float(spearmanr(np.log(rig), test_acc).statistic)
    rho_prod_marginal = float(spearmanr(np.log(prod), test_acc).statistic)

    print("Marginal Spearman correlations:")
    print(f"  log(lambda_hat)   vs test_acc :  rho = {rho_lam_marginal:+.3f}")
    print(f"  log(R_IG)         vs test_acc :  rho = {rho_rig_marginal:+.3f}")
    print(f"  log(lambda*R_IG)  vs test_acc :  rho = {rho_prod_marginal:+.3f}")
    print()

    r_ta = rankdata(test_acc)
    r_lam = rankdata(np.log(lam))
    r_rig = rankdata(np.log(rig))

    # test_acc vs lambda_hat controlling for R_IG
    ta_given_rig = residualize_ranks(r_ta, r_rig)
    lam_given_rig = residualize_ranks(r_lam, r_rig)
    rho_lam_partial = float(np.corrcoef(ta_given_rig, lam_given_rig)[0, 1])

    # test_acc vs R_IG controlling for lambda_hat
    ta_given_lam = residualize_ranks(r_ta, r_lam)
    rig_given_lam = residualize_ranks(r_rig, r_lam)
    rho_rig_partial = float(np.corrcoef(ta_given_lam, rig_given_lam)[0, 1])

    print("Partial Spearman (rank-residualized) correlations:")
    print(f"  test_acc <-> lambda_hat  |  R_IG        :  rho = {rho_lam_partial:+.3f}")
    print(f"  test_acc <-> R_IG        |  lambda_hat  :  rho = {rho_rig_partial:+.3f}")
    print()
    print(
        "Interpretation: this is a descriptive diagnostic on the fitted subset. "
        f"In this run, lambda_hat's marginal correlation with test accuracy is "
        f"{rho_lam_marginal:+.2f}, but its partial correlation after controlling "
        f"for R_IG is {rho_lam_partial:+.2f}. R_IG's marginal correlation is "
        f"{rho_rig_marginal:+.2f}, and its partial correlation after controlling "
        f"for lambda_hat is {rho_rig_partial:+.2f}. Do not treat these partial "
        "correlations as the main Barrett recovery result."
    )


if __name__ == "__main__":
    main()
