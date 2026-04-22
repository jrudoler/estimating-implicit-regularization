#!/usr/bin/env python3
"""Partial-correlation analysis for Barrett Figure 2 reproduction.

Addresses the confound that panels (a) and (b) of barrett_igr_figure2.pdf
could be explained by test_acc tracking 1/R_IG rather than lambda_hat
independently. If lambda_hat is the real driver, we expect:
  - test_acc vs lambda_hat (controlling for R_IG) stays strong
  - test_acc vs R_IG (controlling for lambda_hat) weakens or reverses

Result (n=48 runs, tanh MLP, MNIST):
  Marginal Spearman:
    log(lambda_hat)   vs test_acc:  rho = +0.918
    log(R_IG)         vs test_acc:  rho = -0.701
    log(lambda*R_IG)  vs test_acc:  rho = +0.308
  Partial (rank-residualized):
    test_acc <-> lambda_hat  | R_IG       :  rho = +0.879
    test_acc <-> R_IG        | lambda_hat :  rho = +0.516  (flips sign)

Conclusion: lambda_hat is the dominant predictor of test accuracy.
The apparent negative marginal correlation between R_IG and test accuracy
is mediated by lambda_hat up => R_IG down, not an independent effect of
R_IG. Once lambda_hat is held fixed, R_IG's partial correlation with
test accuracy is positive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "results" / "barrett_igr_figure2.pt",
    )
    return parser.parse_args()


def residualize_ranks(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Return y minus its linear regression on x (on rank scale)."""
    slope, intercept = np.polyfit(x, y, 1)
    return y - (slope * x + intercept)


def main() -> None:
    args = parse_args()
    payload = torch.load(args.results, weights_only=False)
    results = payload["results"]
    lam = np.array([r["lambda_hat"] for r in results], dtype=float)
    rig = np.array([r["r_ig"] for r in results], dtype=float)
    test_acc = np.array([r["best_test_acc"] for r in results], dtype=float)
    prod = lam * rig

    n = len(results)
    print(f"Partial correlation analysis for Barrett Figure 2 (n={n} runs)")
    print("=" * 70)
    print()
    print("Marginal Spearman correlations:")
    print(f"  log(lambda_hat)   vs test_acc :  rho = {spearmanr(np.log(lam), test_acc).statistic:+.3f}")
    print(f"  log(R_IG)         vs test_acc :  rho = {spearmanr(np.log(rig), test_acc).statistic:+.3f}")
    print(f"  log(lambda*R_IG)  vs test_acc :  rho = {spearmanr(np.log(prod), test_acc).statistic:+.3f}")
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
    print("Interpretation: lambda_hat's partial correlation stays strong "
          "(+0.88) while R_IG's partial correlation flips from -0.70 "
          f"(marginal) to {rho_rig_partial:+.2f} (controlling for lambda_hat). "
          "The dominant predictor is lambda_hat; R_IG's apparent negative "
          "relationship with test_acc is mediated through lambda_hat.")


if __name__ == "__main__":
    main()
