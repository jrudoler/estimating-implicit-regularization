#!/usr/bin/env python3
"""Sweep observation-noise sigma and measure bootstrap full-matrix recovery
quality at m=NUM_ENDPOINTS.

For each sigma we:
  1. Generate (X, y, beta) at noise_std=sigma (X and beta shared across sigmas
     via fixed seed; only y/epsilon differs).
  2. Run num_endpoints bootstrap resamples, train each at fixed t, build the
     (theta, -grad) pool.
  3. Recover Q_hat from m=num_endpoints endpoints, record distance to Q_theory.
  4. Repeat for num_pools to get a CI band.

Q_theory only depends on X and t, not sigma — so this isolates sigma's effect
on bootstrap-induced theta variability cleanly.

Output (data/generated/ols_bootstrap_sigma_sweep/results.pt):
    config:        dict of hyperparameters
    sigmas:        [num_sigmas]
    distances:     [num_sigmas, num_pools]
    theta_stds:    [num_sigmas]   per-dim std of bootstrap thetas (first pool only)
    Qt_norm:       float          ||Q_theory||_F (shared comparison scale)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.utils import compute_Q_matrix
from analysis.ols_full_matrix_recovery.pipeline import (
    fit_symmetric_matrix_from_points,
    gd_trajectory,
    loss_grad,
)
from analysis.ols_dgp import (
    DEFAULT_OLS_BETA_SCALE,
    DEFAULT_OLS_N,
    DEFAULT_OLS_P,
    DEFAULT_OLS_SEED,
    sample_ols_design,
    sample_ols_beta,
    make_noisy_linear_response,
)


DEFAULT_EPS = 1e-2
DEFAULT_T_FIXED = 500
DEFAULT_NUM_ENDPOINTS = 100
DEFAULT_NUM_POOLS = 5
DEFAULT_SIGMAS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 14.0, 20.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_OLS_SEED)
    parser.add_argument("--n", type=int, default=DEFAULT_OLS_N)
    parser.add_argument("--p", type=int, default=DEFAULT_OLS_P)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--t-fixed", type=int, default=DEFAULT_T_FIXED)
    parser.add_argument("--num-endpoints", type=int, default=DEFAULT_NUM_ENDPOINTS)
    parser.add_argument("--num-pools", type=int, default=DEFAULT_NUM_POOLS)
    parser.add_argument(
        "--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS,
        help="Observation-noise std values to sweep.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_bootstrap_pool(X, y, num_endpoints, fixed_t, eps, max_epochs, seed_base):
    n, p = X.shape
    thetas = torch.empty(num_endpoints, p)
    targets = torch.empty(num_endpoints, p)
    for k in range(num_endpoints):
        g = torch.Generator().manual_seed(seed_base + k)
        idx = torch.randint(0, n, (n,), generator=g)
        X_b, y_b = X[idx], y[idx]
        traj = gd_trajectory(X_b, y_b, max_epochs, eps)
        theta_stop = traj["theta"][fixed_t]
        thetas[k] = theta_stop
        targets[k] = -loss_grad(X_b, y_b, theta_stop)
    return thetas, targets


def main():
    args = parse_args()

    g_dgp = torch.Generator().manual_seed(args.seed)
    X = sample_ols_design(n=args.n, p=args.p, generator=g_dgp)
    beta = sample_ols_beta(p=args.p, beta_scale=DEFAULT_OLS_BETA_SCALE, generator=g_dgp)
    Q_theory = compute_Q_matrix(X, args.t_fixed, args.eps)
    Qt_norm = float(torch.linalg.norm(Q_theory))
    print(f"||Q_theory||_F = {Qt_norm:.4f}, t={args.t_fixed}, n={args.n}, p={args.p}")

    sigmas = list(args.sigmas)
    distances = torch.zeros(len(sigmas), args.num_pools)
    theta_stds = torch.zeros(len(sigmas))

    for si, sigma in enumerate(sigmas):
        g_y = torch.Generator().manual_seed(args.seed + 7919)
        y = make_noisy_linear_response(X, beta, noise_std=sigma, generator=g_y)
        for pool_idx in range(args.num_pools):
            seed_base = 1_000_000 + pool_idx * 10_000
            thetas, targets = run_bootstrap_pool(
                X, y, args.num_endpoints, args.t_fixed, args.eps, args.t_fixed, seed_base,
            )
            Q_hat = fit_symmetric_matrix_from_points(thetas, targets)
            distances[si, pool_idx] = torch.linalg.norm(Q_hat - Q_theory)
            if pool_idx == 0:
                theta_stds[si] = thetas.std(0).mean()
        print(
            f"  sigma={sigma:5.2f}  theta_std={theta_stds[si].item():.3f}  "
            f"dist mean={distances[si].mean().item():.4f}"
        )

    payload = {
        "config": {
            "seed": args.seed, "n": args.n, "p": args.p, "eps": args.eps,
            "t_fixed": args.t_fixed,
            "num_endpoints": args.num_endpoints,
            "num_pools": args.num_pools,
        },
        "sigmas": torch.tensor(sigmas),
        "distances": distances,
        "theta_stds": theta_stds,
        "Qt_norm": Qt_norm,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
