#!/usr/bin/env python3
"""OLS full-matrix endpoint-recovery experiment.

For a fixed design matrix X, draw many ground-truth weight vectors beta_k
(seed-controlled), train each noisy regression problem
(X, y_k = X beta_k + epsilon_k) by full-batch GD with a callback-selected early
stop, and stack the (theta_k, -grad_k) pairs into a least-squares system. This
recovers the full theoretical regularizer matrix Q_t implied by early-stopped
GD.

Outputs (data/generated/ols_full_matrix_recovery/results.pt):

    config:           dict of hyperparameters
    X:                [N, P] shared design matrix
    theta_pool:       [num_avg_seeds, num_endpoints, P]
    target_pool:      [num_avg_seeds, num_endpoints, P]   (= -grad at stop)
    Q_theory_pool:    [num_avg_seeds, P, P]               (median over endpoints)
    distances_pool:   [num_avg_seeds, num_endpoints]      (||Q_hat_m - Q_theory||
                                                          for m = 1..num_endpoints)
    vis_beta_first:   [P]   ground-truth beta of the first endpoint in pool 0
    vis_y_first:      [N]   y_k for that endpoint
    vis_theta_first:  [P]   model_theta after callback stop for that endpoint

Reproduces the figure pair in
notebooks/linear-regression-trajectory.ipynb (section "Full-matrix endpoint
recovery from independent seeds") with the parameter Q estimated from all
NUM_FULL_MATRIX_ENDPOINTS endpoints rather than a 10-endpoint subset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Make src/ importable so we can use core.utils.compute_Q_matrix.
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.utils import compute_Q_matrix  # noqa: E402

from analysis.ols_dgp import (  # noqa: E402
    DEFAULT_OLS_BETA_SCALE,
    DEFAULT_OLS_N,
    DEFAULT_OLS_NOISE_STD,
    DEFAULT_OLS_P,
    DEFAULT_OLS_SEED,
    make_noisy_linear_response,
    sample_ols_beta,
    sample_ols_design,
)
from analysis.ols_full_matrix_recovery.pipeline import (  # noqa: E402
    Endpoint,
    fit_symmetric_matrix_from_points,
    gd_trajectory,
    loss_grad,
    matrix_relative_distance,
)


# Notebook defaults; exposed via argparse so the rule can override.
DEFAULT_SEED = DEFAULT_OLS_SEED
DEFAULT_P = DEFAULT_OLS_P
DEFAULT_N = DEFAULT_OLS_N
DEFAULT_EPS = 1e-2
DEFAULT_STOP_STEP = 250
DEFAULT_NOISE_STD = DEFAULT_OLS_NOISE_STD
DEFAULT_NUM_ENDPOINTS = 100
DEFAULT_NUM_AVG_SEEDS = 5
SEED_STRIDE = 10_000  # spacing between pools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--p", type=int, default=DEFAULT_P, help="weight dimension")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="design-matrix rows")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="GD step size")
    parser.add_argument(
        "--stop-step",
        type=int,
        default=DEFAULT_STOP_STEP,
        help="Number of GD iterates t at which every endpoint is read off. "
        "No early stopping; the implicit-regularization theory is stated for a "
        "fixed t and we want every endpoint at the same t.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=DEFAULT_NOISE_STD,
        help="Gaussian observation-noise standard deviation in y = X beta + eps.",
    )
    parser.add_argument(
        "--num-endpoints",
        type=int,
        default=DEFAULT_NUM_ENDPOINTS,
        help="distinct beta seeds per pool",
    )
    parser.add_argument(
        "--num-avg-seeds",
        type=int,
        default=DEFAULT_NUM_AVG_SEEDS,
        help="independent pools of beta seeds (averaged for the SE band)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def train_endpoint(
    X: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    stop_step: int,
    noise_std: float,
    generator: torch.Generator,
) -> Endpoint:
    y = make_noisy_linear_response(
        X,
        beta,
        noise_std=noise_std,
        generator=generator,
    )
    traj = gd_trajectory(X, y, stop_step, eps)
    theta_at_stop = traj["theta"][stop_step].clone()
    return Endpoint(
        beta=beta,
        y=y,
        stop_step=stop_step,
        model_theta=theta_at_stop,
        neg_grad_stop=-loss_grad(X, y, theta_at_stop),
        Q_theory_stop=compute_Q_matrix(X, stop_step, eps),
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(False)

    # Shared design matrix across all endpoints and pools (beta and noise vary).
    design_generator = torch.Generator().manual_seed(args.seed)
    X = sample_ols_design(n=args.n, p=args.p, generator=design_generator)

    theta_pool = torch.empty(args.num_avg_seeds, args.num_endpoints, args.p)
    target_pool = torch.empty(args.num_avg_seeds, args.num_endpoints, args.p)
    Q_points_pool = torch.empty(
        args.num_avg_seeds, args.num_endpoints, args.p, args.p
    )
    vis_beta_first = torch.empty(args.p)
    vis_y_first = torch.empty(args.n)
    vis_theta_first = torch.empty(args.p)

    for pool_idx in range(args.num_avg_seeds):
        seed_offset = pool_idx * SEED_STRIDE
        for ep_idx in range(args.num_endpoints):
            g = torch.Generator().manual_seed(args.seed + 100 + seed_offset + ep_idx)
            beta = sample_ols_beta(
                p=args.p,
                beta_scale=DEFAULT_OLS_BETA_SCALE,
                generator=g,
            )
            ep = train_endpoint(
                X,
                beta,
                args.eps,
                args.stop_step,
                args.noise_std,
                g,
            )
            theta_pool[pool_idx, ep_idx] = ep.model_theta
            target_pool[pool_idx, ep_idx] = ep.neg_grad_stop
            Q_points_pool[pool_idx, ep_idx] = ep.Q_theory_stop
            if pool_idx == 0 and ep_idx == 0:
                vis_beta_first.copy_(ep.beta)
                vis_y_first.copy_(ep.y)
                vis_theta_first.copy_(ep.model_theta)
        print(
            f"  pool {pool_idx + 1}/{args.num_avg_seeds}: "
            f"trained {args.num_endpoints} endpoints"
        )

    # Median theoretical Q within each pool (the "visual" theory target shown
    # next to the Q estimate).
    Q_theory_pool = torch.quantile(Q_points_pool, 0.5, dim=1)  # [pools, p, p]

    # Distance-to-theory curves: refit Q with the first m endpoints for
    # m = 1..num_endpoints, measure ||Q_m - Q_theory_pool|| per pool.
    distances_pool = torch.empty(args.num_avg_seeds, args.num_endpoints)
    for pool_idx in range(args.num_avg_seeds):
        for m in range(1, args.num_endpoints + 1):
            Q_m = fit_symmetric_matrix_from_points(
                theta_pool[pool_idx, :m], target_pool[pool_idx, :m]
            )
            distances_pool[pool_idx, m - 1] = matrix_relative_distance(
                Q_m, Q_theory_pool[pool_idx]
            )

    payload = {
        "config": {
            "seed": args.seed,
            "p": args.p,
            "n": args.n,
            "eps": args.eps,
            "stop_step": args.stop_step,
            "noise_std": args.noise_std,
            "num_endpoints": args.num_endpoints,
            "num_avg_seeds": args.num_avg_seeds,
            "seed_stride": SEED_STRIDE,
        },
        "X": X,
        "theta_pool": theta_pool,
        "target_pool": target_pool,
        "Q_theory_pool": Q_theory_pool,
        "distances_pool": distances_pool,
        "vis_beta_first": vis_beta_first,
        "vis_y_first": vis_y_first,
        "vis_theta_first": vis_theta_first,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
