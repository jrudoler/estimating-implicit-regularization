#!/usr/bin/env python3
"""OLS bootstrap endpoint-recovery experiment.

Mirrors ols_full_matrix_recovery/run.py but generates 'endpoints' by
bootstrap-resampling rows of the original (X, y) dataset rather than drawing
independent (beta, epsilon) pairs.  This tests whether bootstrap resampling of
a single dataset can also recover the full theoretical regulariser matrix
Q^{(t)} implied by early-stopped GD.

Each bootstrap endpoint draws n rows with replacement from the original (X, y),
trains full-batch GD on the resampled dataset, and contributes one row to the
stacked system  Q theta_k = -grad_k  used to recover the full symmetric matrix.

The comparison target Q_theory differs between modes:
  - Fixed-t (--stop-step INT): all endpoints stopped at the canonical iterate t;
    Q_theory is Q(t) computed from the *original* X, since bootstrap resamples
    approximate the same eigenstructure as X.
  - Variable-t (no --stop-step): each endpoint is stopped individually by early
    stopping; Q_theory is the per-pool median of the bootstrap-specific
    Q^*(t_k) matrices (same convention as ols_full_matrix_recovery/run.py Panel D).

Multiple 'pools' represent independent draws of bootstrap resample sets from
the same (X, y), giving variance estimates for the distance-to-theory curves.

Inputs
------
    --linear-data    data/generated/linear_regression_ols/results.pt
                     (provides original X, y, and canonical stop_epoch)

Outputs (--output PATH):

    config:           dict of hyperparameters
    Q_theory_pool:    [num_avg_seeds, P, P]   per-pool comparison target
    theta_pool:       [num_avg_seeds, num_endpoints, P]
    target_pool:      [num_avg_seeds, num_endpoints, P]  (= -grad at stop)
    distances_pool:   [num_avg_seeds, num_endpoints]     (||Q_hat_m - Q_theory||)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.utils import compute_Q_matrix  # noqa: E402

from analysis.ols_full_matrix_recovery.pipeline import (  # noqa: E402
    callback_stop_step,
    fit_symmetric_matrix_from_points,
    gd_trajectory,
    loss_grad,
    matrix_relative_distance,
)


DEFAULT_EPS = 1e-2
DEFAULT_MAX_EPOCHS = 2000
DEFAULT_PATIENCE = 5
DEFAULT_MIN_DELTA = 1e-3
DEFAULT_NUM_ENDPOINTS = 100
DEFAULT_NUM_AVG_SEEDS = 10
SEED_STRIDE = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--linear-data",
        type=Path,
        required=True,
        help="Path to linear_regression_ols/results.pt (provides original X, y, stop_epoch).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--eps", type=float, default=DEFAULT_EPS, help="GD step size (must match original run)."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="Maximum GD steps per bootstrap endpoint before forced termination.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early-stopping patience.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=DEFAULT_MIN_DELTA,
        help="Minimum loss decrease to count as an improvement for early stopping.",
    )
    parser.add_argument(
        "--num-endpoints",
        type=int,
        default=DEFAULT_NUM_ENDPOINTS,
        help="Bootstrap resamples per pool.",
    )
    parser.add_argument(
        "--num-avg-seeds",
        type=int,
        default=DEFAULT_NUM_AVG_SEEDS,
        help="Independent pools of bootstrap resamples (for variance of distance curve).",
    )
    parser.add_argument(
        "--stop-step",
        type=int,
        default=None,
        help="Fix every endpoint's stop step to this value (Panel B mode). "
        "Omit to use per-endpoint early stopping (Panel D mode).",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def train_bootstrap_endpoint(
    X: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    max_epochs: int,
    patience: int,
    min_delta: float,
    generator: torch.Generator,
    stop_step_override: int | None = None,
) -> dict:
    """Bootstrap-resample (X, y), train GD, return (theta, neg_grad, Q_theory, stop_step)."""
    n = X.shape[0]
    indices = torch.randint(0, n, (n,), generator=generator)
    X_b = X[indices]
    y_b = y[indices]

    traj = gd_trajectory(X_b, y_b, max_epochs, eps)
    stop_step = (
        stop_step_override
        if stop_step_override is not None
        else callback_stop_step(traj["loss"], patience, min_delta)
    )
    theta_stop = traj["theta"][stop_step]
    return {
        "stop_step": stop_step,
        "model_theta": theta_stop.clone(),
        "neg_grad_stop": -loss_grad(X_b, y_b, theta_stop),
        "Q_theory_stop": compute_Q_matrix(X_b, stop_step, eps),
    }


def main() -> None:
    args = parse_args()
    fixed_t = args.stop_step

    if fixed_t is not None and fixed_t > args.max_epochs:
        raise ValueError(
            f"--stop-step {fixed_t} exceeds --max-epochs {args.max_epochs}; "
            "increase --max-epochs so the trajectory reaches the fixed step."
        )

    lin = torch.load(args.linear_data, weights_only=False)
    X_orig = lin["X"]
    y_orig = lin["y"]
    n, p = X_orig.shape

    # For fixed-t mode: the shared theoretical target is Q(t) from the original X.
    # Bootstrap resamples approximate the same eigenstructure as X, so this is the
    # natural comparison target (mirrors Panel B of the original composite figure).
    Q_theory_orig = compute_Q_matrix(X_orig, fixed_t or lin["config"]["stop_epoch"], args.eps)

    theta_pool = torch.empty(args.num_avg_seeds, args.num_endpoints, p)
    target_pool = torch.empty(args.num_avg_seeds, args.num_endpoints, p)
    Q_points_pool = torch.empty(args.num_avg_seeds, args.num_endpoints, p, p)

    for pool_idx in range(args.num_avg_seeds):
        seed_offset = pool_idx * SEED_STRIDE
        for ep_idx in range(args.num_endpoints):
            g = torch.Generator().manual_seed(args.seed + 100 + seed_offset + ep_idx)
            ep = train_bootstrap_endpoint(
                X_orig,
                y_orig,
                args.eps,
                args.max_epochs,
                args.patience,
                args.min_delta,
                g,
                stop_step_override=fixed_t,
            )
            theta_pool[pool_idx, ep_idx] = ep["model_theta"]
            target_pool[pool_idx, ep_idx] = ep["neg_grad_stop"]
            Q_points_pool[pool_idx, ep_idx] = ep["Q_theory_stop"]
        print(
            f"  pool {pool_idx + 1}/{args.num_avg_seeds}: "
            f"trained {args.num_endpoints} bootstrap endpoints"
        )

    if fixed_t is not None:
        # All endpoints share the same canonical t, so Q_theory is the same for
        # all pools: Q(t) from the original X.
        Q_theory_pool = Q_theory_orig.unsqueeze(0).expand(args.num_avg_seeds, -1, -1).clone()
    else:
        # Variable early stopping: each endpoint has its own bootstrap X_b and t_k.
        # Use the per-pool median of Q^*(t_k) as the comparison target, mirroring
        # the convention used in ols_full_matrix_recovery/run.py Panel D.
        Q_theory_pool = torch.quantile(Q_points_pool, 0.5, dim=1)

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
            "n": n,
            "p": p,
            "eps": args.eps,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "num_endpoints": args.num_endpoints,
            "num_avg_seeds": args.num_avg_seeds,
            "seed_stride": SEED_STRIDE,
            "fixed_t": fixed_t,
            "linear_data_path": str(args.linear_data),
        },
        "Q_theory_orig": Q_theory_orig,
        "Q_theory_pool": Q_theory_pool,
        "theta_pool": theta_pool,
        "target_pool": target_pool,
        "distances_pool": distances_pool,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
