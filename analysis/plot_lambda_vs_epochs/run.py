#!/usr/bin/env python3
"""
Plot estimated scalar ridge λ̂ vs. GD epochs for the linear-regression
implicit-bias experiment. Replaces Table 1 in the manuscript.

Reproduces the notebook setup (seed=56, n=1000, p=5, η=0.005) at a
finer grid of epoch checkpoints, and shows three quantities:

  1. λ̂_iter      : the empirical gradient-matching estimator obtained
                   by running BiasWithMSE to convergence
  2. λ̂_closed   : the same gradient-matching estimator solved
                   analytically (closed-form OLS — only possible
                   because RidgeBias has one parameter)
  3. λ_theory   : tr(Q_t) / p, the theoretical scalar summary of the
                   implicit-bias matrix Q_t

(1) and (2) optimise the same objective; agreement between them shows
the iterative estimator converges to the right thing. Agreement with
(3) shows the empirical estimator recovers the theory it should.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import lightning.pytorch as pl
from cmap import Colormap
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping

# make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from core.utils import compute_Q_matrix          # noqa: E402
from core.models import LinearRegression          # noqa: E402
from core.data import FullBatchDataModule         # noqa: E402
from core.bias import RidgeBias                   # noqa: E402
from core.estimators import BiasWithMSE           # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_FIGURES_DIR = REPO_ROOT / "results" / "figures"


# ── experiment parameters (match the notebook / paper) ─────────────────
SEED = 56
N = 1000
P = 5
EPS = 1e-2          # notebook eps; effective per-step coefficient on (1/n)X^TX
LR = EPS / 2        # SGD lr accounts for factor-of-2 in nn.MSELoss
EPOCH_GRID = [1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 500, 1000]

# bias-fit hyper-parameters (tuned to converge across the full λ range)
BIAS_MAX_EPOCHS = 30_000
BIAS_PATIENCE = 2_000
BIAS_LR = 0.05      # learning rate on β (log λ)
BIAS_INIT_BETA = 0.0  # init λ = e^0 = 1; lets log-domain SGD reach
                      # both ~e^5 (≈150) and ~e^-21 (≈1e-9) ends


def fit_lambda_iterative(theta_star: torch.Tensor,
                          X: torch.Tensor,
                          y: torch.Tensor,
                          device: torch.device) -> float:
    """
    Run the empirical gradient-matching estimator (BiasWithMSE) on a
    fixed predictive model whose weight is theta_star, and return the
    learned scalar λ.
    """
    # build a LinearRegression Lightning model and stick theta_star into it
    model = LinearRegression(input_dim=P, output_dim=1, lr=LR,
                             fit_intercept=False, init_zeros=True).to(device)
    with torch.no_grad():
        model.linear.weight.copy_(theta_star.view(1, -1).to(model.linear.weight))
    model = model.float()  # Lightning + lightning DM expect float32 by default

    bias_model = RidgeBias(enforce_positive=True,
                            init_value=BIAS_INIT_BETA).to(device).float()

    estimator = BiasWithMSE(
        predictive_model=model,
        bias_model=bias_model,
        grad_match_loss_fn=nn.functional.mse_loss,
        lr=BIAS_LR,
        optimizer_cls=torch.optim.Adam,
    )

    dm = FullBatchDataModule(X.float().cpu(), y.float().cpu(), num_workers=0)

    trainer = Trainer(
        max_epochs=BIAS_MAX_EPOCHS,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[EarlyStopping(monitor="train_bias/loss",
                                  patience=BIAS_PATIENCE,
                                  mode="min",
                                  min_delta=0.0)],
        accelerator="auto",
    )
    trainer.fit(estimator, dm)
    return bias_model.get_bias_params()["scale"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_FIGURES_DIR / "lambda_vs_epochs.pdf",
        help="Canonical output path for the manuscript figure.",
    )
    parser.add_argument(
        "--save-data",
        type=Path,
        default=None,
        help=(
            "Optional .pt path. If set, also saves epoch_grid + iter/closed/"
            "theory lambda series so downstream composite plots can reuse them."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(torch.float64)
    torch.set_float32_matmul_precision("high")
    print(f"Running on {device}")

    # ── data generation ───────────────────────────────────────────────
    torch.manual_seed(SEED)
    X = torch.randn(N, P, dtype=torch.float64)
    betas = 3.0 * torch.randn(P, dtype=torch.float64)
    y = X @ betas
    X_d, y_d = X.to(device), y.to(device)

    # OLS sanity check
    theta_ols = torch.linalg.lstsq(X_d, y_d.unsqueeze(1)).solution.squeeze()
    print(f"||θ_OLS - β|| = {(theta_ols - betas.to(device)).norm():.3e}")

    # ── replay full-batch GD; cache θ_t at each checkpoint ────────────
    theta = torch.zeros(P, dtype=torch.float64, device=device)
    XtX_over_n = X_d.T @ X_d / N
    Xty_over_n = X_d.T @ y_d / N

    thetas, grads = {}, {}
    for t in range(1, max(EPOCH_GRID) + 1):
        grad_mse = 2.0 * (XtX_over_n @ theta - Xty_over_n)
        theta = theta - LR * grad_mse
        if t in EPOCH_GRID:
            thetas[t] = theta.clone()
            grads[t] = grad_mse.clone()

    # ── (2) closed-form λ̂ (exact OLS argmin of grad-match objective) ─
    closed_lambdas = []
    for k in EPOCH_GRID:
        th, g = thetas[k], grads[k]
        closed_lambdas.append((-th @ g / (2.0 * (th @ th))).item())

    # ── (3) theoretical λ from Q_t ────────────────────────────────────
    theoretical_lambdas = []
    for k in EPOCH_GRID:
        Q = compute_Q_matrix(X_d, k=k, eps=EPS)
        theoretical_lambdas.append((Q.trace() / P).item())

    # ── (1) iterative λ̂ via BiasWithMSE, run to convergence ──────────
    iter_lambdas = []
    for k in EPOCH_GRID:
        lam_iter = fit_lambda_iterative(thetas[k], X_d, y_d, device)
        iter_lambdas.append(lam_iter)
        print(f"  k={k:>4}  λ_iter={lam_iter:.6g}  "
              f"λ_closed={closed_lambdas[EPOCH_GRID.index(k)]:.6g}  "
              f"λ_theory={theoretical_lambdas[EPOCH_GRID.index(k)]:.6g}")

    # ── plot ───────────────────────────────────────────────────────────
    plt.style.use(str(Path(__file__).resolve().parents[2] / "clean_fig.mplstyle"))
    fig, ax = plt.subplots(figsize=(5.0, 3.4))

    batlow = Colormap("crameri:batlow").to_mpl()
    c_iter, c_closed, c_theory = batlow(0.2), batlow(0.55), batlow(0.85)

    ax.loglog(EPOCH_GRID, iter_lambdas, "o-", color=c_iter,
              label=r"Iterative $\hat{\lambda}_t$ (gradient matching)",
              markersize=6)
    ax.loglog(EPOCH_GRID, closed_lambdas, "x", color=c_closed,
              label=r"Closed-form $\hat{\lambda}_t$",
              markersize=8, markeredgewidth=1.8)
    ax.loglog(EPOCH_GRID, theoretical_lambdas, "--", color=c_theory,
              label=r"Theoretical $\mathrm{tr}(Q_t)/p$",
              linewidth=1.5)

    ax.set_xlabel("Gradient descent epochs $t$")
    ax.set_ylabel(r"Scalar ridge penalty $\hat{\lambda}_t$")
    ax.legend(frameon=False, loc="lower left", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"\nSaved -> {args.out}")
    plt.close(fig)

    if args.save_data is not None:
        args.save_data.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": {
                    "seed": SEED,
                    "p": P,
                    "n": N,
                    "eps": EPS,
                    "lr": LR,
                    "bias_max_epochs": BIAS_MAX_EPOCHS,
                    "bias_patience": BIAS_PATIENCE,
                    "bias_lr": BIAS_LR,
                },
                "epoch_grid": list(EPOCH_GRID),
                "iter_lambdas": list(iter_lambdas),
                "closed_lambdas": list(closed_lambdas),
                "theoretical_lambdas": list(theoretical_lambdas),
            },
            args.save_data,
        )
        print(f"Saved -> {args.save_data}")


if __name__ == "__main__":
    main()
