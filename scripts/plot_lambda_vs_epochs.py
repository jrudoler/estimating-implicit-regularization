#!/usr/bin/env python3
"""
Plot estimated scalar ridge λ̂ vs. GD epochs for linear regression,
replacing Table 1 in the manuscript.

Reproduces the notebook experiment (seed=56, n=1000, p=5, η=0.005)
at a finer grid of epoch checkpoints and overlays the theoretical
scalar λ derived from the Q_t matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping

# ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.models import LinearRegression
from core.data import FullBatchDataModule
from core.bias import RidgeBias
from core.estimators import BiasWithAutodiffLoss
from core.utils import compute_Q_matrix


# ── experiment parameters (match the notebook / paper) ──────────────────
SEED = 56
N = 1000
P = 5
EPS = 1e-2          # notebook eps
LR = EPS / 2        # 0.005, the η used by SGD (accounts for factor-of-2 in MSELoss)
EPOCH_GRID = [5, 10, 20, 50, 100, 150, 200, 300, 500]

# ── data generation ─────────────────────────────────────────────────────
torch.manual_seed(SEED)
X = torch.randn(N, P)
betas = 3.0 * torch.randn(P)
y = X @ betas
dm = FullBatchDataModule(X, y, num_workers=0)

# ── theoretical λ from Q_t ──────────────────────────────────────────────
# The theory: θ_{t+1} = θ_t - (lr) * ∇MSE, where ∇MSE = (2/n)X^T(Xθ-y).
# So the effective step on (1/n)X^TX is 2*lr = eps.
# compute_Q_matrix expects eps such that the update rule is θ -= eps*(X^TX/n)*θ + ...
# so pass eps = 2*LR = EPS.
theoretical_lambdas = []
for k in EPOCH_GRID:
    Q = compute_Q_matrix(X, k=k, eps=EPS)
    # scalar λ that best summarises Q: λ = tr(Q) / p
    lam = Q.trace().item() / P
    theoretical_lambdas.append(lam)
    print(f"  theory k={k}: tr(Q)/p = {lam:.6g}")

# ── estimated λ̂ at each checkpoint ─────────────────────────────────────
estimated_lambdas = []

for max_ep in EPOCH_GRID:
    # fresh model each time (init at zero, same as notebook)
    torch.manual_seed(SEED)
    lm = LinearRegression(input_dim=P, output_dim=1, lr=LR,
                          fit_intercept=False, init_zeros=True)
    trainer = Trainer(
        max_epochs=max_ep,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        accelerator="auto",
    )
    trainer.fit(lm, dm)

    # estimate scalar λ
    torch.manual_seed(SEED)
    bias_model = RidgeBias()
    estimator = BiasWithAutodiffLoss(
        predictive_model=lm,
        bias_model=bias_model,
        predictive_loss_fn=nn.functional.mse_loss,
        grad_match_loss_fn=nn.functional.mse_loss,
        lr=1e-2,
        optimizer_cls=torch.optim.Adam,
    )
    est_trainer = Trainer(
        max_epochs=5000,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[EarlyStopping(monitor="train_bias/loss", patience=150, mode="min")],
        accelerator="auto",
    )
    est_trainer.fit(estimator, dm)

    lam_hat = estimator.bias_model.get_bias_params()["scale"]
    estimated_lambdas.append(lam_hat)
    print(f"epochs={max_ep:>4d}  λ̂={lam_hat:.6g}  λ_theory={theoretical_lambdas[EPOCH_GRID.index(max_ep)]:.6g}")

# ── plot ────────────────────────────────────────────────────────────────
plt.style.use(str(Path(__file__).resolve().parents[1] / "clean_fig.mplstyle"))

fig, ax = plt.subplots(figsize=(4.5, 3.2))

ax.semilogy(EPOCH_GRID, estimated_lambdas, "o-", color="C0", label=r"Estimated $\hat{\lambda}$")
ax.semilogy(EPOCH_GRID, theoretical_lambdas, "s--", color="C3", label=r"Theoretical $\mathrm{tr}(Q_t)/p$")

ax.set_xlabel("Gradient descent epochs")
ax.set_ylabel(r"$\hat{\lambda}$")
ax.legend(frameon=False)

out_dir = Path(__file__).resolve().parents[1] / "figures"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "lambda_vs_epochs.pdf")
print(f"\nSaved → {out_dir / 'lambda_vs_epochs.pdf'}")
plt.close(fig)
