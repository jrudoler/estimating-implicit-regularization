#!/usr/bin/env python3
"""Generate data for the scalar ridge lambda-vs-epochs OLS experiment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.bias import RidgeBias  # noqa: E402
from core.data import FullBatchDataModule  # noqa: E402
from core.estimators import BiasWithMSE  # noqa: E402
from core.models import LinearRegression  # noqa: E402
from core.utils import compute_Q_matrix  # noqa: E402

from analysis.ols_dgp import (  # noqa: E402
    DEFAULT_OLS_N,
    DEFAULT_OLS_NOISE_STD,
    DEFAULT_OLS_P,
    DEFAULT_OLS_SEED,
    sample_ols_problem,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT = REPO_ROOT / "data" / "generated" / "lambda_vs_epochs" / "results.pt"

# Experiment parameters shared with the OLS early-stopping pipeline.
SEED = DEFAULT_OLS_SEED
N = DEFAULT_OLS_N
P = DEFAULT_OLS_P
NOISE_STD = DEFAULT_OLS_NOISE_STD
EPS = 1e-2
LR = EPS / 2
EPOCH_GRID = [1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 500, 1000]

# Bias-fit hyperparameters tuned to converge across the full lambda range.
BIAS_MAX_EPOCHS = 30_000
BIAS_PATIENCE = 2_000
BIAS_LR = 0.05
BIAS_INIT_BETA = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .pt path for the reusable experiment data.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=NOISE_STD,
        help="Gaussian observation-noise standard deviation in y = X beta + eps.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def fit_lambda_iterative(
    theta_star: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> float:
    """Fit the scalar ridge bias by gradient matching for one GD checkpoint."""
    model = LinearRegression(
        input_dim=P,
        output_dim=1,
        lr=LR,
        fit_intercept=False,
        init_zeros=True,
    ).to(device)
    with torch.no_grad():
        model.linear.weight.copy_(theta_star.view(1, -1).to(model.linear.weight))
    model = model.float()

    bias_model = RidgeBias(
        enforce_positive=True,
        init_value=BIAS_INIT_BETA,
    ).to(device).float()
    estimator = BiasWithMSE(
        predictive_model=model,
        bias_model=bias_model,
        grad_match_loss_fn=nn.functional.mse_loss,
        lr=BIAS_LR,
        optimizer_cls=torch.optim.Adam,
    )
    datamodule = FullBatchDataModule(x.float().cpu(), y.float().cpu(), num_workers=0)
    trainer = Trainer(
        max_epochs=BIAS_MAX_EPOCHS,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[
            EarlyStopping(
                monitor="train_bias/loss",
                patience=BIAS_PATIENCE,
                mode="min",
                min_delta=0.0,
            )
        ],
        accelerator="auto",
    )
    trainer.fit(estimator, datamodule)
    scale = bias_model.get_bias_params()["scale"]
    if isinstance(scale, torch.Tensor):
        return float(scale.detach().cpu().item())
    return float(scale)


def generate_payload(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")
    LOGGER.info("Running lambda-vs-epochs data generation on %s", device)

    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(False)

    x, y, betas = sample_ols_problem(
        seed=SEED,
        n=N,
        p=P,
        noise_std=args.noise_std,
    )
    x_device, y_device = x.to(device), y.to(device)

    theta_ols = torch.linalg.lstsq(x_device, y_device.unsqueeze(1)).solution.squeeze()
    ols_error = (theta_ols - betas.to(device)).norm().item()
    LOGGER.info(
        "Empirical OLS sampling error ||theta_ols - beta|| = %.3e (noise_std=%.3g)",
        ols_error,
        args.noise_std,
    )

    theta = torch.zeros(P, device=device)
    xtx_over_n = x_device.T @ x_device / N
    xty_over_n = x_device.T @ y_device / N

    thetas: dict[int, torch.Tensor] = {}
    grads: dict[int, torch.Tensor] = {}
    for t in range(1, max(EPOCH_GRID) + 1):
        grad_mse = 2.0 * (xtx_over_n @ theta - xty_over_n)
        theta = theta - LR * grad_mse
        if t in EPOCH_GRID:
            thetas[t] = theta.clone()
            grads[t] = grad_mse.clone()

    closed_lambdas = []
    for k in EPOCH_GRID:
        th, grad = thetas[k], grads[k]
        closed_lambdas.append((-th @ grad / (2.0 * (th @ th))).item())

    theoretical_lambdas = []
    for k in EPOCH_GRID:
        q_matrix = compute_Q_matrix(x_device, k=k, eps=EPS)
        theoretical_lambdas.append((q_matrix.trace() / P).item())

    iter_lambdas = []
    for index, k in enumerate(EPOCH_GRID):
        lam_iter = fit_lambda_iterative(thetas[k], x_device, y_device, device)
        iter_lambdas.append(lam_iter)
        LOGGER.info(
            "k=%4d lambda_iter=%.6g lambda_closed=%.6g lambda_theory=%.6g",
            k,
            lam_iter,
            closed_lambdas[index],
            theoretical_lambdas[index],
        )

    return {
        "config": {
            "seed": SEED,
            "p": P,
            "n": N,
            "noise_std": args.noise_std,
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
    }


def main() -> None:
    configure_logging()
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(generate_payload(args), args.output)
    LOGGER.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
