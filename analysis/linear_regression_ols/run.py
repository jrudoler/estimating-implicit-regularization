#!/usr/bin/env python3
"""Generate reusable OLS early-stopping experiment data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from core.bias import DiagMatrixRidgeBias
from core.data import FullBatchDataModule
from core.estimators import BiasWithMSE
from core.models import LinearRegression
from core.utils import compute_Q_matrix, compute_beta_closed_form

from analysis.ols_dgp import (
    DEFAULT_OLS_N,
    DEFAULT_OLS_NOISE_STD,
    DEFAULT_OLS_P,
    DEFAULT_OLS_SEED,
    sample_ols_problem,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_OUTPUT = REPO_ROOT / "data" / "generated" / "linear_regression_ols" / "results.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .pt path for the reusable experiment data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_OLS_SEED,
        help="Random seed for the OLS DGP.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=DEFAULT_OLS_P,
        help="Linear regression feature dimension.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_OLS_N,
        help="Synthetic sample count.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=DEFAULT_OLS_NOISE_STD,
        help="Gaussian observation-noise standard deviation in y = X beta + eps.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-2,
        help="Gradient descent step-size parameter used in the notebook.",
    )
    parser.add_argument(
        "--max-epochs-train",
        type=int,
        default=500,
        help="Maximum epochs for the predictive linear model.",
    )
    parser.add_argument(
        "--max-epochs-bias",
        type=int,
        default=5000,
        help="Maximum epochs for the bias estimator.",
    )
    parser.add_argument(
        "--train-patience",
        type=int,
        default=5,
        help="Early-stopping patience for the predictive model.",
    )
    parser.add_argument(
        "--bias-patience",
        type=int,
        default=150,
        help="Early-stopping patience for the bias estimator.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def make_synthetic_problem(
    *,
    seed: int,
    input_dim: int,
    num_samples: int,
    noise_std: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("highest")
    return sample_ols_problem(
        seed=seed,
        n=num_samples,
        p=input_dim,
        noise_std=noise_std,
    )


def train_predictive_model(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    input_dim: int,
    eps: float,
    max_epochs: int,
    patience: int,
) -> tuple[LinearRegression, int]:
    datamodule = FullBatchDataModule(x, y, num_workers=0)
    model = LinearRegression(
        input_dim=input_dim,
        output_dim=1,
        lr=eps / 2,
        fit_intercept=False,
        init_zeros=True,
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accumulate_grad_batches=1,
        log_every_n_steps=1,
        logger=False,
        callbacks=[EarlyStopping(monitor="train/loss", patience=patience, mode="min")],
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule)
    return model, trainer.current_epoch


def compute_beta_iterates(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    input_dim: int,
    eps: float,
    num_steps: int,
) -> torch.Tensor:
    beta_iterates = [torch.zeros(input_dim)]
    for _ in range(1, num_steps + 1):
        beta_prev = beta_iterates[-1]
        beta_next = beta_prev + eps * (x.T @ y - x.T @ x @ beta_prev) / x.shape[0]
        beta_iterates.append(beta_next)
    return beta_iterates[-1]


def train_bias_estimator(
    *,
    predictive_model: LinearRegression,
    x: torch.Tensor,
    y: torch.Tensor,
    max_epochs: int,
    patience: int,
) -> torch.Tensor:
    datamodule = FullBatchDataModule(x, y, num_workers=0)
    bias_estimator = BiasWithMSE(
        predictive_model=predictive_model,
        bias_model=DiagMatrixRidgeBias(dim=x.shape[1]),
        grad_match_loss_fn=torch.nn.functional.mse_loss,
        lr=1e-2,
        optimizer_cls=torch.optim.Adam,
    )
    trainer = Trainer(
        max_epochs=max_epochs,
        accumulate_grad_batches=1,
        log_every_n_steps=1,
        logger=False,
        callbacks=[
            EarlyStopping(monitor="train_bias/loss", patience=patience, mode="min"),
        ],
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(bias_estimator, datamodule)
    return torch.diag(bias_estimator.bias_model.Q.detach())


def generate_payload(args: argparse.Namespace) -> dict[str, Any]:
    x, y, betas = make_synthetic_problem(
        seed=args.seed,
        input_dim=args.input_dim,
        num_samples=args.num_samples,
        noise_std=args.noise_std,
    )
    predictive_model, stop_epoch = train_predictive_model(
        x=x,
        y=y,
        input_dim=args.input_dim,
        eps=args.eps,
        max_epochs=args.max_epochs_train,
        patience=args.train_patience,
    )
    LOGGER.info("Predictive model early-stopped at epoch %d", stop_epoch)
    theta_ols = torch.linalg.lstsq(x, y.unsqueeze(1)).solution.squeeze()
    LOGGER.info(
        "Empirical OLS sampling error ||theta_ols - beta|| = %.3e (noise_std=%.3g)",
        float((theta_ols - betas).norm()),
        args.noise_std,
    )

    beta_from_iterates = compute_beta_iterates(
        x=x,
        y=y,
        input_dim=args.input_dim,
        eps=args.eps,
        num_steps=stop_epoch,
    )
    model_beta = predictive_model.linear.weight.detach()[0]
    if not torch.allclose(beta_from_iterates, model_beta, atol=1e-5):
        raise RuntimeError(
            "Explicit OLS iterate does not match the trained model weights."
        )

    q_matrix = compute_Q_matrix(x, stop_epoch, args.eps)
    q_hat = train_bias_estimator(
        predictive_model=predictive_model,
        x=x,
        y=y,
        max_epochs=args.max_epochs_bias,
        patience=args.bias_patience,
    )
    theta_real = betas.detach().cpu().numpy().round(5)
    theta_hat = model_beta.detach().cpu().numpy().round(5)
    theta_hat_prime = (
        compute_beta_closed_form(x, y, q_hat).detach().cpu().numpy().round(5)
    )

    return {
        "config": {
            "seed": args.seed,
            "input_dim": args.input_dim,
            "num_samples": args.num_samples,
            "noise_std": args.noise_std,
            "eps": args.eps,
            "stop_epoch": stop_epoch,
        },
        "Q": q_matrix.detach().cpu(),
        "q_hat_diag": q_hat.detach().cpu(),
        "theta_real": torch.from_numpy(theta_real),
        "theta_hat": torch.from_numpy(theta_hat),
        "theta_hat_prime": torch.from_numpy(theta_hat_prime),
        "X": x.detach().cpu(),
        "y": y.detach().cpu(),
    }


def main() -> None:
    configure_logging()
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(generate_payload(args), args.output)
    LOGGER.info("Saved %s", args.output)


if __name__ == "__main__":
    main()
