#!/usr/bin/env python3
"""Regenerate the OLS early-stopping component figures from notebooks/linear-regression.ipynb logic."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

from core.bias import DiagMatrixRidgeBias
from core.data import FullBatchDataModule
from core.estimators import BiasWithMSE
from core.models import LinearRegression
from core.utils import compute_Q_matrix, compute_beta_closed_form


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_FIGURES_DIR = REPO_ROOT / "paper" / "figures"
STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
DEFAULT_LAMBDA_OUT = PAPER_FIGURES_DIR / "Lambda_comparison-ols.pdf"
DEFAULT_WEIGHTS_OUT = PAPER_FIGURES_DIR / "predictive_weights_comparison_ols.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--lambda-out",
        type=Path,
        default=DEFAULT_LAMBDA_OUT,
        help="Output path for the learned-vs-theoretical Lambda heatmap.",
    )
    parser.add_argument(
        "--weights-out",
        type=Path,
        default=DEFAULT_WEIGHTS_OUT,
        help="Output path for the predictive-weights comparison heatmap.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=56,
        help="Random seed matching notebooks/linear-regression.ipynb.",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=10,
        help="Linear regression feature dimension.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Synthetic sample count.",
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


def configure_plot_style() -> None:
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))


def make_synthetic_problem(
    *,
    seed: int,
    input_dim: int,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("highest")

    x = torch.randn(num_samples, input_dim)
    betas = 3 * torch.randn(input_dim)
    y = x @ betas
    return x, y, betas


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


def save_lambda_comparison(q: torch.Tensor, q_hat: torch.Tensor, out_path: Path) -> None:
    q_cpu = q.detach().cpu()
    qhat_cpu = q_hat.detach().cpu()
    vmax = max(q_cpu.abs().max().item(), qhat_cpu.abs().max().item())
    vmin = -vmax

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sns.heatmap(
        qhat_cpu.numpy(),
        ax=axes[0],
        cmap="bwr",
        norm=norm,
        square=True,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )
    axes[0].set_title(r"Estimated $\hat{\Lambda}$")
    axes[0].set_xlabel(r"$p$")
    axes[0].set_ylabel(r"$p$")

    sns.heatmap(
        q_cpu.numpy(),
        ax=axes[1],
        cmap="bwr",
        norm=norm,
        square=True,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )
    axes[1].set_title(r"Theoretical $\Lambda$")
    axes[1].set_xlabel(r"$p$")
    axes[1].set_ylabel("")

    for ax in axes:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

    scalar_mappable = plt.cm.ScalarMappable(cmap="bwr", norm=norm)
    scalar_mappable.set_array([])
    fig.colorbar(scalar_mappable, ax=axes, location="right", pad=0.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", out_path)


def save_predictive_weights_comparison(
    *,
    theta_real: np.ndarray,
    theta_hat: np.ndarray,
    theta_hat_prime: np.ndarray,
    out_path: Path,
) -> None:
    vmax = max(
        np.abs(theta_real).max(),
        np.abs(theta_hat).max(),
        np.abs(theta_hat_prime).max(),
    )
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(2, 6))
    for ax, theta, title in zip(
        axes.flatten(),
        [theta_hat_prime, theta_hat, theta_real],
        [r"$\hat{\theta}_{\Lambda}$", r"$\hat{\theta}$", r"$\theta$"],
        strict=True,
    ):
        sns.heatmap(
            theta.reshape(-1, 1),
            cmap="bwr",
            norm=norm,
            square=True,
            cbar=False,
            yticklabels=False,
            xticklabels=False,
            ax=ax,
            linewidths=0.5,
            linecolor="k",
        )
        ax.set_title(title, fontsize=20)
    axes[0].set_ylabel(r"$p$", fontsize=16)

    scalar_mappable = plt.cm.ScalarMappable(cmap="bwr", norm=norm)
    scalar_mappable.set_array([])
    plt.colorbar(
        scalar_mappable,
        ax=list(axes),
        orientation="horizontal",
        fraction=0.02,
        pad=0.05,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", out_path)


def main() -> None:
    configure_logging()
    configure_plot_style()
    args = parse_args()

    x, y, betas = make_synthetic_problem(
        seed=args.seed,
        input_dim=args.input_dim,
        num_samples=args.num_samples,
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

    q = compute_Q_matrix(x, stop_epoch, args.eps)
    q_hat = train_bias_estimator(
        predictive_model=predictive_model,
        x=x,
        y=y,
        max_epochs=args.max_epochs_bias,
        patience=args.bias_patience,
    )

    save_lambda_comparison(q, q_hat, args.lambda_out)

    theta_real = betas.detach().cpu().numpy().round(5)
    theta_hat = model_beta.detach().cpu().numpy().round(5)
    theta_hat_prime = (
        compute_beta_closed_form(x, y, q_hat).detach().cpu().numpy().round(5)
    )
    save_predictive_weights_comparison(
        theta_real=theta_real,
        theta_hat=theta_hat,
        theta_hat_prime=theta_hat_prime,
        out_path=args.weights_out,
    )


if __name__ == "__main__":
    main()
