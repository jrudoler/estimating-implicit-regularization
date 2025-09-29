#!/usr/bin/env python3
"""Run kernel regression with an RBF kernel and bias estimation end-to-end."""

import argparse
import logging
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from accelerate.test_utils.testing import get_backend
from core.bias import DiagMatrixBiasKRR
from core.callbacks import WandBCallback
from core.data import FullBatchDataModule
from core.estimators import BiasWithMSE
from core.models import KernelRegression
from core.utils import is_psd, rbf_kernel_torch
from functools import partial
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kernel regression with RBF kernel and diagonal bias estimation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "figures" / "kernel_regression_rbf",
        help="Directory where plots will be written.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=REPO_ROOT / "logs" / "kernel_regression_rbf.log",
        help="File path for application logs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=56,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1000,
        help="Maximum epochs for the kernel regression trainer.",
    )
    parser.add_argument(
        "--bias-max-epochs",
        type=int,
        default=1200,
        help="Maximum epochs for the bias estimation trainer.",
    )
    return parser.parse_args()


def compute_kernel_ridge_Q_omega(
    K: torch.Tensor,
    eta: float,
    t: int,
    lambda_: float,
    alpha_init: torch.Tensor,
    device: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Q matrix and omega vector for kernel ridge regression."""
    if t <= 0:
        LOGGER.warning(
            "Received non-positive t=%s; using t=1 to avoid singular inverse.", t
        )
        t = 1
    if device is None:
        device = K.device
    K = K.to(device)
    alpha_init = alpha_init.to(device)
    dim = K.shape[0]
    I = torch.eye(dim, device=device)
    C = (1.0 / dim) * (K @ K) + lambda_ * K
    A = I - 2 * eta * C
    A_t = torch.linalg.matrix_power(A, t)
    Q_t = C @ (torch.linalg.inv(I - A_t) - I)
    omega_t = (C + Q_t) @ A_t @ alpha_init
    return Q_t, omega_t


def compute_alpha_closed_form(
    K: torch.Tensor,
    y: torch.Tensor,
    Q: torch.Tensor,
    omega: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Closed-form solution for the dual parameters alpha."""
    n = K.shape[0]
    C = (1 / n) * K @ K + lambda_ * K
    rhs = omega + (1 / n) * K @ y
    lhs = C + Q
    alpha = torch.linalg.solve(lhs, rhs)
    return alpha


def init_wandb_logger(name_suffix: str) -> Optional[WandbLogger]:
    run_name = f"kernel-regression-{name_suffix}"
    return WandbLogger(
        project="inductive-bias", name=run_name, log_model=False, save_dir="logs/"
    )


def save_scatter_plot(x: torch.Tensor, y: torch.Tensor, path: Path) -> None:
    fig, ax = plt.subplots()
    scatter = ax.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", alpha=0.8)
    fig.colorbar(scatter, ax=ax)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Radial Sine Function Data")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved radial sine scatter plot to %s", path)


def save_residual_plot(
    x_feature: torch.Tensor, residuals: dict[str, torch.Tensor], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for label, values in residuals.items():
        ax.plot(x_feature, values, "o", alpha=0.4, label=label)
    ax.legend()
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Residual")
    ax.set_title("Residual Comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    LOGGER.info("Saved residual comparison plot to %s", path)


def configure_logging(log_file: Path) -> None:
    log_handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=log_handlers,
    )


def apply_plot_style() -> None:
    style_path = REPO_ROOT / "clean_fig.mplstyle"
    if style_path.exists():
        plt.style.use(str(style_path))
        LOGGER.info("Loaded matplotlib style from %s", style_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_file)
    LOGGER.info("Starting kernel regression with arguments: %s", args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_plot_style()

    torch.manual_seed(args.seed)
    device, n_devices, _ = get_backend()
    torch.set_float32_matmul_precision("highest")

    N = 500
    p = 2
    X = torch.randn(N, p)
    y = torch.sin(torch.norm(X, p=1, dim=1)) * torch.norm(
        X, p=2, dim=1
    ) + 0.5 * torch.randn(N)
    dm_sin = FullBatchDataModule(X, y, num_workers=3)

    scatter_path = output_dir / "radial_sine_function_data_2d.png"
    save_scatter_plot(X, y, scatter_path)

    eta = 1e-2
    lambda_ = 0.1
    gamma = 100.0
    K = rbf_kernel_torch(X=X, Y=X, gamma=gamma, eps=1e-8, enforce_psd=False)
    assert is_psd(K, tol=1e-8), "Kernel is not positive semidefinite!"

    eigenvalues = torch.linalg.eigvalsh(K @ K / N + lambda_ * K)
    max_eigenvalue = eigenvalues.max().item()
    min_eigenvalue = eigenvalues.min().item()
    LOGGER.info("Max eigenvalue of C: %.10f", max_eigenvalue)
    LOGGER.info("Min eigenvalue of C: %.10f", min_eigenvalue)
    LOGGER.info("Max learning rate: %.10f", 1.0 / max_eigenvalue)
    if eta >= 1.0 / max_eigenvalue:
        raise ValueError("Learning rate is too high for convergence.")

    rbf_kernel_reg = KernelRegression(
        kernel=partial(rbf_kernel_torch, gamma=gamma, eps=1e-8, enforce_psd=False),
        n_train_samples=N,
        fit_intercept=False,
        ridge_lambda=lambda_,
        init_zeros=False,
        lr=eta,
    )

    use_wandb = not args.no_wandb
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    callbacks = [EarlyStopping(monitor="train/loss", patience=50, mode="min")]
    if use_wandb:
        callbacks.append(WandBCallback())
        logger = init_wandb_logger(f"kernel-{timestamp}")
    else:
        logger = None

    rbf_trainer = Trainer(
        max_epochs=args.max_epochs,
        accumulate_grad_batches=1,
        log_every_n_steps=1,
        logger=logger,
        callbacks=callbacks,
        accelerator=device,
        devices=n_devices,
    )
    rbf_trainer.fit(rbf_kernel_reg, dm_sin)
    t = max(rbf_trainer.current_epoch, 1)
    if rbf_trainer.current_epoch <= 0:
        LOGGER.warning(
            "Trainer reported epoch %s; using t=%s to avoid singular matrix inversion.",
            rbf_trainer.current_epoch,
            t,
        )

    alpha = rbf_kernel_reg.kernel_linear.weight.detach().view(-1)

    alpha_init = rbf_kernel_reg.init_weights.view(-1).detach().clone()

    bias_callbacks = [EarlyStopping(monitor="train/loss", patience=50, mode="min")]
    bias_logger = None
    if use_wandb:
        bias_callbacks.append(WandBCallback())
        bias_logger = init_wandb_logger(f"bias-{timestamp}")

    rbf_estimator = BiasWithMSE(
        predictive_model=rbf_kernel_reg,
        bias_model=DiagMatrixBiasKRR(
            Q_t_init=None,
            dim=N,
            t=t,
            K=K,
            alpha_init=alpha_init,
            eta=eta,
            lambda_=lambda_,
        ),
        grad_match_loss_fn=torch.nn.functional.mse_loss,
        lr=1e-2,
        optimizer_cls=torch.optim.Adam,
    )

    bias_trainer = Trainer(
        max_epochs=args.bias_max_epochs,
        accumulate_grad_batches=1,
        log_every_n_steps=5,
        logger=bias_logger,
        callbacks=bias_callbacks,
        accelerator=device,
        devices=n_devices,
    )
    bias_trainer.fit(rbf_estimator, dm_sin)

    Q_kernel_ridge, omega_kernel_ridge = compute_kernel_ridge_Q_omega(
        K, eta=eta, t=t, lambda_=lambda_, alpha_init=alpha_init, device=device
    )
    Q_kernel_ridge = Q_kernel_ridge.detach().cpu().clone()
    omega_kernel_ridge = omega_kernel_ridge.detach().cpu().clone()

    Q_hat_kernel_ridge = torch.diag(rbf_estimator.bias_model.Q_t.detach())
    LOGGER.info("Estimated Q from kernel ridge bias:\n%s", Q_hat_kernel_ridge)
    LOGGER.info("Theoretical Q from kernel ridge regression:\n%s", Q_kernel_ridge)

    theory_alpha = compute_alpha_closed_form(
        K=K, y=y, Q=Q_kernel_ridge, omega=omega_kernel_ridge, lambda_=lambda_
    )
    estimated_alpha = compute_alpha_closed_form(
        K=K, y=y, Q=Q_hat_kernel_ridge, omega=omega_kernel_ridge, lambda_=lambda_
    )
    LOGGER.info("Theoretical alpha from Q bias (first 25):\n%s", theory_alpha[:25])
    LOGGER.info("Estimated alpha from Q bias (first 25):\n%s", estimated_alpha[:25])
    LOGGER.info("Ground truth alpha (first 25):\n%s", alpha[:25])

    theory_pred = theory_alpha @ K
    estimated_pred = estimated_alpha @ K
    true_pred = alpha @ K

    theory_resid = y - theory_pred
    estimated_resid = y - estimated_pred
    true_resid = y - true_pred

    residual_plot_path = output_dir / "residual_comparison.png"
    save_residual_plot(
        X[:, 0],
        {
            "Theoretical Residuals": theory_resid,
            "Estimated Residuals": estimated_resid,
            "True Residuals": true_resid,
        },
        residual_plot_path,
    )

    from sklearn.metrics import mean_squared_error

    theory_rmse = mean_squared_error(y, theory_pred)
    estimated_rmse = mean_squared_error(y, estimated_pred)
    true_rmse = mean_squared_error(y, true_pred)
    LOGGER.info("Theoretical RMSE: %.4f", theory_rmse)
    LOGGER.info("Estimated RMSE: %.4f", estimated_rmse)
    LOGGER.info("True RMSE: %.4f", true_rmse)

    LOGGER.info(
        "Is the estimated Q positive semidefinite? %s",
        is_psd(Q_hat_kernel_ridge),
    )
    LOGGER.info(
        "Is the theoretical Q positive semidefinite? %s",
        is_psd(Q_kernel_ridge),
    )


if __name__ == "__main__":
    main()
