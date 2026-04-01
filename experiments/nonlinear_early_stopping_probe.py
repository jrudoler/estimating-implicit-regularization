#!/usr/bin/env python3
"""Probe implicit bias of early stopping in nonlinear nets.

Trains a depth-2 ReLU net on synthetic regression data WITHOUT any explicit
regularization, checkpoints at multiple training epochs, and at each checkpoint
fits:
  1. A scalar L2 penalty (closed-form least squares)
  2. A power-family penalty (λ, p) via optimization

The OLS theory (Ali et al. 2019) predicts that early stopping acts as L2
regularization whose strength λ decreases over training.  This experiment
tests whether that qualitative prediction extends to nonlinear nets.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
from pathlib import Path
import sys

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.bias import SmoothedPowerBias  # noqa: E402
from core.estimators import vector_to_parameter_views  # noqa: E402

try:
    from experiments.function_class_identifiability import (
        DeepReLURegressor,
        generate_dataset,
        set_seed,
    )
except ModuleNotFoundError:
    from function_class_identifiability import (  # type: ignore
        DeepReLURegressor,
        generate_dataset,
        set_seed,
    )

STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
if STYLE_PATH.exists():
    plt.style.use(str(STYLE_PATH))

LOGGER = logging.getLogger(__name__)


class UnregularizedReLURegressor(DeepReLURegressor):
    """DeepReLURegressor with no val-loss LR scheduler (for training without val set)."""

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.hparams.lr)


def flatten_model_parameters(model: torch.nn.Module) -> Tensor:
    return torch.nn.utils.parameters_to_vector(list(model.parameters())).detach().cpu()


def compute_target_gradient(model: torch.nn.Module, dataset: TensorDataset) -> Tensor:
    """Compute -∇L at current weights (the gradient the regularizer should match)."""
    device = next(model.parameters()).device
    model.eval()
    inputs, targets = dataset.tensors[:2]
    predictions = model(inputs.to(device))
    data_loss = F.mse_loss(predictions, targets.to(device))
    loss_grads = torch.autograd.grad(data_loss, tuple(model.parameters()), create_graph=False)
    target_gradient = -torch.cat([g.reshape(-1) for g in loss_grads]).detach().cpu()
    return torch.nan_to_num(target_gradient, nan=0.0, posinf=0.0, neginf=0.0)


def fit_scalar_l2_closed_form(solution: Tensor, target_gradient: Tensor) -> dict:
    """Closed-form scalar L2: target ≈ λ * 2 * θ  →  λ = (θ · target) / (2 * θ · θ)."""
    theta = solution.detach()
    target = target_gradient.detach()
    theta_dot_target = float(torch.dot(theta, target).item())
    theta_dot_theta = float(torch.dot(theta, theta).item())
    if theta_dot_theta < 1e-12:
        return {"lambda": 0.0, "cosine": 0.0, "residual": 1.0}
    lam = theta_dot_target / (2.0 * theta_dot_theta)
    predicted = 2.0 * lam * theta
    residual = float((target - predicted).norm().item() / max(target.norm().item(), 1e-8))
    cosine = float(
        torch.dot(predicted, target).item()
        / max(predicted.norm().item() * target.norm().item(), 1e-12)
    )
    return {"lambda": lam, "cosine": cosine, "residual": residual}


def fit_power_family(
    solution: Tensor,
    target_gradient: Tensor,
    lr: float = 0.01,
    max_epochs: int = 2000,
    patience: int = 200,
) -> dict:
    """Fit (λ, p) for R(θ) = λ Σ (θ_j² + ε)^(p/2) via Adam."""
    bias = SmoothedPowerBias(
        scale_init=0.01,
        exponent_init=1.8,
        epsilon=1e-6,
        trainable_scale=True,
        trainable_exponent=True,
        p_min=0.5,
        p_max=4.0,
    )
    optimizer = torch.optim.Adam(bias.parameters(), lr=lr)
    target = target_gradient.detach()
    theta = solution.detach().requires_grad_(False)

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    for _ in range(max_epochs):
        optimizer.zero_grad(set_to_none=True)
        predicted = bias.penalty_gradient(theta)
        loss = F.mse_loss(predicted, target)
        loss.backward()
        optimizer.step()
        lv = float(loss.item())
        if lv + 1e-12 < best_loss:
            best_loss = lv
            best_state = {k: v.detach().clone() for k, v in bias.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        bias.load_state_dict(best_state)

    with torch.no_grad():
        predicted = bias.penalty_gradient(theta)
        residual = float((target - predicted).norm().item() / max(target.norm().item(), 1e-8))
        denom = float(predicted.norm().item() * target.norm().item())
        cosine = float(torch.dot(predicted, target).item() / max(denom, 1e-12))
    params = bias.get_bias_params()
    return {
        "lambda": params["scale"],
        "p": params["exponent"],
        "cosine": cosine,
        "residual": residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=12)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[10, 25, 50, 100, 200, 500, 1000, 2000],
    )
    parser.add_argument("--output-dir", type=Path, default=Path("logs/phase2/nonlinear_early_stopping_probe"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures/nonlinear_early_stopping_probe"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    full_dataset, _metadata = generate_dataset(
        n_samples=args.n_samples,
        input_dim=args.input_dim,
        function_class="teacher_relu",
        noise_std=0.0,
        seed=args.seed,
        data_mode="structured",
        input_spectrum="identity",
        input_rank=0,
        input_spectrum_decay=1.5,
        teacher_spectrum="identity",
        teacher_rank=0,
        teacher_spectrum_decay=1.5,
        teacher_activation="relu",
    )
    X, y = full_dataset.tensors[:2]
    if y.ndim == 1:
        y = y.unsqueeze(1)

    n_train = int(0.8 * len(X))
    train_dataset = TensorDataset(X[:n_train], y[:n_train])
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

    max_epoch = max(args.checkpoints)
    LOGGER.info("Training for %d epochs total, checkpointing at %s", max_epoch, args.checkpoints)

    # Train model WITHOUT regularization
    model = UnregularizedReLURegressor(
        input_dim=args.input_dim,
        depth=args.depth,
        width=args.width,
        lr=args.lr,
        explicit_bias_types=[],
        explicit_lambdas=[],
    )
    trainer = pl.Trainer(
        max_epochs=max_epoch,
        logger=False,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        log_every_n_steps=25,
    )

    # We need intermediate checkpoints, so we train in increments
    rows = []
    current_epoch = 0
    for target_epoch in sorted(args.checkpoints):
        additional = target_epoch - current_epoch
        if additional <= 0:
            continue
        trainer.fit_loop.max_epochs = target_epoch
        trainer.fit(model, train_dataloaders=train_loader)
        current_epoch = target_epoch

        solution = flatten_model_parameters(model.network)
        target_grad = compute_target_gradient(model.network, train_dataset)

        # Fit scalar L2
        l2_result = fit_scalar_l2_closed_form(solution, target_grad)
        # Fit power family
        power_result = fit_power_family(solution, target_grad)

        # Training MSE
        with torch.no_grad():
            preds = model(train_dataset.tensors[0])
            train_mse = float(F.mse_loss(preds, train_dataset.tensors[1]).item())

        row = {
            "epoch": target_epoch,
            "train_mse": train_mse,
            "l2_lambda": l2_result["lambda"],
            "l2_cosine": l2_result["cosine"],
            "l2_residual": l2_result["residual"],
            "power_lambda": power_result["lambda"],
            "power_p": power_result["p"],
            "power_cosine": power_result["cosine"],
            "power_residual": power_result["residual"],
        }
        rows.append(row)
        LOGGER.info(
            "epoch=%d  train_mse=%.4f  l2_lambda=%.6f  l2_cos=%.3f  power_lambda=%.6f  power_p=%.3f  power_cos=%.3f",
            target_epoch, train_mse,
            l2_result["lambda"], l2_result["cosine"],
            power_result["lambda"], power_result["p"], power_result["cosine"],
        )

    # Write CSV
    csv_path = args.output_dir / "early_stopping_probe.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %s", csv_path)

    # Plot
    epochs = [r["epoch"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    # Panel 1: L2 lambda vs epoch
    axes[0].plot(epochs, [r["l2_lambda"] for r in rows], "o-", color="#4C72B0", label="scalar L2")
    axes[0].plot(epochs, [r["power_lambda"] for r in rows], "s--", color="#DD8452", label="power family")
    axes[0].set_xlabel("training epoch")
    axes[0].set_ylabel("estimated λ")
    axes[0].set_title("Regularization Strength")
    axes[0].legend(frameon=False)
    axes[0].set_xscale("log")

    # Panel 2: power family exponent vs epoch
    axes[1].plot(epochs, [r["power_p"] for r in rows], "s-", color="#DD8452")
    axes[1].axhline(2.0, color="gray", linestyle="--", linewidth=0.8, label="p = 2 (L2)")
    axes[1].set_xlabel("training epoch")
    axes[1].set_ylabel("estimated p")
    axes[1].set_title("Exponent (Power Family)")
    axes[1].legend(frameon=False)
    axes[1].set_xscale("log")

    # Panel 3: cosine similarity
    axes[2].plot(epochs, [r["l2_cosine"] for r in rows], "o-", color="#4C72B0", label="scalar L2")
    axes[2].plot(epochs, [r["power_cosine"] for r in rows], "s--", color="#DD8452", label="power family")
    axes[2].set_xlabel("training epoch")
    axes[2].set_ylabel("gradient cosine")
    axes[2].set_title("Fit Quality")
    axes[2].legend(frameon=False)
    axes[2].set_xscale("log")

    fig.tight_layout()
    fig_path = args.figure_dir / "early_stopping_probe.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Wrote %s", fig_path)


if __name__ == "__main__":
    main()
