#!/usr/bin/env python3
"""Synthetic evaluation for the GradientSquaredPenaltyEstimator."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.func import functional_call

from lightning.pytorch import Trainer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.bias import GradientSquaredPenaltyScale
from core.estimators import GradientSquaredPenaltyEstimator


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=int, default=512, help="Number of synthetic samples."
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=1,
        help="Input dimensionality for the predictive model.",
    )
    parser.add_argument(
        "--true-weight",
        type=float,
        default=2.0,
        help="Ground-truth linear weight used to generate labels.",
    )
    parser.add_argument(
        "--model-weight",
        type=float,
        default=0.5,
        help="Initial predictive model weight (per feature).",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.05,
        help="Standard deviation of Gaussian label noise.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for the estimator training loop.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=200,
        help="Number of estimator training epochs.",
    )
    parser.add_argument(
        "--predictive-steps",
        type=int,
        default=100,
        help="Gradient descent steps for training the predictive model.",
    )
    parser.add_argument(
        "--bias-lr",
        type=float,
        default=5e-2,
        help="Learning rate for the bias estimator.",
    )
    parser.add_argument(
        "--lambda-init",
        type=float,
        default=-0.1,
        help="Initial value for lambda in the bias model.",
    )
    parser.add_argument(
        "--enforce-positive",
        action="store_true",
        help="Force lambda to stay positive via softplus reparameterisation.",
    )
    parser.add_argument(
        "--seed", type=int, default=17, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save experiment metrics (*.pt).",
    )
    parser.add_argument(
        "--gd-step-size",
        type=float,
        default=0.05,
        help="Gradient descent step size h assumed by the theoretical model.",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="Python logging level."
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def make_dataset(
    n_samples: int,
    input_dim: int,
    true_weight: float,
    noise_std: float,
    seed: int,
) -> TensorDataset:
    """
    Generate a synthetic regression dataset.
    Model: y = Xw + N(0, noise_std^2)

    Args:
        n_samples: Number of samples to generate.
        input_dim: Dimensionality of the input features.
        true_weight: Ground-truth weight for the linear model.
        noise_std: Standard deviation of Gaussian noise added to targets.
        seed: Random seed for reproducibility.
    Returns:
        A TensorDataset containing features and targets.
    """
    gen = torch.Generator().manual_seed(seed)
    features = torch.randn(n_samples, input_dim, generator=gen)
    weight_vec = torch.full((input_dim, 1), true_weight)
    targets = features @ weight_vec
    if noise_std > 0:
        noise = torch.randn(targets.shape, generator=gen)
        targets = targets + noise_std * noise
    # print description of the dataset
    LOGGER.info(
        "Generated dataset with %d samples, input_dim=%d, true_weight=%.2f, noise_std=%.4f",
        n_samples,
        input_dim,
        true_weight,
        noise_std,
    )
    return TensorDataset(features, targets)


def initialise_predictive_model(input_dim: int, weight_value: float) -> nn.Module:
    model = nn.Linear(input_dim, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(weight_value)
    return model


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat([param.detach().reshape(-1) for param in model.parameters()])


def compute_grad_and_hvp(
    predictive_model: nn.Module,
    loss_fn: nn.Module,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    param_clones = OrderedDict(
        (
            name,
            param.clone().detach().requires_grad_(True),
        )
        for name, param in predictive_model.named_parameters()
    )

    predictions = functional_call(predictive_model, param_clones, (features,))
    loss = loss_fn(predictions, targets)

    clone_params_tuple = tuple(param_clones.values())
    grads = torch.autograd.grad(loss, clone_params_tuple, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    hvp_tensors = torch.autograd.grad(
        grads,
        clone_params_tuple,
        grad_outputs=grads,
        retain_graph=True,
    )
    flat_hvp = torch.cat([h.reshape(-1) for h in hvp_tensors])
    return flat_grad.detach(), flat_hvp.detach()


def collect_gradients(
    predictive_model: nn.Module,
    loss_fn: nn.Module,
    dataset: TensorDataset,
) -> Tuple[torch.Tensor, torch.Tensor]:
    features, targets = dataset.tensors
    flat_grad, flat_hvp = compute_grad_and_hvp(
        predictive_model=predictive_model,
        loss_fn=loss_fn,
        features=features,
        targets=targets,
    )
    return flat_grad, flat_hvp


def train_predictive_model(
    model: nn.Module,
    dataset: TensorDataset,
    step_size: float,
    num_steps: int,
) -> dict:
    features, targets = dataset.tensors
    optimizer = torch.optim.SGD(model.parameters(), lr=step_size)
    loss_fn = nn.MSELoss(reduction="mean")

    trajectory: dict[str, list[torch.Tensor]] = {
        "grads": [],
        "hvps": [],
        "deltas": [],
        "residuals": [],
    }

    for step in range(num_steps):
        model.train()
        grad_vec, hvp_vec = compute_grad_and_hvp(
            predictive_model=model,
            loss_fn=loss_fn,
            features=features,
            targets=targets,
        )
        flat_before = flatten_parameters(model)

        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_fn(predictions, targets)
        loss_value = loss.item()
        loss.backward()
        optimizer.step()

        flat_after = flatten_parameters(model)
        delta = flat_after - flat_before
        residual = (-delta / step_size) - grad_vec

        trajectory["grads"].append(grad_vec)
        trajectory["hvps"].append(hvp_vec)
        trajectory["deltas"].append(delta)
        trajectory["residuals"].append(residual)

        if step == 0 or (step + 1) % max(num_steps // 5, 1) == 0:
            LOGGER.info(
                "Predictive GD step %d/%d | loss=%.6f",
                step + 1,
                num_steps,
                loss_value,
            )

    model.eval()
    model.requires_grad_(False)
    return trajectory


def train_estimator(
    estimator: GradientSquaredPenaltyEstimator,
    dataloader: DataLoader,
    max_epochs: int,
) -> None:
    if torch.cuda.is_available():
        accelerator, devices = "gpu", 1
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        accelerator, devices = "mps", 1
    else:
        accelerator, devices = "cpu", 1
    LOGGER.info("Training estimator on %s", accelerator.upper())
    trainer = Trainer(
        max_epochs=max_epochs,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        accelerator=accelerator,
        devices=devices,
    )
    trainer.fit(estimator, train_dataloaders=dataloader)


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    torch.manual_seed(args.seed)

    dataset = make_dataset(
        n_samples=args.samples,
        input_dim=args.input_dim,
        true_weight=args.true_weight,
        noise_std=args.noise_std,
        seed=args.seed,
    )

    predictive_model = initialise_predictive_model(
        input_dim=args.input_dim,
        weight_value=args.model_weight,
    )
    LOGGER.info(
        "Initial predictive weights: %s", predictive_model.weight.detach().view(-1)
    )

    LOGGER.info(
        "Training predictive model with step size %.5f for %d steps",
        args.gd_step_size,
        args.predictive_steps,
    )
    trajectory_stats = train_predictive_model(
        model=predictive_model,
        dataset=dataset,
        step_size=args.gd_step_size,
        num_steps=args.predictive_steps,
    )
    LOGGER.info(
        "Trained predictive weights: %s", predictive_model.weight.detach().view(-1)
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loss_fn = nn.MSELoss(reduction="mean")

    flat_grad, flat_hvp = collect_gradients(
        predictive_model=predictive_model,
        loss_fn=loss_fn,
        dataset=dataset,
    )

    num_params = sum(p.numel() for p in predictive_model.parameters())
    LOGGER.info("Number of parameters in the predictive model: %d", num_params)
    theoretical_lambda = args.gd_step_size * num_params / 4.0
    LOGGER.info(
        "Theoretical lambda (h m / 4) with h=%.5f, m=%d: %.6f",
        args.gd_step_size,
        num_params,
        theoretical_lambda,
    )

    bias_model = GradientSquaredPenaltyScale(
        lambda_init=args.lambda_init,
        enforce_positive=args.enforce_positive,
    )

    estimator = GradientSquaredPenaltyEstimator(
        predictive_model=predictive_model,
        predictive_loss_fn=loss_fn,
        bias_model=bias_model,
        lr=args.bias_lr,
        gd_step_size=args.gd_step_size,
    )

    train_estimator(estimator, dataloader, args.max_epochs)

    with torch.no_grad():
        learned_lambda = estimator.bias_model().detach().cpu().item()
    LOGGER.info("Learned lambda: %.6f", learned_lambda)
    LOGGER.info(
        "Lambda comparison | gradient-matching: %.6f | theory: %.6f",
        learned_lambda,
        theoretical_lambda,
    )

    hvp_norm = flat_hvp.norm().item()
    grad_norm = flat_grad.norm().item()
    LOGGER.info(
        "Gradient norm: %.6f | Hessian-gradient norm: %.6f", grad_norm, hvp_norm
    )

    rel_error = abs(learned_lambda - theoretical_lambda) / (
        abs(theoretical_lambda) + 1e-12
    )
    LOGGER.info("Relative error: %.4f%%", rel_error * 100.0)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lambda_theoretical": torch.tensor([theoretical_lambda]),
            "lambda_estimated": torch.tensor([learned_lambda]),
            "gradient": flat_grad,
            "hessian_grad": flat_hvp,
            "relative_error": torch.tensor([rel_error]),
            "config": vars(args),
        }
        torch.save(payload, args.save)
        LOGGER.info("Saved metrics to %s", args.save)


if __name__ == "__main__":
    main()
