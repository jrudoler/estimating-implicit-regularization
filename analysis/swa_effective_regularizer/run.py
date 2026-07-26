"""Identify candidate effective regularizers for SWA on a small MNIST MLP.

The experiment separates the two changes commonly bundled together as SWA:

1. learning-rate schedule: a standard decayed-SGD tail versus a constant-LR tail;
2. weight averaging: the last iterate versus the average of that same constant-LR
   trajectory.

All three endpoints branch from the same burn-in checkpoint, and both tail
branches see identical minibatch orders.  At every endpoint we apply the paper's
closed-form scalar gradient-matching estimator.  Because averaging is a
post-processing map rather than an additive force during optimization, we also
fit candidate regularizer gradients to the *ablation displacement*

    theta_treatment - theta_control ~= -a grad P(theta_control).

The displacement cosine is scale-free; unlike the fitted ``a``, it does not
require inventing an "effective step size" for SWA.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.swa_utils import AveragedModel

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from core.bias import (  # noqa: E402
    NuclearNormBias,
    RidgeBias,
    SpectralEntropyBias,
    SpectralGapBias,
    StableRankBias,
)
from core.estimators import vector_to_parameter_views  # noqa: E402
from core.models import DeepReLUClassifier  # noqa: E402

LOGGER = logging.getLogger(__name__)

FAMILIES: dict[str, type[nn.Module]] = {
    "ridge": RidgeBias,
    "nuclear_norm": NuclearNormBias,
    "stable_rank": StableRankBias,
    "spectral_entropy": SpectralEntropyBias,
    "spectral_gap": SpectralGapBias,
}


def _load_matched_module() -> Any:
    """Reuse the exact in-memory MNIST split and loader from the dropout audit."""
    path = REPO_ROOT / "analysis" / "closed_form_dropout_matched" / "run.py"
    spec = importlib.util.spec_from_file_location("cf_dropout_matched_run", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MATCHED = _load_matched_module()
load_mnist_in_memory = _MATCHED.load_mnist_in_memory
ChunkLoader = _MATCHED.ChunkLoader
full_batch_loss_grad = _MATCHED.full_batch_loss_grad


def parameter_vector(model: nn.Module) -> Tensor:
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach()


def make_model(args: argparse.Namespace, device: torch.device) -> DeepReLUClassifier:
    return DeepReLUClassifier(
        input_dim=784,
        num_classes=10,
        depth=args.depth,
        width=args.width,
        dropout=0.0,
        batchnorm=False,
        l2_lambda=0.0,
        lr=args.lr,
        momentum=args.momentum,
    ).to(device)


def epoch_order(
    n_train: int, seed: int, epoch: int, device: torch.device
) -> Tensor:
    """Deterministic epoch-specific order, shared exactly across tail branches."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed * 1_000_003 + epoch)
    return torch.randperm(n_train, generator=generator, device=device)


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x_train: Tensor,
    y_train: Tensor,
    order: Tensor,
    batch_size: int,
    *,
    proximal_center: Tensor | None = None,
    proximal_scale: float = 0.0,
) -> float:
    model.train()
    loss_sum = 0.0
    n_train = int(y_train.numel())
    for start in range(0, n_train, batch_size):
        idx = order[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_train[idx]), y_train[idx])
        if proximal_center is not None and proximal_scale > 0.0:
            flat = torch.nn.utils.parameters_to_vector(model.parameters())
            loss = loss + proximal_scale * (flat - proximal_center).pow(2).sum()
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.detach()) * int(idx.numel())
    return loss_sum / n_train


@torch.no_grad()
def training_metrics(
    model: nn.Module, x: Tensor, y: Tensor, batch_size: int
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    loss_sum = 0.0
    correct = 0
    for start in range(0, int(y.numel()), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model(xb)
        loss_sum += float(F.cross_entropy(logits, yb, reduction="sum"))
        correct += int((logits.argmax(dim=1) == yb).sum())
    if was_training:
        model.train()
    n = int(y.numel())
    return {"loss": loss_sum / n, "accuracy": correct / n}


def family_gradient(
    family_name: str,
    model: nn.Module,
    *,
    center: Tensor | None = None,
) -> Tensor:
    """Gradient of a unit-scale candidate penalty at the model parameters."""
    flat = parameter_vector(model).requires_grad_()
    if family_name == "burnin_centered_ridge":
        if center is None:
            raise ValueError("burnin_centered_ridge requires a center")
        # Match RidgeBias's convention P(theta)=||theta||^2.
        penalty = (flat - center.to(flat)).pow(2).sum()
    else:
        family_cls = FAMILIES[family_name]
        family = family_cls(enforce_positive=True, init_value=0.0).to(flat.device)
        structured = vector_to_parameter_views(flat, model)
        penalty = family(flat, structured)
    (gradient,) = torch.autograd.grad(penalty, flat)
    return gradient.detach()


def scalar_projection(gradient: Tensor, target: Tensor) -> dict[str, float | None]:
    """Exact 1-D least squares for ``scale * gradient ~= target``."""
    denom = float(gradient.square().sum())
    target_norm = float(target.norm())
    if denom == 0.0 or target_norm == 0.0:
        return {
            "scale_star": None,
            "scale_star_clamped": None,
            "grad_cosine": None,
            "projection_r2": None,
            "projection_r2_positive": None,
            "residual_ratio": None,
            "residual_ratio_clamped": None,
            "grad_p_norm": denom**0.5,
            "target_norm": target_norm,
        }
    scale = float(gradient @ target) / denom
    scale_positive = max(scale, 0.0)
    cosine = float(F.cosine_similarity(gradient, target, dim=0))
    return {
        "scale_star": scale,
        "scale_star_clamped": scale_positive,
        "grad_cosine": cosine,
        "projection_r2": cosine**2,
        "projection_r2_positive": cosine**2 if cosine > 0.0 else 0.0,
        "residual_ratio": float((scale * gradient - target).norm()) / target_norm,
        "residual_ratio_clamped": (
            float((scale_positive * gradient - target).norm()) / target_norm
        ),
        "grad_p_norm": denom**0.5,
        "target_norm": target_norm,
    }


def candidate_gradients(
    model: nn.Module, burnin_vector: Tensor
) -> dict[str, Tensor]:
    names = [*FAMILIES, "burnin_centered_ridge"]
    return {
        name: family_gradient(name, model, center=burnin_vector) for name in names
    }


def joint_projection(
    gradients: dict[str, Tensor], target: Tensor
) -> dict[str, Any]:
    """Unconstrained projection onto the span of all candidate gradients."""
    names = list(gradients)
    design = torch.stack([gradients[name] for name in names], dim=1)
    # Solve in float64: spectral candidates can differ greatly in scale.
    coef = torch.linalg.lstsq(design.double(), target.double()).solution
    fitted = design.double() @ coef
    target_double = target.double()
    target_norm = float(target_double.norm())
    residual_ratio = (
        float((fitted - target_double).norm()) / target_norm
        if target_norm > 0.0
        else None
    )
    cosine = (
        float(F.cosine_similarity(fitted, target_double, dim=0))
        if float(fitted.norm()) > 0.0 and target_norm > 0.0
        else None
    )
    return {
        "coefficients": {name: float(value) for name, value in zip(names, coef)},
        "residual_ratio": residual_ratio,
        "projection_r2": None if cosine is None else cosine**2,
        "fitted_target_cosine": cosine,
        "rank": int(torch.linalg.matrix_rank(design.double())),
    }


def endpoint_probe(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    x_train: Tensor,
    y_train: Tensor,
    batch_size: int,
    burnin_vector: Tensor,
) -> dict[str, Any]:
    model.eval()
    loss_gradient = full_batch_loss_grad(model, loader)
    theta = parameter_vector(model)
    gradients = candidate_gradients(model, burnin_vector)
    target = -loss_gradient
    return {
        "train": training_metrics(model, x_train, y_train, batch_size),
        "stationarity": {
            "full_batch_grad_norm": float(loss_gradient.norm()),
            "param_norm": float(theta.norm()),
            "relative_grad_norm": float(loss_gradient.norm() / theta.norm()),
        },
        "closed_form": {
            name: scalar_projection(gradient, target)
            for name, gradient in gradients.items()
        },
        "joint_unconstrained": joint_projection(gradients, target),
    }


def displacement_probe(
    source_model: nn.Module,
    target_model: nn.Module,
    burnin_vector: Tensor,
) -> dict[str, Any]:
    """Fit treatment displacement to negative candidate penalty gradients."""
    displacement = parameter_vector(target_model) - parameter_vector(source_model)
    gradients = candidate_gradients(source_model, burnin_vector)
    # A positive coefficient means delta ~= -a grad P, as under explicit
    # regularization.  Therefore the scalar-projection target is delta and the
    # design direction is -grad P.
    directions = {name: -gradient for name, gradient in gradients.items()}
    return {
        "displacement_norm": float(displacement.norm()),
        "relative_displacement_norm": float(
            displacement.norm() / parameter_vector(source_model).norm()
        ),
        "candidates": {
            name: scalar_projection(direction, displacement)
            for name, direction in directions.items()
        },
        "joint_unconstrained": joint_projection(directions, displacement),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


@torch.no_grad()
def recovery_metrics(
    recovered: nn.Module,
    target: nn.Module,
    burnin: nn.Module,
    x_train: Tensor,
    batch_size: int,
) -> dict[str, float]:
    recovered_vector = parameter_vector(recovered)
    target_vector = parameter_vector(target)
    burnin_vector = parameter_vector(burnin)
    target_effect = target_vector - burnin_vector
    recovered_effect = recovered_vector - burnin_vector
    distance = float((recovered_vector - target_vector).norm())
    target_effect_norm = float(target_effect.norm())

    agreement_count = 0
    logit_mse_sum = 0.0
    n = int(x_train.shape[0])
    recovered.eval()
    target.eval()
    for start in range(0, n, batch_size):
        xb = x_train[start : start + batch_size]
        recovered_logits = recovered(xb)
        target_logits = target(xb)
        agreement_count += int(
            (recovered_logits.argmax(dim=1) == target_logits.argmax(dim=1)).sum()
        )
        logit_mse_sum += float(
            F.mse_loss(recovered_logits, target_logits, reduction="sum")
        )

    return {
        "parameter_distance": distance,
        "parameter_distance_over_swa_effect": (
            distance / target_effect_norm if target_effect_norm > 0.0 else float("nan")
        ),
        "burnin_displacement_cosine": float(
            F.cosine_similarity(recovered_effect, target_effect, dim=0)
        ),
        "prediction_agreement": agreement_count / n,
        "logit_mse": logit_mse_sum / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--burnin-epochs", type=int, default=100)
    parser.add_argument("--tail-epochs", type=int, default=100)
    parser.add_argument("--swa-lr", type=float, default=0.005)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional debugging-only limit on the MNIST training split.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    data = load_mnist_in_memory(args.data_root, device)
    x_train = data["x_train"]
    y_train = data["y_train"]
    if args.train_limit is not None:
        x_train = x_train[: args.train_limit]
        y_train = y_train[: args.train_limit]
    n_train = int(y_train.numel())
    loader = ChunkLoader(x_train, y_train, args.batch_size)

    config = {
        "seed": args.seed,
        "depth": args.depth,
        "width": args.width,
        "lr": args.lr,
        "momentum": args.momentum,
        "batch_size": args.batch_size,
        "burnin_epochs": args.burnin_epochs,
        "tail_epochs": args.tail_epochs,
        "swa_lr": args.swa_lr,
        "train_size": n_train,
        "device": str(device),
        "param_count": None,
    }

    started = time.time()
    burnin = make_model(args, device)
    config["param_count"] = sum(p.numel() for p in burnin.parameters())
    optimizer = torch.optim.SGD(
        burnin.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 100, 200], gamma=0.1
    )

    LOGGER.info("Training shared burn-in for %d epochs", args.burnin_epochs)
    for epoch in range(1, args.burnin_epochs + 1):
        order = epoch_order(n_train, args.seed, epoch, device)
        loss = train_epoch(
            burnin, optimizer, x_train, y_train, order, args.batch_size
        )
        scheduler.step()
        if epoch == 1 or epoch % 20 == 0:
            LOGGER.info(
                "burn-in epoch=%d loss=%.6f next_lr=%.3g",
                epoch,
                loss,
                optimizer.param_groups[0]["lr"],
            )

    burnin_vector = parameter_vector(burnin)
    if args.checkpoint_dir is not None:
        save_checkpoint(
            args.checkpoint_dir / "burnin.pt",
            burnin,
            optimizer,
            args.burnin_epochs,
            config,
        )

    # Both branches inherit weights and momentum from exactly the same checkpoint.
    control = copy.deepcopy(burnin)
    high_lr = copy.deepcopy(burnin)
    control_optimizer = torch.optim.SGD(
        control.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0
    )
    high_lr_optimizer = torch.optim.SGD(
        high_lr.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0
    )
    control_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    high_lr_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for group in high_lr_optimizer.param_groups:
        group["lr"] = args.swa_lr

    averaged = AveragedModel(high_lr, use_buffers=False)
    LOGGER.info(
        "Training matched tails for %d epochs (control_lr=%.3g, swa_lr=%.3g)",
        args.tail_epochs,
        control_optimizer.param_groups[0]["lr"],
        high_lr_optimizer.param_groups[0]["lr"],
    )
    for tail_index in range(1, args.tail_epochs + 1):
        epoch = args.burnin_epochs + tail_index
        order = epoch_order(n_train, args.seed, epoch, device)
        control_loss = train_epoch(
            control, control_optimizer, x_train, y_train, order, args.batch_size
        )
        high_lr_loss = train_epoch(
            high_lr, high_lr_optimizer, x_train, y_train, order, args.batch_size
        )
        averaged.update_parameters(high_lr)
        if tail_index == 1 or tail_index % 20 == 0:
            LOGGER.info(
                "tail epoch=%d control_loss=%.6f high_lr_loss=%.6f averages=%d",
                epoch,
                control_loss,
                high_lr_loss,
                int(averaged.n_averaged),
            )

    swa = copy.deepcopy(high_lr)
    swa.load_state_dict(averaged.module.state_dict())
    total_epochs = args.burnin_epochs + args.tail_epochs
    if args.checkpoint_dir is not None:
        save_checkpoint(
            args.checkpoint_dir / "control_decayed_sgd.pt",
            control,
            control_optimizer,
            total_epochs,
            config,
        )
        save_checkpoint(
            args.checkpoint_dir / "high_lr_last_iterate.pt",
            high_lr,
            high_lr_optimizer,
            total_epochs,
            config,
        )
        save_checkpoint(
            args.checkpoint_dir / "swa_average.pt",
            swa,
            None,
            total_epochs,
            config,
        )

    LOGGER.info("Computing full-batch endpoint and displacement diagnostics")
    endpoints = {
        "burnin": endpoint_probe(
            burnin, loader, x_train, y_train, args.batch_size, burnin_vector
        ),
        "control_decayed_sgd": endpoint_probe(
            control, loader, x_train, y_train, args.batch_size, burnin_vector
        ),
        "high_lr_last_iterate": endpoint_probe(
            high_lr, loader, x_train, y_train, args.batch_size, burnin_vector
        ),
        "swa_average": endpoint_probe(
            swa, loader, x_train, y_train, args.batch_size, burnin_vector
        ),
    }
    proximal_scale = endpoints["swa_average"]["closed_form"][
        "burnin_centered_ridge"
    ]["scale_star_clamped"]
    if proximal_scale is None:
        raise RuntimeError("Could not estimate the checkpoint-centered ridge scale")
    LOGGER.info(
        "Retraining from burn-in with fitted proximal scale %.8g", proximal_scale
    )
    explicit_proximal = copy.deepcopy(burnin)
    proximal_optimizer = torch.optim.SGD(
        explicit_proximal.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=0.0,
    )
    proximal_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for group in proximal_optimizer.param_groups:
        group["lr"] = args.swa_lr
    for tail_index in range(1, args.tail_epochs + 1):
        epoch = args.burnin_epochs + tail_index
        order = epoch_order(n_train, args.seed, epoch, device)
        proximal_loss = train_epoch(
            explicit_proximal,
            proximal_optimizer,
            x_train,
            y_train,
            order,
            args.batch_size,
            proximal_center=burnin_vector,
            proximal_scale=proximal_scale,
        )
        if tail_index == 1 or tail_index % 20 == 0:
            LOGGER.info(
                "explicit-proximal epoch=%d objective=%.6f",
                epoch,
                proximal_loss,
            )
    endpoints["explicit_proximal_retrain"] = endpoint_probe(
        explicit_proximal,
        loader,
        x_train,
        y_train,
        args.batch_size,
        burnin_vector,
    )
    if args.checkpoint_dir is not None:
        save_checkpoint(
            args.checkpoint_dir / "explicit_proximal_retrain.pt",
            explicit_proximal,
            proximal_optimizer,
            total_epochs,
            {**config, "fitted_proximal_scale": proximal_scale},
        )

    displacements = {
        "schedule_only": displacement_probe(control, high_lr, burnin_vector),
        "averaging_only": displacement_probe(high_lr, swa, burnin_vector),
        "full_swa": displacement_probe(control, swa, burnin_vector),
    }
    output = {
        "config": config,
        "protocol": {
            "control": "shared burn-in + standard decayed learning-rate tail",
            "high_lr_last_iterate": "shared burn-in + constant high-LR tail, no averaging",
            "swa_average": "equal average of every constant high-LR tail endpoint",
            "n_averaged": int(averaged.n_averaged),
            "fitted_proximal_penalty": (
                f"{proximal_scale:.8g} * ||theta - theta_burnin||^2"
            ),
        },
        "endpoints": endpoints,
        "displacements": displacements,
        "explicit_recovery": recovery_metrics(
            explicit_proximal, swa, burnin, x_train, args.batch_size
        ),
        "run": {"wall_minutes": (time.time() - started) / 60.0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    LOGGER.info("Wrote %s in %.2f minutes", args.output, output["run"]["wall_minutes"])
    LOGGER.info(
        "SWA train loss=%.6f, averaging displacement best positive cosine=%s",
        endpoints["swa_average"]["train"]["loss"],
        max(
            (
                (name, result["grad_cosine"])
                for name, result in displacements["averaging_only"][
                    "candidates"
                ].items()
                if result["grad_cosine"] is not None
            ),
            key=lambda item: item[1],
        ),
    )


if __name__ == "__main__":
    main()
