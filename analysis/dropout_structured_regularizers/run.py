"""Compare structured quadratic and path regularizers at dropout-trained endpoints.

The original Figure 5 estimator asks whether the deterministic-loss gradient at
a dropout-trained endpoint is aligned with a candidate penalty gradient.  This
runner keeps that estimand but replaces iterative coefficient optimization with
exact nonnegative least squares and compares four candidate families on the
same trained model:

* global ridge: one coefficient multiplying ||theta||_2^2;
* layerwise ridge: one coefficient per Linear layer;
* activation-weighted quadratic: one coefficient per Linear layer multiplying
  tr(W diag(E[h^2]) W^T) + ||b||_2^2;
* squared path norm: one coefficient for the sum of squared input-output path
  products, including bias edges.

For protection against the mechanical in-sample advantage of multi-coefficient
families, the training set is deterministically split in half.  Coefficients
fitted to one half's loss gradient are evaluated against the other half's
gradient, in both directions.  Full-data fits are still reported for direct
comparison with Figure 5.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, "src")

from core.data import MNISTLightningDataModule
from core.models import DeepReLUClassifier

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class GradientSummary:
    """A mean loss gradient and activation second moments for one data split."""

    gradient: Tensor
    moments: dict[str, Tensor]
    examples: int
    mean_loss: float


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def linear_layers(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    """Return Linear modules in forward/parameter order."""
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]


def loss_gradient_and_moments(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
) -> GradientSummary:
    """Compute the mean CE gradient and E[input^2] at every Linear layer."""
    device = next(model.parameters()).device
    layers = linear_layers(model)
    moment_sums: dict[str, Tensor] = {}
    moment_counts = {name: 0 for name, _ in layers}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            activations = inputs[0].detach()
            if activations.ndim != 2:
                activations = activations.reshape(activations.shape[0], -1)
            batch_sum = activations.to(torch.float64).square().sum(dim=0)
            if name not in moment_sums:
                moment_sums[name] = torch.zeros_like(batch_sum)
            moment_sums[name].add_(batch_sum)
            moment_counts[name] += activations.shape[0]

        return hook

    for name, module in layers:
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    model.eval()
    model.zero_grad(set_to_none=True)
    total_examples = 0
    total_loss = 0.0
    try:
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss_sum = F.cross_entropy(model(features), labels, reduction="sum")
            loss_sum.backward()
            batch_size = labels.numel()
            total_examples += batch_size
            total_loss += float(loss_sum.detach())
    finally:
        for handle in handles:
            handle.remove()

    if total_examples == 0:
        raise ValueError("cannot estimate a gradient from an empty data split")
    if any(moment_counts[name] != total_examples for name, _ in layers):
        raise RuntimeError(
            "a Linear layer was not called exactly once per example while collecting moments"
        )

    gradient = torch.cat(
        [
            (
                parameter.grad.detach().reshape(-1) / total_examples
                if parameter.grad is not None
                else torch.zeros(
                    parameter.numel(),
                    device=device,
                    dtype=parameter.dtype,
                )
            )
            for parameter in model.parameters()
        ]
    )
    model.zero_grad(set_to_none=True)
    moments = {
        name: (moment_sums[name] / moment_counts[name]).to(
            device=device,
            dtype=module.weight.dtype,
        )
        for name, module in layers
    }
    return GradientSummary(
        gradient=gradient,
        moments=moments,
        examples=total_examples,
        mean_loss=total_loss / total_examples,
    )


def combine_summaries(
    first: GradientSummary,
    second: GradientSummary,
) -> GradientSummary:
    """Form the exact sample-size-weighted union of two disjoint summaries."""
    total = first.examples + second.examples
    weight_first = first.examples / total
    weight_second = second.examples / total
    if first.moments.keys() != second.moments.keys():
        raise ValueError("activation-moment layer sets do not match")
    return GradientSummary(
        gradient=(
            weight_first * first.gradient + weight_second * second.gradient
        ),
        moments={
            name: (
                weight_first * first.moments[name]
                + weight_second * second.moments[name]
            )
            for name in first.moments
        },
        examples=total,
        mean_loss=(
            weight_first * first.mean_loss + weight_second * second.mean_loss
        ),
    )


def flatten_parameter_blocks(
    model: nn.Module,
    blocks: dict[str, Tensor],
) -> Tensor:
    """Flatten named gradient blocks in model parameter order, zero-filling others."""
    chunks = [
        blocks.get(name, torch.zeros_like(parameter)).reshape(-1)
        for name, parameter in model.named_parameters()
    ]
    return torch.cat(chunks)


def global_ridge_bases(model: nn.Module) -> dict[str, Tensor]:
    return {
        "global": torch.cat(
            [2.0 * parameter.detach().reshape(-1) for parameter in model.parameters()]
        )
    }


def layerwise_ridge_bases(model: nn.Module) -> dict[str, Tensor]:
    bases: dict[str, Tensor] = {}
    for layer_name, module in linear_layers(model):
        blocks = {f"{layer_name}.weight": 2.0 * module.weight.detach()}
        if module.bias is not None:
            blocks[f"{layer_name}.bias"] = 2.0 * module.bias.detach()
        bases[layer_name] = flatten_parameter_blocks(model, blocks)
    return bases


def activation_quadratic_bases(
    model: nn.Module,
    moments: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Gradients of per-layer tr(W diag(E[h^2]) W^T) + ||b||^2."""
    bases: dict[str, Tensor] = {}
    for layer_name, module in linear_layers(model):
        second_moment = moments[layer_name]
        if second_moment.numel() != module.weight.shape[1]:
            raise ValueError(
                f"{layer_name}: got {second_moment.numel()} moments for "
                f"{module.weight.shape[1]} input features"
            )
        blocks = {
            f"{layer_name}.weight": (
                2.0 * module.weight.detach() * second_moment.unsqueeze(0)
            )
        }
        if module.bias is not None:
            # This is the augmented-feature formula with a constant input whose
            # second moment is one.
            blocks[f"{layer_name}.bias"] = 2.0 * module.bias.detach()
        bases[layer_name] = flatten_parameter_blocks(model, blocks)
    return bases


def squared_path_basis(model: nn.Module) -> dict[str, Tensor]:
    """Gradient of the squared path norm, with a unit source and bias edges."""
    layers = linear_layers(model)
    if not layers:
        raise ValueError("squared path norm requires at least one Linear layer")
    source = torch.ones(
        layers[0][1].in_features,
        device=layers[0][1].weight.device,
        dtype=layers[0][1].weight.dtype,
    )
    path_mass = source
    for _name, module in layers:
        path_mass = module.weight.square() @ path_mass
        if module.bias is not None:
            path_mass = path_mass + module.bias.square()
    penalty = path_mass.sum()
    parameters = list(model.parameters())
    gradients = torch.autograd.grad(penalty, parameters, allow_unused=True)
    flat = torch.cat(
        [
            (
                gradient.detach().reshape(-1)
                if gradient is not None
                else torch.zeros_like(parameter).reshape(-1)
            )
            for parameter, gradient in zip(parameters, gradients, strict=True)
        ]
    )
    return {"global": flat}


def candidate_bases(
    model: nn.Module,
    moments: dict[str, Tensor],
    fixed_bases: dict[str, dict[str, Tensor]],
) -> dict[str, dict[str, Tensor]]:
    return {
        "global_ridge": fixed_bases["global_ridge"],
        "layerwise_ridge": fixed_bases["layerwise_ridge"],
        "activation_quadratic": activation_quadratic_bases(model, moments),
        "squared_path_norm": fixed_bases["squared_path_norm"],
    }


def evaluate_coefficients(
    target: Tensor,
    bases: dict[str, Tensor],
    coefficients: dict[str, float],
) -> dict[str, float | None]:
    """Evaluate a fixed nonnegative linear combination against a target."""
    target64 = target.detach().to(torch.float64)
    prediction = torch.zeros_like(target64)
    for name, basis in bases.items():
        prediction.add_(basis.detach().to(torch.float64), alpha=coefficients[name])
    target_norm = float(target64.norm())
    prediction_norm = float(prediction.norm())
    residual_norm = float((prediction - target64).norm())
    if target_norm == 0.0:
        raise ValueError("cannot score against a zero target gradient")
    cosine = (
        float(torch.dot(prediction, target64) / (prediction.norm() * target64.norm()))
        if prediction_norm > 0.0
        else None
    )
    return {
        "target_norm": target_norm,
        "prediction_norm": prediction_norm,
        "residual_norm": residual_norm,
        "residual_ratio": residual_norm / target_norm,
        "projection_r2": 1.0 - (residual_norm / target_norm) ** 2,
        "cosine": cosine,
    }


def exact_nnls(
    target: Tensor,
    bases: dict[str, Tensor],
) -> dict[str, object]:
    """Solve a small NNLS problem exactly by enumerating active sets.

    There are at most four layer coefficients in this experiment.  Enumerating
    all 2^K active sets avoids adding an iterative optimizer or a convergence
    tolerance to the estimand.  Columns are normalized before solving.
    """
    names = list(bases)
    if not names:
        raise ValueError("NNLS requires at least one basis vector")
    target64 = target.detach().to(torch.float64)
    columns = [bases[name].detach().to(torch.float64) for name in names]
    norms = torch.tensor(
        [float(column.norm()) for column in columns],
        device=target64.device,
        dtype=torch.float64,
    )
    if bool(torch.any(norms <= 0)):
        bad = [name for name, norm in zip(names, norms, strict=True) if norm <= 0]
        raise ValueError(f"zero-norm regularizer gradients: {bad}")
    normalized = torch.stack(
        [column / norm for column, norm in zip(columns, norms, strict=True)],
        dim=1,
    )
    gram = normalized.T @ normalized
    correlation = normalized.T @ target64

    best_scaled = torch.zeros(len(names), device=target64.device, dtype=torch.float64)
    best_sse = float(torch.dot(target64, target64))
    feasibility_tolerance = 1e-10
    for mask in range(1, 1 << len(names)):
        active = [index for index in range(len(names)) if mask & (1 << index)]
        active_tensor = torch.tensor(active, device=target64.device)
        gram_active = gram.index_select(0, active_tensor).index_select(1, active_tensor)
        corr_active = correlation.index_select(0, active_tensor)
        solution = torch.linalg.lstsq(gram_active, corr_active).solution
        if bool(torch.any(solution < -feasibility_tolerance)):
            continue
        solution = solution.clamp_min(0.0)
        scaled = torch.zeros_like(best_scaled)
        scaled[active_tensor] = solution
        prediction = normalized @ scaled
        sse = float((prediction - target64).square().sum())
        if sse < best_sse:
            best_sse = sse
            best_scaled = scaled

    coefficients_tensor = best_scaled / norms
    coefficients = {
        name: float(value)
        for name, value in zip(names, coefficients_tensor, strict=True)
    }
    metrics = evaluate_coefficients(target64, bases, coefficients)
    positive_tolerance = max(float(best_scaled.max()) * 1e-10, 1e-14)
    active_names = [
        name
        for name, value in zip(names, best_scaled, strict=True)
        if float(value) > positive_tolerance
    ]
    positive_eigenvalues = torch.linalg.eigvalsh(gram)
    condition_number = (
        float(positive_eigenvalues.max() / positive_eigenvalues.min())
        if float(positive_eigenvalues.min()) > 0
        else None
    )
    return {
        "coefficients": coefficients,
        "active": active_names,
        "basis_norms": {
            name: float(norm) for name, norm in zip(names, norms, strict=True)
        },
        "gram_condition_number": condition_number,
        **metrics,
    }


def fit_all_candidates(
    model: nn.Module,
    full: GradientSummary,
    split_a: GradientSummary,
    split_b: GradientSummary,
) -> dict[str, object]:
    fixed = {
        "global_ridge": global_ridge_bases(model),
        "layerwise_ridge": layerwise_ridge_bases(model),
        "squared_path_norm": squared_path_basis(model),
    }
    bases_full = candidate_bases(model, full.moments, fixed)
    bases_a = candidate_bases(model, split_a.moments, fixed)
    bases_b = candidate_bases(model, split_b.moments, fixed)

    output: dict[str, object] = {}
    for family in bases_full:
        full_fit = exact_nnls(-full.gradient, bases_full[family])
        fit_a = exact_nnls(-split_a.gradient, bases_a[family])
        fit_b = exact_nnls(-split_b.gradient, bases_b[family])
        output[family] = {
            "full": full_fit,
            "crossfit": {
                "a_to_b": {
                    "fit": fit_a,
                    "heldout": evaluate_coefficients(
                        -split_b.gradient,
                        bases_a[family],
                        fit_a["coefficients"],  # type: ignore[arg-type]
                    ),
                },
                "b_to_a": {
                    "fit": fit_b,
                    "heldout": evaluate_coefficients(
                        -split_a.gradient,
                        bases_b[family],
                        fit_b["coefficients"],  # type: ignore[arg-type]
                    ),
                },
            },
        }
    return output


def make_split_loaders(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    split_seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(len(dataset), generator=generator).tolist()
    midpoint = len(permutation) // 2
    subsets = (
        Subset(dataset, permutation[:midpoint]),
        Subset(dataset, permutation[midpoint:]),
    )
    loaders = tuple(
        DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
        for subset in subsets
    )
    return loaders  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=2027)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    torch.manual_seed(args.seed)

    datamodule = MNISTLightningDataModule(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    datamodule.prepare_data()
    datamodule.setup("fit")
    model = DeepReLUClassifier(
        input_dim=784,
        num_classes=10,
        depth=args.depth,
        width=args.width,
        dropout=args.dropout,
        batchnorm=False,
        l2_lambda=0.0,
        lr=0.05,
        momentum=0.9,
    )
    trainer = Trainer(
        max_epochs=args.max_epochs,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
        accelerator="auto",
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=args.patience,
            )
        ],
    )
    trainer.fit(model, datamodule=datamodule)
    model.eval()

    train_dataset = datamodule._train_dataset
    if train_dataset is None:
        raise RuntimeError("MNIST training dataset was not initialized")
    loader_a, loader_b = make_split_loaders(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        split_seed=args.split_seed,
    )
    LOGGER.info("computing gradients and activation moments for split A")
    summary_a = loss_gradient_and_moments(model, loader_a)
    LOGGER.info("computing gradients and activation moments for split B")
    summary_b = loss_gradient_and_moments(model, loader_b)
    summary_full = combine_summaries(summary_a, summary_b)
    LOGGER.info("fitting four candidate regularizer families by exact NNLS")
    fits = fit_all_candidates(model, summary_full, summary_a, summary_b)

    parameters = torch.cat(
        [parameter.detach().reshape(-1) for parameter in model.parameters()]
    )
    output = {
        "config": {
            "dropout": args.dropout,
            "seed": args.seed,
            "depth": args.depth,
            "width": args.width,
            "batchnorm": False,
            "l2_lambda": 0.0,
            "lr": 0.05,
            "momentum": 0.9,
            "batch_size": args.batch_size,
            "split_seed": args.split_seed,
            "param_count": parameters.numel(),
        },
        "training": {
            "epochs_run": int(trainer.current_epoch),
        },
        "gradient": {
            "examples": summary_full.examples,
            "mean_loss": summary_full.mean_loss,
            "full_norm": float(summary_full.gradient.norm()),
            "split_a_norm": float(summary_a.gradient.norm()),
            "split_b_norm": float(summary_b.gradient.norm()),
            "parameter_norm": float(parameters.norm()),
            "relative_full_norm": float(
                summary_full.gradient.norm() / parameters.norm()
            ),
            "split_disagreement_ratio": float(
                (summary_a.gradient - summary_b.gradient).norm()
                / summary_full.gradient.norm()
            ),
        },
        "candidates": fits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    headline = {
        family: {
            "full_projection_r2": record["full"]["projection_r2"],
            "crossfit_projection_r2": [
                record["crossfit"]["a_to_b"]["heldout"]["projection_r2"],
                record["crossfit"]["b_to_a"]["heldout"]["projection_r2"],
            ],
        }
        for family, record in fits.items()  # type: ignore[union-attr]
    }
    LOGGER.info("wrote %s\n%s", args.output, json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
