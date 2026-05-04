#!/usr/bin/env python3
"""Run a nonlinear retrain suite for single and multi-component power-geometry recovery."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
import itertools
import json
import logging
import math
from pathlib import Path
import sys
from typing import Sequence

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, random_split

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.bias import SmoothedPowerBias, SmoothedSchattenBias  # noqa: E402
from core.estimators import vector_to_parameter_views  # noqa: E402

try:  # noqa: E402
    from analysis.function_class_identifiability.run import (
        DeepReLURegressor,
        generate_dataset,
        set_seed,
    )
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from function_class_identifiability import (  # type: ignore
        DeepReLURegressor,
        generate_dataset,
        set_seed,
    )


STYLE_PATH = REPO_ROOT / "clean_fig.mplstyle"
if STYLE_PATH.exists():
    plt.style.use(str(STYLE_PATH))

LOGGER = logging.getLogger(__name__)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)


@dataclass(frozen=True)
class GeometryComponentConfig:
    label: str
    family: str
    true_p: float
    target_gradient_scale: float
    fit_p_init: float
    fit_lambda_init: float = 0.01
    epsilon: float = 1e-6


@dataclass(frozen=True)
class GeometryCase:
    name: str
    title: str
    components: tuple[GeometryComponentConfig, ...]


@dataclass(frozen=True)
class ComponentRecovery:
    label: str
    family: str
    true_lambda: float
    estimated_lambda: float
    true_p: float
    estimated_p: float
    lambda_rel_error: float
    p_abs_error: float


@dataclass(frozen=True)
class CaseRecovery:
    case_name: str
    case_title: str
    n_components: int
    gradient_cosine: float
    relative_residual: float
    full_train_mse: float
    full_test_mse: float
    replicate_train_mse_mean: float
    replicate_test_mse_mean: float
    stationarity_residual_mean: float
    components: list[ComponentRecovery]


@dataclass(frozen=True)
class CaseReplicatePool:
    case: GeometryCase
    truth_components: list[dict[str, float | str]]
    parameter_shapes: list[torch.Size]
    full_train_mse: float
    full_test_mse: float
    replicate_train_mses: list[float]
    replicate_test_mses: list[float]
    solutions: Tensor
    target_gradients: Tensor
    stationarity_residuals: list[float]


PowerGeometryModule = SmoothedPowerBias | SmoothedSchattenBias


def build_component_module(
    config: GeometryComponentConfig,
    *,
    scale_init: float,
    exponent_init: float,
    trainable_scale: bool,
    trainable_exponent: bool,
    p_min: float,
    p_max: float,
) -> PowerGeometryModule:
    common_kwargs = dict(
        scale_init=scale_init,
        exponent_init=exponent_init,
        epsilon=config.epsilon,
        trainable_scale=trainable_scale,
        trainable_exponent=trainable_exponent,
        p_min=p_min,
        p_max=p_max,
    )
    if config.family == "elementwise":
        return SmoothedPowerBias(**common_kwargs)
    if config.family == "spectral":
        return SmoothedSchattenBias(**common_kwargs)
    raise ValueError(f"Unsupported geometry family: {config.family}")


def component_penalty_gradient(
    module: PowerGeometryModule,
    flattened_params: Tensor,
    structured_params: Sequence[Tensor],
) -> Tensor:
    if isinstance(module, SmoothedPowerBias):
        return module.penalty_gradient(flattened_params)
    if isinstance(module, SmoothedSchattenBias):
        return module.penalty_gradient(flattened_params, list(structured_params))
    raise TypeError(f"Unsupported geometry module: {type(module)!r}")


class CompositeGeometryBias(nn.Module):
    def __init__(
        self,
        labels: Sequence[str],
        families: Sequence[str],
        components: Sequence[PowerGeometryModule],
    ) -> None:
        super().__init__()
        if len(labels) != len(components) or len(families) != len(components):
            raise ValueError("labels, families, and components must have the same length.")
        self.labels = list(labels)
        self.families = list(families)
        self.components = nn.ModuleList(list(components))

    def forward(
        self,
        flattened_params: Tensor,
        structured_params: Sequence[Tensor],
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        total = torch.zeros(
            (), device=flattened_params.device, dtype=flattened_params.dtype
        )
        component_values: dict[str, Tensor] = {}
        for label, module in zip(self.labels, self.components):
            value = module(flattened_params, list(structured_params))
            total = total + value
            if return_components:
                component_values[label] = value
        if return_components:
            return total, component_values
        return total

    def penalty_gradient(
        self,
        flattened_params: Tensor,
        structured_params: Sequence[Tensor],
    ) -> Tensor:
        total = torch.zeros_like(flattened_params)
        for module in self.components:
            total = total + component_penalty_gradient(module, flattened_params, structured_params)
        return torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)

    def get_component_params(self) -> list[dict[str, float | str]]:
        estimates: list[dict[str, float | str]] = []
        for label, family, module in zip(self.labels, self.families, self.components):
            params = module.get_bias_params()
            estimates.append(
                {
                    "label": label,
                    "family": family,
                    "scale": float(params["scale"]),
                    "exponent": float(params["exponent"]),
                }
            )
        return estimates


class GeometryRegularizedDeepReLURegressor(DeepReLURegressor):
    def __init__(
        self,
        input_dim: int,
        depth: int,
        width: int,
        lr: float,
        geometry_bias: CompositeGeometryBias,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            depth=depth,
            width=width,
            lr=lr,
            explicit_bias_types=[],
            explicit_lambdas=[],
        )
        self.geometry_bias = geometry_bias
        self.save_hyperparameters(ignore=["geometry_bias"])

    def _explicit_regularization(self) -> tuple[Tensor, dict[str, Tensor]]:
        flat_params = torch.nn.utils.parameters_to_vector(list(self.network.parameters()))
        structured_params = vector_to_parameter_views(flat_params, self.network)
        total, components = self.geometry_bias(
            flat_params,
            structured_params,
            return_components=True,
        )
        return total, components

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
            },
        }


class CompositeGeometryEstimator(nn.Module):
    def __init__(
        self,
        parameter_shapes: Sequence[torch.Size],
        component_configs: Sequence[GeometryComponentConfig],
        *,
        p_min: float,
        p_max: float,
    ) -> None:
        super().__init__()
        self.parameter_shapes = [torch.Size(shape) for shape in parameter_shapes]
        self.parameter_numels = [int(math.prod(shape)) for shape in self.parameter_shapes]
        geometry_modules = [
            build_component_module(
                config,
                scale_init=config.fit_lambda_init,
                exponent_init=config.fit_p_init,
                trainable_scale=True,
                trainable_exponent=True,
                p_min=p_min,
                p_max=p_max,
            )
            for config in component_configs
        ]
        self.geometry = CompositeGeometryBias(
            labels=[config.label for config in component_configs],
            families=[config.family for config in component_configs],
            components=geometry_modules,
        )

    def _vector_to_views(self, vector: Tensor) -> list[Tensor]:
        views: list[Tensor] = []
        pointer = 0
        for shape, numel in zip(self.parameter_shapes, self.parameter_numels):
            views.append(vector[pointer : pointer + numel].view(shape))
            pointer += numel
        return views

    def predicted_gradients(self, solutions: Tensor) -> Tensor:
        gradients: list[Tensor] = []
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        for solution in solutions:
            flat_params = solution.to(device=device, dtype=dtype)
            structured_params = self._vector_to_views(flat_params)
            gradients.append(self.geometry.penalty_gradient(flat_params, structured_params))
        return torch.stack(gradients, dim=0)

    def get_component_estimates(self) -> list[dict[str, float | str]]:
        return self.geometry.get_component_params()


def cosine_between(vec_a: Tensor, vec_b: Tensor) -> float:
    safe_a = torch.nan_to_num(vec_a.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_b = torch.nan_to_num(vec_b.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    denom = float(safe_a.norm().item() * safe_b.norm().item())
    if denom <= 1e-12:
        return 0.0
    return float(torch.dot(safe_a, safe_b).item() / denom)


def make_resample_indices(
    n_total: int,
    mode: str,
    sample_fraction: float,
    generator: torch.Generator,
) -> Tensor:
    sample_size = max(1, int(round(sample_fraction * n_total)))
    if mode == "subsample":
        return torch.randperm(n_total, generator=generator)[:sample_size]
    if mode == "bootstrap":
        return torch.randint(0, n_total, (sample_size,), generator=generator)
    raise ValueError(f"Unsupported resample mode: {mode}")


def dataset_from_indices(dataset: TensorDataset, indices: Tensor) -> TensorDataset:
    inputs, targets = dataset.tensors[:2]
    return TensorDataset(inputs[indices], targets[indices])


def evaluate_split_metrics(
    model: GeometryRegularizedDeepReLURegressor,
    dataset: TensorDataset,
) -> tuple[float, float]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        inputs, targets = dataset.tensors[:2]
        predictions = model(inputs.to(device))
        mse_loss = F.mse_loss(predictions, targets.to(device))
        regularization, _ = model._explicit_regularization()
        total_loss = mse_loss + regularization
    return float(mse_loss.item()), float(total_loss.item())


def flatten_model_parameters(model: nn.Module) -> Tensor:
    return torch.nn.utils.parameters_to_vector(list(model.parameters())).detach().cpu()


def compute_target_gradient(model: nn.Module, dataset: TensorDataset) -> Tensor:
    device = next(model.parameters()).device
    model.eval()
    inputs, targets = dataset.tensors[:2]
    predictions = model(inputs.to(device))
    data_loss = F.mse_loss(predictions, targets.to(device))
    loss_grads = torch.autograd.grad(
        data_loss,
        tuple(model.parameters()),
        create_graph=False,
    )
    target_gradient = -torch.cat([grad.reshape(-1) for grad in loss_grads]).detach().cpu()
    return torch.nan_to_num(target_gradient, nan=0.0, posinf=0.0, neginf=0.0)


def compute_stationarity_residual(
    target_gradient: Tensor,
    solution: Tensor,
    parameter_shapes: Sequence[torch.Size],
    component_configs: Sequence[GeometryComponentConfig],
    true_lambdas: Sequence[float],
) -> float:
    """Measure ||(-∇L) - ∇R|| / ||∇L||.  Returns 0.0 at a perfect stationary point."""
    gt_geometry = build_ground_truth_geometry(component_configs, true_lambdas)
    structured = []
    pointer = 0
    for shape in parameter_shapes:
        numel = int(math.prod(shape))
        structured.append(solution[pointer : pointer + numel].view(shape))
        pointer += numel
    gt_reg_gradient = gt_geometry.penalty_gradient(solution, structured)
    target_norm = float(target_gradient.norm().item())
    if target_norm < 1e-12:
        return 0.0
    return float((target_gradient - gt_reg_gradient).norm().item() / target_norm)


def build_ground_truth_geometry(
    component_configs: Sequence[GeometryComponentConfig],
    true_lambdas: Sequence[float],
) -> CompositeGeometryBias:
    return CompositeGeometryBias(
        labels=[config.label for config in component_configs],
        families=[config.family for config in component_configs],
        components=[
            build_component_module(
                config,
                scale_init=true_lambda,
                exponent_init=config.true_p,
                trainable_scale=False,
                trainable_exponent=False,
                p_min=0.25,
                p_max=4.0,
            )
            for config, true_lambda in zip(component_configs, true_lambdas)
        ],
    )


def calibrate_component_scales(
    predictive_model: nn.Module,
    component_configs: Sequence[GeometryComponentConfig],
) -> list[float]:
    flat_params = torch.nn.utils.parameters_to_vector(list(predictive_model.parameters())).detach()
    structured_params = vector_to_parameter_views(flat_params, predictive_model)
    scales: list[float] = []
    for config in component_configs:
        component = build_component_module(
            config,
            scale_init=1.0,
            exponent_init=config.true_p,
            trainable_scale=False,
            trainable_exponent=False,
            p_min=0.25,
            p_max=4.0,
        )
        grad_vec = component_penalty_gradient(component, flat_params, structured_params)
        grad_norm = float(grad_vec.norm().item())
        scales.append(config.target_gradient_scale / max(grad_norm, 1e-8))
    return scales


def train_model(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    *,
    input_dim: int,
    depth: int,
    width: int,
    lr: float,
    component_configs: Sequence[GeometryComponentConfig],
    true_lambdas: Sequence[float],
    max_epochs: int,
    patience: int,
    accelerator: str,
    seed: int,
) -> GeometryRegularizedDeepReLURegressor:
    set_seed(seed)
    _ = val_dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=False,
        num_workers=0,
    )
    model = GeometryRegularizedDeepReLURegressor(
        input_dim=input_dim,
        depth=depth,
        width=width,
        lr=lr,
        geometry_bias=build_ground_truth_geometry(component_configs, true_lambdas),
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=False,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[EarlyStopping(monitor="train/loss", mode="min", patience=patience)],
        log_every_n_steps=25,
    )
    trainer.fit(model, train_dataloaders=train_loader)
    return model


def fit_composite_geometry(
    *,
    parameter_shapes: Sequence[torch.Size],
    component_configs: Sequence[GeometryComponentConfig],
    solutions: Tensor,
    target_gradients: Tensor,
    p_min: float,
    p_max: float,
    lr: float,
    max_epochs: int,
    patience: int,
) -> tuple[list[dict[str, float | str]], float, float]:
    estimator = CompositeGeometryEstimator(
        parameter_shapes=parameter_shapes,
        component_configs=component_configs,
        p_min=p_min,
        p_max=p_max,
    )
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)

    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    for _ in range(max_epochs):
        optimizer.zero_grad(set_to_none=True)
        predicted = estimator.predicted_gradients(solutions)
        loss = F.mse_loss(predicted, target_gradients, reduction="mean")
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        if loss_value + 1e-12 < best_loss:
            best_loss = loss_value
            best_state = {
                name: tensor.detach().clone()
                for name, tensor in estimator.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        estimator.load_state_dict(best_state)

    with torch.no_grad():
        predicted = estimator.predicted_gradients(solutions)
        residual = float(
            (target_gradients - predicted).norm().item()
            / max(target_gradients.norm().item(), 1e-8)
        )
        cosine = cosine_between(predicted.reshape(-1), target_gradients.reshape(-1))
    return estimator.get_component_estimates(), residual, cosine


def fit_composite_geometry_closed_form(
    *,
    parameter_shapes: Sequence[torch.Size],
    component_configs: Sequence[GeometryComponentConfig],
    solutions: Tensor,
    target_gradients: Tensor,
) -> tuple[list[dict[str, float | str]], float, float]:
    """Closed-form λ solver for the known-p case.

    When the exponent p is known for each component, the gradient-matching
    problem ``target ≈ Σ_i λ_i ∇R_i(θ; p_i)`` is linear in λ and can be
    solved via least-squares instead of iterative optimization.
    """
    n_components = len(component_configs)

    # Build component modules at the true (known) exponent with unit scale.
    modules = [
        build_component_module(
            config,
            scale_init=1.0,
            exponent_init=config.true_p,
            trainable_scale=False,
            trainable_exponent=False,
            p_min=0.25,
            p_max=4.0,
        )
        for config in component_configs
    ]

    # Compute the design matrix: each column is a component's gradient
    # stacked across all solutions.
    def _vector_to_views(vector: Tensor) -> list[Tensor]:
        views: list[Tensor] = []
        pointer = 0
        for shape in parameter_shapes:
            numel = int(math.prod(shape))
            views.append(vector[pointer : pointer + numel].view(shape))
            pointer += numel
        return views

    columns: list[Tensor] = []
    for module in modules:
        col_parts: list[Tensor] = []
        for solution in solutions:
            structured = _vector_to_views(solution)
            col_parts.append(component_penalty_gradient(module, solution, structured))
        columns.append(torch.cat(col_parts, dim=0))

    # G: (n_solutions * n_params, n_components)
    G = torch.stack(columns, dim=1).detach()
    y = target_gradients.reshape(-1).detach()

    # Solve via least-squares: λ = argmin ||y - G λ||²
    result = torch.linalg.lstsq(G, y.unsqueeze(1))
    lambdas = result.solution.squeeze(1)

    # Compute metrics
    predicted = (G @ lambdas.unsqueeze(1)).squeeze(1)
    residual = float((y - predicted).norm().item() / max(y.norm().item(), 1e-8))
    cosine_val = cosine_between(predicted, y)

    estimates = [
        {
            "label": config.label,
            "family": config.family,
            "scale": float(lambdas[i].item()),
            "exponent": config.true_p,
        }
        for i, config in enumerate(component_configs)
    ]
    return estimates, residual, cosine_val


def align_estimates(
    truth_components: Sequence[dict[str, float | str]],
    estimated_components: Sequence[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    if len(truth_components) != len(estimated_components):
        raise ValueError("truth_components and estimated_components must have matching lengths.")
    n_components = len(truth_components)
    best_perm: tuple[int, ...] | None = None
    best_cost = float("inf")
    for permutation in itertools.permutations(range(n_components)):
        cost = 0.0
        valid = True
        for truth_idx, estimated_idx in enumerate(permutation):
            truth = truth_components[truth_idx]
            estimated = estimated_components[estimated_idx]
            if truth["family"] != estimated["family"]:
                valid = False
                break
            log_scale_error = abs(
                math.log(
                    max(float(estimated["scale"]), 1e-8)
                    / max(float(truth["true_lambda"]), 1e-8)
                )
            )
            cost += abs(float(estimated["exponent"]) - float(truth["true_p"])) + 0.25 * log_scale_error
        if valid and cost < best_cost:
            best_cost = cost
            best_perm = permutation
    if best_perm is None:
        raise RuntimeError("Failed to align estimated components with the truth.")
    return [dict(estimated_components[idx]) for idx in best_perm]


def per_component_target_scale(base_scale: float, n_components: int) -> float:
    return base_scale / math.sqrt(max(n_components, 1))


def mean_or_nan(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def build_cases(base_scale: float, epsilon: float) -> list[GeometryCase]:
    def component(
        label: str,
        family: str,
        true_p: float,
        fit_p_init: float,
        n_components: int,
    ) -> GeometryComponentConfig:
        return GeometryComponentConfig(
            label=label,
            family=family,
            true_p=true_p,
            target_gradient_scale=per_component_target_scale(base_scale, n_components),
            fit_p_init=fit_p_init,
            fit_lambda_init=0.01,
            epsilon=epsilon,
        )

    return [
        GeometryCase(
            name="single_l2",
            title="Single Active Elementwise L2",
            components=(component("l2", "elementwise", 2.0, 1.6, 1),),
        ),
        GeometryCase(
            name="single_l1",
            title="Single Active Elementwise L1",
            components=(component("l1", "elementwise", 1.0, 1.7, 1),),
        ),
        GeometryCase(
            name="single_nuclear",
            title="Single Active Nuclear Norm",
            components=(component("nuclear", "spectral", 1.0, 1.5, 1),),
        ),
        GeometryCase(
            name="multi_l1_l2",
            title="Mixed Elementwise L1 + L2",
            components=(
                component("l1", "elementwise", 1.0, 0.8, 2),
                component("l2", "elementwise", 2.0, 2.3, 2),
            ),
        ),
        GeometryCase(
            name="multi_l2_nuclear",
            title="Mixed Elementwise L2 + Nuclear",
            components=(
                component("l2", "elementwise", 2.0, 2.2, 2),
                component("nuclear", "spectral", 1.0, 1.1, 2),
            ),
        ),
        GeometryCase(
            name="multi_l1_l2_nuclear",
            title="Mixed Elementwise L1 + L2 + Nuclear",
            components=(
                component("l1", "elementwise", 1.0, 0.8, 3),
                component("l2", "elementwise", 2.0, 2.2, 3),
                component("nuclear", "spectral", 1.0, 1.1, 3),
            ),
        ),
    ]


def collect_case_replicate_pool(
    case: GeometryCase,
    *,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    test_dataset: TensorDataset,
    input_dim: int,
    depth: int,
    width: int,
    lr: float,
    max_epochs: int,
    patience: int,
    accelerator: str,
    resample_mode: str,
    n_replicates: int,
    sample_fraction: float,
    seed: int,
) -> CaseReplicatePool:
    set_seed(seed)
    reference_model = DeepReLURegressor(
        input_dim=input_dim,
        depth=depth,
        width=width,
        lr=lr,
        explicit_bias_types=[],
        explicit_lambdas=[],
    )
    true_lambdas = calibrate_component_scales(reference_model.network, case.components)
    truth_components = [
        {
            "label": config.label,
            "family": config.family,
            "true_lambda": true_lambda,
            "true_p": config.true_p,
        }
        for config, true_lambda in zip(case.components, true_lambdas)
    ]

    full_model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=input_dim,
        depth=depth,
        width=width,
        lr=lr,
        component_configs=case.components,
        true_lambdas=true_lambdas,
        max_epochs=max_epochs,
        patience=patience,
        accelerator=accelerator,
        seed=seed,
    )
    full_train_mse, _ = evaluate_split_metrics(full_model, train_dataset)
    full_test_mse, _ = evaluate_split_metrics(full_model, test_dataset)
    full_solution = flatten_model_parameters(full_model.network)
    full_target_gradient = compute_target_gradient(full_model.network, train_dataset)
    param_shapes = [p.shape for p in full_model.network.parameters()]

    full_stationarity = compute_stationarity_residual(
        full_target_gradient, full_solution, param_shapes, case.components, true_lambdas,
    )
    LOGGER.info("Full-model stationarity residual: %.4f", full_stationarity)

    replicate_solutions: list[Tensor] = [full_solution]
    replicate_targets: list[Tensor] = [full_target_gradient]
    replicate_train_mses: list[float] = []
    replicate_test_mses: list[float] = []
    stationarity_residuals: list[float] = [full_stationarity]

    n_total = len(train_dataset)
    base_generator = torch.Generator().manual_seed(seed + 100_000)
    for _ in range(n_replicates):
        replicate_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=base_generator).item())
        indices = make_resample_indices(
            n_total=n_total,
            mode=resample_mode,
            sample_fraction=sample_fraction,
            generator=torch.Generator().manual_seed(replicate_seed),
        )
        replicate_train_dataset = dataset_from_indices(train_dataset, indices)
        replicate_model = train_model(
            train_dataset=replicate_train_dataset,
            val_dataset=val_dataset,
            input_dim=input_dim,
            depth=depth,
            width=width,
            lr=lr,
            component_configs=case.components,
            true_lambdas=true_lambdas,
            max_epochs=max_epochs,
            patience=patience,
            accelerator=accelerator,
            seed=replicate_seed,
        )
        train_mse, _ = evaluate_split_metrics(replicate_model, replicate_train_dataset)
        test_mse, _ = evaluate_split_metrics(replicate_model, test_dataset)
        replicate_train_mses.append(train_mse)
        replicate_test_mses.append(test_mse)
        rep_solution = flatten_model_parameters(replicate_model.network)
        rep_target = compute_target_gradient(replicate_model.network, replicate_train_dataset)
        replicate_solutions.append(rep_solution)
        replicate_targets.append(rep_target)
        stationarity_residuals.append(
            compute_stationarity_residual(
                rep_target, rep_solution, param_shapes, case.components, true_lambdas,
            )
        )

    solution_stack = torch.stack(replicate_solutions, dim=0)
    target_stack = torch.stack(replicate_targets, dim=0)
    return CaseReplicatePool(
        case=case,
        truth_components=truth_components,
        parameter_shapes=param_shapes,
        full_train_mse=full_train_mse,
        full_test_mse=full_test_mse,
        replicate_train_mses=replicate_train_mses,
        replicate_test_mses=replicate_test_mses,
        solutions=solution_stack,
        target_gradients=target_stack,
        stationarity_residuals=stationarity_residuals,
    )


def recover_case_from_pool(
    pool: CaseReplicatePool,
    *,
    n_replicates: int,
    estimation_lr: float,
    estimation_max_epochs: int,
    estimation_patience: int,
    p_min: float,
    p_max: float,
    use_closed_form: bool = False,
) -> CaseRecovery:
    if n_replicates < 0:
        raise ValueError("n_replicates must be non-negative.")
    if n_replicates > len(pool.replicate_train_mses):
        raise ValueError(
            "Requested more replicates than were collected in the case pool: "
            f"{n_replicates} > {len(pool.replicate_train_mses)}."
        )
    solution_stack = pool.solutions[: n_replicates + 1]
    target_stack = pool.target_gradients[: n_replicates + 1]
    if use_closed_form:
        estimated_components, residual, cosine = fit_composite_geometry_closed_form(
            parameter_shapes=pool.parameter_shapes,
            component_configs=pool.case.components,
            solutions=solution_stack,
            target_gradients=target_stack,
        )
    else:
        estimated_components, residual, cosine = fit_composite_geometry(
            parameter_shapes=pool.parameter_shapes,
            component_configs=pool.case.components,
            solutions=solution_stack,
            target_gradients=target_stack,
            p_min=p_min,
            p_max=p_max,
            lr=estimation_lr,
            max_epochs=estimation_max_epochs,
            patience=estimation_patience,
        )
    aligned_estimates = align_estimates(pool.truth_components, estimated_components)

    component_results = [
        ComponentRecovery(
            label=str(truth["label"]),
            family=str(truth["family"]),
            true_lambda=float(truth["true_lambda"]),
            estimated_lambda=float(estimate["scale"]),
            true_p=float(truth["true_p"]),
            estimated_p=float(estimate["exponent"]),
            lambda_rel_error=abs(float(estimate["scale"]) - float(truth["true_lambda"]))
            / max(abs(float(truth["true_lambda"])), 1e-8),
            p_abs_error=abs(float(estimate["exponent"]) - float(truth["true_p"])),
        )
        for truth, estimate in zip(pool.truth_components, aligned_estimates)
    ]
    return CaseRecovery(
        case_name=pool.case.name,
        case_title=pool.case.title,
        n_components=len(pool.case.components),
        gradient_cosine=cosine,
        relative_residual=residual,
        full_train_mse=pool.full_train_mse,
        full_test_mse=pool.full_test_mse,
        replicate_train_mse_mean=mean_or_nan(pool.replicate_train_mses[:n_replicates]),
        replicate_test_mse_mean=mean_or_nan(pool.replicate_test_mses[:n_replicates]),
        stationarity_residual_mean=mean_or_nan(pool.stationarity_residuals[: n_replicates + 1]),
        components=component_results,
    )


def run_case(
    case: GeometryCase,
    *,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    test_dataset: TensorDataset,
    input_dim: int,
    depth: int,
    width: int,
    lr: float,
    max_epochs: int,
    patience: int,
    accelerator: str,
    resample_mode: str,
    n_replicates: int,
    sample_fraction: float,
    estimation_lr: float,
    estimation_max_epochs: int,
    estimation_patience: int,
    p_min: float,
    p_max: float,
    seed: int,
) -> CaseRecovery:
    pool = collect_case_replicate_pool(
        case,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        input_dim=input_dim,
        depth=depth,
        width=width,
        lr=lr,
        max_epochs=max_epochs,
        patience=patience,
        accelerator=accelerator,
        resample_mode=resample_mode,
        n_replicates=n_replicates,
        sample_fraction=sample_fraction,
        seed=seed,
    )
    return recover_case_from_pool(
        pool,
        n_replicates=n_replicates,
        estimation_lr=estimation_lr,
        estimation_max_epochs=estimation_max_epochs,
        estimation_patience=estimation_patience,
        p_min=p_min,
        p_max=p_max,
    )


def plot_case_result(result: CaseRecovery, output_path: Path) -> None:
    labels = [component.label for component in result.components]
    x_positions = np.arange(len(labels))
    true_lambdas = np.array([component.true_lambda for component in result.components], dtype=float)
    estimated_lambdas = np.array([component.estimated_lambda for component in result.components], dtype=float)
    true_ps = np.array([component.true_p for component in result.components], dtype=float)
    estimated_ps = np.array([component.estimated_p for component in result.components], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    width = 0.36

    axes[0].bar(x_positions - width / 2.0, true_lambdas, width=width, label="true", color="#4C72B0")
    axes[0].bar(x_positions + width / 2.0, estimated_lambdas, width=width, label="estimated", color="#DD8452")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylabel("lambda")
    axes[0].set_title("Scale Recovery")
    axes[0].legend(frameon=False)

    axes[1].bar(x_positions - width / 2.0, true_ps, width=width, label="true", color="#4C72B0")
    axes[1].bar(x_positions + width / 2.0, estimated_ps, width=width, label="estimated", color="#55A868")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylabel("p")
    axes[1].set_title("Exponent Recovery")

    fig.suptitle(
        (
            f"{result.case_title}\n"
            f"cosine={result.gradient_cosine:.3f}, residual={result.relative_residual:.3f}, "
            f"test_mse={result.full_test_mse:.3f}"
        ),
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_suite_summary(results: Sequence[CaseRecovery], output_path: Path, title: str) -> None:
    component_rows = [
        {
            "case_name": result.case_name,
            "label": component.label,
            "family": component.family,
            "true_lambda": component.true_lambda,
            "estimated_lambda": component.estimated_lambda,
            "true_p": component.true_p,
            "estimated_p": component.estimated_p,
        }
        for result in results
        for component in result.components
    ]
    if not component_rows:
        return

    unique_cases = sorted({row["case_name"] for row in component_rows})
    color_map = plt.get_cmap("tab10")
    case_colors = {
        case_name: color_map(index % 10)
        for index, case_name in enumerate(unique_cases)
    }

    lambda_pairs = np.array(
        [(row["true_lambda"], row["estimated_lambda"]) for row in component_rows],
        dtype=float,
    )
    p_pairs = np.array(
        [(row["true_p"], row["estimated_p"]) for row in component_rows],
        dtype=float,
    )
    lambda_min = float(min(lambda_pairs.min(), 0.0))
    lambda_max = float(lambda_pairs.max())
    p_min_val = float(min(p_pairs.min(), 0.5))
    p_max_val = float(max(p_pairs.max(), 2.5))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for row in component_rows:
        axes[0].scatter(
            row["true_lambda"],
            row["estimated_lambda"],
            color=case_colors[row["case_name"]],
            s=60,
            alpha=0.9,
        )
        axes[1].scatter(
            row["true_p"],
            row["estimated_p"],
            color=case_colors[row["case_name"]],
            s=60,
            alpha=0.9,
        )
    axes[0].plot([lambda_min, lambda_max], [lambda_min, lambda_max], color="black", linewidth=1.0, linestyle="--")
    axes[0].set_xlabel("true lambda")
    axes[0].set_ylabel("estimated lambda")
    axes[0].set_title("Scale Recovery")

    axes[1].plot([p_min_val, p_max_val], [p_min_val, p_max_val], color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xlabel("true p")
    axes[1].set_ylabel("estimated p")
    axes[1].set_title("Exponent Recovery")

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=case_colors[case_name], label=case_name)
        for case_name in unique_cases
    ]
    axes[1].legend(handles=handles, frameon=False, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_component_csv(results: Sequence[CaseRecovery], output_path: Path) -> None:
    rows = [
        {
            "case_name": result.case_name,
            "case_title": result.case_title,
            "n_components": result.n_components,
            "gradient_cosine": result.gradient_cosine,
            "relative_residual": result.relative_residual,
            "full_train_mse": result.full_train_mse,
            "full_test_mse": result.full_test_mse,
            "replicate_train_mse_mean": result.replicate_train_mse_mean,
            "replicate_test_mse_mean": result.replicate_test_mse_mean,
            "stationarity_residual_mean": result.stationarity_residual_mean,
            "label": component.label,
            "family": component.family,
            "true_lambda": component.true_lambda,
            "estimated_lambda": component.estimated_lambda,
            "true_p": component.true_p,
            "estimated_p": component.estimated_p,
            "lambda_rel_error": component.lambda_rel_error,
            "p_abs_error": component.p_abs_error,
        }
        for result in results
        for component in result.components
    ]
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=12)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-spectrum", type=str, default="identity")
    parser.add_argument("--input-rank", type=int, default=0)
    parser.add_argument("--input-spectrum-decay", type=float, default=1.5)
    parser.add_argument("--teacher-spectrum", type=str, default="identity")
    parser.add_argument("--teacher-rank", type=int, default=0)
    parser.add_argument("--teacher-spectrum-decay", type=float, default=0.0)
    parser.add_argument("--teacher-hidden-dim", type=int, default=16)
    parser.add_argument("--teacher-activation", type=str, default="relu", choices=["relu", "tanh", "linear"])
    parser.add_argument("--target-scale", type=float, default=1.0)

    parser.add_argument("--regularizer-epsilon", type=float, default=1e-6)
    parser.add_argument("--target-gradient-scale", type=float, default=0.3)
    parser.add_argument("--suite", type=str, default="all", choices=["single", "multi", "all"])
    parser.add_argument("--resample-mode", type=str, default="bootstrap", choices=["subsample", "bootstrap"])
    parser.add_argument("--n-replicates", type=int, default=4)
    parser.add_argument("--sample-fraction", type=float, default=1.0)

    parser.add_argument("--estimation-lr", type=float, default=5e-2)
    parser.add_argument("--estimation-max-epochs", type=int, default=2000)
    parser.add_argument("--estimation-patience", type=int, default=200)
    parser.add_argument("--p-min", type=float, default=0.5)
    parser.add_argument("--p-max", type=float, default=3.0)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/phase2/nonlinear_multi_geometry_suite"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("figures/nonlinear_multi_geometry_suite"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset, _ = generate_dataset(
        n_samples=args.n_samples,
        input_dim=args.input_dim,
        function_class="teacher_relu",
        noise_std=args.noise_std,
        seed=args.seed,
        data_mode="structured",
        input_rank=args.input_rank if args.input_rank > 0 else args.input_dim,
        input_spectrum=args.input_spectrum,
        input_spectrum_decay=args.input_spectrum_decay,
        teacher_rank=args.teacher_rank if args.teacher_rank > 0 else min(args.input_dim, args.teacher_hidden_dim),
        teacher_spectrum=args.teacher_spectrum,
        teacher_spectrum_decay=args.teacher_spectrum_decay,
        teacher_hidden_dim=args.teacher_hidden_dim,
        teacher_activation=args.teacher_activation,
        target_scale=args.target_scale,
    )
    test_size = int(len(dataset) * args.test_fraction)
    val_size = int(len(dataset) * args.val_fraction)
    train_size = len(dataset) - val_size - test_size
    train_split, val_split, test_split = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    full_inputs, full_targets = dataset.tensors
    train_indices = torch.as_tensor(train_split.indices, dtype=torch.long)
    val_indices = torch.as_tensor(val_split.indices, dtype=torch.long)
    test_indices = torch.as_tensor(test_split.indices, dtype=torch.long)
    train_dataset = TensorDataset(full_inputs[train_indices], full_targets[train_indices])
    val_dataset = TensorDataset(full_inputs[val_indices], full_targets[val_indices])
    test_dataset = TensorDataset(full_inputs[test_indices], full_targets[test_indices])

    cases = build_cases(
        base_scale=args.target_gradient_scale,
        epsilon=args.regularizer_epsilon,
    )
    if args.suite == "single":
        cases = [case for case in cases if len(case.components) == 1]
    elif args.suite == "multi":
        cases = [case for case in cases if len(case.components) > 1]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    results: list[CaseRecovery] = []
    for case_index, case in enumerate(cases):
        LOGGER.info("Running case %s (%d/%d)", case.name, case_index + 1, len(cases))
        result = run_case(
            case,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            input_dim=args.input_dim,
            depth=args.depth,
            width=args.width,
            lr=args.lr,
            max_epochs=args.max_epochs,
            patience=args.patience,
            accelerator=accelerator,
            resample_mode=args.resample_mode,
            n_replicates=args.n_replicates,
            sample_fraction=args.sample_fraction,
            estimation_lr=args.estimation_lr,
            estimation_max_epochs=args.estimation_max_epochs,
            estimation_patience=args.estimation_patience,
            p_min=args.p_min,
            p_max=args.p_max,
            seed=args.seed + 1_000 * case_index,
        )
        results.append(result)

        case_json_path = args.output_dir / f"{case.name}.json"
        case_json_path.parent.mkdir(parents=True, exist_ok=True)
        with case_json_path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(result), handle, indent=2)
        plot_case_result(result, args.figure_dir / f"{case.name}.pdf")
        LOGGER.info(
            "Case %s complete | cosine=%.4f residual=%.4f mean_component_lambda_err=%.4f mean_component_p_err=%.4f",
            case.name,
            result.gradient_cosine,
            result.relative_residual,
            float(np.mean([component.lambda_rel_error for component in result.components])),
            float(np.mean([component.p_abs_error for component in result.components])),
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite_json_path = args.output_dir / "suite_results.json"
    with suite_json_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(result) for result in results], handle, indent=2)
    write_component_csv(results, args.output_dir / "component_recovery.csv")

    single_results = [result for result in results if result.n_components == 1]
    multi_results = [result for result in results if result.n_components > 1]
    plot_suite_summary(single_results, args.figure_dir / "single_summary.pdf", "Single-Component Geometry Recovery")
    plot_suite_summary(multi_results, args.figure_dir / "multi_summary.pdf", "Multi-Component Geometry Recovery")

    LOGGER.info("Saved suite JSON to %s", suite_json_path)
    LOGGER.info("Saved component CSV to %s", args.output_dir / "component_recovery.csv")
    LOGGER.info("Saved figures to %s", args.figure_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
