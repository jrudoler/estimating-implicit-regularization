#!/usr/bin/env python3
"""Estimate a global smoothed-power regularizer from retrained nonlinear solutions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset, random_split

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.bias import SmoothedPowerBias  # noqa: E402
from core.estimators import vector_to_parameter_views  # noqa: E402

try:  # noqa: E402
    from analysis.function_class_identifiability.run import (
        DeepReLURegressor,
        generate_dataset,
        set_seed,
    )
except ModuleNotFoundError:  # pragma: no cover - script-style fallback
    from function_class_identifiability import (
        DeepReLURegressor,
        generate_dataset,
        set_seed,
    )


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PowerGeometrySummary:
    true_lambda: float
    true_p: float
    full_estimated_lambda: float
    full_estimated_p: float
    full_relative_residual: float
    full_gradient_cosine: float
    collection_estimated_lambda: float
    collection_estimated_p: float
    collection_relative_residual: float
    collection_gradient_cosine: float
    full_train_mse: float
    full_test_mse: float
    replicate_train_mse_mean: float
    replicate_test_mse_mean: float
    n_replicates: int
    sample_fraction: float


class PowerRegularizedDeepReLURegressor(DeepReLURegressor):
    def __init__(
        self,
        input_dim: int,
        depth: int,
        width: int,
        lr: float,
        regularizer_scale: float,
        regularizer_exponent: float,
        regularizer_epsilon: float,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            depth=depth,
            width=width,
            lr=lr,
            explicit_bias_types=[],
            explicit_lambdas=[],
        )
        self.power_regularizer = SmoothedPowerBias(
            scale_init=regularizer_scale,
            exponent_init=regularizer_exponent,
            epsilon=regularizer_epsilon,
            trainable_scale=False,
            trainable_exponent=False,
        )

    def _explicit_regularization(self) -> tuple[Tensor, dict[str, Tensor]]:
        flat_params = torch.nn.utils.parameters_to_vector(list(self.network.parameters()))
        regularization = self.power_regularizer(flat_params, structured_params=None)
        return regularization, {self.power_regularizer.bias_name: regularization}

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


class PowerGeometryEstimator(nn.Module):
    def __init__(
        self,
        lambda_init: float,
        p_init: float,
        epsilon: float,
        p_min: float,
        p_max: float,
    ) -> None:
        super().__init__()
        self.regularizer = SmoothedPowerBias(
            scale_init=lambda_init,
            exponent_init=p_init,
            epsilon=epsilon,
            trainable_scale=True,
            trainable_exponent=True,
            p_min=p_min,
            p_max=p_max,
        )

    def predicted_gradients(self, solutions: Tensor) -> Tensor:
        return self.regularizer.penalty_gradient(solutions)

    def get_estimates(self) -> dict[str, float]:
        return self.regularizer.get_bias_params()


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


def evaluate_split_metrics(model: PowerRegularizedDeepReLURegressor, dataset: TensorDataset) -> tuple[float, float]:
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


def calibrate_power_lambda(
    predictive_model: nn.Module,
    exponent: float,
    epsilon: float,
    target_scale: float,
) -> float:
    flat_params = (
        torch.nn.utils.parameters_to_vector(list(predictive_model.parameters()))
        .detach()
        .requires_grad_(True)
    )
    _ = vector_to_parameter_views(flat_params, predictive_model)
    bias = SmoothedPowerBias(
        scale_init=1.0,
        exponent_init=exponent,
        epsilon=epsilon,
        trainable_scale=False,
        trainable_exponent=False,
    )
    grad_vec = bias.penalty_gradient(flat_params).detach()
    grad_norm = float(torch.nan_to_num(grad_vec, nan=0.0, posinf=0.0, neginf=0.0).norm().item())
    return target_scale / max(grad_norm, 1e-8)


def train_model(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    input_dim: int,
    depth: int,
    width: int,
    lr: float,
    regularizer_scale: float,
    regularizer_exponent: float,
    regularizer_epsilon: float,
    max_epochs: int,
    patience: int,
    accelerator: str,
    seed: int,
) -> PowerRegularizedDeepReLURegressor:
    set_seed(seed)
    _ = val_dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=False,
        num_workers=0,
    )
    model = PowerRegularizedDeepReLURegressor(
        input_dim=input_dim,
        depth=depth,
        width=width,
        lr=lr,
        regularizer_scale=regularizer_scale,
        regularizer_exponent=regularizer_exponent,
        regularizer_epsilon=regularizer_epsilon,
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


def fit_power_geometry(
    solutions: Tensor,
    target_gradients: Tensor,
    lambda_init: float,
    p_init: float,
    epsilon: float,
    p_min: float,
    p_max: float,
    lr: float,
    max_epochs: int,
    patience: int,
) -> tuple[dict[str, float], float, float]:
    estimator = PowerGeometryEstimator(
        lambda_init=lambda_init,
        p_init=p_init,
        epsilon=epsilon,
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
    return estimator.get_estimates(), residual, cosine


def run_study(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    test_dataset: TensorDataset,
    input_dim: int,
    depth: int,
    width: int,
    lr: float,
    gt_lambda: float,
    gt_p: float,
    regularizer_epsilon: float,
    max_epochs: int,
    patience: int,
    accelerator: str,
    resample_mode: str,
    n_replicates: int,
    sample_fraction: float,
    estimation_lr: float,
    estimation_max_epochs: int,
    estimation_patience: int,
    lambda_init: float,
    p_init: float,
    p_min: float,
    p_max: float,
    seed: int,
) -> PowerGeometrySummary:
    full_model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=input_dim,
        depth=depth,
        width=width,
        lr=lr,
        regularizer_scale=gt_lambda,
        regularizer_exponent=gt_p,
        regularizer_epsilon=regularizer_epsilon,
        max_epochs=max_epochs,
        patience=patience,
        accelerator=accelerator,
        seed=seed,
    )
    full_train_mse, _ = evaluate_split_metrics(full_model, train_dataset)
    full_test_mse, _ = evaluate_split_metrics(full_model, test_dataset)
    full_solution = flatten_model_parameters(full_model.network)
    full_target_gradient = compute_target_gradient(full_model.network, train_dataset)

    full_estimated, full_residual, full_cosine = fit_power_geometry(
        solutions=full_solution.unsqueeze(0),
        target_gradients=full_target_gradient.unsqueeze(0),
        lambda_init=lambda_init,
        p_init=p_init,
        epsilon=regularizer_epsilon,
        p_min=p_min,
        p_max=p_max,
        lr=estimation_lr,
        max_epochs=estimation_max_epochs,
        patience=estimation_patience,
    )

    replicate_solutions: list[Tensor] = []
    replicate_targets: list[Tensor] = []
    replicate_train_mses: list[float] = []
    replicate_test_mses: list[float] = []

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
            regularizer_scale=gt_lambda,
            regularizer_exponent=gt_p,
            regularizer_epsilon=regularizer_epsilon,
            max_epochs=max_epochs,
            patience=patience,
            accelerator=accelerator,
            seed=replicate_seed,
        )
        train_mse, _ = evaluate_split_metrics(replicate_model, replicate_train_dataset)
        test_mse, _ = evaluate_split_metrics(replicate_model, test_dataset)
        replicate_train_mses.append(train_mse)
        replicate_test_mses.append(test_mse)
        replicate_solutions.append(flatten_model_parameters(replicate_model.network))
        replicate_targets.append(compute_target_gradient(replicate_model.network, replicate_train_dataset))

    solution_stack = torch.stack(replicate_solutions, dim=0)
    target_stack = torch.stack(replicate_targets, dim=0)
    collection_estimated, collection_residual, collection_cosine = fit_power_geometry(
        solutions=solution_stack,
        target_gradients=target_stack,
        lambda_init=lambda_init,
        p_init=p_init,
        epsilon=regularizer_epsilon,
        p_min=p_min,
        p_max=p_max,
        lr=estimation_lr,
        max_epochs=estimation_max_epochs,
        patience=estimation_patience,
    )

    return PowerGeometrySummary(
        true_lambda=gt_lambda,
        true_p=gt_p,
        full_estimated_lambda=float(full_estimated["scale"]),
        full_estimated_p=float(full_estimated["exponent"]),
        full_relative_residual=full_residual,
        full_gradient_cosine=full_cosine,
        collection_estimated_lambda=float(collection_estimated["scale"]),
        collection_estimated_p=float(collection_estimated["exponent"]),
        collection_relative_residual=collection_residual,
        collection_gradient_cosine=collection_cosine,
        full_train_mse=full_train_mse,
        full_test_mse=full_test_mse,
        replicate_train_mse_mean=float(sum(replicate_train_mses) / len(replicate_train_mses)),
        replicate_test_mse_mean=float(sum(replicate_test_mses) / len(replicate_test_mses)),
        n_replicates=n_replicates,
        sample_fraction=sample_fraction,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=2048)
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
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

    parser.add_argument("--gt-lambda", type=float, default=0.05)
    parser.add_argument("--gt-p", type=float, default=1.5)
    parser.add_argument("--regularizer-epsilon", type=float, default=1e-6)
    parser.add_argument("--auto-balance-gt-lambda", action="store_true", default=False)
    parser.add_argument("--gt-lambda-scale", type=float, default=0.05)

    parser.add_argument("--resample-mode", type=str, default="bootstrap", choices=["subsample", "bootstrap"])
    parser.add_argument("--n-replicates", type=int, default=6)
    parser.add_argument("--sample-fraction", type=float, default=1.0)

    parser.add_argument("--estimation-lr", type=float, default=5e-2)
    parser.add_argument("--estimation-max-epochs", type=int, default=4000)
    parser.add_argument("--estimation-patience", type=int, default=200)
    parser.add_argument("--lambda-init", type=float, default=0.01)
    parser.add_argument("--p-init", type=float, default=2.0)
    parser.add_argument("--p-min", type=float, default=0.5)
    parser.add_argument("--p-max", type=float, default=3.0)

    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("logs/phase2/nonlinear_power_retrain_geometry_summary.json"),
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

    gt_lambda = float(args.gt_lambda)
    if args.auto_balance_gt_lambda:
        reference_model = DeepReLURegressor(
            input_dim=args.input_dim,
            depth=args.depth,
            width=args.width,
            lr=args.lr,
            explicit_bias_types=[],
            explicit_lambdas=[],
        )
        gt_lambda = calibrate_power_lambda(
            predictive_model=reference_model.network,
            exponent=args.gt_p,
            epsilon=args.regularizer_epsilon,
            target_scale=args.gt_lambda_scale,
        )

    summary = run_study(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        input_dim=args.input_dim,
        depth=args.depth,
        width=args.width,
        lr=args.lr,
        gt_lambda=gt_lambda,
        gt_p=args.gt_p,
        regularizer_epsilon=args.regularizer_epsilon,
        max_epochs=args.max_epochs,
        patience=args.patience,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        resample_mode=args.resample_mode,
        n_replicates=args.n_replicates,
        sample_fraction=args.sample_fraction,
        estimation_lr=args.estimation_lr,
        estimation_max_epochs=args.estimation_max_epochs,
        estimation_patience=args.estimation_patience,
        lambda_init=args.lambda_init,
        p_init=args.p_init,
        p_min=args.p_min,
        p_max=args.p_max,
        seed=args.seed,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2)

    LOGGER.info(
        "NonlinearPowerGeometry | teacher_activation=%s depth=%d width=%d resample_mode=%s n_replicates=%d sample_fraction=%.3f true_lambda=%.6f true_p=%.6f full_estimated_lambda=%.6f full_estimated_p=%.6f full_relative_residual=%.6f full_gradient_cosine=%.6f collection_estimated_lambda=%.6f collection_estimated_p=%.6f collection_relative_residual=%.6f collection_gradient_cosine=%.6f full_train_mse=%.6f full_test_mse=%.6f replicate_train_mse_mean=%.6f replicate_test_mse_mean=%.6f",
        args.teacher_activation,
        args.depth,
        args.width,
        args.resample_mode,
        args.n_replicates,
        args.sample_fraction,
        summary.true_lambda,
        summary.true_p,
        summary.full_estimated_lambda,
        summary.full_estimated_p,
        summary.full_relative_residual,
        summary.full_gradient_cosine,
        summary.collection_estimated_lambda,
        summary.collection_estimated_p,
        summary.collection_relative_residual,
        summary.collection_gradient_cosine,
        summary.full_train_mse,
        summary.full_test_mse,
        summary.replicate_train_mse_mean,
        summary.replicate_test_mse_mean,
    )
    LOGGER.info(
        "  Full fit | lambda_true=%.6f lambda_hat=%.6f p_true=%.6f p_hat=%.6f",
        summary.true_lambda,
        summary.full_estimated_lambda,
        summary.true_p,
        summary.full_estimated_p,
    )
    LOGGER.info(
        "  Collection fit | lambda_true=%.6f lambda_hat=%.6f p_true=%.6f p_hat=%.6f",
        summary.true_lambda,
        summary.collection_estimated_lambda,
        summary.true_p,
        summary.collection_estimated_p,
    )
    LOGGER.info("  Saved summary to %s", args.output_json)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
