#!/usr/bin/env python3
"""Retrain-per-resample study for the structured nonlinear teacher-student pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Dict, Sequence

import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping
import numpy as np
from scipy.optimize import nnls
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from function_class_identifiability import (  # noqa: E402
    BiasWithMSENormalized,
    DeepReLURegressor,
    build_bias_module,
    calibrate_explicit_lambdas,
    compute_target_and_bias_gradients,
    cosine_between,
    generate_dataset,
    get_scale_params,
    JointBias,
    parse_csv_floats,
    parse_csv_list,
    set_seed,
)
from core.estimators import BiasWithMSE  # noqa: E402


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrainSummary:
    full_pair_cosine: float
    full_design_condition: float
    full_ols_mean_rel_error: float
    full_ols_max_rel_error: float
    full_train_mse: float
    full_test_mse: float
    replicate_train_mse_mean: float
    replicate_test_mse_mean: float
    single_resample_mean_rel_error: float
    single_resample_max_rel_error: float
    aggregate_mean_rel_error: float
    aggregate_max_rel_error: float
    stacked_mean_rel_error: float
    stacked_max_rel_error: float
    stacked_relative_residual: float
    stacked_design_condition: float
    replicate_pair_cosine_mean: float
    replicate_pair_cosine_std: float


def train_bias_estimator(
    predictive_model: torch.nn.Module,
    train_dataset: TensorDataset,
    bias_types: Sequence[str],
    estimator_mode: str,
    bias_lr: float,
    bias_max_epochs: int,
    bias_patience: int,
    accelerator: str,
) -> Dict[str, float]:
    estimators = [build_bias_module(name, trainable=True) for name in bias_types]
    joint_bias = JointBias(estimators)
    if estimator_mode == "normalized":
        bias_estimator = BiasWithMSENormalized(
            predictive_model=predictive_model.eval(),
            bias_model=joint_bias,
            grad_match_loss_fn=F.mse_loss,
            optimizer_cls=torch.optim.Adam,
            lr=bias_lr,
            bias_lr=bias_lr,
        )
    elif estimator_mode == "standard":
        bias_estimator = BiasWithMSE(
            predictive_model=predictive_model.eval(),
            bias_model=joint_bias,
            grad_match_loss_fn=F.mse_loss,
            optimizer_cls=torch.optim.Adam,
            lr=bias_lr,
        )
    else:
        raise ValueError(f"Unsupported estimator mode: {estimator_mode}")

    batch_size = min(len(train_dataset), 256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    bias_trainer = pl.Trainer(
        max_epochs=bias_max_epochs,
        logger=False,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[EarlyStopping(monitor="train_bias/loss", mode="min", patience=bias_patience)],
        log_every_n_steps=25,
    )
    bias_trainer.fit(bias_estimator, train_dataloaders=train_loader)
    if estimator_mode == "normalized":
        return {name: float(value) for name, value in bias_estimator.get_estimated_lambdas().items()}
    return get_scale_params(joint_bias, bias_types)


def build_design_matrix(
    bias_gradients: Dict[str, Tensor],
    bias_names: Sequence[str],
) -> Tensor:
    return torch.stack([bias_gradients[name] for name in bias_names], dim=1)


def design_condition_number(design: Tensor) -> float:
    gram = design.T @ design
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return float(torch.linalg.cond(gram + 1e-8 * eye).item())


def mean_relative_error(
    estimated: Dict[str, float],
    ground_truth: Dict[str, float],
    bias_names: Sequence[str],
) -> tuple[float, float]:
    total_rel_error = 0.0
    max_rel_error = 0.0
    for bias_name in bias_names:
        true_value = float(ground_truth.get(bias_name, 0.0))
        est_value = float(estimated.get(bias_name, 0.0))
        rel_error = abs(est_value - true_value) / max(abs(true_value), 1e-8)
        total_rel_error += rel_error
        max_rel_error = max(max_rel_error, rel_error)
    return total_rel_error / max(len(bias_names), 1), max_rel_error


def solve_least_squares_lambdas(
    target_gradient: Tensor,
    bias_gradients: Dict[str, Tensor],
    bias_names: Sequence[str],
) -> tuple[Dict[str, float], float]:
    design = build_design_matrix(bias_gradients, bias_names)
    solution = torch.linalg.lstsq(design, target_gradient).solution
    reconstructed = design @ solution
    residual = float(
        (target_gradient - reconstructed).norm().item() / max(target_gradient.norm().item(), 1e-8)
    )
    return (
        {name: float(solution[idx].item()) for idx, name in enumerate(bias_names)},
        residual,
    )


def solve_nonnegative_lambdas(
    target_gradient: Tensor,
    bias_gradients: Dict[str, Tensor],
    bias_names: Sequence[str],
) -> tuple[Dict[str, float], float]:
    design = build_design_matrix(bias_gradients, bias_names)
    solution_np, residual_norm = nnls(
        design.detach().cpu().numpy(),
        target_gradient.detach().cpu().numpy(),
    )
    solution = torch.from_numpy(np.asarray(solution_np)).to(
        device=target_gradient.device,
        dtype=target_gradient.dtype,
    )
    reconstructed = design @ solution
    residual = float(
        (target_gradient - reconstructed).norm().item() / max(target_gradient.norm().item(), 1e-8)
    )
    return (
        {name: float(solution[idx].item()) for idx, name in enumerate(bias_names)},
        residual,
    )


def solve_normalized_nonnegative_lambdas(
    target_gradient: Tensor,
    bias_gradients: Dict[str, Tensor],
    bias_names: Sequence[str],
    eps: float = 1e-8,
) -> tuple[Dict[str, float], float]:
    design = build_design_matrix(bias_gradients, bias_names)
    column_norms = design.norm(dim=0).clamp_min(eps)
    normalized_design = design / column_norms.unsqueeze(0)
    solution_np, _ = nnls(
        normalized_design.detach().cpu().numpy(),
        target_gradient.detach().cpu().numpy(),
    )
    normalized_solution = torch.from_numpy(np.asarray(solution_np)).to(
        device=target_gradient.device,
        dtype=target_gradient.dtype,
    )
    raw_solution = normalized_solution / column_norms
    reconstructed = design @ raw_solution
    residual = float(
        (target_gradient - reconstructed).norm().item() / max(target_gradient.norm().item(), 1e-8)
    )
    return (
        {name: float(raw_solution[idx].item()) for idx, name in enumerate(bias_names)},
        residual,
    )


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
    inputs, targets = dataset.tensors
    return TensorDataset(inputs[indices], targets[indices])


def average_lambda_dicts(
    lambda_dicts: Sequence[Dict[str, float]],
    bias_names: Sequence[str],
) -> Dict[str, float]:
    return {
        name: float(sum(entry.get(name, 0.0) for entry in lambda_dicts) / len(lambda_dicts))
        for name in bias_names
    }


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    tensor = torch.tensor(list(values), dtype=torch.float32)
    return float(tensor.std(unbiased=True).item())


def evaluate_split_metrics(model: DeepReLURegressor, dataset: TensorDataset) -> tuple[float, float]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        inputs, targets = dataset.tensors
        predictions = model(inputs.to(device))
        mse_loss = F.mse_loss(predictions, targets.to(device))
        regularization, _ = model._explicit_regularization()
        total_loss = mse_loss + regularization
    return float(mse_loss.item()), float(total_loss.item())


def train_model(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    input_dim: int,
    depth: int,
    width: int,
    lr: float,
    explicit_bias_types: Sequence[str],
    explicit_lambdas: Sequence[float],
    max_epochs: int,
    patience: int,
    accelerator: str,
    seed: int,
) -> DeepReLURegressor:
    set_seed(seed)
    batch_size = min(len(train_dataset), 256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=min(len(val_dataset), 256), shuffle=False, num_workers=0)
    model = DeepReLURegressor(
        input_dim=input_dim,
        depth=depth,
        width=width,
        lr=lr,
        explicit_bias_types=explicit_bias_types,
        explicit_lambdas=explicit_lambdas,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=False,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[EarlyStopping(monitor="val/loss", mode="min", patience=patience)],
        log_every_n_steps=25,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    return model


def run_retrain_study(
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    test_dataset: TensorDataset,
    input_dim: int,
    depth: int,
    width: int,
    lr: float,
    explicit_bias_types: Sequence[str],
    explicit_lambdas: Sequence[float],
    ground_truth: Dict[str, float],
    max_epochs: int,
    patience: int,
    accelerator: str,
    resample_mode: str,
    n_replicates: int,
    sample_fraction: float,
    estimator_mode: str,
    bias_lr: float,
    bias_max_epochs: int,
    bias_patience: int,
    seed: int,
) -> RetrainSummary:
    full_model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=input_dim,
        depth=depth,
        width=width,
        lr=lr,
        explicit_bias_types=explicit_bias_types,
        explicit_lambdas=explicit_lambdas,
        max_epochs=max_epochs,
        patience=patience,
        accelerator=accelerator,
        seed=seed,
    )
    full_train_mse, _ = evaluate_split_metrics(full_model, train_dataset)
    full_test_mse, _ = evaluate_split_metrics(full_model, test_dataset)
    full_target_gradient, full_bias_gradients, _ = compute_target_and_bias_gradients(
        full_model.network.eval(),
        train_dataset.tensors,
        explicit_bias_types,
    )
    full_estimated = train_bias_estimator(
        predictive_model=full_model.network,
        train_dataset=train_dataset,
        bias_types=explicit_bias_types,
        estimator_mode=estimator_mode,
        bias_lr=bias_lr,
        bias_max_epochs=bias_max_epochs,
        bias_patience=bias_patience,
        accelerator=accelerator,
    )
    full_ols_mean_rel_error, full_ols_max_rel_error = mean_relative_error(
        full_estimated,
        ground_truth,
        explicit_bias_types,
    )

    per_replicate_estimated: list[Dict[str, float]] = []
    stacked_targets: list[Tensor] = []
    stacked_designs: list[Tensor] = []
    replicate_pair_cosines: list[float] = []
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
            explicit_bias_types=explicit_bias_types,
            explicit_lambdas=explicit_lambdas,
            max_epochs=max_epochs,
            patience=patience,
            accelerator=accelerator,
            seed=replicate_seed,
        )
        train_mse, _ = evaluate_split_metrics(replicate_model, replicate_train_dataset)
        test_mse, _ = evaluate_split_metrics(replicate_model, test_dataset)
        replicate_train_mses.append(train_mse)
        replicate_test_mses.append(test_mse)
        target_gradient, bias_gradients, _ = compute_target_and_bias_gradients(
            replicate_model.network.eval(),
            replicate_train_dataset.tensors,
            explicit_bias_types,
        )
        estimated = train_bias_estimator(
            predictive_model=replicate_model.network,
            train_dataset=replicate_train_dataset,
            bias_types=explicit_bias_types,
            estimator_mode=estimator_mode,
            bias_lr=bias_lr,
            bias_max_epochs=bias_max_epochs,
            bias_patience=bias_patience,
            accelerator=accelerator,
        )
        per_replicate_estimated.append(estimated)
        stacked_targets.append(target_gradient)
        stacked_designs.append(build_design_matrix(bias_gradients, explicit_bias_types))
        replicate_pair_cosines.append(abs(cosine_between(bias_gradients[explicit_bias_types[0]], bias_gradients[explicit_bias_types[1]])))

    single_errors = [mean_relative_error(estimated, ground_truth, explicit_bias_types) for estimated in per_replicate_estimated]
    single_mean_rel_error = float(sum(item[0] for item in single_errors) / len(single_errors))
    single_max_rel_error = float(sum(item[1] for item in single_errors) / len(single_errors))
    aggregate_estimated = average_lambda_dicts(per_replicate_estimated, explicit_bias_types)
    aggregate_mean_rel_error, aggregate_max_rel_error = mean_relative_error(
        aggregate_estimated,
        ground_truth,
        explicit_bias_types,
    )
    stacked_design = torch.cat(stacked_designs, dim=0)
    stacked_target = torch.cat(stacked_targets, dim=0)
    stacked_relative_residual = float("nan")
    stacked_mean_rel_error = float("nan")
    stacked_max_rel_error = float("nan")

    return RetrainSummary(
        full_pair_cosine=abs(cosine_between(full_bias_gradients[explicit_bias_types[0]], full_bias_gradients[explicit_bias_types[1]])),
        full_design_condition=design_condition_number(build_design_matrix(full_bias_gradients, explicit_bias_types)),
        full_ols_mean_rel_error=full_ols_mean_rel_error,
        full_ols_max_rel_error=full_ols_max_rel_error,
        full_train_mse=full_train_mse,
        full_test_mse=full_test_mse,
        replicate_train_mse_mean=float(sum(replicate_train_mses) / len(replicate_train_mses)),
        replicate_test_mse_mean=float(sum(replicate_test_mses) / len(replicate_test_mses)),
        single_resample_mean_rel_error=single_mean_rel_error,
        single_resample_max_rel_error=single_max_rel_error,
        aggregate_mean_rel_error=aggregate_mean_rel_error,
        aggregate_max_rel_error=aggregate_max_rel_error,
        stacked_mean_rel_error=stacked_mean_rel_error,
        stacked_max_rel_error=stacked_max_rel_error,
        stacked_relative_residual=stacked_relative_residual,
        stacked_design_condition=design_condition_number(stacked_design),
        replicate_pair_cosine_mean=float(sum(replicate_pair_cosines) / len(replicate_pair_cosines)),
        replicate_pair_cosine_std=std(replicate_pair_cosines),
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
    parser.add_argument("--gt-bias-types", type=str, default="ridge,nuclear_norm")
    parser.add_argument("--gt-lambdas", type=str, default="0.05,0.05")
    parser.add_argument("--auto-balance-gt-lambdas", action="store_true", default=False)
    parser.add_argument("--gt-lambda-scale", type=float, default=0.05)
    parser.add_argument("--resample-mode", type=str, default="bootstrap", choices=["subsample", "bootstrap"])
    parser.add_argument("--n-replicates", type=int, default=6)
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--estimator-mode", type=str, default="normalized", choices=["standard", "normalized"])
    parser.add_argument("--bias-lr", type=float, default=0.05)
    parser.add_argument("--bias-max-epochs", type=int, default=2000)
    parser.add_argument("--bias-patience", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    gt_bias_types = parse_csv_list(args.gt_bias_types)
    gt_lambdas = parse_csv_floats(args.gt_lambdas)
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

    if args.auto_balance_gt_lambdas:
        reference_model = DeepReLURegressor(
            input_dim=args.input_dim,
            depth=args.depth,
            width=args.width,
            lr=args.lr,
            explicit_bias_types=gt_bias_types,
            explicit_lambdas=gt_lambdas,
        )
        gt_lambdas = calibrate_explicit_lambdas(reference_model.network, gt_bias_types, args.gt_lambda_scale)
    gt_map = {name: value for name, value in zip(gt_bias_types, gt_lambdas)}

    summary = run_retrain_study(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        input_dim=args.input_dim,
        depth=args.depth,
        width=args.width,
        lr=args.lr,
        explicit_bias_types=gt_bias_types,
        explicit_lambdas=gt_lambdas,
        ground_truth=gt_map,
        max_epochs=args.max_epochs,
        patience=args.patience,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        resample_mode=args.resample_mode,
        n_replicates=args.n_replicates,
        sample_fraction=args.sample_fraction,
        estimator_mode=args.estimator_mode,
        bias_lr=args.bias_lr,
        bias_max_epochs=args.bias_max_epochs,
        bias_patience=args.bias_patience,
        seed=args.seed,
    )

    LOGGER.info(
        "NonlinearRetrain | teacher_activation=%s depth=%d width=%d teacher_spectrum=%s gradient_dataset=train resample_mode=%s estimator_mode=%s full_cos=%.6f full_cond=%.6f full_ols_mean_rel_error=%.6f single_mean_rel_error=%.6f aggregate_mean_rel_error=%.6f stacked_mean_rel_error=%.6f stacked_relative_residual=%.6f replicate_cos_mean=%.6f replicate_cos_std=%.6f full_train_mse=%.6f full_test_mse=%.6f replicate_train_mse_mean=%.6f replicate_test_mse_mean=%.6f",
        args.teacher_activation,
        args.depth,
        args.width,
        args.teacher_spectrum,
        args.resample_mode,
        args.estimator_mode,
        summary.full_pair_cosine,
        summary.full_design_condition,
        summary.full_ols_mean_rel_error,
        summary.single_resample_mean_rel_error,
        summary.aggregate_mean_rel_error,
        summary.stacked_mean_rel_error,
        summary.stacked_relative_residual,
        summary.replicate_pair_cosine_mean,
        summary.replicate_pair_cosine_std,
        summary.full_train_mse,
        summary.full_test_mse,
        summary.replicate_train_mse_mean,
        summary.replicate_test_mse_mean,
    )
    for bias_name in gt_bias_types:
        LOGGER.info("  %s | true=%.6f", bias_name, gt_map[bias_name])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
